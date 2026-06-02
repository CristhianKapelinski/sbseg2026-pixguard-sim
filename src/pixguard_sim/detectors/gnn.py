"""Graph-neural-network baseline detector (GraphSAGE-style).

The conventional baselines (logistic regression, random forest, gradient
boosting, XGBoost) score each event from its own features only. A graph
neural network is the natural baseline for multi-hop laundering and MED-2.0
refund chains because it propagates a node's signal to its neighbours, so a
mule deep in a chain can inherit risk from upstream accounts.

This module implements a two-layer mean-aggregation graph convolution
(GraphSAGE-style) in plain PyTorch over the account-interaction graph induced
by the event stream. It runs on the available GPU when one is present and
falls back to CPU otherwise; the device is detected, never hardcoded. The
model is wrapped behind the same :class:`~pixguard_sim.detectors.base.Detector`
protocol as every other baseline, so the harness scores it on identical
inputs.

Scaling a GNN to production-volume graphs (mini-batch neighbour sampling,
multi-relational edges) is a research direction in its own right and is left to
future work; here the GNN serves as a representative graph-aware baseline on
the benchmark graphs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pixguard_sim.detectors.base import Detector
from pixguard_sim.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def torch_available() -> bool:
    """Return whether PyTorch is importable (the GNN's only heavy dependency)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class GraphSageDetector(Detector):
    """A two-layer GraphSAGE-style fraud detector over the payment graph.

    The node set is the union of payer and payee accounts in the frame. Node
    features are the mean of the schema feature columns over the events that
    touch each account, so an account's representation summarises its activity.
    Edges connect payer to payee per event. Training is transductive over the
    training events' accounts; scoring an evaluation frame builds the induced
    graph for those events and reads off the per-event payee node score.
    """

    name = "gnn_sage"

    def __init__(
        self,
        seed: int,
        inference_budget_ms: int = 50,
        hidden_dim: int = 32,
        epochs: int = 60,
        lr: float = 0.01,
        name: str = "gnn_sage",
    ) -> None:
        """Initialise the GNN detector.

        Args:
            seed: RNG seed for reproducible weight init and training.
            inference_budget_ms: Decision latency on the relative timeline.
            hidden_dim: Hidden layer width.
            epochs: Training epochs (full-batch over the induced graph).
            lr: Adam learning rate.
            name: Detector name used in reports.
        """
        self.seed = seed
        self.inference_budget_ms = inference_budget_ms
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.name = name
        self._model = None
        self._device = None
        self._feat_mean = None
        self._feat_std = None

    # -- graph construction -------------------------------------------------

    def _build_graph(self, frame: pd.DataFrame):
        """Build node features, a symmetric adjacency, and a per-event payee map.

        Returns:
            ``(x, adj_norm, node_index, payee_node_of_event)`` as torch tensors
            on the detector's device, where ``payee_node_of_event[i]`` is the
            node id whose score is read off for event ``i``.
        """
        import torch

        accounts = pd.unique(
            pd.concat([frame["payer_account"], frame["payee_account"]],
                      ignore_index=True)
        )
        node_index = {int(a): i for i, a in enumerate(accounts)}
        n_nodes = len(node_index)

        feats = frame[list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
        payer = frame["payer_account"].map(node_index).to_numpy()
        payee = frame["payee_account"].map(node_index).to_numpy()

        # Node feature = mean of incident-event features (payer or payee side).
        acc = np.zeros((n_nodes, feats.shape[1]), dtype="float64")
        cnt = np.zeros(n_nodes, dtype="float64")
        for side in (payer, payee):
            np.add.at(acc, side, feats)
            np.add.at(cnt, side, 1.0)
        cnt[cnt == 0] = 1.0
        node_feat = acc / cnt[:, None]

        # Standardise features with train-time statistics.
        if self._feat_mean is None:
            self._feat_mean = node_feat.mean(axis=0)
            self._feat_std = node_feat.std(axis=0) + 1e-6
        node_feat = (node_feat - self._feat_mean) / self._feat_std

        # Symmetric, self-looped, degree-normalised adjacency (GCN-style mean agg).
        src = np.concatenate([payer, payee, np.arange(n_nodes)])
        dst = np.concatenate([payee, payer, np.arange(n_nodes)])
        idx = torch.tensor(np.stack([src, dst]), dtype=torch.long,
                           device=self._device)
        vals = torch.ones(idx.shape[1], device=self._device)
        adj = torch.sparse_coo_tensor(idx, vals, (n_nodes, n_nodes)).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        dinv = (1.0 / deg).unsqueeze(1)

        x = torch.tensor(node_feat, dtype=torch.float32, device=self._device)
        payee_t = torch.tensor(payee, dtype=torch.long, device=self._device)
        return x, adj, dinv, payee_t

    # -- detector protocol --------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> GraphSageDetector:
        """Train the GNN transductively on the training events' graph."""
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("gnn device=%s", self._device)

        x, adj, dinv, payee = self._build_graph(frame)
        y_event = torch.tensor(
            frame["is_fraud"].to_numpy(), dtype=torch.float32, device=self._device
        )
        in_dim = x.shape[1]
        model = _SageNet(in_dim, self.hidden_dim).to(self._device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Class weight to counter heavy imbalance.
        pos = float(y_event.sum().item())
        neg = float(len(y_event) - pos)
        pos_weight = torch.tensor(
            [neg / max(pos, 1.0)], device=self._device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            node_logits = model(x, adj, dinv)
            event_logits = node_logits[payee]
            loss = loss_fn(event_logits, y_event)
            loss.backward()
            opt.step()
        self._model = model
        logger.info("gnn trained: final_loss=%.4f nodes=%d", float(loss.item()),
                    x.shape[0])
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Score events by the GNN probability of their payee node."""
        import torch

        if self._model is None:
            raise RuntimeError("detector not fitted")
        x, adj, dinv, payee = self._build_graph(frame)
        self._model.eval()
        with torch.no_grad():
            node_logits = self._model(x, adj, dinv)
            probs = torch.sigmoid(node_logits[payee])
        return probs.detach().cpu().numpy().astype("float64")


def _make_sagenet(in_dim: int, hidden_dim: int):
    """Construct the underlying network (deferred import of torch)."""
    return _SageNet(in_dim, hidden_dim)


try:  # Define the nn.Module only when torch is present.
    import torch
    from torch import nn

    class _SageNet(nn.Module):
        """Two mean-aggregation graph-convolution layers plus a node head."""

        def __init__(self, in_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.lin1 = nn.Linear(in_dim * 2, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)
            self.act = nn.ReLU()

        def _agg(self, h, adj, dinv):
            """Degree-normalised neighbour mean aggregation."""
            return torch.sparse.mm(adj, h) * dinv

        def forward(self, x, adj, dinv):
            h = self.act(self.lin1(torch.cat([x, self._agg(x, adj, dinv)], dim=1)))
            h = self.act(self.lin2(torch.cat([h, self._agg(h, adj, dinv)], dim=1)))
            return self.head(h).squeeze(1)

except ImportError:  # pragma: no cover - torch is an optional extra
    _SageNet = None  # type: ignore[assignment, misc]
