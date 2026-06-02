# Pinned, lean image for reproducing the in-repo PixGuard-Sim experiments.
# The cross-generator experiments (E3/E5/E6) need the public datasets mounted at
# --data-dir; the GraphSAGE baseline (E7) wants a CUDA runtime, so both are run
# outside this lean image. Build:
#   docker build -t pixguard-sim .
# Run the main-claim reproduction (writes to a mounted results/ + logs/):
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/logs:/app/logs" \
#       pixguard-sim run --experiments E1 E2 E4
FROM python:3.12-slim

# Avoid interactive prompts and keep the image lean.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only the runtime dependency set (pinned in pyproject) plus the
# package itself. Dev extras are not installed in the image.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

# Default config and experiment directories are created at runtime; mount
# results/ and logs/ to persist outputs on the host.
ENTRYPOINT ["pixguard-sim"]
CMD ["run", "--experiments", "E1", "E2", "E4"]
