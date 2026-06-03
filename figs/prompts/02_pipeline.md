# Image 2 — PixGuard-Sim testbed pipeline (icon infographic)

- **Save the PNG as:** `paper/figs/pipeline_pixguard.png`
- **Aspect ratio:** 16:9
- **Where it goes:** the "Method / Evaluation" section (system figure)
- **Background the AI needs:** this is an evaluation testbed — fraud event data is
  produced by generators, fed to any fraud detector, and scored by a harness that
  measures both accuracy and whether the decision lands within a time deadline.
- **Note:** carries short labels; if garbled, regenerate or fix text in an editor.

```prompt
Flat vector infographic, horizontal left-to-right pipeline on a pure white
background, clean academic style, thin monoline icons joined by thin arrows with
small triangular arrowheads, four stages.
STAGE 1 (left), a vertical group of three small source icons stacked together,
each a little factory/gear "generator": an in-house Pix generator (a Brazilian
instant-payment "Pix" style spark/lightning coin icon), and two external dataset
cylinders; bracket them with a brace; caption: "Generators (in-house + 2 public)".
ARROW into STAGE 2: a spreadsheet/stream icon showing rows of events with four tiny
column glyphs (a money amount, a phone/device, a new-payee person, a speedometer
for velocity); caption: "Labeled event stream (4 features)".
ARROW into STAGE 3: a swappable "detector" slot drawn as a socket holding
interchangeable chips — show a small tree (random forest), a line/curve (logistic),
a graph of nodes (GNN), and a speech-bubble brain (LLM) hovering as options that
plug into the socket; caption: "Any detector".
ARROW into STAGE 4: a measuring instrument / dashboard box combining a gauge AND a
stopwatch, signifying accuracy measured together with decision latency against a
deadline; caption: "Deadline-aware harness".
ARROW into final OUTPUT card: a small report with a bar-chart, a checkmark, and
little error-bar whiskers (confidence intervals); caption: "Report: PR-AUC,
recall, pre-deadline fraction, 95% CIs".
Palette: deep navy #1F3A5F for lines, icons and captions; teal #2A9D8F accent on
the harness gauge+stopwatch; amber #E9C46A on the detector socket; light gray
fills; white background; rounded corners; subtle shadows; clean sans-serif for the
short captions only. 16:9 aspect ratio, evenly spaced stages, strong horizontal
flow, generous white space. Only the short captions named above — no sentences.
```
