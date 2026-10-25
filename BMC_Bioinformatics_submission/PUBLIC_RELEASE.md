# Minimal manuscript and reference release

Read `manuscript_revised.pdf` for the current article draft.

This public snapshot contains a research draft, its five main figure PDFs,
editable main tables, supplementary narrative/PDF, diagnostic summaries, and
the compact `../reference/` implementation. It is not a journal submission.

The manuscript is approximately 10% briefer and centers on pooled fusion,
interaction attention, generalization, and explanation controls. Historical
Davis test exposure remains prominent. The new KIBA benchmark and additional
Davis validation folds are **in progress**. Their results have not been filled
in or inferred from published baselines.

Compile the main paper from this directory with:

```sh
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

All required main table inputs are included. Main figures are uploaded
separately, as described in the figure legends. The supplementary PDF is
self-contained; its editable narrative is Markdown. The larger local
submission folder also contains supplementary LaTeX, source data, SVG and
600-dpi PNG/TIFF alternatives, archives, and QA records. Those large assets,
encoder features, model weights, and third-party datasets are not represented
as publicly deposited by this limited commit. Persistent full-asset deposit
and author declarations remain submission requirements.

The reference runner requires the separately archived frozen-feature assets
and official benchmark data. See its README for the feature contract,
dependency requirements, local baseline source pins, leakage audit, selection
rules, and test gate. `reference/registration.json` records the actual frozen
protocol and implementation hashes before new KIBA test scoring.
