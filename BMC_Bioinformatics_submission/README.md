# BINDRA manuscript draft for BMC Bioinformatics

Start with `manuscript_revised.pdf`. The selected title is **BINDRA: evaluating
pooled fusion and interaction attention for protein-ligand affinity prediction**.

This minimal public snapshot includes the manuscript PDF, editable Markdown and
LaTeX, bibliography, main table sources, six main figure PDF/SVG pairs, and
supplementary methods with their table and figure dependencies. The compact
[reference implementation](../reference/README.md) is provided separately.

## Study scope

Affinity benchmarking is restricted to completed Davis warm, cold-drug, and
cold-protein experiments. The validation-selected pooled BINDRA ensemble and
retained no-FiLM BINDRA-iBAM ensemble each contain three models. Historical Davis
test inspection, selection on one validation fold, and literature-only external
comparisons remain explicit. Unfinished independent-benchmark and additional-fold
training was discontinued on October 25, 2026 and is excluded from the paper.

Completed attention-contact, bond-map, screening, generation, and docking analyses
retain their negative findings. Generation and docking details are supplementary.
BindingDB supplies ligand pretraining and supporting molecular edits; it is not
an additional reported affinity benchmark.

## Compile the included document sources

From this directory, with a LaTeX installation providing the packages named in
the document preambles:

```sh
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error Additional_file_1_supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error Additional_file_1_supplement.tex
```

The article PDF embeds all six main figures with their captions, including the
FAK1-P4N iBAM and bond-map case study. Figure PDF/SVG files are also provided
separately. Main and supplementary LaTeX require the included figure PDFs. Compilation does not run
training, sampling, docking, or new analyses. American English is used in authored
prose, with no em dashes. The main manuscript includes a compact glossary.

The small `tables/fak_case_*.csv` exports contain both attention directions,
the physical bond map, bond weights, and construct/PDB residue mappings.
`provenance/fak_case_sources.json` records source identities and display rules.
`source/render_fak_case.py` renders from the separately archived frozen attention
arrays and requires NumPy, pandas, scikit-learn, RDKit, Matplotlib, and Pillow.
It performs no model inference or training.

## Assets and author review

The full local research package also contains the editable Word manuscript,
600-dpi PNG/TIFF figures, frozen predictions, all generated draws and rejection
records, numeric attention matrices, source-data mappings, checkpoints, and
render-review records. Those bulk assets and third-party encoder/data files are
not included in this minimal commit. Persistent repository deposits and access
or redistribution terms remain necessary; local files are not a public deposit.

Author contribution roles, funder involvement, acknowledgments, final approval,
and originality/exclusive-submission declarations remain to be completed.
This is a manuscript draft for author review, not a submitted article.
