# BINDRA draft for BMC Bioinformatics

Start with `manuscript_revised.pdf` for reading or `manuscript.md` for prose editing.
`manuscript.tex` is the editable journal-submission source. This folder is a
scientific manuscript draft, not a submitted article or a claim of author approval.

The draft uses American English and no em dashes in authored prose. It keeps
DTA performance primary, preserves the validation-selected pooled BINDRA model
and no-FiLM BINDRA-iBAM companion, and retains mixed and negative findings.
Author names, affiliation, correspondence, funding, and competing interests
come from the information provided by the corresponding project owner.

## Contents

- `manuscript_revised.pdf`, `manuscript.tex`, `manuscript.md`: main article with references, diagnostic tables and figure legends.
- `figures/`: five main figures in PDF/SVG and 600-dpi PNG/lossless LZW TIFF; six selected supplementary PDF figures.
- `tables/`: five main and three supplementary CSV/LaTeX tables and full source-linked machine-readable results.
- `additional_files/`: revised supplementary methods PDF/source and four numbered ZIP uploads containing full tables, frozen predictions, all generator draws, and the complete figure collection.
- `references.bib`, `references/`: editable bibliography and primary-source metadata.
- `cover_letter_draft.md`, `submission_checklist.md`, `author_information.json`: author-facing submission materials and unresolved confirmations.
- `provenance/`: original model/data identities, numeric review, evidence inventory, source snapshot and manuscript source mappings.
- `source/`: reproducible preparation scripts and manuscript prose source.
- `qa/`: compilation logs, page renders and final review records; these are internal checks, not submission uploads.

## Rebuild

From this folder, using the original project environment:

```powershell
..\.venv\Scripts\python.exe source/prepare_data.py
..\.venv\Scripts\python.exe source/prepare_references.py
..\.venv\Scripts\python.exe source/revision_diagnostics.py
..\.venv\Scripts\python.exe source/build_manuscript.py
..\.venv\Scripts\python.exe source/revision_figures.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa Additional_file_1_supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa Additional_file_1_supplement.tex
```

The compiled files are initially written under `qa/`. The final review script
copies checked PDFs to their delivery locations and records their hashes.
Plain LaTeX compilation does not require Python or the original model weights.
Data preparation requires the canonical local evidence tree and does not run
training, inference, sampling, or docking. It does not modify frozen artifacts.

Main figures preserve the original vector sources and embedded fonts. The
600-dpi raster versions preserve their physical dimensions; TIFF compression
is lossless. At final production width, confirm readability with the journal.
No table is flattened into an image.

## Reviewer revision

The main prose is 6062 words versus 6711 previously, a 9.7% reduction using
the same source-body counter. The paper centers on pooled versus attention
prediction. Main tables include crossed-bootstrap intervals, paired contrasts,
detection-plateau diagnostics, within-target ranking, and absolute attention
controls. Generation, docking, the glossary, and development details are
supplementary. The historical Davis test-exposure disclosure remains prominent.

The independent KIBA reruns and two additional Davis validation folds are
**in progress**, not completed results. See `../reference/README.md` and the
locked protocol. This draft is not ready to submit until those results and the
author actions are resolved. The minimal GitHub release contains manuscript
text, editable tables, references, and the reference implementation. Large
figures, model/data assets, and local QA records require separate deposits.

## Before submission

Resolve the clearly marked author contribution, funder-role, acknowledgment,
approval/originality and data-deposit fields. Deposit the full reproducibility
assets, including large raw maps, SDFs and checkpoints, in an appropriate
repository with access/license conditions. The supplied local inventory is not
a public DOI. Confirm all authors' approval and follow the final checklist.
The current journal guidance is linked in `submission_checklist.md`.

## Reduced execution budget

The active protocol now schedules 36 neural fits rather than 212: 24 KIBA fits
and 12 fixed-configuration Davis confirmations. KIBA uses three shared learning
rates, dropout 0.1, and one validation fold; three final seeds remain. The two
leading Davis models retain their existing selected settings across two extra
folds. The public runtime amendment discloses initial validation exposure and
preserves the ongoing fit. There is no full eight-arm stability claim and no
new generation/docking campaign.
