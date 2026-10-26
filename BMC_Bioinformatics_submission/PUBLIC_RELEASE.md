# Davis-only manuscript and reference release

`manuscript_revised.pdf` is the current BMC Bioinformatics article draft. It reports completed Davis warm/cold architecture comparisons and supporting attention and case studies. No independent affinity benchmark, local external-model rerun, or additional validation-fold confirmation is claimed.

The public reference code supplies reusable research utilities. The presence of a runner, source pins, or an archived protocol does not establish that an experiment was completed. Unfinished independent-benchmark training was discontinued on October 25, 2026; its partial results are excluded from this manuscript.

The manuscript uses five main figure PDFs, editable tables, numbered references, supplementary methods, and source-linked results. Large model weights, frozen encoder assets, raw chemistry and per-head matrices require separate persistent deposits and redistribution terms. Local paths do not constitute public availability. Author declarations and final approval remain unresolved.

Compile from this directory with `pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex` twice, with the included table files. Supplementary LaTeX also requires its included figure PDFs. Full rebuild and review instructions are in `README.md`.
