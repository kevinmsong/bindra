# Revision record

The manuscript now focuses on controlled architectural evaluation. This is a
revised research draft, not a completed independent-validation report.

| Request | Implemented revision | Remaining evidence |
|---|---|---|
| Independent benchmark and controlled reruns | Frozen official KIBA protocol; fresh supervised BINDRA models; pinned official GraphDTA-GIN and MGraphDTA; ridge/random forest; shared partitions, validation-only selection, single/ensemble reporting; separate pretraining and supervised exposure audit | Training and test evaluation are running. No new benchmark metrics are claimed. |
| Reduce generation emphasis | New architectural title and abstract; concise negative findings in main text; detailed generation/docking methods, tables, figures, and failures in supplement | No claim of useful target-conditioned design or experimental binding. |
| Promote decisive uncertainty and diagnostics | Main Tables 2-5 include individual-model variation, ensemble intervals, paired MSE/CI contrasts, plateau errors, within-target ranking, absolute attention AP, prevalence, permutation, and gradient controls | Two additional Davis validation folds confirm the two leading fixed configurations with three seeds; exhaustive eight-arm retuning was removed in a disclosed runtime amendment. |
| Scientific narrative and consistent terminology | BINDRA and BINDRA-iBAM used consistently; historical test-exposure disclosure retained; glossary and development records moved to supplement; attention-contact correspondence replaces validation language | Author declarations and persistent full-data/model deposits remain unresolved. |

Main source prose decreased from 6711 to 6062 words, approximately 9.7%, using
the same counter before references and expanded tables. The structured
abstract contains 244 words. Main figures are editable PDF/SVG with 600-dpi
PNG/TIFF alternatives in the local submission package.

The KIBA chemical audit covers 2068 ligands against the actual 649364-molecule
ligand-pretraining corpus: 2067 have no normalized identity, parent, or scaffold
overlap; one unresolved chemical mapping remains unknown. Protein-language
model pretraining exposure remains unknown. KIBA is an independent dataset,
not a claim that public labels were historically inaccessible.

The public `reference/` implementation was tested against the existing GitHub
base without the uncommitted project changes. The limited release excludes
weights, datasets, raw chemistry, large run archives, and internal QA logs.
Its benchmark and additional-fold results must be incorporated before journal
submission. Existing frozen research checkpoints and predictions are preserved.

Runtime-budget amendment: at the user's request, the queue was reduced from
212 to 36 neural fits and 40 to 16 classical fits after initial validation
inspection and before any KIBA test scoring. All neural methods retain the
same reduced tuning budget and three final seeds. Two-model Davis confirmation
replaces full eight-arm foldwise retuning; no broader stability claim is made.
