# BINDRA

**Binding INference Design Retrieval and Attention**

BINDRA compares pooled protein-ligand fusion with interaction attention for
binding-affinity prediction. The [reference implementation](reference/) provides
the downstream architectures, frozen-feature interface, metric utilities,
validation-only training, and ordered three-member prediction aggregation.

## Davis results

| Model | Role | Test MSE | Test CI |
|---|---|---:|---:|
| BINDRA | Validation-selected pooled ensemble | 0.19663 | 0.90802 |
| BINDRA-iBAM | Retained attention ensemble without FiLM | 0.19586 | 0.91253 |

Each ensemble averages pKd predictions from seeds 20261001, 20261002, and 20261003
on the official 5,010-pair warm test. Selection used one fixed validation fold.
Earlier Davis test inspection makes the standard-test comparison retrospective.
Published external results provide protocol-qualified literature context.

iBAM means **Interpretable Binding Attention Map**. Residue and bond attention,
structural controls, retrospective screening, molecular editing, and docking are
supporting analyses described in the manuscript and supplement.

## Source release

This repository contains the reference implementation, tests, and lightweight
protocol records. Manuscript and supplementary documents, compiled PDFs, and
submission artwork are maintained separately in the local authoring package.

The completed study uses Davis for affinity evaluation. Encoder weights, trained
model checkpoints, feature stores, raw data, and large generated-draw archives are
separate assets with their own access and redistribution terms.

## Run the reference checks

Use Python 3.12 from the repository root:

```sh
python -m pip install -r reference/requirements.txt
python -m pytest reference/tests -q
```

The synthetic checks require neither encoder weights nor benchmark downloads.
An official-data check is skipped when those assets are absent. The reference
code does not launch a study queue on import. The [ensemble helper](reference/ensemble.py)
preserves observation order, validates the three members and pKd scale, and
returns member predictions, arithmetic mean pKd, and descriptive sample SD.

Full experiments require original frozen encoder assets, compatible feature
caches, and the declared data partitions. Checkpoints, raw data, feature stores,
and large generated-draw archives are separate assets, as detailed in the paper.
Archived KIBA and additional-fold runners are development history, not completed
comparisons reported by the manuscript.

Code is distributed under the [MIT license](LICENSE). Third-party data and encoder
weights retain their own terms. Earlier DeepDTA-iBAM work is available in its
[archival repository](https://github.com/kevinmsong/DeepDTA-iBAM).
