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

## Paper and source data

- [Manuscript PDF](J_Cheminform_submission/manuscript.pdf)
- [Readable manuscript](J_Cheminform_submission/manuscript.md)
- [Supplementary PDF](J_Cheminform_submission/additional_files/Additional_file_1_supplement.pdf)
- [Supplementary source](J_Cheminform_submission/additional_files/Additional_file_1_supplement.md)
- [Architecture diagram](J_Cheminform_submission/figures/Figure_1_architecture.svg)
- [FAK1-P4N iBAM and bond-map case](J_Cheminform_submission/figures/Figure_6_FAK1_case.pdf)
- [Release contents and availability](J_Cheminform_submission/README.md)
- [Reference API and usage](reference/README.md)

The current draft is prepared for Journal of Cheminformatics. Earlier journal
versions remain in repository history. Selected result tables accompany the
paper; complete model/data deposits and final author approval remain separate
submission steps.

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
