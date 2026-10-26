# BINDRA

**Binding INference Design Retrieval and Attention**

BINDRA compares pooled protein-ligand fusion with interaction attention for
binding-affinity prediction. The completed manuscript study uses Davis, with
DTA prediction as its primary outcome. The compact [reference implementation](reference/)
provides the downstream architectures, metric utilities, frozen-feature interface,
and validation-only training code.

## Reported Davis results

| Model | Role | Test MSE | Test CI |
|---|---|---:|---:|
| BINDRA | Validation-selected pooled ensemble | 0.19663 | 0.90802 |
| BINDRA-iBAM | Retained attention ensemble without FiLM | 0.19586 | 0.91253 |

Each result averages pKd predictions from three independently trained models on
the official 5,010-pair warm test. Architecture selection used one fixed validation
fold across three seeds. Earlier Davis test inspection limits confirmatory claims.
Literature values are author-reported comparisons, not controlled local reruns.
These results do not establish universal state-of-the-art performance.

iBAM means **Interpretable Binding Attention Map**. Attention-contact analyses,
bond-map controls, retrospective screening, and an independently trained diffusion
generator are supporting studies. Generation recovered no measured analogs in
9,000 draws across the strict five-query, three-series Kd subset. Attention and
docking results do not establish causal binding mechanisms or experimental binding.

## Manuscript and reference code

- [Manuscript PDF](BMC_Bioinformatics_submission/manuscript_revised.pdf)
- [Readable manuscript source](BMC_Bioinformatics_submission/manuscript.md)
- [Supplementary methods and analyses](BMC_Bioinformatics_submission/additional_files/Additional_file_1_supplement_revised.pdf)
- [Architecture diagram](BMC_Bioinformatics_submission/figures/Figure_1_architecture.svg)
- [FAK1-P4N iBAM and bond-map case](BMC_Bioinformatics_submission/figures/Figure_6_FAK1_case.pdf)
- [Reference code and usage](reference/README.md)
- [Public release scope](BMC_Bioinformatics_submission/PUBLIC_RELEASE.md)

The manuscript is a draft for author review. Contribution roles, funder involvement,
acknowledgments, final author approval, and persistent full-data/model deposits
remain to be completed before journal submission.

## Run the reference checks

Use Python 3.12 from a source checkout:

```sh
python -m pip install -r reference/requirements.txt
python -m pytest reference/tests -q
```

The reference models reuse the tracked `bindra.fusion` modules. The synthetic
checks require neither encoder weights nor benchmark downloads; one optional
official-data indexing check is skipped when benchmark assets are absent. Full experiments
require separately obtained official data, original frozen encoder assets, and
checksum-compatible feature caches. Those assets are not included in this minimal
release; see the reference README for their interface and availability limits.

The KIBA and additional Davis-fold protocol files are retained as development
history. That experiment queue was discontinued on October 25, 2026, and its
partial results are excluded from the manuscript. Installing the reference code or running its tests does not launch the study queue.

Code is distributed under the [MIT license](LICENSE). Third-party data and encoder
weights retain their respective licenses. Earlier DeepDTA-iBAM work remains at
[its archival repository](https://github.com/kevinmsong/DeepDTA-iBAM).
