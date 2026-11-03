# BINDRA

**Binding INference Design Retrieval and Attention**

BINDRA studies how protein and ligand features should be combined to predict
binding affinity. This repository provides the reference models, training
utilities, evaluation metrics, and three-model prediction averaging.

## Models

- **BINDRA** is the validation-selected pooled model. It combines separately
  summarized protein and ligand features.
- **BINDRA-iBAM** is the attention companion. It exchanges information between
  protein residues and ligand features before prediction.

Both use shared, frozen ESM-C protein features and LiGhT-mini ligand features.
An **Interpretable Binding Attention Map (iBAM)** visualizes how the attention
model weights protein residues and ligand tokens.

## Davis results

| Model | MSE ↓ | CI ↑ |
|---|---:|---:|
| BINDRA | 0.19663 | 0.90802 |
| BINDRA-iBAM | 0.19586 | 0.91253 |

Results use the official 5,010-pair Davis warm test. Each ensemble equally averages
pKd predictions from seeds 20261001, 20261002, and 20261003. Lower mean squared error
(MSE) and higher concordance index (CI) are better. Architecture selection used one
validation fold; historical test inspection makes these results retrospective.

## Quick start

Use Python 3.12 from the repository root:

```sh
python -m pip install -r reference/requirements.txt
python -m pytest reference/tests -q
```

Tests use synthetic inputs without encoder weights or benchmark downloads. The
optional official-data check is skipped when benchmark assets are unavailable.

## Reference code

- [Models](reference/models.py): pooled and attention architectures.
- [Training](reference/train.py): validation-only selection and resumable checkpoints.
- [Ensembles](reference/ensemble.py): member and ordering checks, mean pKd, and member SD.
- [Usage and feature format](reference/README.md): inputs, setup, and reproduction scope.

Trained checkpoints, encoder weights, benchmark data, and feature caches are
separate assets. Manuscripts and PDFs are maintained outside this repository.

## License

Code is distributed under the [MIT license](LICENSE). Third-party data and encoder
weights retain their own terms.
