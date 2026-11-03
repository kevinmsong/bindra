# Controlled BINDRA reference

This compact research implementation supplies the downstream pooled and attention
architectures over frozen encoder features. It is separate from the legacy
platform API. It does not include benchmark data, encoder weights, trained affinity
or generator checkpoints, or frozen prediction files.

## Completed study and archived protocol

The current manuscript reports completed Davis comparisons and supporting case
studies. **BINDRA** is the validation-selected pooled architecture (C-matched);
**BINDRA-iBAM** is its retained no-FiLM attention companion (F). Each reported
prediction ensemble averages three independently trained members in pKd space.
See [the manuscript](../J_Cheminform_submission/manuscript.md) for protocols,
results, uncertainty, historical test exposure, and limitations.

The separate KIBA benchmark, local external/classical baseline reruns, and
additional Davis-fold confirmation were discontinued on October 25, 2026. Their
partial results are not manuscript evidence. `protocol.json`, `registration.json`,
`upstream_sources.json`, and the implementation/runtime amendments preserve the
planned protocol and its development history. Their presence, and the runner's
ability to execute them, do not imply completed evaluation. Existing protocol and
implementation identities have not been rewritten for this documentation update.

## Install and check

Use Python 3.12 from the repository root:

```sh
python -m pip install -r reference/requirements.txt
python -m pytest reference/tests -q
```

The checks use synthetic examples and do not start study training or download
benchmark data. One optional official-fold indexing check is skipped when the
benchmark assets are absent. GPU experiments require an appropriate CUDA-enabled
PyTorch build.
Optional execution of the archived graph-baseline runners also requires
`torch-geometric` and the official code versions pinned in `upstream_sources.json`.
Those methods were not completed as local manuscript comparisons.

## Included implementation

- `models.py`: eight downstream architecture arms, including pooled BINDRA and
  BINDRA-iBAM. Their trainable prediction-parameter counts are 2,455,221 and
  2,228,293 per member. Shared fusion primitives come from tracked
  `bindra.fusion` modules.
- `data.py`: official pair/fold indexing, frozen-feature validation, baseline
  inputs, and classical features. Davis labels use pKd; KIBA uses the explicit
  `KIBA_score` scale. These scales are never interchanged.
- `train.py`: validation-only checkpoint selection, regression/ranking metrics,
  optimizer and random-state checkpointing, and matching-manifest resumption.
- `ensemble.py`: ordered arithmetic aggregation of three declared pKd prediction
  members, with per-member values and descriptive sample SD.
- `exposure.py`: chemical pretraining-exposure audits. Failed or ambiguous
  normalization remains unknown rather than being labeled unexposed.
- `study.py`: the archived, gated benchmark runner. Test scoring requires frozen
  selection, durations, and exposure audits.
- `io.py` and `lease.py`: atomic writes and execution coordination.

`models.py` preserves the tested downstream parameterization. It constructs no
protein or ligand encoder. New supervised models initialize fresh downstream
weights over original frozen ESM-C 600M and five-epoch LiGhT features, rather than
Davis-adapted member states. Internal features from independently trained members
must not be averaged as though their coordinates were interchangeable.

## Three-member Davis prediction aggregation

After predicting each member independently with `reference.train.predict`, retain
each prediction array's original observation identifiers:

```python
from reference.ensemble import DAVIS_SEEDS, MemberPredictions, aggregate_davis

# predictions_by_seed and pair_ids_by_seed are the exported member records.
members = [
    MemberPredictions(seed, pair_ids_by_seed[seed], predictions_by_seed[seed], "pKd")
    for seed in DAVIS_SEEDS
]
result = aggregate_davis(members)
```

Use three independent members of the same architecture. The default seeds are
20261001, 20261002, and 20261003. The helper rejects missing or repeated members,
different observation IDs or row orders, nonfinite values, and label scales other
than pKd. It preserves the supplied common row order and sorts member columns by
seed. `mean_pKd` is the equal arithmetic mean of pKd values. `member_SD_pKd` uses
the sample standard deviation across members (`ddof=1`); it is descriptive and
does not provide calibrated coverage. Each member must supply its own original
IDs so that misordered arrays are detectable.

This helper loads no weights, averages no latent features, and launches no
training. Reproducing trained predictions still requires the separately obtained
checkpoints and compatible frozen feature caches described below.

## Frozen-feature interface

The feature directory contains `manifest.json` with schema
`bindra-followup-frozen-encoders-v1`, the protein encoder contract, and ligand
checkpoint hash. Records are `protein/SHA256(sequence).pt` and
`ligand/SHA256(SMILES).pt`. Protein records contain residue `features`; ligand
records contain `tokens`, `physical_mask`, and three `graph_states`. Each record
includes its input hash and encoder identity. Special and padding masks remain
explicit. Only computations upstream of trainable modules may be cached.

Training utilities retain BF16, fused AdamW, gradient accumulation, length buckets,
cached graphs, and bounded frozen-feature caches. Gathered inputs and masks must
match the original padded representations. Resume requires matching data,
configuration, implementation, optimizer, and random-state identities.

The encoder weights, full feature stores, and official benchmark data must be
obtained separately under their applicable licenses. Persistent full-asset deposit
identifiers are pending in the manuscript availability statement. A local workspace
path is not a public asset deposit, and this release alone does not reproduce the
trained ensemble's predictions.

## Historical benchmark runner

These commands describe opt-in execution of the archived protocol, not work
completed for the manuscript or an instruction to resume it automatically:

```sh
python -m reference.study --stage freeze --out runs/new-benchmark
python -m reference.exposure --corpus PRETRAINING_SMILES_GZ --reference-table CHEMICAL_REFERENCE_PARQUET --reference-manifest REFERENCE_JSON --out runs/new-benchmark/exposure
python -m reference.study --stage neural --features FROZEN_KIBA_FEATURES --out runs/new-benchmark
python -m reference.study --stage classical --out runs/new-benchmark
python -m reference.study --stage evaluate --features FROZEN_KIBA_FEATURES --out runs/new-benchmark
```

The preserved protocol uses official DeepDTA setting1 indexing, one KIBA validation
fold, a common three-configuration neural tuning budget, and three final seeds.
Ridge is deterministic and is not represented as three independent fits. Inputs,
external pretraining, released graph encodings, and departures from upstream
selection policies remain explicit. The separate Davis confirmation schedule used
fixed fold-0-selected configurations on two additional folds; it was not completed
and cannot support a selection-stability claim.
