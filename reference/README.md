# Controlled BINDRA reference

This compact research implementation reproduces the downstream pooled and
attention architectures over frozen encoder tokens. It is separate from the
legacy platform API. It does not include licensed data, encoder weights, run
outputs, or a claim that unfinished experiments have been completed.

`models.py` preserves the tested downstream parameterization. The pooled and
no-FiLM attention models have 2455221 and 2228293 fitted parameters. Shared
fusion primitives come from this repository's `bindra.fusion` modules. Cached
inputs use the original ESM-C 600M and five-epoch LiGhT representations; no
Davis-adapted member states initialize new benchmark supervision.

## Independent benchmark

`protocol.json` specifies official KIBA setting1 folds, three learning rates
at fixed dropout 0.1, one validation fold (official fold 0), and three final
seeds. The runtime-budget amendment followed initial validation inspection
and preceded every KIBA test prediction. All neural methods receive the same
reduced tuning budget. GraphDTA-GIN and MGraphDTA use pinned official source, checked against
`upstream_sources.json`; they are local reruns under this controlled protocol,
not transcriptions of paper metrics. Optimizer, selection, input cap and
normalization departures from upstream scripts are explicit. All models use
the same rows and first 1000 protein residues. Baselines use their released
graph encodings; BINDRA's external pretraining remains a resource difference.
Ridge/random forest use Morgan-2048 fingerprints and 420 sequence-composition
features fitted on the same partitions. Ridge is deterministic and is not
presented as three independent fits.

Install the project dependencies and `torch-geometric`, `networkx`, `scipy`,
`scikit-learn`, `pandas`, `pyarrow`, `psutil`, and `pytest`. Obtain official DeepDTA data
under `data/external/{davis,kiba}` with their original fold files. Obtain the
separately archived frozen feature assets and checkpoint identities described
in the manuscript availability statement. Public deposit identifiers remain
pending; a local path is not a public release.

```powershell
python -m reference.study --stage freeze --out runs/reviewer-kiba
python -m reference.exposure --corpus PATH_TO_PRETRAINING_SMILES_GZ --reference-table PATH_TO_CHEMICAL_REFERENCE_PARQUET --reference-manifest PATH_TO_REFERENCE_JSON --out runs/reviewer-kiba/exposure
python -m reference.study --stage neural --features PATH_TO_FROZEN_KIBA_FEATURES --out runs/reviewer-kiba
python -m reference.study --stage classical --out runs/reviewer-kiba
python -m reference.study --stage evaluate --features PATH_TO_FROZEN_KIBA_FEATURES --out runs/reviewer-kiba
```

The protocol records source hashes before new scoring. Every model selection
and final refit, plus the exposure audit, must finish before test scoring.
Outputs identify `KIBA_score` explicitly; pKd magnitude errors are never used
for KIBA. Per-seed and arithmetic three-model ensemble results are separate.
Missing/ambiguous chemical normalization is unknown, not unexposed. This is
an independent public dataset, not a claim of historically secret labels.

## Frozen feature interface and speed

The feature directory contains `manifest.json` with schema
`bindra-followup-frozen-encoders-v1`, the exact protein encoder contract and
ligand checkpoint hash. Files are `protein/SHA256(sequence).pt` and
`ligand/SHA256(SMILES).pt`. Protein records contain residue `features`; ligand
records contain `tokens`, `physical_mask` and three `graph_states`. Each record
includes its input hash and encoder identity. Special/padding masks remain
explicit; trainable adapter/fusion states must never be cached.

Training retains BF16, fused AdamW, gradient accumulation, length buckets,
cached graphs and a bounded frozen GPU feature cache. GPU gathers must match
the original padded inputs and masks exactly. Optimizer and RNG state are
checkpointed. Resume requires matching data, configuration and implementation.
Run only one GPU worker at a time. Runtime estimates require profiling actual
fits; maximum epoch counts are not an ETA.

`implementation_amendment.json` records an I/O-only correction after four
completed tuning epochs: atomic file replacement retries transient Windows
sharing locks. Model, partition, hyperparameter, and selection rules did not
change. Training resumed from the saved optimizer/RNG state; no KIBA test
predictions existed at the amendment. The registration preserves both protocol
identities and the original/current I/O implementation hashes.

Additional Davis confirmation compares only pooled BINDRA and BINDRA-iBAM
on folds 1 and 2, with three seeds and their existing fold-0-selected
learning rate 0.0003 and dropout 0.1. This requires 12 fits, with no new
hyperparameter sweep and no repeated Davis test scoring. It tests the ranking
of two fixed configurations, not eight-arm architecture-selection stability.
The original fold-0 decision and historical exposure remain disclosed.

`runtime_amendment.json` records the reduction from 212 to 36 neural fits
(24 KIBA and 12 Davis) and 40 to 16 classical fits. The ongoing first fit's
manifest, weights, optimizer, and RNG bytes were preserved after epoch 24.
The active protocol controls the queue; a `policy_reuse.json` sidecar records
compatibility with its original fitting policy. Optimization, source, data,
seed, and model checks remain enforced on resume. No test metrics informed
this user-requested runtime reduction.

The command requires the separately archived full-sequence Davis feature store:

```powershell
python -m reference.study --stage davis-stability --davis-features PATH_TO_DAVIS_FEATURES --out runs/reviewer-kiba
python -m pytest reference/tests -q
```
