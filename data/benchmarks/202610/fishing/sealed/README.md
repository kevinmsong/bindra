# Sealed labels

These are evaluation labels. They exist so metrics can be computed **after predictions are frozen**, and for nothing else.

Do not read them from training, scaler-fitting, hyperparameter selection or calibration code. Access goes through `bindra.eval.benchmarks.lock.load_sealed_labels`, which requires an explicit authorisation and a reason, and appends every access to `access_log.jsonl`.
