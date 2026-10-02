# Migration map: DeepDTA-iBAM → BINDRA

The specification's first required action is a migration map from existing
components to proposed replacements. This is it.

**Source:** `C:\Users\kevin\OneDrive\Documents\Research\DeepDTA-iBAM`
(github.com/kevinmsong/DeepDTA-iBAM, `main` @ `1b4fd7c`), frozen as the archival
record of its IEEE Access and J. Cheminform. submissions.

**Audit date:** 2026-10-01

## What the legacy system actually is

A single-endpoint regressor trained on **KIBA scores** (not pKd — the legacy
`affinity` column holds raw KIBA values around 11.7):

- Protein: ESM-C-600M, frozen, embeddings cached offline (1152-dim)
- Ligand: an in-repo 78-feature GAT built from RDKit, **not** KPGT
- Fusion: bidirectional atom↔residue cross-attention, 3 layers, 8 heads, dim 512
- Head: one scalar affinity output
- Generation: a Gaussian DDPM over continuous atom feature vectors
- Scale: 2,068 unique ligands, 229 unique proteins

## Component disposition

| Legacy component | Location | Disposition |
|---|---|---|
| `ProteinESM` | `models/rmse_model.py:162-273` | **Ported + fixed** → `bindra/encoders/protein_esmc.py` |
| `MolecularGAT`, `smiles_to_graph` | `models/rmse_model.py:43-159,836-1067` | **Port as ablation** → `bindra/encoders/ligand_gat.py` |
| — | — | **New** → `bindra/encoders/ligand_kpgt.py` |
| `ProteinAdapter`, `AdapterBlock` | `models/rmse_model.py:276-341` | **Port mostly as-is** → `bindra/fusion/adapters.py` |
| `BidirectionalCrossAttention` | `models/rmse_model.py:344-456` | **Port** → `bindra/fusion/cross_attention.py` |
| `AffinityHead` | `models/rmse_model.py:459-507` | **Rework** → `bindra/heads/affinity.py` |
| `SimpleGraphDiffusion` | `models/rmse_model.py:542-642` | **Replace** → `bindra/generation/digress/` |
| `training/engine.py` | whole file | **Port, high reuse** → `bindra/training/engine.py` |
| `pairwise_logistic_ranking_loss` | `training/engine.py:150-162` | **Port + fix grouping** → `bindra/training/losses/ranking.py` |
| `TokenBudgetBatchSampler` et al. | `data/datasets.py:111-246` | **Port** → `bindra/data/batching.py` |
| `GraphCache`, `ProteinEmbeddingCache` | `data/cache_builders.py:21-48` | **Rewrite** → `bindra/features/cache.py` |
| `data/splits.py` | whole file | **Extend** → `bindra/data/splits.py` |
| `utils/metrics.py` | whole file | **Port** (verify on port) → `bindra/eval/metrics.py` |
| `config_profiles.py` | whole file | **Port the pattern** → `bindra/config/profiles.py` |

## Defects carried forward as fixes

These are not stylistic preferences; each one can corrupt results silently.

### 1. Special tokens stripped by guessing — **fixed**

`ProteinESM._align_embeddings` (`models/rmse_model.py:220-235`) inferred which
rows were special tokens by comparing lengths, slicing `[1:1+L]` when the output
was at least two longer, and **zero-padding** when it was shorter. A tokenizer
emitting an unexpected token would shift every residue silently.

BINDRA uses `tokenizer.get_special_tokens_mask(..., already_has_special_tokens=True)`,
which marks special positions exactly, and **raises** on any count mismatch.
Verified: ESM-C returns 35 positions for a 33-residue sequence; the mask
identifies both `<cls>` and `<eos>`.

### 1b. The output tensor was chosen by attribute fallback — **fixed**

`_embed_chunks_esmc` took `outputs.embeddings`, falling back to
`outputs.hidden_states`, and raised only if both were absent. That is a guess
about which tensor holds the residue states.

It matters more than it looks. The native `EsmcModel` exposes both
`last_hidden_state` (after the final LayerNorm) and `last_hidden_state_prenorm`
(the raw residual stream). Measured on this machine for a 33-residue sequence,
they differ by a **max of ~6.2e3**. Reading the wrong one would not crash; it
would silently produce a cache of garbage.

The legacy code happened to land on the right tensor — `ESMC(...).embeddings`
matches `last_hidden_state` to within 0.007 — but by luck, not by design.
BINDRA names `last_hidden_state` explicitly and raises if it is absent, and a
test asserts the two tensors differ sharply so the reason is not forgotten.

### 1c. The checkpoint was a moving target — **fixed**

`ESMC.from_pretrained("esmc_600m")` resolves a mutable HF repo through a
deprecated shim. The installed `esm` also changed under us: the legacy project
had 3.2.1, this one resolved 3.4.1, in which `ESMC` is a compatibility wrapper
that *realigns* `hidden_states` onto the old layout.

BINDRA pins `biohub/ESMC-600M` @ `fcc9d36f5fff97d9bb69a36e3252de61cc78c968`,
records the state-dict SHA-256 (`7855e032…8538`) and asserts it in a test.

It also records which acceleration libraries are installed. ESM-C falls back to
pure-PyTorch LayerNorm, attention and RoPE when transformer-engine, xformers
and flash-attn are absent — as they are here — and warns that this changes
results numerically. That set is part of the cache key, so installing
flash-attn later invalidates the cache rather than silently disagreeing with it.

### 2. Residue numbering lost when windowing — **fixed**

`_stitch_chunks` (`data/cache_builders.py:123-139`) averaged overlapping windows
into a dense array with no record of source positions. `ProteinFeatures` now
carries `residue_indices`, so a long protein's attention map can be labelled
with real positions.

### 3. Cache keys ignored the encoder — **fixed**

Keys were `sha1(smiles_or_sequence)[:16]` (`data/cache_builders.py:16-18`).
Changing encoder, checkpoint or precision silently reused stale features.
`EncoderMetadata.cache_key()` now hashes the input *and* encoder name,
checkpoint, output dim, precision, schema version, revision, weights checksum
and preprocessing policy. Tested: changing any one of them changes the key.

### 4. Caches load entirely into RAM — **rewrite pending**

`GraphCache.__init__` and `ProteinEmbeddingCache.__init__` call `torch.load` on
the whole cache. Today that is 367 MB for 229 proteins and 118 MB for 2,068
ligands. Projected to a filtered BindingDB snapshot (estimates, to be replaced
by a measured pilot): ~10 GB of protein features and ~50 GB of KPGT token
features. Disk is fine (565 GB free); RAM is not. Replacement is sharded
memmap with streamed batches.

### 5. Ranking groups by target alone — **fix pending**

`train_one_epoch` passes `batch["target_sequence"]` as the grouping key
(`training/engine.py:632`). The specification forbids exactly this: *"A missing
assay identifier must not cause all records for that target to become one giant
ranking group."* The group key must be target + endpoint + assay-comparability.

### 6. `DataParallel` across heterogeneous GPUs — **fixed by removal**

`maybe_parallelize_model` (`training/engine.py:207-214`) wraps the model across
every visible CUDA device. On this machine that shards batches across a 16 GB
5060 Ti and a 4 GB T1200. BINDRA is single-GPU and selects by capability.

The companion hazard was confirmed accidentally during this work. A throwaway
script that hardcoded `cuda:1` — without importing `bindra` first, so
`CUDA_DEVICE_ORDER=PCI_BUS_ID` was never set — landed on the **4 GB T1200** and
died with `CUDA out of memory ... GPU 1 has a total capacity of 4.00 GiB`.
Under PCI-bus ordering `cuda:1` is the 16 GB card; under CUDA's default
ordering it is not. This is precisely why `bindra.runtime.device` selects by
capability and memory, and why the environment variable is set in
`bindra/__init__.py` before any CUDA context can be created.

### 7. Invalid SMILES became an all-zero graph — **fix pending**

`smiles_to_graph` (`models/rmse_model.py:847-855`) returns zeros on RDKit
failure, so an unparseable molecule becomes a silent training example.
Standardization must quarantine with a reason instead.

## Why the legacy diffusion branch is replaced, not kept

`SimpleGraphDiffusion` is a Gaussian DDPM over continuous 78-dim atom feature
vectors. It cannot serve as the analog generator because:

- **Bonds are never generated.** `sample()` takes `reference_adj` as an input
  and holds it fixed (`models/rmse_model.py:624`). Only atom features move.
- **There is no decoder.** It emits a continuous `[nodes, 78]` tensor. Nothing
  maps that back to a valid molecule, and an arbitrary vector is not invertible
  into one.
- **Conditioning is a single pooled protein vector** via FiLM — no seed ligand,
  no scaffold constraint, no endpoint.

The specification anticipates this case and permits retaining a latent diffusion
implementation only after demonstrating an explicit decoder, reconstruction
quality, conditional denoising and a valid reverse process. This one has none of
those, so it moves to `bindra/generation/legacy_latent.py` as a labelled
ablation and is replaced by discrete graph diffusion.

## Legacy baseline: what can actually be reproduced

`checkpoints/` holds only `max_rmse_cluster_diffusion_member_0.safetensors`
(158 MB) plus its optimizer sidecar, while the profile declares
`ensemble_size=5`. `_load_saved_best_ensemble` (`training/engine.py:515-536`)
requires all five members and raises otherwise.

So the reproducible legacy baseline is **single-member inference only**.
Recorded validation metrics from the checkpoint's metadata sidecar, in KIBA
units: RMSE 0.473, MAE 0.298, CI 0.868, R² 0.689, Pearson 0.867, epoch 47,
`target_mean` 11.719, `target_std` 0.837.

This is recorded as a known limitation rather than resolved by retraining.
