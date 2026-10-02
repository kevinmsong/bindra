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

### 7. Invalid SMILES became an all-zero graph — **fixed**

`smiles_to_graph` (`models/rmse_model.py:847-855`) returns an all-zero `[N, 78]`
tensor with an identity adjacency on RDKit failure. That is not a null value: it
is a valid-looking molecule made of nothing. It trains, it contributes gradient,
and downstream it is indistinguishable from a real single-atom species.

`bindra.encoders.ligand_graph.featurize_molecule` raises `LigandGraphError`
instead, and standardization quarantines with a reason upstream of it. Asserted by
`test_unparseable_smiles_raises_instead_of_returning_a_null_molecule`.

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

---

## Stage 2 addendum: defects found while curating (2026-10-02)

Six more silent-failure modes, found by running the pipeline on the real corpus
rather than by reading code. Recorded here because each one would have produced
plausible-looking output.

### 8. `doctor` validated a different code path than the cache run

`_check_esmc` used `from esm.models.esmc import ESMC`,
`ESMC.from_pretrained("esmc_600m")` and `output.embeddings` — the unpinned name,
the deprecated shim and the attribute guess that defects 1b and 1c above record as
*fixed*. The command that gates a 14 GiB encode was therefore exercising none of
the fixes. Now routed through `ESMCProteinEncoder`, with `--verify-weights` to
compare the state-dict hash against the pinned digest.

### 9. The weights checksum was missing from cache keys

`EncoderMetadata.weights_checksum` defaulted to `None` whenever metadata came from
`encode()`, so the checksum that exists to distinguish a corrupted or locally
overridden checkpoint from the pinned one was absent from the very keys it was
meant to protect. It also meant metadata built by a caller disagreed with metadata
attached to features, which the cache correctly refused as two contracts.
`metadata()` now resolves the real checksum by default.

### 10. Lower-case sequences silently became all-`<unk>`

34 of 10,822 constructs in release 202610 are written entirely in lower case. ESM-C
tokenizes lower-case residues to `<unk>`, and `<unk>` is in `all_special_ids`, so
special-token stripping removed **every position** of the protein. The adapter's
strict count check (defect 1, fixed) caught it and raised rather than caching 437
`<unk>` embeddings — the guard earning its keep. Sequences are now canonicalized at
ingest, and the encoder rejects non-canonical input with a message that names the
cause instead of reporting "0 residue positions".

### 11. Global alignment made the coverage criterion inert

The split policy originally pinned global (Needleman–Wunsch) alignment for the
40% identity / 80% coverage rule. Measured against same-UniProt positives and
cross-UniProt negatives:

| mode | sensitivity | false-positive rate |
|---|---|---|
| global | 0.867 | 0.043 |
| semi-global | 0.987 | 0.000 |

Global forces end-to-end alignment, which lifts background identity for unrelated
proteins to a median of 0.252 and pins coverage near 1.0, so the 80% criterion
filtered nothing. On a 500-construct sample it produced **one component containing
all 500 sequences** — cold-target evaluation would have been impossible, and the
tempting "fix" would have been to weaken the threshold. Semi-global produced 394
components, largest 54 (10.8%).

### 12. A conservative margin made leakage worse, not better

Optimal alignments are rarely unique — Biopython reports 2,695,680 co-optimal
alignments for one real pair, with identity counts differing by enough to straddle
the threshold. Adding a 0.08 identity margin looked like the safe response. It
dropped the effective threshold to 0.32, inside the negative-pair identity
distribution (p90 = 0.385), turning 134 qualifying edges into 473 and the largest
component from 10.8% into 53.8%. Single linkage amplifies a blanket margin.
Resolution: no margin, plus an audit that re-aligns near-threshold cross-partition
pairs with an independent implementation and quarantines real violations.

### 13. Folding fingerprints is not conservative

The cross-partition similarity screen folds 2048-bit fingerprints to 512 bits, on
the usual argument that folding can only over-report similarity. That argument is
wrong: two bits set in *the same* molecule can collide into one folded position,
shrinking that molecule's popcount and the intersection with it. Measured, 0.4% of
random pairs have folded Tanimoto *below* true Tanimoto, worst shortfall 0.017.
The screen is now described as a heuristic ranker, gated by the provably sound
popcount bound `Tanimoto <= min(|A|,|B|) / max(|A|,|B|)`, with the shortfall
measured and reported alongside every audit.

### Also corrected

- The dedup key omitted assay conditions, so two records from one paper reporting
  the same value at different pH collapsed as re-imports.
- `parasail`'s 16-bit kernel silently returns score 0 on the 34,350-residue
  construct; only the `_sat` kernels are used and saturation raises.
- The census counted unique sequences with salted `hash()`, so its 11,018 was not
  reproducible across runs.

### 14. "Resumable" was a documentation claim, not an implementation

`build_relation_table` was documented as resumable, and `bindra prepare-data
homology` advertised it in its help text. It was not: results accumulated in a
list in memory and were written only at the end.

The cost was concrete. The first full run reached **block 181 of 271 — 52,124,380
of 58,530,790 pairs, 89% — with roughly 15 minutes of work left**, hit a two-hour
wall-clock limit, and lost all of it.

Now each block's edges are written to `protein_relation/blocks/block-NNNNN.npy`
through a temporary file and an atomic rename, so a file's existence means it is
complete. A rerun loads what is there and computes only the remainder. A `key.json`
records the sequence-set hash, block size and alignment policy, and resuming
against anything else is refused — block indices are meaningless across a
different corpus, so stitching them would silently mix results for different
pairs.

Verified: deleting half the checkpoints and rerunning reproduces the full run's
edge array byte-identically; a fully cached rerun takes 0.07 s.

The general lesson is the one this file exists for: a property stated in a
docstring and never exercised is not a property. The same applies to the claims
above — each one has a test behind it.

### 15. The 78-feature atom layout has four defects, reproduced on purpose

Found while porting `smiles_to_graph` for ablation 1. The specification's
ablation 1 is "the existing ESMC + GAT model retrained on the same curated
data", which is only a controlled comparison if the ligand features are the
*same numbers*. So these are reproduced bit-for-bit rather than corrected, and
`bindra/encoders/ligand_graph.py` counts each one so its reach is a measurement
instead of a worry. Prevalence below is over a 4,000-molecule random sample of
the curated corpus (median 30 heavy atoms).

| Defect | What it does | Prevalence in 4,000 molecules |
|---|---|---|
| `ACCEPTOR_EXCLUDES_TERMINAL_OH` | Slot 75 ("acceptor") excludes any N/O bearing a hydrogen at degree 1, removing hydroxyl and primary-amine oxygen/nitrogen from the acceptor set. A hydroxyl oxygen is a textbook acceptor *and* donor. | **2,738 atoms** |
| `SMALL_RING_COLLAPSE` | The bond ring-size block maps sizes 3, 4 and 5 all onto slot 6 (6 → slot 7, 7 → slot 8), so cyclopropane, cyclobutane and cyclopentane bonds are indistinguishable. The *atom* ring-size block does separate 3/4/5/6, so the two blocks disagree about what a ring size is. | **2,901 molecules (73%)** |
| `GASTEIGER_NAN` | A NaN Gasteiger charge leaves all four charge slots zero, while an *absent or unparseable* property sets the "slightly positive" bin and an infinity bins as a genuine extreme. Three failure modes, three different encodings. | **86 atoms** |
| `UNKNOWN_SYMBOL` | The symbol one-hot covers only C, N, O, S, F, P, Cl, Br, I, with no "other" bin, so boron and platinum share an all-zero symbol block. | **12 atoms** |

The first two are the ones that matter: together they touch most molecules in the
corpus. Neither is a reason to deviate here — changing them would make ablation 1
a comparison of two different inputs rather than of two encoders — but any BINDRA
arm that is *not* reproducing the legacy model should use a corrected layout, and
that choice must be stated wherever the arms are compared.

A related structural change was made rather than reproduced: the legacy function
returns a dense `[N, N, 12]` bond tensor, quadratic in atom count. The port stores
bonds as `[E, 12]` with an `edge_index`, and `MolecularGraph.dense_views()`
reconstructs the legacy tensors for the equivalence test. The numbers are identical;
only the layout differs.

### 16. DGL cannot run on this GPU, and the recorded reason was wrong

`docs/STATUS.md` recorded the KPGT blocker as DGL "stops at Python 3.12 / no
longer ships Windows CUDA wheels". Both specifics are false: DGL 2.2.1 *is* the one
2.x release that ships a `cp312-win_amd64` wheel. The real constraint is narrower
and permanent — 2.2.1 is DGL's final release (2024-05-13), its `graphbolt`
extension loads by exact torch version and ships DLLs only up to torch 2.3.0,
torch 2.3.0 is a cu121 build with no `sm_120`, and this GPU is compute capability
12.0. DGL therefore cannot execute on it, and no future wheel will change that.

The environment is still built, CPU-only, at `C:/bindra/envs/kpgt` — as a
numerical oracle for verifying a pure-PyTorch port, nothing more. Recipe, the
seven forced dependency pins and what each one fixes are in `envs/kpgt/README.md`.
The lesson is the same as defect 14's: a blocker recorded from a plausible cause
rather than a measurement sends the design the wrong way. Here it would have
argued for abandoning KPGT's architecture when what is actually needed is a port.

