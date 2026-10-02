# BINDRA status ledger

The specification is explicit: *"Do not mark all requested functionality complete
merely because interfaces exist. Separate implemented, smoke-tested, trained,
benchmarked, and experimentally validated status."*

This file is that separation. It is the authoritative answer to "does BINDRA do X
yet?" — not the presence of a module, not a passing import.

**Last updated:** 2026-10-01 (BindingDB snapshot acquired; ESM-C moved to the native pinned API)

## Status vocabulary

| Status | Meaning |
|---|---|
| `absent` | Not written. |
| `scaffolded` | Interface and contract exist; the body raises `NotImplementedError`. |
| `implemented` | Logic written, unit-tested in isolation. Says nothing about real data. |
| `smoke-tested` | Runs end to end on real inputs at small scale. |
| `trained` | A model was actually fit on the curated corpus. |
| `benchmarked` | Measured on locked, prespecified evaluation panels. |
| `validated` | Confirmed by **experimental measurement**, not by another model. |

Nothing below is `validated`. Nothing below is `trained`.

## Priority 0 — environment and legacy preservation

| Component | Status | Evidence |
|---|---|---|
| CUDA environment (torch 2.11.0+cu128) | smoke-tested | `bindra doctor`: selects RTX 5060 Ti, sm_120, 49.4 TFLOP/s bf16 |
| GPU selection by capability, not index | implemented | `bindra/runtime/device.py`; rejects the 4 GB T1200 |
| ESM-C-600M forward pass on GPU | smoke-tested | `bindra doctor --esmc`: `(1, 35, 1152)` bf16 |
| KPGT isolated environment | absent | Blocked, see below |
| Legacy baseline reproduction | absent | Single-member only, see blockers |

## Priority 1 — endpoint-aware BindingDB dataset

| Component | Status | Evidence |
|---|---|---|
| Endpoint / p-value conversion | implemented | `bindra/data/endpoints.py`, 41 tests incl. the spec's required table |
| Censoring relations and reversal | implemented | Round-trip and strictness tests pass |
| Snapshot provenance + verification | implemented | `bindra/data/bindingdb/snapshot.py`, 14 tests |
| TSV schema + header validation | implemented | `bindra/data/bindingdb/schema.py`, verified vs. real file |
| Snapshot census | smoke-tested | 3.24M rows scanned; `docs/DATA_PROFILE.md` |
| BindingDB snapshot acquired | smoke-tested | Release 202610, see below |
| BindingDB ingestion (parsing) | absent | Snapshot in hand; parser not written |
| Standardization and QC | absent | |
| Deduplication / assay comparability | absent | |
| Immutable splits (cold/double-cold/temporal) | absent | |
| Curation report | absent | |

## Priority 2 — cross-attention, iBAM, retrieval

| Component | Status | Evidence |
|---|---|---|
| `ProteinFeatures` / `LigandFeatures` contracts | implemented | `bindra/features/types.py` |
| ESM-C adapter (native API, pinned, post-LN) | implemented | `bindra/encoders/protein_esmc.py`, 39 tests |
| Cache-key discipline (encoder identity in key) | implemented | Tested: any encoder change invalidates the key |
| KPGT adapter | absent | Blocked on the isolated environment |
| Sharded feature cache | absent | Legacy design does not scale, see plan |
| Cross-attention fusion | absent | Port target identified |
| Per-endpoint heads (pKd/pKi/pIC50) | absent | |
| iBAM export and plots | absent | |
| Retrieval projections / ANN index | absent | |

## Priorities 3–5 and Upgrades A–C

| Component | Status |
|---|---|
| Discrete graph diffusion (DiGress-style) | absent |
| Graph-to-molecule decoder | absent |
| Measured analog-delta learning (Upgrade A) | absent |
| Locked stress-test suite (Upgrade B) | absent |
| Prospective study support (Upgrade C) | absent |
| Ensemble + conformal calibration | absent |
| Applicability-domain reporting | absent |

## Resolved

### BindingDB snapshot — acquired

Release **202610**, `BindingDB_All_202610_tsv.zip`, from
`https://www.bindingdb.org/rwd/bind/downloads/`, server-dated 2026-09-30.
Provenance (release, URL, size, SHA-256, download time, archive members,
parser version) is recorded in `data/snapshots/bindingdb_202610/provenance.json`
and re-verifiable with `verify_snapshot()`.

The release reports 3,248,380 measurements, 1,446,269 compounds and 11,516
targets. Note that the `Download.jsp` wrapper URL returns **HTTP 200 with an
HTML page** rather than the archive; `record_snapshot()` rejects anything
lacking a zip signature, because `curl` exits 0 on that page.

### ESM-C — now pinned and on the native API

The deprecated `ESMC` class is no longer used. Resolution:

- **Native API.** `EsmcModel` + `EsmcTokenizer`, not the `ESMC` shim (which
  wraps `EsmcForMaskedLM` and realigns `hidden_states` onto the pre-3.3 layout).
- **Pinned artifact.** `biohub/ESMC-600M` @ revision
  `fcc9d36f5fff97d9bb69a36e3252de61cc78c968`, with the state-dict SHA-256
  `7855e032…8538` recorded and asserted by a test.
- **Post-LayerNorm state selected by name.** `EsmcModel` exposes both
  `last_hidden_state` (post final LayerNorm) and `last_hidden_state_prenorm`.
  Measured difference on this machine: **max ~6.2e3**. The legacy adapter fell
  back through `embeddings` → `hidden_states` until something was non-None;
  that happened to land on the right tensor, but only by luck. A test now
  asserts the two differ sharply, documenting why there is no fallback.
- **Kernel backends recorded.** transformer-engine, xformers and flash-attn are
  all absent here, so ESM-C uses pure-PyTorch LayerNorm/attention/RoPE. The
  package warns this changes numerics, so the installed set is part of the
  cache key: installing flash-attn later invalidates cached features instead of
  silently disagreeing with them.

Feature schema version is `protein-esmc-v2-postln-native`, so nothing cached
under the old adapter can be mistaken for this contract.

## Active blockers

1. **KPGT / DGL dependency conflict.** KPGT needs DGL, which stops at Python
   3.12 / torch ~2.4 and no longer ships Windows CUDA wheels; the training
   stack is torch 2.11+cu128. Resolution: a separate CPU-only Python 3.12
   environment runs frozen KPGT and exchanges versioned cached arrays. Not yet
   built. KPGT pretrained weights not yet downloaded.

2. **Legacy ensemble is incomplete.** `checkpoints/` in the legacy repo holds
   only `max_rmse_cluster_diffusion_member_0`, but that profile declares
   `ensemble_size=5`, and the legacy loader requires all five. The reproducible
   legacy baseline is therefore **single-member inference only**. Recorded
   legacy validation metrics, in KIBA units, not reproduced here yet:
   RMSE 0.473, CI 0.868, R² 0.689, epoch 47.

## Decisions pending (must be made before modelling)

- **Temporal cutoff year.** Coverage measured (see `docs/DATA_PROFILE.md`);
  2022 or 2023 both leave a substantial future panel. Not chosen.
- **Ligand cache strategy.** KPGT token states for all 1.44M molecules would be
  ~100 GB. Requires a pilot encode before committing.

## Deliberately not claimed

- No BINDRA model has been trained. There are no BINDRA performance numbers.
- No retrieval, generation, or iBAM results exist.
- DIP2A is an exploratory application. No DIP2A binding assay is assumed to
  exist, and no compound is treated as a validated DIP2A binder.
