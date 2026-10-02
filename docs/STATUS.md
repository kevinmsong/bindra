# BINDRA status ledger

The specification is explicit: *"Do not mark all requested functionality complete
merely because interfaces exist. Separate implemented, smoke-tested, trained,
benchmarked, and experimentally validated status."*

This file is that separation. It is the authoritative answer to "does BINDRA do X
yet?" — not the presence of a module, not a passing import.

**Last updated:** 2026-10-02 (curation, chemical standardization, target
constructs, protein homology grouping and the GPU protein feature cache)

## Status vocabulary

| Status | Meaning |
|---|---|
| `absent` | Not written. |
| `scaffolded` | Interface and contract exist; the body raises `NotImplementedError`. |
| `implemented` | Logic written, unit-tested in isolation. Says nothing about real data. |
| `smoke-tested` | Runs end to end on real inputs at small scale. |
| `run-on-corpus` | Executed over the whole snapshot, with recorded artifacts. Still says nothing about model quality. |
| `trained` | A model was actually fit on the curated corpus. |
| `benchmarked` | Measured on locked, prespecified evaluation panels. |
| `validated` | Confirmed by **experimental measurement**, not by another model. |

Nothing below is `validated`. Nothing below is `trained`. **No BINDRA model
exists and there are no BINDRA performance numbers.**

## Priority 0 — environment and legacy preservation

| Component | Status | Evidence |
|---|---|---|
| CUDA environment (torch 2.11.0+cu128) | smoke-tested | `bindra doctor`: RTX 5060 Ti, sm_120, 48.6 TFLOP/s bf16 |
| GPU selection by capability, not index | implemented | `bindra/runtime/device.py`; rejects the 4 GB T1200 |
| ESM-C-600M forward pass on GPU | run-on-corpus | `doctor --esmc --verify-weights` now routes through the real adapter |
| KPGT isolated environment | built, CPU only, **job done** | Served as the numerical oracle for the pure-PyTorch port; kept only for re-verification. `envs/kpgt/README.md` |
| Legacy baseline reproduction | absent | Single-member only, see blockers |

## Priority 1 — endpoint-aware BindingDB dataset

| Component | Status | Evidence |
|---|---|---|
| Endpoint / p-value conversion | implemented | `bindra/data/endpoints.py`, 41 tests incl. the spec's required table |
| Censoring relations and reversal | implemented | Round-trip and strictness tests pass |
| Snapshot provenance + verification | implemented | `bindra/data/bindingdb/snapshot.py`, 14 tests |
| TSV schema + header validation | implemented | `bindra/data/bindingdb/schema.py`, verified vs. the real file |
| Curation policy as hashed data | implemented | `bindra/config/profiles.py`; thresholds derived from the measured error model |
| Table I/O, atomic + content-hashed | implemented | `bindra/data/tables.py`, 23 tests |
| **BindingDB ingestion** | **run-on-corpus** | 3,243,660 rows → **2,798,222 observations**, 0 id collisions, 241 s. Reproduces the census exactly |
| **Chemical standardization** | **run-on-corpus** | 1,438,824 unique SMILES → **1,436,561 standardized**, 2,263 quarantined, 26 min over 14 cores |
| **Target constructs** | **run-on-corpus** | 10,820 valid constructs; 348 point mutants detected offline with **91.4% independent corroboration** |
| **Deduplication / assay comparability** | **run-on-corpus** | **2,739,920 curated observations**; 52,856 duplicate imports; 165,020 comparability groups |
| **Protein homology grouping** | **run-on-corpus** | All **58,530,790** pairs aligned exactly; 62,884 qualifying edges; two groupings published (40% strict, 60% cold-target) because the graph percolates at 40% |
| **Curation report** | implemented | [`CURATION_REPORT.md`](CURATION_REPORT.md) + `curation_report.json` |
| Immutable split manifests | implemented | `bindra/data/splits.py` + `bindra/cli/make_splits.py`, 37 tests |
| **Temporal split** | **run-on-corpus** | train 2,170,213 / val 48,723 / cal 59,095 / **test 227,514** (1,247 targets); 12,000 straddling pairs routed to a repeated-measurement panel; audit passes with **0 future labels informing the model** and 0 re-imports in test |
| **All six split regimes over the corpus** | **run-on-corpus** | random_pair, cold_drug, cold_target (60%), cold_target_strict (40%), double_cold, temporal. **Zero violations across 62.3M cross-partition homology checks**; all six manifests re-verify |
| **Seven locked benchmark panels** | **run-on-corpus** | activity_cliff, selectivity, censored_selectivity, fishing, and three temporal panels; labels sealed with an access log |
| Benchmark locking + sealed labels | implemented | `bindra/eval/benchmarks/lock.py`, 17 tests incl. a monkeypatched leak guard that is itself proven able to fire |
| **Activity-cliff panel** | implemented | `activity_cliff.py`, 16 tests; endpoint-specific magnitude and a `cliff_kind` diagnostic category |
| **Temporal panels** | implemented | `temporal.py`, 16 tests; three separate panels plus a `pretraining_exposure` non-claim |
| **Selectivity panel** | implemented | `selectivity.py`, 15 tests; unknowns never negatives, censored sides stay intervals |
| **Fishing panel** | implemented | `fishing.py`, 18 tests; two control tiers, homologue-aware decoy exclusion |
| Shared fingerprint + property stores | implemented | `similarity.py`, `properties.py`; built once, reused by three consumers |

## Priority 2 — cross-attention, iBAM, retrieval

| Component | Status | Evidence |
|---|---|---|
| `ProteinFeatures` / `LigandFeatures` contracts | implemented | `bindra/features/types.py` |
| ESM-C adapter (native API, pinned, post-LN) | implemented | `bindra/encoders/protein_esmc.py`, 41 tests |
| Cache-key discipline (encoder identity in key) | implemented | Weights checksum now always in the key; any encoder change invalidates it |
| **Sharded feature cache** | **implemented** | `bindra/features/cache.py`, 25 tests; ragged bf16, bit-exact, stale-cache refusal |
| **Protein feature cache populated** | **run-on-corpus** | **10,820/10,820 constructs, 14.023 GiB across 44 shards**, 40 min on the 5060 Ti at 2,709 residues/s, peak VRAM 2.36 GiB. All 261 over-window constructs encoded through the windowing path |
| **Cached vs live agreement** | **run-on-corpus** | `bindra cache verify --sample 0.01`: **108/108 re-encoded constructs bit-identical**, not merely within tolerance |
| **Ligand graph featurization** | **implemented** | `bindra/encoders/ligand_graph.py`, 95 tests; legacy 78-atom/12-bond layout reproduced **bit-identically** against the legacy repository's own executed `smiles_to_graph`, on 22 probe molecules and 400 real curated ones |
| **Ligand encoder seam** | **implemented** | `bindra/encoders/ligand.py`; `LigandEncoder` protocol + registry, so the fusion stack never names an arm. Segment ops (softmax/sum/mean/max) with overflow and empty-segment behaviour tested |
| **GAT ligand encoder (ablation 1)** | **implemented** | `bindra/encoders/ligand_gat.py`, 38 tests; GATv2 with edge-conditioned attention, self-loops, per-layer per-head attention exposed. Batch/single equivalence and permutation equivariance asserted; CUDA and CPU agree to 1e-4 |
| **KPGT featurization (DGL-free)** | **implemented** | `bindra/encoders/kpgt_graph.py`; triplet graph, 137-dim atom and 14-dim bond vectors, RDKit path fingerprint. Graph structure, every feature column and the edge ordering **bit-identical** to the reference |
| **LiGhT ported to pure PyTorch** | **implemented** | `bindra/encoders/kpgt_light.py`, 44 tests. Verified against the DGL reference at ten staged intermediates; 4.8e-08 (small) / 6.9e-06 (base, 92M params) end to end. Runs on the RTX 5060 Ti at **1,469 mol/s**, 997 MiB peak |
| **KPGT descriptors (vendored)** | **implemented** | `bindra/encoders/kpgt_descriptors/`; 4 files from the reference, Apache-2.0, LICENSE carried. Matches the original to **1e-15** on identical column sets. Vendored not depended on: the pretrained weights were fitted against this exact CDF normalization |
| **KPGT ligand-encoder adapter** | **implemented** | `bindra/encoders/ligand_kpgt.py`, 71 tests with the GAT arm. Registered as `"kpgt"`; emits **bond-level tokens** (2 atoms each) plus 2 knowledge tokens that are non-physical so iBAM will not draw them. `from_pretrained` loads `strict=True` and keys the cache on a content hash |
| KPGT **weights** | absent | One manual browser download; figshare is behind a JS challenge. Until then the arm is verified but not pretrained |
| **Frozen-ligand vector cache** | **implemented** | `bindra/features/ligand_cache.py`, 27 tests with the fine-tuning suite. Fixed-width bf16, sharded, resumable, encoder identity in the key. Refuses to cache a trainable encoder. Unused in the chosen regime, and kept because `n_trainable_layers=0` is a supported configuration |
| **KPGT partial fine-tuning** | **implemented** | Last layer unfrozen by default; embeddings and lower layers stay frozen so the representation does not move out from under them. Gradient reach asserted exactly |
| KPGT wired into `BindraModel` | absent | The model still constructs `GATLigandEncoderModule` directly. The two arms take different batch types, so this is a real change, not a parameter. **This is the next step** |
| **Shared adapters** | **implemented** | `bindra/fusion/adapters.py`; widths read off encoder metadata, nothing hardcodes 1152. `attribute_gradients` discharges "log which parameters receive gradients from each loss" by backpropagating each loss in isolation; a stray `detach()` is caught by test rather than by inspection |
| **Bidirectional cross-attention** | **implemented** | `bindra/fusion/cross_attention.py`, 41 tests with the adapters. Two independent attention blocks, asserted **not** to be transposes of one another; applied in parallel so the returned maps describe the inputs. Padding invariance asserted end to end; an all-masked side yields zeros rather than NaN |
| **Per-endpoint heads (pKd/pKi/pIC50)** | **implemented** | `bindra/heads/affinity.py`; three independent trunks, unbounded output. No target exists in the module, so a fabricated zero label is not expressible; gradient to an unmeasured endpoint asserted to be exactly 0 |
| **Censored affinity loss** | **implemented** | `bindra/training/losses/affinity.py`, 49 tests. Huber delta = **measured sigma_rep per endpoint** (0.945/0.641/0.671), so residuals inside the replicate noise are treated as noise. One-sided penalties for bounds; Tier B structurally confined to the auxiliary term; exact RMSE and bound-violation rate reported separately |
| **Assembled model** | **implemented** | `bindra/model.py`, 26 tests. **Tiny-data overfit: exact RMSE 7.740 → 0.0264** over 400 steps on 8 pairs across all three endpoints (predicting the label mean gives > 2.0, so the bar excludes the trivial solution). Nonzero shared-module gradients verified per task against the prefixes the model itself declares. Save/reload reproduces inference exactly; atom renumbering changes nothing |
| **iBAM numeric export** | **implemented** | `bindra/ibam/export.py`, 50 tests with the plots. Native residue x token **and** derived residue x atom views; bond tokens split equally between endpoints per the spec and labelled derived (the flag cannot be turned off). Non-physical token mass is preserved, never dropped. Residue numbering uses original-sequence indices, so the 261 over-window constructs are not misplaced. Aggregation policy fixed in advance and content-hashed into every export |
| **iBAM plots** | **implemented** | `bindra/ibam/plots.py`; viridis-only (not configurable), PDF+SVG with optional 600-dpi PNG, zoomed residue windows, molecular depiction mapped through `std_to_output` so contributions land on the right atoms. Disclaimer and aggregation hash burned into every figure. Colour scale anchored at zero so a near-uniform map renders as uniform rather than being stretched into apparent structure |
| Retrieval projections / ANN index | absent | `pair_representation` is the intended source |

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

## What this stage established about the corpus

These are measurements, not design intentions, and several of them constrain what
any later result can mean. Full detail in [`CURATION_REPORT.md`](CURATION_REPORT.md).

- **Measurement error.** Pooled within-publication replicate SD is **0.945 p-units
  for pKd**, 0.641 for pKi, 0.671 for pIC50. pKd is the *noisiest* endpoint as well
  as the scarcest, so a pKd RMSE near 0.95 sits at the noise floor rather than
  being good. Every curation threshold is derived from these numbers rather than
  taken from convention — including the ranking margin, which the specification
  starts at 0.5 and which is **below the noise floor of every endpoint**.
- **Dependence.** 10,474 targets, but a Kish effective sample size of **572**
  target units, with the top 1% of targets carrying 34.2% of observations.
  Intervals must come from a cluster bootstrap over protein components.
- **No assay identifier exists.** Not a missing value — the column is absent from
  all 640. Every within-assay quantity rests on a named proxy, and a record with no
  citation of any kind forms a singleton group so it cannot enter ranking.
- **Censoring is informative and bidirectional**, piling up on round assay limits
  (IC50 > 10,000 nM; IC50 < 100 nM). Dropping it would thin both tails of the label
  distribution, and it makes conformal exchangeability questionable across assay
  regimes.
- **Patent data is 41.6% of the corpus** and its replicate SD is *lower* than
  literature's, which indicates re-statement rather than independent measurement.
- **Roughly half of activity cliffs are unanswerable by a topological
  fingerprint.** Of 747 cliffs found in a 4,000-group sample, 325 are
  stereochemical (ECFP4 without chirality cannot see them) and 40 are
  fingerprint-blind homologues differing in chain length, ring size or linker
  length — at the same median magnitude as the 382 ordinary substitution cliffs.
  `cliff_kind` is therefore a prespecified diagnostic category, and cliff metrics
  must be reported per kind rather than pooled.

## Split and panel sizes, as built

| Regime | train | val | cal | test | quarantine |
|---|---|---|---|---|---|
| random_pair | 1,917,944 | 273,992 | 273,992 | 273,992 | 0 |
| cold_drug | 1,825,626 | 181,674 | 181,674 | 550,946 | 0 |
| cold_target (60%) | 1,917,944 | 273,992 | 273,992 | 273,992 | 0 |
| cold_target_strict (40%) | 2,695,321 | 14,867 | 14,866 | **14,866** | 0 |
| double_cold | 1,125,513 | 18,734 | 20,317 | 104,669 | 1,470,687 |
| temporal | 2,170,213 | 48,723 | 59,095 | 227,514 | 198,748 + 35,627 repeated |

| Panel | Size |
|---|---|
| activity_cliff | 111,532 pairs (129,874 candidates dropped as not in test) |
| selectivity | 1,716,186 exact triples (incl. swapped controls) |
| censored_selectivity | 1,161,343 interval triples |
| selectivity unknowns | 1,009,606 (never negatives) |
| fishing | 3,721 targets, 81,971 actives, 37,187 measured inactives, 1,926,939 presumed-control decoys |
| temporal | 227,514 strict future / 35,627 repeated / 75,819 re-import control |

`cold_target_strict` is tiny by construction, not by accident: the 93% giant
component has to go somewhere, so it goes to train. That contrast is the point of
building both.

Fishing reports a **decoy shortfall of 122,336** of ~2.05M requested: the pool
could not fill the quota inside the declared property tolerances. Reported rather
than fixed by loosening the match.

## Decisions that overturned a default, with the evidence

1. **Semi-global alignment, not global.** Against same-UniProt positives and
   cross-UniProt negatives: semi-global reached sensitivity 0.987 at a 0.000
   false-positive rate; global reached 0.867 at 0.043. Global forces end-to-end
   alignment, which pushes background identity for unrelated proteins to a median
   of 0.252 and pins coverage near 1.0 so the 80% criterion filters nothing. On a
   500-construct sample, global produced **one component holding all 500
   sequences** — cold-target evaluation would have been impossible. Semi-global
   produced 394 components, largest 54 (10.8%).
2. **No identity ambiguity margin.** Optimal alignments are rarely unique
   (Biopython reports 2,695,680 co-optimal alignments for one real pair), so a
   conservative margin looked right. Measured, a 0.08 margin turned 134 qualifying
   edges into 473 and the largest component from 10.8% to 53.8%. Near-threshold
   cross-partition pairs are audited individually instead.
3. **Tautomer canonicalization disabled.** 96.5 ms/mol against ~4 ms/mol for the
   whole rest of the pipeline — 2.8 h instead of 7 min — and InChI's
   mobile-hydrogen layer already collapses the common classes. Available as a flag.
4. **IC50 Tier B keyed on curation source class, never on PubChem AID presence.**
   68% of rows carry an AID, including ~545,254 whose provenance is a US patent.
5. **Two cold-target regimes, not one.** At 40% identity single linkage puts
   10,066 of 10,820 constructs (93.0%) in one component covering 98.4% of
   observations, leaving 1.6% for a test set. The graph shatters between 0.50 and
   0.60 and is flat above it, so `cold_target` groups at 60% and
   `cold_target_strict` at 40%. The looser regime's audit publishes the 17,981
   cross-partition pairs at 40-60% identity that it permits to straddle.

## Defects found and fixed in this stage

- **`doctor` validated a different code path than the cache run.** It used the
  deprecated `ESMC` shim and `output.embeddings`, the three defects
  [`MIGRATION.md`](MIGRATION.md) records as fixed. Now routes through
  `ESMCProteinEncoder`.
- **The weights checksum was absent from cache keys** whenever metadata came from
  `encode()`, because it defaulted to `None`. A corrupted or locally overridden
  checkpoint would have produced entries indistinguishable from pinned ones.
- **34 constructs are written in lower case**, and ESM-C maps lower-case residues
  to `<unk>`, which is a *special* token — so special-token stripping removed every
  residue of the protein. The adapter's strict guard caught it rather than caching
  437 `<unk>` embeddings. Sequences are now canonicalized at ingest and the encoder
  rejects non-canonical input with a precise message.
- **The dedup key omitted assay conditions**, so two records from one paper
  reporting the same value at different pH were collapsed as re-imports.
- **`parasail`'s 16-bit kernel silently saturates** (score 0) on the
  34,350-residue construct. Only the `_sat` kernels are used and a saturated
  result raises.
- **The census used salted `hash()`** for unique-sequence counting, so the recorded
  11,018 was not reproducible across runs.

## Active blockers

1. **DGL / KPGT — RESOLVED.** KPGT now runs, on the RTX 5060 Ti, with no DGL
   anywhere in `bindra`. See the ledger rows above and `envs/kpgt/README.md`.

   The blocker was resolved by **porting LiGhT to pure PyTorch** rather than by
   finding a working DGL. The port is verified stage by stage against the
   reference implementation executing in the CPU-only DGL environment: max
   absolute deviation 4.8e-08 (small config) and 6.9e-06 (base, 92M parameters),
   which is float32 reassociation, and the base state dict loads with
   `strict=True` and zero key mismatches.

   **Two corrections to what was previously recorded here.** This section first
   said DGL "stops at Python 3.12 / no longer ships Windows CUDA wheels", which
   was wrong; it was then corrected to "DGL cannot run on this GPU at any
   version", which was also not quite right:

   - This machine has **two** CUDA devices — the RTX 5060 Ti (sm_120, 16 GiB)
     **and** a T1200 Laptop GPU (sm_75, 4 GiB). DGL's cu121 build cannot target
     sm_120, but it *could* have run on the T1200. The option was rejected on
     merits (4 GiB, two generations old, permanent dependency on a package last
     released May 2024), not on impossibility.
   - torch's device *indices are not stable between runs* on this machine: the
     5060 Ti appeared at `cuda:0` and then at `cuda:1` within one session, and
     nvidia-smi orders the two cards the opposite way from torch. Any selection
     by index is unsafe. `bindra/runtime/device.py` already selects by
     requirement and documents this; everything must go through it.

   Remaining for a *pretrained* KPGT, neither of which blocks the architecture:

   - **Weights are not downloaded.** They are on figshare
     (`figshare.com/s/d488f30c23946cf6898f`, file `base.pth`) behind an AWS WAF
     JavaScript challenge that curl cannot clear, and no HuggingFace or GitHub
     mirror exists. This needs one manual browser download. Until then the port
     is verified only against randomly initialised weights — which is a complete
     verification of the *forward pass*, since numerical equivalence does not
     depend on the values.
   - **The 200-dim descriptor input is not wired.** KPGT's second knowledge node
     takes `RDKit2DNormalized` descriptors from vendored, Apache-2.0,
     DGL-free code in the reference repo. It needs `pandas_flavor` and one shim
     (`scipy.stats.gilbrat` was renamed `gibrat`). The 512-bit fingerprint input
     is already ported (`kpgt_graph.rdk_fingerprint`).

2. **Legacy ensemble is incomplete.** `checkpoints/` in the legacy repo holds only
   `max_rmse_cluster_diffusion_member_0`, but that profile declares
   `ensemble_size=5` and the legacy loader requires all five. The reproducible
   legacy baseline is therefore **single-member inference only**. Recorded legacy
   validation metrics, in KIBA units, not reproduced here: RMSE 0.473, CI 0.868,
   R² 0.689, epoch 47.

3. **Protein encoding is overhead-bound, not GPU-bound.** Measured GPU utilisation
   during the full encode is ~8%: the adapter encodes one sequence per forward pass
   and most constructs are short (median 454 residues). Batching by token budget is
   the obvious optimisation and is not implemented; the current cost is tolerable
   because it runs once.

## Training throughput

Measured end to end on the real `random_pair` validation partition, RTX 5060 Ti,
bf16 autocast, both encoders frozen. **124 -> 328 pairs/s, 257 -> 97 minutes per
epoch over 1,917,944 training pairs.** Where it came from, largest first:

| change | effect |
|---|---|
| `pad_token_states` rewritten as two scatters | **48.9 ms -> 0.55 ms** at batch 256 |
| ligand featurization: numpy edge build, ring-size scan, array conversion | 2.69 -> **1.57 ms/mol** |
| protein padding kept in bf16 until the device | 48 ms/batch and half the host-to-device bytes |
| three GPU synchronisations removed from the step | ~6 ms each |
| bf16 autocast | 1.3x |
| prefetch threads overlapping the batch build | 238 -> 328 pairs/s |

Two findings worth keeping, because both contradicted a plausible guess:

- **The syncs were the whole problem, not the arithmetic.** `pad_token_states`
  looped over molecules calling `BatchedGraphs.graph_slice`, which does
  `int()` on a CUDA tensor — two device-to-host stalls per molecule, 190
  microseconds each, 49 ms for a batch of 256. The profiler put `aten::item` at
  65% of CPU time. `collate_graphs` now precomputes the padding map in numpy,
  where the data already lives, so the forward pass never asks the GPU a
  question it must stop to answer.
- **Skipping the attention maps saved nothing.** A `need_attention=False` path
  was added on the theory that materialising `[B, H, Lq, Lk]` per layer
  dominated memory. Measured: 1.00x speed, 1.02x memory — ligand token counts
  are ~13-36, so the maps are small. The flag is kept because it is correct and
  iBAM needs the explicit path anyway, but it is not an optimisation.

Step cost is governed by `batch_pairs x padded_residues`, which is exactly what
`BatchingPolicy.residue_budget` controls: ~110 ms per step at 65,536 slots
regardless of how those slots are divided. Raising the budget gives diminishing
returns (1,166 -> 1,330 -> 1,401 pairs/s for 1x -> 2x -> 4x memory), so the
default stays at 65,536 and leaves headroom.

**Next lever, not taken:** ligand featurization is still 72% of the batch build.
It is a deterministic function of the SMILES, so a graph cache would remove it
entirely — ~11.5 GiB in fp16, or ~0.6 GiB bit-packed, since 75 of the 78 atom
channels are binary and the three continuous ones are functions of the element.
Not built, because the format is a real design choice and the GPU is now the
bottleneck anyway.

## Decisions still pending

- **Both encoders fine-tune their last layer.** Defaults are
  `KPGTConfig.n_trainable_layers = 1` and
  `ESMCProteinEncoder(n_trainable_layers=1)`.

  | | trainable | of total | per effective target |
  |---|---|---|---|
  | KPGT, last of 12 blocks | 7,087,872 | 7.7% of 91.9M | — |
  | ESM-C, last of 36 blocks + final norm | 15,933,312 | 2.8% of 573.6M | 27,855 |
  | **combined** | **23,021,184** | | |

  Joint forward+backward peaks at **3.0 GiB of 15.9**, so memory is not the
  constraint. Three layers each was measured first and stepped back to one; the
  comparison is kept because it shows where the cost actually is:

  | | ESM-C trainable | throughput | per effective target |
  |---|---|---|---|
  | 1 block (chosen) | 15,933,312 | 13.9 prot/s | 27,855 |
  | 3 blocks | 47,797,632 | 14.5 prot/s | 83,562 |

  **Cutting from three unfrozen blocks to one buys no speed at all** (13.9 vs
  14.5 proteins/s, within noise). The full 36-block *forward* is unavoidable
  either way, and backward through one block instead of three is a small part of
  the step. What it buys is 3x fewer trainable protein parameters against the
  axis where this corpus is poorest.

  *Both caches remain invalid in this regime, and that is enforced.* A cached
  feature is valid only because it is a deterministic function of its input; one
  trainable block ends that. `ESMCProteinEncoder.assert_cacheable` and
  `KPGTLigandEncoder.assert_cacheable` refuse, `encode_to_cache` and
  `bindra cache proteins` call them, and `n_trainable_layers` is in both cache
  keys. `bindra cache proteins` constructs its encoder with
  `n_trainable_layers=0` **explicitly** rather than inheriting a default,
  because caching requires frozen and that path should state its own
  requirement. Freezing is applied at construction, not only after a checkpoint
  load — an injected model previously skipped it entirely, since
  `_ensure_loaded` returns early when one is supplied.

  *ESM-C fine-tuning still needs target-grouped batching to be tractable.* The
  arithmetic is unchanged by the layer count, because it is driven by how many
  distinct proteins a batch forces through the encoder:

  | batching | protein forwards / epoch | hours / epoch |
  |---|---|---|
  | one forward per observation | 2,739,920 | **54.6** |
  | de-duplicated within a random batch | ~2,160,000 | **43** |
  | target-grouped, 4 targets per batch | 85,622 | 1.7 |
  | target-grouped, 1 target per batch | 21,406 | 0.4 |

  Random batching is infeasible. Target-grouped batching is cheap but correlates
  gradients within a batch and interacts with the comparability-group structure
  curation built. `data/batching.py` does not exist yet and this is its hardest
  requirement.

- **Open risk: 15.9M protein parameters against 572 effective targets.**
  Recorded rather than silently accepted. The corpus has 10,474 observed targets
  but a Kish effective sample size of **572 target units**, so one unfrozen
  ESM-C block is ~27,900 trainable parameters per effective independent target.
  The protein axis is where this corpus is poorest, and the usual failure is the
  encoder fitting the targets rather than the interaction. The split audit cannot
  detect it: a model that memorised a target family still scores well on held-out
  *ligands* for those targets. Only `cold_target` and `double_cold` would show
  it, so any run with ESM-C unfrozen must report those alongside the random
  split, and the gap between them is the diagnostic. The ligand side does not
  carry this risk: 1,436,561 distinct molecules against 7.1M trainable KPGT
  parameters.

- **Double-cold partition sizing.** At 70/10/10/10 on both axes the retained share
  is ~0.52 and the test partition receives ~1% of interactions. A two-level cascade
  is specified in `SplitPolicy` but has not been exercised on the corpus.
- **Temporal validation/calibration source.** The cutoff is 2023. A nested variant
  (train ≤ 2022, val 2023-H1, cal 2023-H2, test ≥ 2024) is configured so that
  tuning and calibration also respect a cutoff; the alternative keeps all of ≤ 2023
  for training and draws val/cal by grouped random within it.
- **Ligand cache strategy.** Now a decision with numbers behind it, measured on
  4,000 real curated molecules (median 30 heavy atoms, max 327):

  | Quantity | Measured |
  |---|---|
  | Featurization | **308 mol/s** single core (3.25 ms/mol) → 78 min for 1,436,561 molecules on one core |
  | GAT forward, batch 1024 | **9,189 mol/s** on the 5060 Ti, peak 652 MiB → 5.0 min per epoch over 2,739,920 observations |
  | GAT forward, batch 256 | 3,207 mol/s → 14.2 min per epoch |
  | Cached graphs, naive float32 | **17.44 GiB** projected |

  Token states are *not* cacheable here: the ligand encoder is trained, so its
  activations change every step. Graphs are. The open question is whether graphs
  are cached at all, and the measurement makes the trade-off concrete rather than
  speculative: 15 featurization workers give ~4,620 mol/s, which is **below** the
  9,189 mol/s the GPU can consume, so on-the-fly featurization would bottleneck
  the ligand path in isolation. Whether it bottlenecks *training* depends on the
  full step cost (protein side, cross-attention, backward), which does not exist
  yet, so the decision waits for a measured step time rather than being guessed now.

  If caching is chosen, 17.44 GiB is not the real cost: 75 of the 78 atom channels
  are binary, and the three continuous ones (mass, atomic number, covalent radius)
  are each pure functions of the element, so an atom compresses to ~11 bytes from
  312 — roughly 0.6 GiB for the corpus. That is a design note, not an implemented
  format.

## Deliberately not claimed

- **No BINDRA model has been trained. There are no BINDRA performance numbers.**
  The tiny-data overfit (RMSE 7.740 → 0.0264 on 8 pairs) is a **wiring test and
  nothing else**: it shows gradient reaches every shared module and that the
  ligand signal affects the output. It is deliberately memorisation of 8 points by
  a model with far more parameters, says nothing about generalisation, and must
  never be quoted as a result.
- The GAT arm is still the ligand encoder actually wired into the model. KPGT's
  architecture is ported and verified, but it is not yet an encoder arm: the
  descriptor input is unwired and the published weights are not downloaded. Until
  both are done, **no claim about KPGT performance is supported**, and the port
  is verified for numerical equivalence only — which says nothing about accuracy.
  The encoder in use is recorded in `EncoderMetadata.name` and in every cache key.
- The attention maps the fusion stack retains are **not** contact predictions.
  `FusionOutput.contact_map` is a derived aggregation of two differently-oriented
  distributions; its rows do not sum to 1 and it has been compared to no
  structural ground truth.
- No retrieval, generation, or iBAM results exist.
- Protein grouping is exact at the stated threshold, but 40% identity / 80%
  coverage is an operational threshold and not proof that shared domains or
  families are absent across partitions.
- DIP2A is an exploratory application. No DIP2A binding assay is assumed to exist,
  and no compound is treated as a validated DIP2A binder.
