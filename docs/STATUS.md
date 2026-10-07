# BINDRA status ledger

The specification is explicit: *"Do not mark all requested functionality complete
merely because interfaces exist. Separate implemented, smoke-tested, trained,
benchmarked, and experimentally validated status."*

This file is that separation. It is the authoritative answer to "does BINDRA do X
yet?" — not the presence of a module, not a passing import.

**Last updated:** 2026-10-06 (first two complete training runs; a trained model
and val-partition numbers now exist, and the locked panels are shown to be
unusable for a `random_pair`-trained model)

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

Nothing below is `validated`. Nothing below is `benchmarked`: no model has been
scored on the locked panels, and [the panels cannot score either run that
exists](#why-the-locked-panels-cannot-score-these-runs). Two models are
`trained`, on the `random_pair` regime only, and **the only BINDRA performance
numbers that exist are on a validation partition of that one regime** — see
[Trained models and what they measured](#trained-models-and-what-they-measured).
No number below speaks to held-out ligands, unseen targets, or a future cutoff.

## Priority 0 — environment and legacy preservation

| Component | Status | Evidence |
|---|---|---|
| CUDA environment (torch 2.11.0+cu128) | smoke-tested | `bindra doctor`: RTX 5060 Ti, sm_120, 48.6 TFLOP/s bf16 |
| GPU selection by capability, not index | implemented | `bindra/runtime/device.py`; rejects the 4 GB T1200. Selection is correct, but the *driver* is intermittently not: see the note below this table |
| ESM-C-600M forward pass on GPU | run-on-corpus | `doctor --esmc --verify-weights` now routes through the real adapter |
| KPGT isolated environment | built, CPU only, **job done** | Served as the numerical oracle for the pure-PyTorch port; kept only for re-verification. `envs/kpgt/README.md` |
| Legacy baseline reproduction | absent | Single-member only, see blockers |

**Observed 2026-10-06: the GPU intermittently faults under a correct selection,
and the symptom is misleading.** Three successive evaluation runs of the same
checkpoint, with identical arguments, failed twice and then succeeded. Two
distinct symptoms, same cause: a silent process death (exit code 4, no
traceback), and a `torch.OutOfMemoryError` reporting **"0 bytes is free"** of
15.90 GiB while PyTorch itself held only 35.33 MiB. Polling the card every 4 s
through the run that succeeded showed peak usage of **~1.4 GiB against ~14.7 GiB
free throughout**, so this workload uses under a tenth of the card and the OOM
message is spurious. `nvidia-smi` across the same window showed the T1200
appearing, vanishing and returning, i.e. device enumeration itself flapping.

Two things this is *not*, both checked and ruled out: it is not the wrong device
(`cuda:1` **is** the 5060 Ti under the `CUDA_DEVICE_ORDER=PCI_BUS_ID` that
`bindra/__init__.py` sets, and pinning by UUID did not stop the failures), and
it is not the multiprocessing loader (it fails with `--workers 0` too).
**Treat an OOM on this machine as suspect until the free-memory figure is
checked against a poll**; retrying the identical command is a legitimate first
response. Two asymmetries worth closing when this is next touched:
`DeviceRequirement` has a `require_uuid` field that `cli/evaluate.py` does not
pass, and `cli/train.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
while `cli/evaluate.py` does not.

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
| KPGT wired into `BindraModel` | **implemented** | `bindra/encoders/ligand_arm.py` adds the module-level arm contract (`LigandArm`, `FusionTokens`); `ligand_kpgt_arm.py` adds `KPGTLigandArmModule` and `KPGTLigandBatch`. `BindraConfig.ligand` now takes either config and reads widths through `token_dim`/`graph_dim`, so the model never names an arm. Both arms run end to end and gradients reach every shared prefix. `tests/test_ligand_arm.py`, 23 tests; token counts asserted against **bond** counts, not triplet-count-minus-two, so dropping the wrong two nodes would fail. Full suite 1,082 passed after the refactor, and the trained `v2-mid` checkpoint still loads `strict` at 9,254,787 parameters |
| Ligand featurization behind the seam | **implemented** | `bindra/encoders/ligand_featurizers.py`; `LigandBatchBuilder` carries the arm *name* so it and its featurizer stay picklable for the process pool, which a bound method or lambda would not. `PairDataset` takes a `ligand_builder`; `build_batch` dispatches through it and catches each arm's own featurization errors. `bindra train --ligand-arm {gat,kpgt}` derives the featurizer from the same config object the model is built from, so the two cannot disagree |
| **LiGhT-mini — trainable now, no weights needed** | **implemented** | `KPGTConfig.mini()`: `d_g_feats=192`, 4 layers, 8 heads, **2,232,352 trainable** against the GAT arm's 2,101,120 (6% apart) and 9,324,195 whole-model against v2-mid's 9,254,787 (0.8% apart), so an arm comparison is about bond-level vs atom-level tokens rather than capacity. Trains **from scratch**: a `from_scratch` regime unfreezes the embeddings and attention biases, which no setting of `n_trainable_layers` reaches. Reported as **`ligand_light_mini`**, never anything containing KPGT — see the naming note below this table. `--ligand-arm kpgt` |
| KPGT **base** as a training option | **removed from the CLI** | `--kpgt-size` and `--ligand-trainable-layers` are gone; `--ligand-arm kpgt` builds LiGhT-mini and nothing else. Without `base.pth` the base shape can only be partial fine-tuning over randomly initialised weights, which `cmd_fit` refused anyway, so the flag's one reachable outcome was an error message. **The configuration is still expressible in library code** — `KPGTConfig()` defaults to the published 91.9M shape and `load_pretrained` loads the checkpoint `strict=True` — because that is what the port's verification rests on and what the CLI returns to when the weights arrive |
| **Shared adapters** | **implemented** | `bindra/fusion/adapters.py`; widths read off encoder metadata, nothing hardcodes 1152. `attribute_gradients` discharges "log which parameters receive gradients from each loss" by backpropagating each loss in isolation; a stray `detach()` is caught by test rather than by inspection |
| **Bidirectional cross-attention** | **implemented** | `bindra/fusion/cross_attention.py`, 41 tests with the adapters. Two independent attention blocks, asserted **not** to be transposes of one another; applied in parallel so the returned maps describe the inputs. Padding invariance asserted end to end; an all-masked side yields zeros rather than NaN |
| **Per-endpoint heads (pKd/pKi/pIC50)** | **implemented** | `bindra/heads/affinity.py`; three independent trunks, unbounded output. No target exists in the module, so a fabricated zero label is not expressible; gradient to an unmeasured endpoint asserted to be exactly 0 |
| **Censored affinity loss** | **implemented** | `bindra/training/losses/affinity.py`, 49 tests. Huber delta = **measured sigma_rep per endpoint** (0.945/0.641/0.671), so residuals inside the replicate noise are treated as noise. One-sided penalties for bounds; Tier B structurally confined to the auxiliary term; exact RMSE and bound-violation rate reported separately |
| **Assembled model** | **implemented** | `bindra/model.py`, 26 tests. **Tiny-data overfit: exact RMSE 7.740 → 0.0264** over 400 steps on 8 pairs across all three endpoints (predicting the label mean gives > 2.0, so the bar excludes the trivial solution). Nonzero shared-module gradients verified per task against the prefixes the model itself declares. Save/reload reproduces inference exactly; atom renumbering changes nothing |
| **iBAM numeric export** | **implemented** | `bindra/ibam/export.py`, 50 tests with the plots. Native residue x token **and** derived residue x atom views; bond tokens split equally between endpoints per the spec and labelled derived (the flag cannot be turned off). Non-physical token mass is preserved, never dropped. Residue numbering uses original-sequence indices, so the 261 over-window constructs are not misplaced. Aggregation policy fixed in advance and content-hashed into every export |
| **iBAM plots** | **implemented** | `bindra/ibam/plots.py`; viridis-only (not configurable), PDF+SVG with optional 600-dpi PNG, zoomed residue windows, molecular depiction mapped through `std_to_output` so contributions land on the right atoms. Disclaimer and aggregation hash burned into every figure. Colour scale anchored at zero so a near-uniform map renders as uniform rather than being stretched into apparent structure |
| **Pair batching** | **implemented** | `bindra/data/batching.py`, 28 tests. Residue-budgeted and length-bucketed, forced by a protein length range of 454 median to 34,350 max: a fixed batch padded to the longest member would be 5 GiB of features alone. 2.2% padding waste on the real distribution; the 34,350-residue construct gets a batch of one rather than being dropped |
| **Training engine** | **trained** | Two complete 10-epoch runs on `random_pair`; `bindra/training/engine.py`, 33 tests. AdamW, cosine schedule with warmup, bf16 autocast, EMA evaluated in place of raw weights, early stopping on exact RMSE. Resumable: model, optimizer, scheduler, EMA, epoch, step, best metric and both RNG streams, written atomically. **Refuses to train on or select against anything but `train` and `val`**, and imports no sealed-label machinery (asserted on the import graph) |
| Retrieval projections / ANN index | absent | `pair_representation` is the intended source |

**What remains before the KPGT arm can be trained.** The model seam and the
data pipeline are both done; two items are left, and the first is not a code
change at all:

1. **The published weights are still not downloaded**, which blocks the *base*
   configuration only, which is no longer selectable from the CLI — `--ligand-arm kpgt` trains LiGhT-mini today without them. Two
   failure modes are now refused by `cmd_fit` rather than trained:

   - `frozen_and_untrained()` — no trainable layers and no checkpoint, i.e. a
     fixed random projection. Loud and obvious.
   - `partially_frozen_at_random()` — **the quiet one**. Partial fine-tuning
     with no checkpoint loaded pins most of the encoder to its random
     initialisation while the trailing blocks learn to compensate. It trains,
     and it reports a loss curve, so nothing visibly breaks. Measured at the
     default `n_trainable_layers=1`: **82,463,960 parameters frozen at random**.
     Even at the maximum `n_trainable_layers=12` it is still 4,497,368, because
     no setting of that flag reaches the embeddings — which is why
     `from_scratch` had to be a separate regime rather than a larger count.

   `load_pretrained` loads `base.pth` with `strict=True` when it arrives. The
   download is **the only item a person has to do**: one manual browser pass
   through figshare's JavaScript challenge.
2. **Whether the descriptors need a cache is unmeasured.** ~200 RDKit
   descriptors per molecule are this arm's dominant cost, and they are now
   computed **in the existing process pool** — the right place, since the pool
   exists precisely for GIL-bound RDKit work and the payload back is ~3 KiB a
   molecule against a protein tensor of ~144 MiB that never crosses. Whether
   eight cores are *enough* is a measurement that has not been taken. Deciding
   the cache format before measuring would repeat the mistake this ledger
   records twice elsewhere: the ligand-graph cache was deferred for exactly
   this reason, and the two failed throughput hypotheses above were both
   plausible guesses that measurement overturned. The descriptors are
   cacheable if needed (deterministic in the SMILES), unlike the token states,
   which move every step once any layer is trainable.

**Why the from-scratch arm is called LiGhT-mini and not any kind of KPGT.**
The authors describe KPGT as two separable things: *"a novel graph transformer
architecture, LiGhT, and a knowledge-guided pre-training strategy"*. This
project has the first and, without `base.pth`, cannot have the second — and the
pre-training is the contribution the acronym names. A label like `kpgt_scratch`
would keep the acronym while negating one of its letters, asking every reader to
hold a correction in mind, and a results row would still be filed mentally
against published KPGT numbers. `ligand_light_mini` gives a reader nothing to
index against, which is the point.

It is not a hedge: the knowledge inputs survive. The 512-bit fingerprint and the
200 descriptors still condition every forward pass and still occupy two nodes per
molecule, because they are inputs to the architecture. Only the objective that
*reconstructed* them is gone. The name reaches `EncoderMetadata.name` and every
provenance record, so a results table states which arm produced it.

**One decision was taken rather than deferred, and it is worth knowing about.**
KPGT's two knowledge nodes per molecule are **excluded from the fusion token
axis and retained in the graph readout**. Padding them in would give the protein
a token to cross-attend to whose content is a fingerprint vector, producing an
attention map with mass on something that cannot be drawn on a molecule — the
thing `TokenType.is_physical` exists to prevent. They still reach the pair
representation through `graph_states`, which KPGT's readout builds as
`fp node ++ md node ++ mean of the rest`. Virtual-atom triplets (`vavn == -1`)
are kept as physical: they carry one real atom, and the reference's own readout
counts them via `indicators <= 0`.

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
     JavaScript challenge that curl cannot clear. This needs one manual browser
     download. Until then the port is verified only against randomly initialised
     weights — which is a complete verification of the *forward pass*, since
     numerical equivalence does not depend on the values.

     **Re-checked 2026-10-06, four routes, all dead.** Recorded so the search is
     not repeated: the share URL returns HTTP 202 with a zero-byte body (the
     challenge); `api.figshare.com` ignores `private_link` on an article listing
     and the article id is not derivable from the token; the GitHub release
     `v1.0.0` carries **no assets**, only a source tag; the Zenodo archive
     (`10.5281/zenodo.8418818`) is the 573 KB **source zip**, not the ~350 MB
     checkpoint; and a HuggingFace model search for KPGT returns nothing. The
     repo README still points at the same figshare link, so upstream has not
     moved it.
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

## First real training run, and two optimisation hypotheses that failed

*Superseded as a result by the completed runs in [Trained models and what they
measured](#trained-models-and-what-they-measured). Kept for the throughput
measurements and the two failed hypotheses, which still hold.*

`bindra train --regime random_pair` reaches **1,917,944 pairs in 22,044 batches,
1.9% padding**, and learns: over the first 5,601 steps loss fell 9.80 -> 2.06 and
exact RMSE 6.50 -> 2.08, with all three endpoint heads improving together
(Kd 2.86x, Ki 3.66x, IC50 2.88x of their measured noise floors). The run was
stopped by request at 25% of epoch 0.

Sustained throughput over the two runs that later completed was **196-207
pairs/s on train and 279-334 on val, so 2.6-2.7 h per training epoch** — inside
the range this section projected, at the low end.

Sustained throughput is **~150-200 pairs/s, so an epoch is 2.7-3.5 hours**, not
the ~1.6 h a warm-slice benchmark suggested. Two attempts to close that gap did
not work, and both are recorded because the negative result is the useful part:

- **Moving the 15 GiB protein cache off OneDrive to local disk did not help.**
  Measured 149 pairs/s from `C:/bindra/cache/proteins` against 196 from the
  OneDrive path. The I/O hypothesis is not supported; the earlier 328 pairs/s
  came from a small `val` slice with a different batch composition, not from a
  warmer cache.
- **Raising the residue budget OOMs on real batches.** 65,536 peaks at 6.66 GiB,
  but 131,072 exhausts 15.9 GiB. A synthetic sweep had suggested 5.44 GiB at
  that budget; real molecules carry far more atoms than the repeated aspirin it
  used, and the ligand side scales with batch size. 65,536 stands.

So the remaining headroom is **not** in the configuration. The profile still
says ligand featurization is 72% of the CPU batch build, and it is a
deterministic function of the SMILES, so a graph cache is the next real lever.

**Mid-epoch checkpointing was added in response to this run.** It had written
nothing when stopped 42 minutes in, because `save_checkpoint` only ran at epoch
boundaries -- a cadence sized against a projected 97-minute epoch rather than
the measured one. `checkpoint_every_steps` defaults to 500, and a mid-epoch
checkpoint resumes *inside* its epoch, skipping the batches already seen rather
than replaying and double-weighting them.

**Learning rate and batch size are untuned.** 3e-4 with cosine decay and 5%
warmup is a plausible default, not a validated choice, and no sweep has been
run. Model size is no longer untested: the capacity step from 5.73M to 9.25M
parameters was taken against a baseline and won on every metric, as recorded
below. That is one comparison at one width, not a tuned width.

## Trained models and what they measured

Two runs completed the full 10-epoch budget on `random_pair`, both with **both
encoders frozen** (`n_trainable_layers=0`), so the protein features come from
the verified cache and the open risk about 15.9M trainable protein parameters
against 572 effective targets does **not** apply to either.

| | `gat-frozen-v1` | `v2-mid` |
|---|---|---|
| parameters | 5,728,771 | 9,254,787 |
| model dim / ligand hidden | 256 | 320 |
| final val exact RMSE | 1.1666 | **1.0809** |
| bound-violation rate | 0.4888 | **0.4388** |

Both plateaued in the last two epochs (v1 1.1678 → 1.1666; v2-mid 1.0830 →
1.0809), so the 10-epoch budget was roughly right rather than a premature stop.
A third run, `gat-frozen-v1-new`, **diverged** — val RMSE climbed 2.40 → 2.97 →
3.39 over three epochs — and is retained as a failure, not a candidate.

`v2-mid` is the kept model. Scored from `runs/eval-v2-mid-e9.json`, EMA weights,
on a fixed 30,000-pair slice of the `random_pair` **val** partition (23,873
exact labels; the same slice as every earlier eval, so the rows are comparable):

| | v1 ep9 | v2-mid ep4 | **v2-mid ep9** |
|---|---|---|---|
| RMSE | 1.1802 | 1.1972 | **1.0899** |
| MAE | 0.8900 | 0.9087 | **0.8177** |
| concordance index | 0.7437 | 0.7392 | **0.7678** |
| Spearman | 0.6648 | 0.6559 | **0.7183** |
| AUC@100nM | 0.8198 | 0.8169 | **0.8498** |

Per endpoint at epoch 9, beside the measured replicate-noise floor:

| endpoint | n | RMSE | floor | × floor | fold error | CI | rho |
|---|---|---|---|---|---|---|---|
| IC50 | 17,966 | 1.0493 | 0.671 | 1.56× | 11.2× | 0.773 | 0.729 |
| Ki | 5,433 | 1.2047 | 0.641 | 1.88× | 16.0× | 0.751 | 0.681 |
| Kd | 474 | 1.2108 | 0.945 | 1.28× | 16.2× | 0.721 | 0.610 |

Three cautions that the table cannot carry. Ki remains the weakest endpoint
relative to its floor. Kd is nominally closest at 1.28×, but that is the
*noisiest* floor (0.945) and only 474 labels, so it is the least trustworthy row
here, not the best result. And the intermediate epoch-4 checkpoint scored *worse
than v1* (1.1972 vs 1.1802), which is what a half-trained checkpoint looks like
and not a regression — quoting it as one would be wrong.

**6,127 censored labels in the slice are excluded from every number above.**
They are 20% of the slice. Bound-violation rate is reported only in the training
history, so the eval RMSE describes the exact-label subset alone.

### Why the locked panels cannot score these runs

`benchmarked` is defined above as measurement on the locked panels, and neither
run can earn it. The three affinity-shaped panels are built from the **temporal**
split's test partition; `random_pair` draws its training set at random across all
dates, so the panels are largely training data for a `random_pair` model.
Measured by observation id against `random_pair`'s own train ids:

| panel | n | share already in `random_pair` train |
|---|---|---|
| panel_strict_future | 227,514 | **60.2%** |
| panel_repeated_measurement | 35,627 | **100.0%** |
| panel_reimport_control | 75,819 | **67.7%** |

So `bindra evaluate --panels` was **not** run against either model, and should
not be. It would produce a contaminated number and the cost is irreversible: the
call opens sealed labels and writes to each panel's access log, which cannot be
un-written. Reaching `benchmarked` requires a model trained on the **temporal**
regime, whose train partition respects the 2022-12-31 cutoff the panels were
defined against. The panels are not at fault and need no rebuild; they were
defined for a cutoff this model never respected.

## Decisions still pending

- **Retiring the GAT arm is deferred until `base.pth` is downloaded**
  (decided 2026-10-06). Removing it was considered and held, for three reasons
  worth keeping so the question is not reopened from scratch:

  1. `random_pair-v2-mid`'s state dict is `ligand_encoder.input_projection.*`
     and `cli/evaluate.py::_load_model` rebuilds it through `GATConfig`.
     Deleting the class orphans the only trained checkpoint and makes every
     number in this ledger unreproducible.
  2. Without the published weights, a KPGT run is 7,087,872 trainable
     parameters on top of ~84.8M frozen **at random initialisation** — fitting
     a head to a fixed random projection, which is worse than the small but
     fully trainable GAT arm. Training all 12 blocks instead would learn, at
     ~2.46 h/epoch, while forfeiting the pretrained representation that is the
     reason to prefer KPGT at all.
  3. The specification names the legacy GAT as **ablation 1**, so it is a
     required comparison rather than a fallback.

  The seam already delivers what removal would: `--ligand-arm kpgt` selects the
  arm and nothing downstream names either one. Once the weights land this
  becomes a clean one-pass change.

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

- **The numbers that exist cover one regime and one partition.** `v2-mid` is
  trained and scored on `random_pair` **val** only. Nothing here measures held-out
  ligands (`cold_drug`), unseen targets (`cold_target`, `double_cold`), or a
  future cutoff (`temporal`), and a `random_pair` number cannot stand in for any
  of them: random splitting puts near-duplicate ligands against the same target on
  both sides. **No model has been scored on the locked panels**, for the
  contamination reason recorded above, so nothing is `benchmarked`.
- **The `random_pair` test and cal partitions remain unopened**, as do all seven
  sealed panels. Only `val` has been scored, which is the partition that is free
  to repeat.
- The tiny-data overfit (RMSE 7.740 → 0.0264 on 8 pairs) is a **wiring test and
  nothing else**: it shows gradient reaches every shared module and that the
  ligand signal affects the output. It is deliberately memorisation of 8 points by
  a model with far more parameters, says nothing about generalisation, and must
  never be quoted as a result. It is superseded as evidence of training by the
  completed runs, and is retained only as the wiring check it was.
- The GAT arm is the only ligand encoder any result came from. KPGT is now a
  real arm — wired into `BindraModel`, fed by the batch pipeline, descriptor
  input included — but **the published weights are still not downloaded**, so
  constructing it gives randomly initialised KPGT and **no claim about KPGT
  performance is supported**. The port is verified for numerical equivalence
  only, which says nothing about accuracy. Nothing has been trained with it: a
  frozen untrained arm is refused outright by `bindra train`, and an unfrozen
  one would be learning from a random initialisation. The arm in use is recorded
  in `EncoderMetadata.name`, in `BindraConfig.to_dict()["ligand_arm"]` and in
  every cache key.
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
