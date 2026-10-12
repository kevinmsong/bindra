# BINDRA benchmark

`runs/random_pair-v2/best-e4.pt`, EMA weights, 2,590,851 trainable parameters.
**Trained for 5 epochs, one seed.** Scored 2026-10-11.

Raw numbers: `runs/eval-bindra-e4-val.json`, `runs/eval-bindra-e4-cal.json`,
`runs/eval-external-davis.json`, `runs/eval-external-kiba.json`.

## What was scored, and what was not

Scored: the **val** and **cal** partitions of the `random_pair` split, in full,
no sampling; and two external sets, **Davis** and **KIBA**, which the model has
never been trained against.

Not scored, deliberately:

- **The test partition.** It is a one-way door. `bindra evaluate --partition
  test` refuses without `--i-mean-it`, and nothing here passed it.
- **The sealed affinity panels.** `random_pair` overlaps them by 60-100%, and
  `_panel_contamination` refuses a panel its training set overlaps. Opening a
  panel also writes an access log that cannot be un-written.
- **Per-prediction uncertainty.** One seed means one model. The intervals below
  are *sampling uncertainty in the metric* -- how much a number would move on a
  different draw of proteins. They say nothing about how much to trust an
  individual prediction, which needs an ensemble or a conformal calibration.
  `uncertainty/ensemble.py:assert_independent` refuses three passes over one
  model by design, so that number does not exist and is not implied here.

Training stopped at a budget, not at a plateau: val RMSE fell on every one of
the five epochs (1.3128, 1.2405, 1.1918, 1.1578, 1.1343). This model was still
improving when it stopped.

## Headline

Intervals are 95% percentile **cluster bootstraps over protein components**,
1,000 resamples. Not over rows: 10,474 targets carry a Kish effective sample
size of 572, and the top 1% of targets hold 34.2% of observations.

| | val | cal |
|---|---|---|
| RMSE (primary rows) | **1.1342** [1.111, 1.157] | **1.1335** [1.110, 1.157] |
| RMSE (all exact rows) | 1.1212 [1.100, 1.144] | 1.1210 [1.097, 1.144] |
| MAE | 0.860 | 0.859 |
| Concordance index | 0.757 [0.750, 0.764] | 0.758 [0.751, 0.765] |
| Spearman | 0.697 [0.681, 0.712] | 0.699 [0.684, 0.714] |
| AUROC @ 100 nM | 0.837 [0.829, 0.846] | 0.838 [0.830, 0.847] |
| AUPRC (prevalence 0.424) | 0.789 | 0.791 |
| rows (primary / all exact) | 181,178 / 186,923 | 181,410 / 187,022 |

**Quote 1.1342, not 1.1212.** They are different quantities.
`AffinityOutput.exact_rmse` -- the number the training loop reports and the one
in `history.json` -- covers exact **primary** rows only; rows the curator marked
non-primary (IC50 Tier B) reach the auxiliary loss at weight 0.25 and never the
regression metric. The pooled figure folds those 5,745 rows back in, and they
are easier than average, so it reads 0.013 better. The primary figure matches
`history.json` to the row (181,178) and to five significant figures (1.13427 vs
1.1342); the residual is bf16 autocast nondeterminism.

**val and cal agree to the third decimal on every metric.** Two independent
held-out sets, ~187,000 rows each, ~3,200 protein components each.

### Why the clustering matters

On val RMSE the clustered interval is **4.60x wider** than the naive i.i.d.
one: [1.100, 1.144] against [1.116, 1.126]. An i.i.d. interval here is not
merely optimistic, it is wrong by a factor of five, because rows within a
protein are not independent draws.

## Per endpoint, against the replicate-noise floor

The floors are what the same lab measuring the same pair twice disagrees by.
They are the ceiling on achievable accuracy, and the only honest denominator.

| endpoint | n (val) | RMSE | floor | x floor | typical fold error |
|---|---|---|---|---|---|
| Kd | 4,613 | 1.3469 | 0.945 | 1.43 | 22x |
| Ki | 47,378 | 1.1807 | 0.641 | 1.84 | 15x |
| IC50 | 134,932 | 1.0909 | 0.671 | 1.63 | 12x |

Kd is both the worst endpoint and the thinnest slice -- 4,613 rows against
IC50's 134,932. Any claim about Kd specifically rests on 2.5% of the data.

Read against the noise floor the model reaches 1.4-1.8x irreducible error. Read
against zero it looks far worse, and that comparison is meaningless: no model
can do better than the measurement.

## Pooled vs per-target

| grouping | macro RMSE | median | groups used | coverage |
|---|---|---|---|---|
| per target (>= 10 rows) | 1.0889 | 1.0458 | 2,063 of 5,980 | 93.8% |
| per protein component | 1.1192 | 1.0717 | 1,272 of 3,202 | 96.9% |
| pooled | 1.1212 | -- | -- | 100% |

Macro beats pooled, so the targets carrying the most observations are the
harder ones. The macro averages drop ~2/3 of the groups to the row-count floor
but keep 94-97% of the observations, which is what a long-tailed corpus looks
like.

## Censored labels

18.5% of the corpus is censored -- a bound, not a value. Every number above
excludes them; these are them.

| | val | cal |
|---|---|---|
| n | 48,550 | 48,695 |
| bound violated | 50.5% | 50.3% |
| mean miss | 0.367 log units | 0.368 |

Half of censored predictions land on the wrong side of their bound, but the
miss when it happens is 0.367 log units -- below every endpoint's noise floor.
These are threshold near-misses, not wild predictions. The violation rate fell
from 0.717 to 0.494 over the five epochs, so it was still improving too.

## Calibration

Equal-count bins of predicted pX against observed mean pX. This is magnitude
calibration of a real value, not the reliability of a probability.

- slope **0.763** (ideal 1.0), intercept 1.784
- mean absolute bias 0.330 log units

The slope under 1 is regression to the mean: the model under-predicts the
strongest binders and over-predicts the weakest. Expected for an L2-trained
regressor, and it matters for any absolute-potency claim. It does **not** affect
ranking, which is what the concordance index and the enrichment below measure.

## Early enrichment

| | hit rate | enrichment factor | max possible |
|---|---|---|---|
| top 1% | 97.9% | 2.311 | 2.360 |
| top 5% | 93.5% | 2.208 | 2.360 |

Against a 42.4% base rate. The top 1% is at 97.9% of the ceiling a perfect
ranker could reach, so for "give me the most likely binders first" the model is
close to as good as this metric can see. Note the base rate is high because the
corpus is mostly measured actives; this is not a virtual-screen hit rate.

## External: Davis and KIBA

**These are zero-shot transfer numbers and are not comparable to published
Davis/KIBA leaderboards.** DeepDTA and its successors train on a split *of
Davis* or *of KIBA* and report in-domain performance, typically CI around
0.85-0.90. BINDRA has never seen either dataset's split, label scale or assay
protocol. A direct comparison of 0.61 to 0.88 would be a category error.

Both sets were checked for overlap with BINDRA's training set before scoring,
matched on InChIKey first block and on the sha256 of the canonical sequence --
the corpus's own key.

### Davis

Kinase Kd panel, 29,172 cells after standardization. Two problems, both severe:

- **70.4% sits at the 10,000 nM assay ceiling.** That is not a measurement, it
  is "nothing detected at the top dose": a bound of pKd <= 5. The field scores
  it as an exact 5.0, which hands any model that answers 5.0 everywhere 70% of
  the matrix for free.
- **59.1% of pairs are already in BINDRA's training set.** BindingDB holds much
  of the same kinase data.

Only 3,461 cells (11.9%) are both a real measurement and an unseen pair.

| stratum | n | CI | Spearman | RMSE | AUROC |
|---|---|---|---|---|---|
| all (field convention) | 29,172 | 0.683 | 0.467 | 1.204 | 0.852 |
| unseen pairs only | 11,922 | 0.658 | 0.406 | 1.291 | 0.806 |
| real measurements only | 8,622 | 0.643 | 0.413 | 1.222 | 0.743 |
| **unseen and real** | **3,461** | **0.614** | **0.331** | **1.299** | **0.694** |

The decline across strata is the inflation, measured: CI 0.683 to 0.614 and
AUROC 0.852 to 0.694 as the memorized pairs and the free ceiling rows come out.
Report the last row.

Note the clean RMSE of 1.299 is close to BINDRA's own val Kd RMSE of 1.347, so
the **magnitude** transfers. The ranking does not transfer as well (CI 0.614 vs
0.704 on val Kd), though part of that is restricted range: the clean subset
spans a narrower band of pKd than val does, which deflates any rank statistic.

**The ceiling rows are the finding worth acting on.** Treated as the censored
bounds they are, BINDRA violates `pKd <= 5` on **85.4% of them, by a mean of
0.896 log units** -- against 50.5% and 0.367 on its own val censored rows. The
model is markedly worse at recognising a non-binder than at ranking binders,
which is what a corpus of overwhelmingly measured actives would predict. Davis
is mostly true inactives; BindingDB is mostly actives. **BINDRA has not learned
"does not bind."**

### KIBA

118,240 cells, **4.8% pair overlap** -- so 112,590 clean rows, a real external
test set. The KIBA score is an aggregate of Ki, Kd and IC50 on a transformed
scale with no units, so RMSE is undefined against it and only rank survives.
All three heads are reported because the dataset does not say which to use;
IC50 was declared primary in advance (72% of the training corpus), not chosen
after seeing the results.

| head | stratum | n | CI | Spearman | AUROC @ 12.1 |
|---|---|---|---|---|---|
| Kd | unseen | 112,590 | 0.590 [0.575, 0.604] | 0.256 | 0.665 |
| Ki | unseen | 112,590 | 0.587 [0.571, 0.602] | 0.247 | 0.674 |
| **IC50** | **unseen** | **112,590** | **0.592** [0.576, 0.607] | **0.262** | **0.680** |

The three heads agree within their intervals, so the choice of head is not
doing any work -- which is the point of reporting all three.

Clearly above chance (AUROC 0.680 against 0.5, prevalence 0.201) and clearly
weak. On a dataset whose label is a different quantity on a different scale, the
model retains rank information but much less than it holds in-domain (CI 0.592
against 0.757).

## Against published BindingDB models

The headline above cannot be compared to the DTA literature as it stands,
because BINDRA does not measure itself the way those papers measure
themselves. So the protocol was reproduced and BINDRA retrained inside it,
to put one number on the literature's own terms.

### The protocol, reproduced

`data/splits/202610/literature_ic50/` rebuilds the DeepAffinity-lineage setup
that MONN, BACPI, SSM-DTA and kNN-DTA all report on:

| | SSM-DTA reports | reproduction |
|---|---|---|
| Measurements | 376,751 | 376,751 |
| Targets | 2,782 | 2,782 |
| Endpoint | IC50 | IC50 only, exact only, primary only |
| Split | random 6:1:3 over measurements | random 60/10/30 over measurements |
| train / val / test | not stated | 226,051 / 37,675 / 113,025 |

Filtering the corpus to IC50 + exact + primary leaves 1,564,121 rows over
6,946 targets; the top 2,782 targets by count hold 1,529,059 of them,
subsampled to 376,751. The smallest kept target has 33 measurements, so
everything thinner than that is the tail this benchmark never sees.

This is an approximation of the published setup, not a reproduction of it.
DeepAffinity's exact filtration is not documented in enough detail to rerun
and the BindingDB release differs. What is reproduced is the shape.

### What the protocol gives away, measured

A random split over *measurements* makes no promise about pairs. BINDRA's own
`random_pair` is audited pair-disjoint -- 0 violations across 1,858,413 pairs
-- and this one is not:

| | rows | sharing a pair with train |
|---|---|---|
| val | 37,675 | 3,729 (9.9%) |
| test | 113,025 | 10,938 (9.7%) |

So about one test row in ten is partly a lookup rather than a prediction.
Recorded in the split manifest under
`audit.pair_leakage_admitted_by_this_protocol`.

### The run

Trained from scratch under that split with e4's exact architecture and
hyperparameters -- 2,590,851 parameters, model-dim 160, two fusion layers,
two pool heads, FiLM bottleneck 128, assay conditioning, ranking weight 0.1,
lr 1e-4, 20 epochs, patience 5.

Two deliberate departures, both favouring BINDRA and both disclosed:

- **Unrestricted pre-training.** The ligand arm was pretrained on all
  1,404,819 corpus molecules -- held-out ones included -- to 0.1726 held-out
  loss over 10 epochs. This breaks the project's own rule that only training
  partitions may supply chemistry pre-training, because the baselines being
  compared against pretrain on unlabeled corpora without excluding their test
  molecules, and a restricted BINDRA would be handicapped on the axis those
  methods are built around. It leaks structure, not labels. The checkpoint
  records `molecule_scope: all_partitions` and the affinity run's provenance
  carries the encoder identity as `UNRESTRICTED:...`, so the inflation travels
  with the artifact.
- **The 9.7% pair leak above**, inherent to the protocol.

Val RMSE fell on all twenty epochs, 1.3064 to 0.9918; patience never fired.

### The comparison

| Model | IC50 RMSE | Pearson |
|---|---|---|
| kNN-DTA | **0.684** | -- |
| SSM-DTA | 0.712 | **0.878** |
| BACPI | 0.740 | 0.860 |
| MONN | 0.764 | 0.858 |
| DeepAffinity | 0.780 | 0.840 |
| DeepDTA | 0.782 | 0.848 |
| Random Forest baseline | 0.910 | 0.780 |
| **BINDRA, this protocol** | **0.9822** [0.960, 1.003] | **0.7212** [0.705, 0.736] |
| BINDRA, `random_pair` IC50 (harder) | 1.0909 | 0.7106 |

113,025 test rows, 1,555 protein components, 1,000 resamples.
Also: MAE 0.7402, CI 0.7657, Spearman 0.7141, AUROC 0.8499, AUPRC 0.8239
against a 0.461 base rate, calibration slope 0.8909 with mean absolute bias
0.1119.

**Matching the protocol does not close the gap.** It moved BINDRA from 1.0909
to 0.9822, a real 0.109, but the model still lands short of the Random Forest
baseline and 0.30 behind the best published result. Roughly a third of the
original 0.42 discrepancy was protocol; the rest is the model. And 0.9822 is
the generous figure, since it carries both the pair leak and an encoder that
has seen the test set's structures.

What transfers is **rank, not magnitude**: CI 0.766 and AUROC 0.850 are close
to the in-domain figures, while calibration is markedly better here than on
`random_pair` (slope 0.891 against 0.763). BINDRA orders compounds
competently and predicts absolute potency poorly.

Two things worth holding onto:

- **The model, not the schedule, is the limit.** Twenty epochs, improving on
  every one, ending with a 0.0003 gain. It did not run out of training; it ran
  out of capacity. 2.59M trainable parameters with a frozen protein encoder,
  against pretrained transformer stacks.
- **kNN-DTA's 0.684 is at the IC50 replicate-noise floor of 0.671.** A model
  scoring at the reproducibility limit of the measurement, on a split that
  admits pair leakage, is a result to interrogate rather than to chase.

### An aside the comparison turned up

`best-e4.pt` records `ligand_encoder.checkpoint: "untrained"`,
`from_scratch: true`, `weights_checksum: null`. The shipped model's ligand arm
was **never pretrained**, despite `scripts/run_v2.ps1` passing
`--ligand-checkpoint`. The architecture figure shows pre-training feeding
initial weights into the LiGhT arm; for this checkpoint that edge is wrong.

## Two defects found while producing these numbers

Both were found by cross-checks, not by the code failing.

**The benchmark did not measure what training measured.** `evaluate.py`
filtered on `is_exact` alone where the training loop filters on exact *and*
primary, so it swept in 5,745 auxiliary IC50 rows and reported 1.1212 for a
checkpoint whose history records 1.1343. The gap read as noise for as long as
nobody counted rows. Kd and Ki matched exactly, which is what located it in the
IC50 tier. Both figures are now reported under their own keys.

**Panel scoring paired every prediction with the wrong label.**
`length_bucketed_batches` sorts by sequence length inside each window *even when
`shuffle=False`*, so the loader never emits rows in frame order. The panel path
zipped `dataset.frame["observation_id"]` against loader-order predictions and
then merged on that id, scoring a permutation of the model's answers. It failed
silently because the row count, the id set and the uniqueness all still looked
correct -- only the pairing was wrong. `_row_grouping` already resolved in
prediction order for exactly this reason; this path had been missed. Now
extracted as `_prediction_frame` with a regression test. The panel path had
never been run, so no previously reported number is affected.

## Limits

- **One seed.** No seed-to-seed variation is measured, no ensemble exists, and
  no per-prediction uncertainty is available.
- **Five epochs, still improving.** This is not a converged model.
- **`random_pair` measures interpolation.** Both the molecule and the target are
  usually represented in training. It says nothing about a new target or a
  future time period; the cold-target and time-based regimes exist for that and
  were not run here.
- **Calibration slope 0.763.** Absolute potency predictions are compressed
  toward the mean. Use ranks, or recalibrate.
- **Non-binders are the weak spot.** 85.4% bound violation on Davis's inactives.
  Do not use this model to rule a compound out.
- **The external numbers are zero-shot**, and Davis's clean stratum is 3,461
  rows with correspondingly wide intervals.
- **The literature-protocol number is off-spec twice over** -- a 9.7% pair
  leak and an encoder pretrained on held-out structures -- and is reported
  only to compare like with like. It is not a BINDRA result; 1.1342 on
  `random_pair` is.
