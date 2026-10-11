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
