# BindingDB 202610 — snapshot profile

A first-pass census of the acquired snapshot. **This is not the curation
report** the specification requires: nothing here is standardized, deduplicated
or assay-grouped. It establishes how much data each downstream filter has to
work with, and records the file's quirks so the parser is written against
measured facts rather than assumptions.

Produced by `scripts/census_snapshot.py`; full output in
`data/snapshots/bindingdb_202610/census.json`. Scan of the 8.4 GiB TSV takes
~137 s streaming.

## Snapshot identity

| | |
|---|---|
| Release | 202610 |
| File | `BindingDB_All_202610_tsv.zip`, 594,284,548 bytes |
| SHA-256 | `e968f90e900c7833d5e6eceae282d0ca3b75201448bccaf4df6b09cc410f7d76` |
| Source | `https://www.bindingdb.org/rwd/bind/downloads/` |
| Server-dated | 2026-09-30 |
| Uncompressed | `BindingDB_All.tsv`, 8.4 GiB, 640 columns |

## Census

3,243,660 data rows. The release page advertises 3,248,380 measurements,
1,446,269 compounds and 11,516 targets; the small differences are expected,
since a row can carry more than one endpoint and the target count includes
multichain complexes.

| | measured |
|---|---|
| Rows | 3,243,660 |
| Unique ligand SMILES (raw, unstandardized) | 1,438,824 |
| Unique chain-1 sequences (raw) | 11,018 |
| Rows with no affinity endpoint | 271,395 (8.4%) |
| Rows with no SMILES | 75 |

### Affinity endpoints

| Endpoint | Present | Exact | Censored | Censored % | Single-chain | Rejected |
|---|---|---|---|---|---|---|
| Ki | 623,701 | 542,684 | 80,983 | 13.0% | 594,309 | 34 |
| IC50 | 2,220,490 | 1,749,813 | 470,504 | 21.2% | 2,075,852 | 173 |
| Kd | 131,531 | 82,481 | 49,017 | **37.3%** | 128,349 | 33 |

Three things follow directly:

- **Censoring is not a corner case.** 600,504 of 2,975,722 affinity
  observations (20.2%) are bounds. Dropping them discards a fifth of the data;
  training on them as exact values corrupts the same fifth.
- **Kd is the scarcest and most censored endpoint** — 131,531 observations,
  37.3% of them bounds, leaving only 82,481 exact pKd values. Any claim that
  rests on pKd rests on a much smaller base than the headline 3.2M.
- **IC50 dominates by volume** (75% of all observations). Since IC50 is
  assay-dependent, volume is not interchangeable with quality, and the
  separate-heads design matters more than the totals suggest.

Only 240 values were rejected outright, all non-positive concentrations. The
numeric data is clean; the difficulty is in provenance and comparability, not
in parsing.

### Target chains

| Chains | Rows |
|---|---|
| 1 | 3,056,340 (94.2%) |
| 2 | ~180,000 |
| 3+ | small tail, up to 50 |

Restricting the initial supervised set to unambiguous single-protein targets
costs ~5.8% of rows.

### Dates

97.3% of rows carry a parseable publication date; 2.7% do not and are excluded
from any temporal claim. Range 1965–2026.

Cumulative share of dated rows by cutoff year:

| Cutoff | Train share | Rows after cutoff |
|---|---|---|
| ≤ 2020 | 73.7% | 831,196 |
| ≤ 2021 | 80.1% | 629,368 |
| ≤ 2022 | 85.3% | 464,357 |
| ≤ 2023 | 90.5% | 298,719 |
| ≤ 2024 | 94.1% | 187,185 |

**Pending decision.** The specification requires choosing the temporal cutoff
*before modelling* and recording it in the benchmark configuration. 2022 or
2023 both leave a substantial future panel. Not yet chosen.

## File quirks that must be handled

1. **640 columns = 40 prefix + 50 chain blocks × 12.** Every row carries all 50
   blocks regardless of its real chain count. Reading "chain 1" without
   consulting `Number of Protein Chains in Target` silently converts a
   multi-protein complex into a single-protein observation.

2. **Relations are embedded in the value cell** (`">10000"`), not in a column
   of their own. A naive float parse drops or corrupts 20.2% of observations.

3. **Dates are US `M/D/YYYY`**, not ISO. Taking the first four characters of
   `8/30/1996` yields `8/30`, which is not a year. This bug occurred during the
   first census run and classified all 3.24M rows as unknown-date — it would
   have silently emptied the temporal panel rather than raising.
   `parse_bindingdb_date` and a regression test now cover it.

4. **`Date in BindingDB` is an import date, not an evidence date.** Temporal
   splits use `Date of publication`.

5. **The `Download.jsp` URL is not a download.** It answers HTTP 200 with a
   12 KB HTML page, and `curl` exits 0. The real files live under
   `/rwd/bind/downloads/`. `record_snapshot()` rejects any archive without a
   zip signature.

## Scale implications

Revises the planning estimates upward — the raw corpus is larger than assumed:

| | plan estimate | measured |
|---|---|---|
| Unique ligands | ~700k | 1,438,824 raw |
| Unique targets | ~8k | 11,018 raw |

Protein feature cache remains modest: ~11k sequences × ~1.2 MB (bf16, 1152-dim)
≈ **13 GB**. The ligand cache is the problem: KPGT token states for 1.44M
molecules would be ~100 GB, roughly double the earlier estimate. Options, to be
decided after a measured pilot encode:

- cache graph-level readouts for the full library and token states only for
  molecules in the supervised set;
- restrict the initial supervised corpus by endpoint and assay confidence;
- store token states at fp16 with per-molecule truncation to a token budget.

None of this is settled. The pilot encode must come before committing to a
full-corpus encode, as the specification requires.
