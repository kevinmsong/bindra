# BindingDB 202610 - curation report

Generated 2026-10-02 17:51 UTC from profile `default` (`7c6be7537fcc43d19ffcd060`).

Every count below is read from the stage that produced it. This is the report the specification asks for; `docs/DATA_PROFILE.md` is the earlier pre-curation census and is not a substitute for it.

## Counts after every filter

| Stage | Observations |
|---|---|
| Source rows | 3,243,660 |
| Rows with no affinity endpoint | 271,395 |
| Rows carrying EC50 (excluded by design) | 270,787 |
| Observations after row->endpoint expansion | 2,975,722 |
|   quarantined at ingest | 177,500 |
|   passed structural screening | 2,798,222 |
| Unusable after standardization/construct join | 5,446 |
| Duplicate imports collapsed | 52,856 |
| **Curated observations** | **2,739,920** |

### Ingest quarantine reasons

| Reason | Observations |
|---|---|
| multichain | 176,972 |
| value_rejected | 240 |
| no_inchikey | 224 |
| no_smiles | 64 |

## Endpoints

| Endpoint | Curated | Exact | Censored | Censored % | sigma_rep |
|---|---|---|---|---|---|
| pKd | 123,092 | 79,053 | 44,039 | 35.8% | 0.945 |
| pKi | 581,226 | 505,925 | 75,301 | 13.0% | 0.641 |
| pIC50 | 2,035,602 | 1,613,872 | 421,730 | 20.7% | 0.671 |

pKd, pKi and pIC50 are kept separate throughout. IC50 is an assay-dependent potency endpoint and is never relabelled as Kd. EC50 is excluded from the affinity regression entirely.

## Measurement error, and the thresholds derived from it

`sigma_rep` is the pooled within-group standard deviation of exact p-values over within-publication groups (ligand, target, endpoint, citation) with at least two members.

| Endpoint | sigma_rep | df | Ranking margin | Cliff \|dp\| | Discordance mod/high |
|---|---|---|---|---|---|
| pKd | 0.945 | 13,427 | 0.94 | 1.89 | 1.89 / 2.62 |
| pKi | 0.641 | 57,841 | 0.64 | 1.28 | 1.28 / 1.78 |
| pIC50 | 0.671 | 203,756 | 0.67 | 1.34 | 1.34 / 1.86 |

Three consequences that change what results can mean:

- **pKd is the noisiest endpoint, not the cleanest** (0.945 against 0.641 for pKi), and also the scarcest. Any pKd claim rests on a small and noisy base, and an RMSE near 0.95 on pKd is at the noise floor rather than good.
- **The specification's starting 0.5 p-unit ranking margin is below the noise floor of every endpoint.** It is raised to one sigma_rep; ordering a pair that differs by less than measurement error is training on noise.
- **Patent-derived sigma_rep is lower than literature-derived** (Kd 0.707 against 1.019). That indicates re-statement of a single table value rather than independent measurement, so patent replicates are less independent, not more precise.

sigma_rep is a within-publication estimate and therefore a **lower bound** on true measurement error; between-laboratory reproducibility is larger and is not estimated here.

## Dependence structure

| Unit | Observed | Kish effective n | Largest single unit | Top 1% share |
|---|---|---|---|---|
| Targets | 10,474 | **572** | 22,999 | 34.2% |
| Ligands | 1,182,431 | 73,945 | 3,094 | 13.7% |

10,474 targets correspond to **572 effective independent target units**. Confidence intervals must therefore come from a cluster bootstrap over protein components, and per-target macro averages must be reported beside pooled metrics, which are dominated by roughly a hundred targets.

## Chemical standardization

| Quantity | Count |
|---|---|
| Unique raw SMILES | 1,438,824 |
| Standardized | 1,436,561 |
| Quarantined | 2,263 |
| Unique identity keys (exact stereochemistry) | 1,433,489 |
| Unique leakage keys (parent connectivity) | 1,338,687 |
| Unique Murcko scaffolds | 481,853 |
| Counterions removed | 1,128 |
| Molecules carrying a formal charge | 15,424 |
| Molecules with unspecified stereocentres | 308,851 |

Standardization outcomes:

| Status | Molecules |
|---|---|
| OK | 1,436,561 |
| PARSE_FAILED | 2,088 |
| MULTIPLE_ORGANIC_FRAGMENTS | 101 |
| SANITIZE_FAILED | 52 |
| ONLY_INORGANIC | 22 |

## Targets and constructs

| Quantity | Count |
|---|---|
| Unique constructs | 10,822 |
| Unique UniProt accessions | 9,149 |
| Accessions with more than one construct | 467 |
| Constructs with detected point mutations | 349 |
| Constructs that are substrings of another | 561 |
| Residue length min / median / max | 7 / 454 / 34350 |
| Total residues | 6,536,409 |
| Constructs over the 2,046-residue encoder window | 261 |

Mutation calls are derived by aligning constructs that share a UniProt accession, with no network access. Free-text target-name tokens are an **independent** signal, so their agreement validates the method: 259 of 303 comparable constructs agree exactly and 18 more agree on the substitution at a constant positional offset, giving **91.4% agreement on the substitution itself**. The offsets are construct-relative versus canonical numbering, which cannot be reconciled offline, and which is also why the reference construct is not claimed to be wild type.

## Protein grouping

| Quantity | Count |
|---|---|
| Constructs | 10,820 |
| Pairs aligned (exhaustive, exact) | 58,530,790 |
| Edges stored (identity >= 0.20) | 39,633,303 |
| Qualifying edges at 40% / 80% | 62,884 |
| UniProt-sharing edges | 7,118 |
| Substring containment edges | 1,390 |

### The target graph percolates at the specified threshold

This is the most consequential measurement in the grouping stage. At the specification's 40% identity / 80% coverage, single linkage does not produce protein families -- it produces one connected mass. Average degree is about 11.6, far above the percolation point.

| Identity | Qualifying edges | Components | Largest | Largest % | Observations outside largest |
|---|---|---|---|---|---|
| 0.40 | 62,884 | 666 | 10,066 | 93.0% | 44,683 |
| 0.50 | 33,728 | 3,968 | 4,422 | 40.9% | 1,138,822 |
| 0.60 | 25,754 | 5,764 | 241 | 2.2% | 2,653,974 |
| 0.70 | 21,957 | 6,319 | 155 | 1.4% | 2,680,629 |
| 0.80 | 18,381 | 6,901 | 108 | 1.0% | 2,697,907 |
| 0.90 | 12,911 | 7,714 | 108 | 1.0% | 2,710,253 |

At 40% a cold-target test set can draw on **1.6% of observations**. The graph shatters between 0.50 and 0.60, and everything from 0.60 upward looks much the same -- so 60% sits on a plateau rather than on a slope, which is why it is not a tuned number.

Two groupings are therefore published, and both are used:

| Grouping | Threshold | Components | Largest | Largest % | Used by |
|---|---|---|---|---|---|
| `protein_group_strict` | 40% | 665 | 10,068 | 93.0% | `cold_target_strict`; bounds novel-target claims |
| `protein_group` | 60% | 5,757 | 241 | 2.2% | `cold_target`, `double_cold`; the powered measurement |

The looser regime's cost is published rather than implied: its audit records every cross-partition construct pair with identity in [0.40, 0.60) -- 17,981 pairs for `cold_target` and 10,165 for `double_cold`. Those are the homologues the 60% threshold permits to straddle a partition, and no claim about novel targets should rest on that regime alone.

Alignment is exhaustive and exact -- every pair, no prefilter -- and a sample is re-aligned with an independent implementation. Optimal alignments are rarely unique, so identity counts can differ between two correct implementations at the same optimal score; only the qualification decision is asserted. Measured identity difference: median 0.0037, p90 0.0955, max 0.3333 over 240 pairs, with 0 pair(s) that the oracle qualifies and the table does not.

## Assay comparability

**BindingDB 202610 contains no assay identifier and no assay description column anywhere in its 640 columns.** This is not a missing value; the field does not exist. Every within-assay quantity in BINDRA therefore rests on a named proxy, recorded per observation in `comparability_scope`.

| Quantity | Count |
|---|---|
| Comparability groups | 165,020 |
| Median group size | 4.0 |
| 90th percentile group size | 34.0 |
| Largest group | 2,675 |
| Singleton groups | 46,647 |
| Observations with no citation of any kind | 1,474 |

An observation with no PMID, DOI, patent number or assay identifier takes its own id as its citation key and therefore forms a **group of one**. The specification forbids a missing identifier collapsing a target into one giant ranking group, and this is the mechanism that prevents it.

| Citation kind | Observations |
|---|---|
| pmid | 1,477,889 |
| patent | 1,160,964 |
| pubchem_aid | 71,408 |
| doi | 28,185 |
| none | 1,474 |

| Comparability scope | Observations |
|---|---|
| literature_assay_proxy | 1,505,028 |
| patent_table_proxy | 1,162,010 |
| bioassay_aid | 71,408 |
| unknown_singleton | 1,474 |

| Metadata confidence | Observations |
|---|---|
| medium | 2,633,933 |
| low | 73,997 |
| high | 31,990 |

`high` is rare because it requires both reported conditions, and pH is populated for only 6.5% of rows and temperature for 6.1%. Ranking supervision deliberately requires only `medium` so the objective is not starved by that scarcity. Patent-sourced records are capped at `medium` however complete their other fields, because one patent examples table can hold thousands of values under more than one protocol.

## Duplicates, replicates and disagreement

| Quantity | Count |
|---|---|
| Duplicate imports collapsed | 52,856 |
| Identical value under a different citation (kept, flagged) | 440,150 |
| Replicate groups | 2,562,767 |
| Replicate groups with more than one exact value | 127,652 |
| Internally contradictory censoring | 19,490 |

Replicate spread, in p-units, over groups holding more than one exact value:

| median | mean | p90 | max | > 0.5 | > 1.0 | > 2.0 |
|---|---|---|---|---|---|---|
| 0.602 | 0.854 | 2.041 | 9.516 | 159,541 | 94,041 | 30,429 |

| Discordance | Observations |
|---|---|
| none | 2,675,353 |
| high | 34,954 |
| moderate | 29,613 |

A group disagreeing beyond its endpoint's repeatability limit keeps its median but is flagged and barred from analog-pair and activity-cliff construction: a delta computed from mutually inconsistent measurements means nothing. Such groups are never dropped and never silently averaged. Censored bounds are never pooled into a median.

## Per-task eligibility

The specification asks how many observations remain eligible for regression, within-assay ranking, retrieval supervision and analog-pair training, separately.

| Task | Eligible observations | Share |
|---|---|---|
| analog pair training | 2,114,625 | 77.2% |
| primary regression | 2,672,606 | 97.5% |
| retrieval supervision | 2,739,920 | 100.0% |
| within assay ranking | 2,620,781 | 95.7% |

| IC50 tier | Observations |
|---|---|
| A | 1,968,288 |
| na | 704,318 |
| B | 67,314 |

`na` is Ki and Kd, which the specification treats as high-confidence binding constants and does not tier. Tier B is excluded from the primary affinity metric and from matched-pair construction but may supervise retrieval and within-assay ranking, where the signal is relative order inside a group.

## Policy register

Every curation choice, with the measurement behind it. Several overturned a default, and those are marked.

| Choice | Decision | Basis |
|---|---|---|
| Endpoints kept separate | pKd, pKi, pIC50 never merge; EC50 excluded | Specification |
| Censoring | One-sided bound, never trained as an exact value | 541,070 censored observations |
| Plausibility window | Disabled | Policy decision; recorded |
| Formal charges | Preserved on the identity path | Specification |
| Ambiguous mixtures | Quarantined, never resolved by largest fragment | 101 molecules |
| **Tautomer canonicalization** | **Disabled** (overturns the default) | 96.5 ms/mol against ~4 ms/mol for the rest of the pipeline; InChI's mobile-hydrogen layer already collapses the common classes |
| Acyclic scaffold fallback | The molecule's own leakage key, not a shared bucket | 6,565 molecules |
| Construct identity | sha256 of the exact construct, not the UniProt accession | Specification |
| Sequence case | Canonicalized to upper case | 34 constructs were lower case; ESM-C maps lower case to <unk>, a special token, which would strip every residue |
| **Alignment mode** | **Semi-global** (overturns global) | Sensitivity 0.987 at 0.000 false positives, against 0.867 at 0.043 for global; global collapsed a 500-construct sample into one component |
| **Identity ambiguity margin** | **Zero** (overturns a conservative margin) | A 0.08 margin turned 134 qualifying edges into 473 and the largest component from 10.8% to 53.8%; near-threshold pairs are audited individually instead |
| Linkage | Single linkage; a large component is reported, never split | Splitting would reintroduce the homology the regime excludes |
| Comparability key | target + endpoint + citation + condition bucket + source class | No assay identifier exists in the file |
| Missing citation | Forms a singleton group | 1,474 observations |
| pH / temperature | Excluded from the key as components; missing is its own bucket | Populated for 6.5% / 6.1% of rows |
| Duplicate key | Byte-exact value cell within one comparability group | 52,856 collapsed |
| Discordance threshold | 2.77 x sigma_rep, endpoint-specific | Measured; a fixed 0.5 would flag a third of replicated groups |
| IC50 Tier B | Keyed on curation source class, never on PubChem AID presence | 68% of rows carry an AID, including ~545k patent rows |
| Split seed | Declared constant 20261001, all seeds logged | Seed shopping must be visible |

## Not claimed

- No model has been trained on this corpus and no performance number exists.
- `sigma_rep` is a within-publication lower bound on measurement error.
- 40% identity / 80% coverage is an operational threshold, not proof that shared domains or protein families are absent across partitions.
- No within-assay claim rests on a real assay identifier, because the field does not exist in this source.
- Mutation calls identify that two constructs differ at a position, not which one is wild type.
- Computational decoys, where used later, are presumed controls with unknown binding status and never receive affinity labels.

