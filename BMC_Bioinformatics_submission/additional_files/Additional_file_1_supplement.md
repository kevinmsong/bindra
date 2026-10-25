# Supplementary methods and analyses for BINDRA

Kevin Song, John Zhang, Lei Ye, and Jianyi Zhang

## Supplementary protocol details

This supplement accompanies the BMC Bioinformatics manuscript draft and uses the same frozen primary pooled C-matched and no-FiLM F ensembles. The arithmetic means are computed in pKd space. Neither the supporting analyses nor packaging update affinity weights. All primary and companion member seeds are 20261001, 20261002, and 20261003.

### Dataset identity and scope

The official warm test contains 5010 pairs, with rows sorted by released pair index rather than the arbitrary order within the test-fold list. Fold membership and labels remain unchanged. The fitting/validation allocation uses official folds 1 through 4 and fold 0, respectively. All five training folds enter fixed-duration refits. Three-seed replication is not five-fold cross-validation.

The cold-drug split excludes normalized identities and quarantines 884 rows implicated by the identity/parent leakage audit. It does not silently substitute a scaffold split. The cold-protein split excludes exact sequence identities, not all homologous proteins. Separate model ensembles are trained for these protocols. Nearest-training chemical and sequence similarities describe their degree of novelty. Preparation of a calibration partition does not establish that calibration is valid: the reported calibration analysis was withheld because full-pool warm models lack independent rows and normalized-pair duplication requires stronger isolation in the noncanonical protocols.

Earlier adapted-encoder Davis experiments and preliminary test inspection remain historical exposure. Final validation-based architecture selection cannot make that history disappear. All eight warm-arm validation records and all executed individual-model metrics are supplied, including unsuccessful variants. The frozen original ESM-C and five-epoch ligand checkpoint are used for these controlled experiments; historical encoder adaptation is not mixed into the new model identities.

### Metric definitions

For measured pKd y and prediction p, MSE is the mean of (y - p) squared, RMSE is its square root, and MAE is the mean absolute difference. CI averages correct-order credit over unequal measured-label pairs, with one-half credit for prediction ties. Within-target macro CI is the unweighted mean over target sequences with comparable pairs. It is distinct from pooled CI and from a mean weighted by target-pair counts.

For the r_m-squared diagnostics, define r2 as squared Pearson correlation and r02 as the zero-intercept fit coefficient in the measured-label direction. The conventional diagnostic is r2 multiplied by (1 - sqrt(abs(r2 - r02))). The DeepDTA implementation instead uses r2 multiplied by (1 - sqrt(abs(r2 squared - r02 squared))). Both are exported under explicit names, with the exact implementation in the source snapshot. They must not be silently substituted or described as the same statistic.

The active threshold is measured pKd >= 7. Average precision is the recall-increment-weighted mean of precision. Trapezoidal precision-recall area uses linear interpolation between the computed precision-recall points. They are separate metrics. Historical auc.jar integration is not assumed identical. For a candidate pool of size N and active count A, EF1% is (a/k)/(A/N), where k is max(1, ceiling(0.01N)) and a is the active count in the top k candidates. Recall@k is the recovered active fraction using min(k, N) candidates. Therefore, Recall@100 can be one in a pool with fewer than 100 candidates without demonstrating strong early enrichment.

Crossed bootstrap intervals resample exact construct-sequence and Murcko-scaffold groups, retain the product of group resampling weights, and use common weights in paired model comparisons. The warm contrast includes 379 sequence groups and 68 Murcko-series groups. These units do not capture all family or series dependence. The F-minus-C-matched warm MSE-difference interval spans -0.01464 to 0.01410. Seed standard deviations describe variation between fitted runs and are not these bootstrap intervals. No significance test is performed against literature point values.

### Literature provenance and comparability

The numeric registry is frozen through October 17, 2026; bibliography metadata were checked during draft preparation. The copied literature-only CSV excludes the historical adapted BINDRA row. Its original source remains unchanged. Existing registry statements referring to BINDRA five-fold selection describe historical experiments, not the controlled single-validation-fold design in this manuscript.

DeepDTA and released GraphDTA folds have direct file verification, but the upstream GraphDTA training script's test-based checkpoint-selection option remains a qualification. WideDTA, MGraphDTA, and DeepDTAGen claim canonical-style folds without independent observation-hash equivalence. FusionDTA aggregation and exact selection were not fully verified. NHGNN-DTA used an 80/10/10 random split with 3006 test pairs, and SSM-DTA used a 70/10/20 split. Predicted protein contacts, binding-site supervision, and affinity-derived graphs are additional inputs or supervision, even when inference begins with sequences and SMILES. No unverified protocol is promoted into a matched ranking.

**Table 6. Selected author-reported Davis results by protocol.**

| Method | Year | MSE | CI | Qualification |
|---|---|---|---|---|
| DeepDTA [2] | 2018 | 0.261 | 0.878 | A |
| WideDTA [3] | 2019 | 0.262 | 0.886 | B |
| DGraphDTA [5] | 2020 | 0.202 | 0.904 | C |
| GraphDTA (GIN) [4] | 2021 | 0.229 | 0.893 | A |
| FusionDTA (KD, large) [7] | 2022 | 0.205 | 0.914 | B |
| MGraphDTA [6] | 2022 | 0.207 | 0.900 | B |
| NHGNN-DTA [9] | 2023 | 0.196 | 0.914 | D1 |
| SSM-DTA [8] | 2023 | 0.219 | 0.890 | D2 |
| HGTDP-DTA [10] | 2024 | 0.142 | 0.910 | C, preprint |
| DeepDTAGen [19] | 2025 | 0.214 | 0.890 | B |
| HCAF-DTA [11] | 2025 | 0.198 | 0.908 | C, preprint |
| CurvFlow-DTA [12] | 2026 | 0.185 | 0.902 | C, preprint |

All values are author-reported, not local reruns. A: official fold files verified, selection and aggregation qualified; the released GraphDTA training script includes test-based selection. B: exact fold identity, selection or aggregation not fully verified; WideDTA is a preprint version. C: additional predicted structural or affinity-graph inputs; not input-matched. D1: random 80/10/10 split, 3006 test pairs. D2: random 70/10/20 split. Preprint labels refer to the versions reviewed. BINDRA uses external pretraining and prediction averaging, so no cross-paper significance or universal superiority claim is made.

## Structural mappings and controls

The panel preserves its original selection by quality and coverage. Each of the 30 complexes has a chemical mapping and construct-to-author residue map, including missing-residue and insertion-code handling. Eligibility requires a noncovalent small-molecule complex, resolution <= 2.5 angstrom, usable coordinates, and unambiguous sequence/chemical correspondence. At most three complexes represent any target group. The final panel has ten target groups and no selection shortfall.

For each member, last-layer attention is averaged over heads. Member maps are combined only after residue and ligand coordinates are aligned. Individual heads, both attention directions, and knowledge-token mass remain available. Contact scores use physical tokens and resolved residues. Minimum heavy-atom distance <= 4 angstrom defines the primary contact; 3.5 and 5 angstrom provide sensitivity. A large residue AP can coexist with poor residue-atom or ligand-atom agreement; those outcomes remain separate.

Uniform contact prevalence and permutation baselines provide reference values. Target-group bootstrap intervals equally weight the ten target groups. Head/member agreement, representation reordering, progressive parameter randomization, swaps in predictions, and protein-feature gradients assess different aspects of model behavior. Ten matched random masks accompany 10% residue-feature masks. These are feature perturbations, not physical mutations, and can be outside the training distribution. Integrated gradients, all masking fractions, and coordinate-aligned explanation-map swaps were deferred and are not presented as completed.

### FAK1 and P4N

PDB 6YOJ supplies the detailed FAK1-P4N structural illustration at 1.361-angstrom resolution. Residue indices in the numeric exports map to PDB author numbering. Min-max displays, where used, are explicitly labeled and accompanied by raw weights. The F ensemble's pKd prediction is 5.41667 for this modeled input. Ki and IC50 source annotations remain assay-qualified and are not treated as trained prediction endpoints or numerically interchangeable pKd labels.

The complete Kd inventory for the focal construct contains 12 unique ligands, including three actives, four weak compounds, and five intermediate/discordant cases. The seven-compound active/weak subset is below the primary fishing threshold. It is an exploratory example and does not replace one of the coverage-selected measured-negative panels.

**Figure S1. FAK1-P4N residue attention and crystal contacts.** Raw attention and the physical contact reference are aligned to the modeled construct. Any normalized display is labeled. Neither correspondence nor a high attention weight establishes causal binding importance.

[Open figure](../figures/Supplementary_Figure_1_FAK_contacts.pdf)

**Figure S2. P4N ligand-only bond attention.** The shared frozen ligand encoder supplies this target-independent map before cross-attention. Physical bond-token weights and knowledge-token mass are retained separately.

[Open figure](../figures/Supplementary_Figure_2_FAK_bonds.pdf)

### Bond attention is a shared ligand representation

Bond maps precede target conditioning. The three affinity members share the same frozen original ligand encoder, so repeating its map across members does not create independent encoder evidence. Controls summarize 27 unique ligands in 25 chemical components. Full-matrix and supported-edge correlations are labeled separately; the latter were added after inspecting full-matrix behavior and are secondary analyses. Atom-order sensitivity and persistence under layer randomization are limitations. Whole-encoder randomization generally produces larger map changes. Topological masking, symmetric atom/bond mappings, knowledge mass, and mapped analog edits accompany the exports.

**Figure S3. Ligand bond-map sensitivity controls.** Twenty-seven unique ligands across 25 chemical components. Full and supported-edge analyses are labeled. Correlation persistence under layer randomization and atom-order sensitivity limit interpretation; shared member entries are not independent ligand encoders.

[Open figure](../figures/Supplementary_Figure_3_bond_controls.pdf)

## Screening interpretation

Measured weak compounds require exact eligible Kd observations at pKd <= 5, and actives require pKd >= 7. Intermediate and discordant measurements remain outside the binary measured subset. Presumed decoys have no fabricated measured affinity labels. Full pools contain 1348 galectin-3, 329 beta-2 adrenergic receptor, and 227 KEAP1 candidates; measured subsets contain 73, 25, and 38. The decoy shortfalls are zero, 21, and 111, respectively.

Target-only predictions, seed-assisted percentile-rank averaging, fingerprint similarity, property similarity, and fixed random ranking are exported for both ensemble roles. Seed-assisted methods receive information not supplied to the target-only predictor. Chemical-component bootstrap units use transitive identity, parent, and scaffold connections. Some resamples contain only one class; intervals therefore report finite-resample counts. Small numbers of components can make these intervals unstable. The three panels are outside the kinase-focused Davis target domain, and no general screening benefit is inferred from them.

## Generator data and evaluation

The raw cohort contains 93454 directed source edits across fitting and validation. Graph quality control retains 41682 fitting and 24819 validation edits and exports 26953 exclusions. Related molecules, reverse edits, and connected chemical series remain together. Each series contributes at most 32 directed pairs. Product graphs provide denoising targets, not conditioning features. Target and seed feature caches are bound to the frozen F ensemble and do not average separately learned latent bases.

Six conditions each use three training seeds. Structured context uses actual mapped seed-atom iBAM information. Uniform context removes localization while retaining the context interface; shuffled context permutes atom-to-context assignments without changing atom identity. Global-only removes local attention context. Seed-only removes every target-derived feature, including information leaked through target length. Target-shuffled uses a different target under the fixed permutation. Capacity and draw budgets are matched.

Requested-delta support would require source-verified same-assay training groups with at least 20 undirected edits, ten molecules, and three series, within the training fifth-to-95th-percentile delta range. No group qualifies, so requested-delta conditioning is disabled. Proxy assay context does not establish measured potency improvement.

Every query/run exports 100 draws and a completion manifest. Raw validity requires a supported chemical reconstruction under the recorded sampling policy. The repair policy is none, so repaired validity cannot exceed raw validity. Uniqueness is the number of unique valid chemical identities divided by the number of valid draws within a query/run. Scaffold retention is reported over all draws. Recovery compares generated identities with eligible held-out measured analog identities; it is not a predicted-score improvement. Novelty against generator training, ligand pretraining, and Davis supervision is reported separately.

The strict Kd subset has five queries across three series, generating 9000 draws across six conditions and three seeds. All conditions have 100% validity and scaffold retention and zero measured-analog recovery. The broader 50-query set adds 18 Ki and 27 IC50 queries. The structured arm's 88.37% overall raw validity is an equal-query/equal-seed mean, whereas bootstrap interval centers in the summary table equally weight chemical series. These estimands should not be combined as though they were identical. No claim is made that exact-reference-set novelty means absence from public chemistry.

## Native and generated docking

Native redocking uses three seeds and exhaustiveness 32 with the same preparation and search settings across comparisons. A receptor must have at least two top-ranked poses with symmetry-aware RMSD <= 2 angstrom and acceptable pose checks. The failed 5Y80 case remains reported. FAK1 has no eligible held-out generator query, and the generated analysis covers only the qualifying 9F3V-associated query.

**Table S2. Native redocking outcomes across search seeds.**

| PDB | Ligand | Top-pose RMSD in seed order (angstrom) | Pose checks | Qualified |
|---|---|---|---|---|
| 6YOJ | P4N | 0.385 / 0.371 / 0.361 | 3/3 | True |
| 5Y80 | IRE | 3.858 / 3.935 / 3.891 | 0/3 | False |
| 9F3V | A1H90 | 11.318 / 1.098 / 1.165 | 2/3 | True |

**Figure S4. Native-ligand redocking by receptor and search seed.** Symmetry-aware RMSD is computed in the experimental receptor frame. Overall pose checks and a top-ranked RMSD cutoff of 2 angstrom determine the receptor gate in at least two seeds. The failed 5Y80 case is retained.

[Open figure](../figures/Supplementary_Figure_4_native_docking.pdf)

The 90 generated selections represent compound instances within each query, condition, and training-seed run. Fifty-five fail preparation and 35 yield 105 repeated-search poses, with 91 passing overall PoseBusters checks. The selection policy is hash-based and independent of affinity or docking scores. Pose success is conditional on preparation, not a success fraction over all generated or selected compounds. Preparation failures, stereochemistry/geometry checks, and search outputs are retained. Docking scores are hypotheses about computed poses, not measurements of binding or potency.

## Reproducibility and unresolved submission actions

The research package's read-only checker verifies frozen member hashes, official fold membership, ensemble arithmetic, numeric metrics, table provenance, all 18 generator checkpoint/sample identities, and the figure export contract. Source, environment, optimizer, EMA, and RNG records support matching resumption. Numerical gates cover accepted Inductor/Triton segments; rejected batch-conditioning acceleration is not represented as used.

The additional files contain machine-readable results and all generated-draw records, but not the entire 4.35-GB canonical evidence tree. Checkpoints, original per-head arrays, full chemical SDF exports, and licensed third-party assets require an appropriate repository and access/licensing statement. The local checksum inventory identifies these assets. The paper remains a submission draft until the authors confirm contribution roles, funder involvement, acknowledgments, exclusive-submission/originality statements, approval, and persistent code/data identifiers.

## Detailed supporting methods and results

### Retrospective compound fishing

We froze target eligibility, candidate pools, seeds, ranking rules, and exposure audits before scoring. Primary measured-negative pools required at least ten actives at pKd >= 7 and ten weak compounds at pKd <= 5 after seed exclusion. Intermediate or discordant measurements were not converted to negative labels. Three eligible construct panels mapped to galectin-3 (P17931), beta-2 adrenergic receptor (P07550), and KEAP1 (Q14145). Exact sequences and normalized chemical identifiers accompany the source tables.

Each candidate was scored exhaustively by both frozen ensembles. Target-only pKd rankings were compared with Morgan fingerprint similarity to the seed, standardized property similarity, a fixed random ranking, and seed-assisted retrieval. Seed-assisted retrieval equally averaged full-pool pKd and similarity percentile ranks. The extra information supplied to seed-assisted methods was explicit. Exact seeds and duplicate chemical identities were removed. Unlabeled compounds were not assigned measured affinity labels.

Measured weak compounds and property-matched presumed decoys were analyzed separately. The decoy target was 25 per active, with shortfalls retained. Metrics included full-pool EF1%, AP, Recall@10, Recall@50, Recall@100, candidate counts, and prevalence. Chemical-component bootstrap intervals preserved related compounds, with the number of finite resamples reported when a resample lacked both classes. FAK1's measured Kd pool did not meet the primary coverage requirement and remained exploratory.

### Independent analog generation

We built endpoint-labeled analog edits from exact measurements with compatible protein constructs and available assay context. Chemical series, shared molecules, reverse pairs, and related edits remained together across partitions. Each series contributed at most 32 directed pairs, selected deterministically. Exposed evaluation components and graph failures were excluded and recorded. Source-verified assay groups were distinguished from proxy comparability groups; the latter do not establish measured potency improvements.

The diffusion generator had separate parameters, optimizer, exponential moving average, and random-number-generator state. The entire F ensemble remained frozen. Conditioning used the actual seed graph and tokens, endpoint, target residue features, member-specific fusion features, and seed-atom-aligned iBAM context. Generator-owned projections preserved independent member coordinate systems. Product features were excluded from clean conditioning. Requested-delta conditioning was disabled because no source-verified same-assay groups met the training-support criteria.

The validated compact generator used dimension 128, three layers, four attention heads, a 64-atom cap, 500 forward diffusion steps, and 100 exact-posterior reverse evaluations. Target features were pooled to 32 residue tokens per member. Training used batch size 64, learning rate 0.0001, weight decay 0.01, at most 20 epochs, and patience three. Endpoint- and series-balanced validation denoising loss selected checkpoints. Six capacity-matched conditions were trained with three seeds each: structured iBAM, global-only, uniform iBAM, shuffled iBAM, seed-only, and target-shuffled. Seed-only removed all target-derived information. Shuffling changed seed-atom-to-context assignments rather than atom identities.

We selected 50 eligible held-out queries without using model performance: five strict Davis-construct Kd queries across three chemical series, followed by 18 Ki and 27 IC50 queries allocated by endpoint and target coverage. Every query received 100 draws per condition and seed. Evaluation included raw validity, uniqueness among valid molecules within a query/run, scaffold retention, seed-copy rate, diversity, seed similarity, and measured-analog recovery. Novelty against generator training, ligand pretraining, and Davis supervision was assessed separately using the declared chemical-identity normalization. Supported scaffold stereochemistry was preserved, unspecified new centers were flagged, and repair was disabled. Affinity-model scores were outside generator optimization.

### Bounded docking assessment

We used AutoDock Vina 1.2.7, Meeko 0.8.0, and PoseBusters 0.6.5 [25, 26, 27]. Native redocking used three search seeds and exhaustiveness 32. A receptor qualified if the top-ranked pose had symmetry-aware heavy-atom root-mean-square deviation (RMSD) <= 2 angstrom and passed pose checks in at least two seeds. Ligands were compared in the experimental receptor frame, not superimposed independently to remove pose error.

Generated-compound docking was restricted to qualifying receptors with eligible held-out Kd queries. Up to five valid, unique, generator-training-novel compounds per query, condition, and training seed were selected by hash, not affinity or docking score. Preparation failures, poses, geometry checks, and docking scores were retained. Pose-level checks were conditional on successful preparation and repeated search poses were not independent compounds. No docking result was interpreted as experimental binding confirmation.

### Retrospective screening did not establish general ranking superiority

The three measured-negative panels contained 73, 25, and 38 candidates after filtering, with active prevalences 0.699, 0.520, and 0.316. Their full pools with presumed decoys contained 1348, 329, and 227 candidates, respectively. Two pools fell short of the requested decoy ratio, by 21 and 111 compounds; these shortfalls were retained.

On the measured panels, primary BINDRA AP was 0.533, 0.408, and 0.487 for galectin-3, beta-2 adrenergic receptor, and KEAP1. F achieved 0.681, 0.513, and 0.324. Fingerprint similarity achieved 0.983, 0.920, and 0.720. Seed-assisted retrieval improved some model rankings but did not establish consistent superiority over similarity or property controls (supplementary screening table). Because the latter methods received a known active seed, their results answer a different information setting from target-only prediction. Both information settings remained explicitly labeled.

These panels extend beyond the kinase-focused Davis supervision and should not be read as evidence of broadly calibrated affinity prediction. Full-pool EF1%, recall, candidate labels, exposure strata, and chemical-component intervals, including unstable or nonfinite resamples, are retained in Additional file 2. FAK1 remained exploratory rather than replacing a less favorable eligible screening example.

**Table 7. Measured-negative screening average precision.**

| Target | N | Actives | Active fraction | BINDRA AP | iBAM AP | Seed FP AP |
|---|---|---|---|---|---|---|
| Galectin-3 | 73 | 51 | 0.699 | 0.533 | 0.681 | 0.983 |
| Beta-2 adrenergic receptor | 25 | 13 | 0.520 | 0.408 | 0.513 | 0.920 |
| KEAP1 | 38 | 12 | 0.316 | 0.487 | 0.324 | 0.720 |

AP is average precision, not trapezoidal PR-AUC. BINDRA and iBAM columns are target-only pKd rankings; the fingerprint method receives an active seed. Exact seeds and duplicate identities are removed. These retrospective non-kinase panels do not establish general screening superiority. Presumed decoys, seed-assisted/property/random methods, full-pool EF1%, recall and chemical-component intervals remain in Additional file 2.

### Valid generation did not translate into measured-analog recovery

Graph quality control retained 41682 fitting and 24819 validation directed edits from 93454 source pairs. All 18 generator fits and 900 query/run evaluations completed, producing 90000 draws. Across all 50 queries, structured conditioning had equal-seed mean raw validity 88.37%. In the primary five-query strict Kd subset, all six conditions achieved 100% raw validity and scaffold retention without repair (supplementary generator table; supplementary generator figure). Structured within-query uniqueness was 99.27% in this subset.

None of the six conditions recovered a held-out measured analog in the 9000 strict-Kd draws. All conditions therefore tied at zero on the primary recovery outcome. The structured condition's high validity and exact training-set novelty did not establish that target information improved biological design. Uniform and target-shuffled conditions could attain lower validation denoising losses than structured conditioning, reinforcing the distinction between optimizing a denoising objective and demonstrating useful conditional generation.

The strict stratum represented five queries across only three chemical series. It was insufficient for a broad target-coverage claim, and the zero recovery result remains conditional on this measured support and sampling budget. Ki and IC50 strata, novelty against each reference set, seed-copy rates, rejection reasons, diversity, and per-seed losses accompany the full exports. No generated compound was described as experimentally validated or as a measured potency improvement.

**Table 8. Generator outcomes across conditioning controls.**

| Condition | All-query validity (%) | Strict-Kd validity (%) | Strict-Kd uniqueness (%) | Strict-Kd recovery (%) |
|---|---|---|---|---|
| Structured iBAM | 88.37 | 100.00 | 99.27 | 0.00 |
| Global only | 88.90 | 100.00 | 99.00 | 0.00 |
| Uniform iBAM | 88.26 | 100.00 | 99.33 | 0.00 |
| Shuffled iBAM | 88.19 | 100.00 | 99.20 | 0.00 |
| Seed only | 89.38 | 100.00 | 99.20 | 0.00 |
| Target shuffled | 89.04 | 100.00 | 99.00 | 0.00 |

Equal-query/equal-seed means; three seeds per condition and 100 draws/query/run. The full set has 50 queries and 90000 draws. Strict Kd has five queries, three chemical series and 9000 draws; scaffold retention is 100% in every condition. Uniqueness is conditional on valid draws within a query/run. Repair is none. Endpoint/series intervals are separate exports; exact novelty does not establish public chemical novelty or binding.

### Docking supported a narrow geometry assessment

Native redocking qualified 6YOJ in three of three seeds and 9F3V in two of three; 5Y80 qualified in none. FAK1 had no eligible held-out generator query, so it contributed native redocking only. Generated docking was consequently limited to the qualifying 9F3V-associated query.

Among 90 hash-selected compound instances, 55 failed preparation and 35 produced three search poses each. Of these 105 poses, 91 passed overall PoseBusters checks and 14 did not (supplementary docking figure). No Vina search process failed. The 86.7% pose-QC pass rate is conditional on preparation, whereas preparation succeeded for 38.9% of selected instances. Reporting only the former would conceal substantial attrition. Docking scores and geometry checks did not establish binding, affinity improvement, or coverage across the broader query set.

### Reproducibility and computational cost

The six final warm-test member exports were reproduced bitwise in the replacement session, and the arithmetic ensemble and official fold membership passed verification. The audit checked all 18 generator checkpoint identities and all 90000 draw exports. All 76 replacement-study figures passed exact-PDF visual review and format/provenance checks. The compact architecture overview was checked against actual checkpoint dimensions and parameter counts.

With frozen features cached, mean warm full-pool training time per member was 982 seconds for C-matched and 2366 seconds for F in the recorded environment. Mean per-member cached-feature inference throughput was approximately 3929 and 2298 pairs per second, respectively. These measurements exclude upstream encoder generation, are hardware- and batch-dependent, and do not represent end-to-end screening throughput. The evidence package preserves software versions, numerical gate results, resumption state, failures, and a checksum inventory.


## Glossary of key terms

**Affinity and pKd.** Kd is an equilibrium dissociation constant; smaller values indicate tighter binding under the measured conditions. pKd is -log10(Kd in molar units), so a larger pKd indicates tighter binding. A one-unit pKd difference corresponds to a tenfold Kd difference. Ki is an inhibition constant. IC50 is the concentration producing 50% inhibition under specified assay conditions. Neither quantity is automatically interchangeable with Kd.

**Concordance index.** A measure of whether predictions put unequal measured affinities in the correct order. It describes ranking and can favor a model differently from MSE, which measures numerical prediction error.

**Average precision.** A summary of precision across increasing recall. It depends on the fraction of actives in the candidate pool, so AP values from different pools require prevalence and protocol context.

**Warm and cold evaluation.** Warm evaluation holds out pairs while allowing their molecules and proteins to appear in other training pairs. Cold-drug and cold-protein evaluation exclude chemical identities or exact protein sequences, respectively, from supervised training.

**Interaction attention and bond maps.** An iBAM summarizes cross-attention between protein residues and ligand tokens. A bond map summarizes attention within the ligand encoder. These maps describe model emphasis; a high weight alone does not establish physical contact or causal importance.

**Scaffold and analog.** A scaffold is a mapped molecular core retained during editing. An analog is a related molecule produced by changing parts of the seed structure. Scaffold retention is a chemical constraint, not proof that biological activity is retained.

**Denoising loss and recovery.** Denoising loss measures how well a generator predicts clean graph information from corrupted training examples. Measured-analog recovery asks whether generated molecules match held-out observed analog identities. A lower denoising loss does not guarantee better recovery or binding.

**Docking and pose quality.** Docking proposes a ligand orientation in a receptor. RMSD compares a proposed native-ligand pose with its crystal coordinates, and geometry checks test physical plausibility. Neither a plausible pose nor a docking score establishes experimental binding.


## Development and numerical reproducibility record

The historical adapted-encoder experiments are distinct from these frozen-encoder comparisons. Earlier BF16 replay differences were documented; the later verification reproduced the six warm-member exports bitwise. A proposed faster batch conditioner failed equivalence and was rejected. Artifact checks, review sessions, and chronological implementation records are reproducibility evidence rather than independent biological validation.

**Figure S5. Generator outcomes across conditioning controls.** Five strict Kd queries across three series; recovery is zero in every condition. Broader endpoint results remain separate.

[Open figure](../figures/Supplementary_Figure_5_generator.pdf)

**Figure S6. Generated-compound preparation and pose checks.** Preparation fails for 55/90 selected instances. The 35 prepared instances yield 105 poses, 91 passing checks. Pose success is conditional on preparation.

[Open figure](../figures/Supplementary_Figure_6_generated_docking.pdf)

## Independent-benchmark runtime amendment

Before any KIBA test scoring, the user requested a smaller execution budget after 24 completed epochs of the first validation fit. Every neural method now receives the same three learning rates at dropout 0.1 on official validation fold 0; three final seeds remain. Additional Davis analysis compares only the two leading fixed configurations across folds 1 and 2 with three seeds. It is not a full architecture-selection stability experiment. The original protocol, partial validation exposure, unchanged optimization settings, and preserved checkpoint identities remain in the public reference registration and runtime amendment. No new generator or docking runs were added.
