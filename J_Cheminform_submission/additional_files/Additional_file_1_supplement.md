# Supplementary methods and analyses for BINDRA

Kevin Song<sup>1</sup>, John Zhang<sup>1</sup>, Lei Ye<sup>1</sup>, and Jianyi Zhang<sup>*1</sup>

<sup>1</sup>Department of Biomedical Engineering, University of Alabama at Birmingham, Birmingham, Alabama 35294, United States

*Email: jayzhang@uab.edu

## Guide to supplementary files

This supplement contains methods, six figures, eleven tables, and references. The accompanying files preserve full-precision results and the records needed to inspect the completed analyses.

**Additional file 2.** Additional_file_2_results_tables.zip contains result tables, the literature registry, architecture and individual-model metrics, screening and structural controls, docking checks, and FAK1 case CSVs.

**Additional file 3.** Additional_file_3_Davis_predictions.zip contains predictions, member manifests, partitions, exclusions, and similarity audits. C-matched denotes BINDRA and F denotes BINDRA-iBAM. Parquet files require pandas or Apache Arrow.

**Additional file 4.** Additional_file_4_generator_draws.zip contains all 90000 draws, including invalid outcomes and rejection reasons, queries, graph exclusions, configurations, and completion records.

**Additional file 5.** Additional_file_5_reviewed_figures.zip supplies current figures in PDF, SVG, and 600-dpi PNG/TIFF under current_figures. Historical exports remain separate. Figure S1 uses 37 physical tokens; the historical 39-token profile includes knowledge tokens.

**Supporting artifacts.** supporting_artifacts supplies candidate rankings, construct sequences, all 30 structural mappings, mean attention/contact arrays, and docking outputs. The artifact catalog and PACKAGE_MANIFEST.json identify paths, software versions, provenance, and SHA-256 checksums.

**Assets for separate deposit.** Checkpoints, optimizer states, original per-head arrays, full SDF exports, and licensed inputs remain separate-deposit assets, as identified in the inventory.

## Supplementary protocol details

This supplement describes the primary pooled BINDRA ensemble (C-matched) and retained BINDRA-iBAM ensemble without FiLM (F). Both average pKd predictions from member seeds 20261001, 20261002, and 20261003. Affinity weights remained fixed during all supporting analyses.

### Dataset identity and scope

The official warm test contains 5010 pairs, sorted by released pair index rather than test-fold list order. Fold membership and labels are unchanged. Official folds 1 through 4 supplied fitting pairs and fold 0 supplied validation pairs. Fixed-duration refits used all five training folds. Three-seed replication used this single validation partition.

The cold-drug split separates normalized chemical identities and excludes 884 rows flagged by the identity/parent leakage audit; scaffolds can recur across partitions. The cold-protein split separates exact sequences while allowing homologous proteins. Each protocol used separately trained ensembles, with nearest-training chemical or sequence similarity reported. Calibration was withheld because warm full-pool models lacked independent calibration observations and normalized-pair duplication required stronger separation in cold protocols.

Earlier Davis experiments with adapted encoders included test inspection, so the current results are retrospective despite validation-based architecture selection. All eight warm-arm validation records and all executed individual-model metrics are supplied, including unsuccessful variants. The controlled experiments used the original frozen ESM-C and five-epoch ligand checkpoints; models with historical encoder adaptation were analyzed separately.

### Detailed affinity partitions and network configuration

In fitting, validation, calibration, and test order, the cold-drug split retained 20332, 3094, 3094, and 2652 pairs, with 884 additional rows retained in the audit exclusion table. The cold-protein split retained 21080, 2992, 2992, and 2992 pairs. Group allocation was label-independent. Cold-drug identities were normalized chemical identities; cold-protein identities were exact amino acid sequences. Calibration eligibility was assessed separately from partition assignment.

The ESM-C input window covered at most 2046 residues, with 256-residue overlap. Residue vectors shared by multiple windows were averaged. Cached encoder representations were tied to encoder and preprocessing identities. The ligand-pretraining audit used the eligible pretraining SMILES rather than the complete BindingDB download; failed chemical normalization remained unknown in the exposure audit.

Adapters projected features to 192 dimensions. Pooled BINDRA concatenated five readouts and used a regression head mapping 960 to 1648 to 256 to one pKd output. BINDRA-iBAM used feed-forward width 576 within its two cross-attention layers. Its six readouts comprised fused protein and ligand features, three global ligand features, and the protein summary computed before attention. Its regression head mapped 1152 to 256 to one output. The global ligand features and pre-attention protein summary bypassed the attention layers. The grouped-gate comparison arm applied pair-dependent scalar gates to interaction, ligand-global, and protein-only readout groups, initialized to preserve their input scale.

The selected first-seed tuning run contributed to the three-seed validation summary. Each architecture's rounded-up median epoch of minimum validation MSE set its full-pool refit duration. Runtime optimizations included length bucketing, graph trimming, and bounded upstream caches in addition to the mixed precision, fused optimizer, and numerically checked acceleration described in the main text. Checkpoints retained optimizer and random states for resumption under matching settings.

### Metric definitions

For measured pKd y and prediction p, MSE is the mean of (y - p) squared, RMSE is its square root, and MAE is the mean absolute difference. CI averages correct-order credit over unequal measured-label pairs, assigning one-half credit to prediction ties. Within-target macro CI averages target sequences with comparable pairs equally; pooled CI and averages weighted by target-pair counts measure different quantities.

For r_m-squared, r2 is squared Pearson correlation and r02 is the zero-intercept fit coefficient in the measured-label direction. The conventional diagnostic is r2 multiplied by (1 - sqrt(abs(r2 - r02))); DeepDTA instead uses r2 multiplied by (1 - sqrt(abs(r2 squared - r02 squared))). Both are explicitly named in the exports and implemented in the source snapshot. They are not interchangeable.

The active threshold is measured pKd >= 7. Average precision is the recall-increment-weighted mean of precision. Trapezoidal precision-recall area uses linear interpolation between the computed precision-recall points. The exports report these metrics separately; equivalence to the historical auc.jar integration was not assessed. For a candidate pool of size N and active count A, EF1% is (a/k)/(A/N), where k is max(1, ceiling(0.01N)) and a is the active count in the top k candidates. Recall@k is the recovered active fraction using min(k, N) candidates. Recall@100 equals one when all candidates in a pool smaller than 100 are included; early enrichment therefore requires the other ranking metrics.

Crossed bootstrap intervals resample exact construct-sequence and Murcko-scaffold groups, retain the product of group resampling weights, and use common weights in paired model comparisons. The warm contrast includes 379 sequence groups and 68 Murcko-series groups. Some dependence within protein families or chemical series may remain after this resampling. The F-minus-C-matched warm MSE-difference interval spans -0.01464 to 0.01410. Seed standard deviations describe variation between fitted runs, separately from bootstrap uncertainty. Literature point values are compared descriptively.

### Literature provenance and comparability

The numerical registry covers literature through October 17, 2026, with metadata checked during manuscript preparation. Its literature-only CSV excludes the historical adapted BINDRA row while preserving the original source. Registry references to BINDRA five-fold selection concern historical experiments; the controlled study used one validation fold.

The DeepDTA and released GraphDTA fold files were verified directly. The upstream GraphDTA training script permits test-based checkpoint selection, a selection-policy difference recorded in the registry. WideDTA, MGraphDTA, and DeepDTAGen report canonical-style folds; observation-level fold equivalence was not independently verified. FusionDTA aggregation and exact selection were not fully verified. NHGNN-DTA used an 80/10/10 random split with 3006 test pairs, and SSM-DTA used a 70/10/20 split. Predicted protein contacts, binding-site supervision, and affinity-derived graphs are additional inputs or supervision, even when inference begins with sequences and SMILES. Results from unverified protocols are not treated as matched comparisons.

**Table S1. Author-reported Davis results for all reviewed model variants.**

| Method | Year | MSE | CI | Qualification |
|---|---|---|---|---|
| DeepDTA [2] | 2018 | 0.261 | 0.878 | A |
| WideDTA [3] | 2019 | 0.262 | 0.886 | B |
| DGraphDTA [5] | 2020 | 0.202 | 0.904 | C |
| GraphDTA (GAT) [4] | 2021 | 0.232 | 0.892 | A |
| GraphDTA (GAT_GCN) [4] | 2021 | 0.245 | 0.881 | A |
| GraphDTA (GCN) [4] | 2021 | 0.254 | 0.880 | A |
| GraphDTA (GIN) [4] | 2021 | 0.229 | 0.893 | A |
| FusionDTA [7] | 2022 | 0.208 | 0.913 | B |
| FusionDTA (KD, large) [7] | 2022 | 0.205 | 0.914 | B |
| MGraphDTA [6] | 2022 | 0.207 | 0.900 | B |
| MFR-DTA [31] | 2023 | 0.221 | 0.905 | C |
| NHGNN-DTA [9] | 2023 | 0.196 | 0.914 | D1 |
| SSM-DTA [8] | 2023 | 0.219 | 0.890 | D2 |
| HGTDP-DTA [10] | 2024 | 0.142 | 0.910 | C, preprint |
| MDF-DTA [32] | 2024 | 0.172 | 0.912 | C |
| ProSmith [33] | 2024 | 0.186 | 0.911 | D1 |
| DeepDTAGen [34] | 2025 | 0.214 | 0.890 | B |
| HCAF-DTA [11] | 2025 | 0.198 | 0.908 | C, preprint |
| CurvFlow-DTA [12] | 2026 | 0.185 | 0.902 | C, preprint |

All 19 values are author-reported, not local reruns. A: official fold files verified, selection and aggregation qualified; the released GraphDTA training script includes test-based selection. B: exact fold identity, selection or aggregation not fully verified; WideDTA is a preprint version. C: additional structural inputs, binding-site supervision or affinity-derived graphs; not input-matched. D1: alternate random 80/10/10 settings. D2: random 70/10/20 split. Exact split and resource differences are detailed in the source-linked registry. Preprint labels refer to the versions reviewed. BINDRA uses external pretraining and prediction averaging; no cross-paper significance or universal superiority claim is made.

## Structural mappings and controls

Structural quality and target coverage determined complex selection. All 30 complexes have chemical and construct-to-author residue mappings, including missing residues and insertion codes. Eligibility required noncovalent small-molecule binding, resolution <= 2.5 angstrom, usable coordinates, and unambiguous sequence/chemical correspondence. The panel reached ten target groups, with at most three complexes per group and no selection shortfall.

For each member, last-layer attention was averaged over heads; member maps were combined after residue and ligand coordinate alignment. Individual heads, both directions, and knowledge-token mass remain available. Contact scores used physical tokens and resolved residues, with minimum heavy-atom distance <= 4 angstrom as primary and 3.5 and 5 angstrom for sensitivity analyses. Residue, residue-atom, and ligand-atom agreement were evaluated separately.

Contact prevalence and residue-score permutations provide reference values. Target-group bootstrap intervals equally weight the ten target groups. Agreement across heads and members, atom reordering, progressive parameter randomization, prediction changes after swaps, and protein-feature gradients assess different aspects of model behavior. Ten matched random masks accompany 10% residue-feature masks. These are feature perturbations, not physical mutations, and can be outside the training distribution. Integrated gradients, additional masking fractions, and coordinate-aligned explanation-map swaps were not evaluated.

### FAK1 and P4N

PDB 6YOJ provides the FAK1-P4N example at 1.361-angstrom resolution. Exported residue indices map to PDB author numbering. Labeled min-max displays accompany raw attention weights. BINDRA-iBAM predicted pKd 5.41667 for this input. Source Ki and IC50 annotations retain their assay context; they are neither trained prediction endpoints nor interchangeable pKd labels.

The focal construct has 12 unique ligands with Kd measurements: three actives, four weak compounds, and five intermediate/discordant cases. Its seven-compound active/weak subset fell below the primary screening requirement and was analyzed as exploratory alongside eligible measured-negative panels.

The main-text case figure shows the raw physical-token iBAM alongside ligand-only bond attention and a protein-to-ligand bond summary. These matrices have different query/key axes and normalization domains. For 6YOJ, residue AP was 0.17648, but derived residue-atom AP was 0.04178 for ligand-to-protein attention and 0.00785 for protein-to-ligand attention, against contact prevalence 0.00784. Pairwise localization therefore differed between the two attention directions. In addition, 31 of 34 ligand heavy atoms contacted the protein, giving atom-contact prevalence 0.91176. Ligand-atom AP was 0.86535 for ligand-to-protein attention and 0.85919 for protein-to-ligand attention, below that reference. Atom-level scores are therefore interpreted against this high-prevalence reference, separately from the residue-level results. The full matrices, mapped author residue numbers, and physical-token mappings accompany the case figure; min-max display scaling, where used, is separate from quantitative scoring.

**Figure S1. FAK1-P4N physical-token attention and derived atom projection.** PDB 6YOJ, at 1.361-angstrom resolution, has 259 resolved residues within the 282-residue modeled construct. The upper profile averages ligand-to-protein attention over 37 physical bond tokens and is min-max scaled only for display; gray regions lack crystal coordinates. Contact residues use the 4-angstrom cutoff. The lower panel is a derived protein-to-ligand atom projection: each bond-token weight is divided equally between its mapped endpoint atoms. White circles mark crystal atom-residue contacts. This projection is not native atom-level attention. PDB author numbering and unscaled weights are exported; no display scaling enters contact scoring.

[Open figure](../figures/Supplementary_Figure_1_FAK_contacts.pdf)

**Figure S2. P4N ligand-only bond attention.** The heatmap shows attention between physical ligand tokens in the shared frozen LiGhT-mini encoder, before protein conditioning. Rows are receiving tokens and columns are source tokens; weights are unitless. Knowledge tokens are excluded from this physical-token display, and their attention mass is exported separately. The map summarizes target-independent attention within the ligand representation and is shared by the affinity members.

[Open figure](../figures/Supplementary_Figure_2_FAK_bonds.pdf)

### Bond attention is a shared ligand representation

Bond maps precede target conditioning. The three affinity members share one frozen ligand encoder, so repeated maps are not independent encoder replications. Controls cover 27 unique ligands in 25 chemical components. Full-matrix and supported-edge correlations are separate; supported-edge analyses followed inspection of full-matrix behavior and are secondary. Maps showed atom-order sensitivity and persistence under layer randomization, with larger changes after whole-encoder randomization. Exports include topology controls, symmetry-aware atom/bond mappings, knowledge-token mass, and mapped analog edits.

**Figure S3. Ligand bond-map sensitivity controls.** Controls cover 27 unique ligands in 25 connected chemical groups. This figure evaluates mapped, supported off-diagonal attention entries. Panel A assesses atom-order stability, and panel B progressively randomizes ligand-encoder parameters. Points show per-ligand summaries; crosses show chemical-group means. All affinity members use the same ligand encoder. Full-matrix results are reported separately; randomization helps distinguish learned weighting from patterns preserved by the representation.

[Open figure](../figures/Supplementary_Figure_3_bond_controls.pdf)

## Exploratory methods

Screening examined transfer beyond the kinase-focused Davis domain. Molecular generation and docking were exploratory extensions of the affinity comparison. Joint prediction-generation frameworks already exist, including DeepDTAGen [34]. Diffusion and scaffold constraints can propose related graphs, but chemical validity alone does not demonstrate useful target conditioning [24, 25].

### Retrospective compound screening

We froze target eligibility, candidate pools, seeds, ranking rules, and exposure audits before scoring. Primary measured-negative panels required at least ten actives at pKd >= 7 and ten weak compounds at pKd <= 5 after seed exclusion. Measurements had to be exact, eligible Kd observations. Intermediate or discordant measurements were excluded from the binary subset, and unlabeled compounds were not assigned affinity labels. Three eligible constructs mapped to galectin-3 (P17931), beta-2 adrenergic receptor (P07550), and KEAP1 (Q14145). Exact sequences and normalized chemical identifiers accompany the source tables. FAK1's measured Kd pool did not meet the coverage requirement and remained exploratory.

Both frozen ensembles scored every candidate. Target-only pKd rankings were compared with Morgan similarity to a known active seed, standardized property similarity, fixed random ranking, and seed-assisted retrieval. The latter equally averaged full-pool pKd and similarity percentile ranks. Seed-informed methods were identified separately because target-only prediction lacked that information. Seeds and duplicate chemical identities were removed.

Measured weak compounds and property-matched presumed decoys were analyzed separately. The decoy target was 25 per active, with shortfalls retained. We reported full-pool EF1%, AP, Recall@10/50/100, candidate counts, and prevalence. Bootstrap units combined transitive chemical-identity, normalized-parent, and scaffold connections. Some resamples contained only one class; interval records therefore retained the number of finite resamples. Small component counts could make intervals unstable.

### Independent analog generation

We built endpoint-labeled analog edits from exact measurements with compatible protein constructs and available assay context. The source cohort contained 93454 directed edits. Graph quality control retained 41682 fitting and 24819 validation edits, with 26953 exclusions recorded. Chemical series, shared molecules, reverse pairs, and related edits remained together across partitions. Each series contributed at most 32 directed pairs, selected deterministically. Exposed evaluation components were excluded and recorded. Source-verified assay groups were distinguished from proxy comparability groups, which indicate source-level comparability rather than verified within-assay potency differences.

The graph-diffusion generator had its own parameters, optimizer, exponential moving average (EMA), and random-number-generator (RNG) state. BINDRA-iBAM remained frozen. Conditioning used the actual seed graph/tokens, endpoint, target residue features, member-specific fusion features, and mapped seed-atom iBAM context. Generator projections kept member feature spaces separate. Product graphs supplied denoising targets only. Cached features were bound to ensemble and preprocessing identities.

Six capacity- and draw-matched conditions each used three training seeds. Structured iBAM used the actual mapped seed-atom context. Uniform iBAM removed localization while retaining the context interface. Shuffled iBAM permuted atom-to-context assignments without changing atom identity. Global-only removed local attention context. Seed-only removed every target-derived signal, including target length. Target-shuffled substituted a different target through a fixed permutation.

The compact generator used dimension 128, three layers, four attention heads, a 64-atom cap, 500 forward diffusion steps, and 100 exact-posterior reverse evaluations. Target features were pooled to 32 residue tokens per member. Training used batch size 64, learning rate 0.0001, weight decay 0.01, at most 20 epochs, and patience three. Endpoint- and series-balanced validation denoising loss selected checkpoints. Requested-delta conditioning required source-verified same-assay training groups with at least 20 undirected edits, ten molecules, and three series, and was restricted to their training fifth-to-95th-percentile delta range. No group qualified, so this conditioning was disabled.

Fifty held-out queries were selected without model-performance criteria: five strict Davis-construct Kd queries across three series, then 18 Ki and 27 IC50 queries allocated by endpoint and target coverage. Each received 100 draws per condition and seed, with complete sample records and a completion manifest. Affinity scores were excluded from generator optimization. Supported scaffold stereochemistry was preserved, unspecified new centers flagged, and repair disabled.

Raw validity required a supported chemical reconstruction under the recorded sampling policy. Uniqueness was the number of unique valid chemical identities divided by valid draws within a query/run. Scaffold retention used all draws as its denominator. Recovery measured the fraction of eligible held-out analog identities recovered per query/run, not predicted affinity improvement or the fraction of generated draws. Evaluation also included seed-copy rate, diversity, and seed similarity. Novelty was assessed separately against generator training, ligand pretraining, and Davis supervision using the declared chemical-identity normalization. These reference-set comparisons do not assess novelty across all public chemistry. Equal-query/equal-seed means and equally weighted chemical-series interval estimates summarize different quantities and were labeled separately.

### Bounded docking assessment

We used AutoDock Vina 1.2.7, Meeko 0.8.0, and PoseBusters 0.6.5 [26, 27, 28]. Native redocking used three search seeds and exhaustiveness 32, with identical preparation and search settings across comparisons. A receptor qualified if the top-ranked pose had symmetry-aware heavy-atom root-mean-square deviation (RMSD) <= 2 angstrom and passed pose checks in at least two seeds. Ligands were compared in the experimental receptor frame, without independently superimposing them to remove pose error.

Generated-compound docking used qualifying receptors with eligible held-out Kd queries. Up to five valid, unique, generator-training-novel compounds per query, condition, and training seed were selected by hash, independently of affinity or docking scores. Preparation failures, poses, stereochemistry/geometry checks, and search outputs were retained. Pose checks were conditional on preparation; repeated poses were not independent compounds. Neither docking scores nor geometry checks established measured binding or potency.

## Exploratory results

### Native receptor qualification

Native redocking qualified 6YOJ in three of three seeds and 9F3V in two of three; 5Y80 qualified in none. The unsuccessful receptor remained reported. FAK1 had no eligible held-out generator query and therefore contributed native redocking only. Generated docking was limited to the qualifying 9F3V-associated query.

**Table S2. Native redocking outcomes across search seeds.**

| PDB | Ligand | RMSD by search seed (angstrom) | Pose checks | Qualified |
|---|---|---|---|---|
| 6YOJ | P4N | 0.385 / 0.371 / 0.361 | 3/3 | True |
| 5Y80 | IRE | 3.858 / 3.935 / 3.891 | 0/3 | False |
| 9F3V | A1H90 | 11.318 / 1.098 / 1.165 | 2/3 | True |

Three native-ligand searches per receptor, exhaustiveness 32. RMSD is symmetry-aware heavy-atom displacement in the experimental receptor frame. Qualification requires RMSD <= 2 angstrom and acceptable pose checks in at least two seeds. Search order and complete checks are in docking_native_searches.csv.

**Figure S4. Native-ligand redocking by receptor and search seed.** Three search seeds per receptor used exhaustiveness 32. Points show symmetry-aware heavy-atom RMSD for the top-ranked native-ligand pose in the experimental receptor frame; the dashed line marks 2 angstrom. Circles indicate poses that passed PoseBusters checks; crosses indicate failures. A receptor qualifies only when RMSD is at most 2 angstrom and pose checks pass in at least two seeds. 6YOJ qualified in three seeds, 9F3V in two, and 5Y80 in none. Failed redocking remains reported.

[Open figure](../figures/Supplementary_Figure_4_native_docking.pdf)

### Retrospective screening across three targets

The measured-negative panels contained 73 galectin-3, 25 beta-2 adrenergic receptor, and 38 KEAP1 candidates after filtering, with active prevalences 0.699, 0.520, and 0.316. Full pools with presumed decoys contained 1348, 329, and 227 candidates, respectively. Decoy shortfalls were zero, 21, and 111 compounds and were retained.

On the measured panels, primary BINDRA AP was 0.533, 0.408, and 0.487 for the three targets. BINDRA-iBAM achieved 0.681, 0.513, and 0.324, while fingerprint similarity achieved 0.983, 0.920, and 0.720. Seed-assisted retrieval improved some model rankings, with relative performance varying across targets and similarity or property controls. These comparisons preserve the information difference between seed-informed controls and target-only prediction.

These three non-kinase panels characterize retrospective transfer performance; broader screening utility and affinity calibration require independent evaluation. Additional file 2 and candidate-level artifacts provide full-pool EF1%, recall, labels, exposure strata, and chemical-component intervals, including unstable or nonfinite resamples. FAK1 remained exploratory.

**Table S3. Measured-negative screening average precision.**

| Target | N | Actives | Active fraction | BINDRA AP | iBAM AP | Seed FP AP |
|---|---|---|---|---|---|---|
| Galectin-3 | 73 | 51 | 0.699 | 0.533 | 0.681 | 0.983 |
| Beta-2 adrenergic receptor | 25 | 13 | 0.520 | 0.408 | 0.513 | 0.920 |
| KEAP1 | 38 | 12 | 0.316 | 0.487 | 0.324 | 0.720 |

AP is average precision, not trapezoidal PR-AUC. BINDRA and iBAM columns are target-only pKd rankings; the fingerprint method receives an active seed. Exact seeds and duplicate identities are removed. These retrospective non-kinase panels do not establish general screening superiority. Presumed decoys, seed-assisted/property/random methods, full-pool EF1%, recall and chemical-component intervals remain in Additional file 2.

### Chemical validity and measured-analog recovery

All 18 generator fits and 900 query/run evaluations completed, producing 90000 draws. Across all 50 queries, structured conditioning had equal-query/equal-seed mean raw validity of 88.37%. In the primary five-query strict Kd subset, all six conditions achieved 100% raw validity and scaffold retention without repair. Structured within-query uniqueness was 99.27% in that subset.

None of the six conditions recovered a held-out measured analog in the 9000 strict-Kd draws across three chemical series. High validity and exact training-set novelty therefore did not translate into measured-analog recovery in this subset. Uniform and target-shuffled conditions could attain lower validation denoising losses than structured conditioning, illustrating why generation quality requires direct evaluation beyond denoising loss.

Supporting Ki and IC50 validity, uniqueness, and recovery appear separately in Table S10 and the numerical exports. The primary strict-Kd recovery result was zero within the evaluated subset and sampling budget. Additional file 4 includes seed-copy rates, rejection reasons, diversity, novelty by reference set, per-seed losses, and all draws. Experimental binding and measured potency improvement remain untested.

**Table S4. Generator outcomes across conditioning controls.**

| Condition | All-query validity (%) | Strict-Kd validity (%) | Strict-Kd uniqueness (%) | Strict-Kd recovery (%) |
|---|---|---|---|---|
| Structured iBAM | 88.37 | 100.00 | 99.27 | 0.00 |
| Global only | 88.90 | 100.00 | 99.00 | 0.00 |
| Uniform iBAM | 88.26 | 100.00 | 99.33 | 0.00 |
| Shuffled iBAM | 88.19 | 100.00 | 99.20 | 0.00 |
| Seed only | 89.38 | 100.00 | 99.20 | 0.00 |
| Target shuffled | 89.04 | 100.00 | 99.00 | 0.00 |

Equal-query/equal-seed means; three seeds per condition and 100 draws/query/run. The full set has 50 queries and 90000 draws. Strict Kd has five queries, three chemical series and 9000 draws; scaffold retention is 100% in every condition. Uniqueness is conditional on valid draws within a query/run. Repair is none. Endpoint/series intervals are separate exports; exact novelty does not establish public chemical novelty or binding.

**Figure S5. Generator outcomes across conditioning controls.** Six conditioning approaches each use three training seeds and 100 draws per query across five strict Kd queries from three chemical series, totaling 9000 draws. Bars show raw validity without repair; markers show measured-analog recovery. Every condition reached full validity and zero recovery in this subset. These outcomes distinguish valid molecular graphs from recovery of measured analogs. Broader endpoint results are reported separately.

[Open figure](../figures/Supplementary_Figure_5_generator.pdf)

### Generated compound preparation and pose quality

Among 90 hash-selected compound instances, 55 failed preparation and 35 produced three search poses each. Of these 105 poses, 91 passed overall PoseBusters checks and 14 did not. No Vina search process failed. Preparation succeeded for 38.9% of selected instances; the 86.7% pose-QC pass rate was conditional on preparation and excluded those failures from its denominator. Repeated poses did not increase the number of distinct tested compounds.

This one-query assessment quantifies preparation losses and computed geometry. Measured binding, affinity improvement, and performance across the broader query set require separate evaluation. Supplementary artifacts retain all failures, checks, and poses.

**Figure S6. Generated-compound preparation and pose checks.** The analysis covers one eligible held-out query associated with receptor 9F3V. Panel A counts 90 hash-selected compound instances: 55 failed preparation and 35 were prepared. Panel B counts 105 repeated-search poses from those 35 instances: 91 passed overall PoseBusters checks and 14 failed. Preparation success is 38.9% of selected instances; the 86.7% pose-QC rate is conditional on preparation. Repeated poses are not independent compounds, and neither docking nor geometry checks establish binding.

[Open figure](../figures/Supplementary_Figure_6_generated_docking.pdf)

## Reproducibility and data availability

The reproducibility checker verified frozen member hashes, official fold membership, ensemble arithmetic, numerical metrics, table provenance, all 18 generator checkpoint/sample identities, and figure export requirements. The six final warm-test member exports were reproduced bitwise. Historical adapted-encoder experiments and their earlier BF16 reproduction discrepancies are documented separately. Inductor/Triton acceleration was retained only where numerical equivalence checks passed; a proposed batch conditioner that failed those checks was excluded.

Source code, environment versions, and optimizer, EMA, and RNG records support resumption under matching settings. With upstream features cached, mean warm full-pool training time per member was 982 seconds for pooled BINDRA (C-matched) and 2366 seconds for BINDRA-iBAM (F); per-member inference throughput was approximately 3929 and 2298 pairs per second. These hardware- and batch-dependent measurements describe predictor execution from cached features and exclude upstream encoding.


## Scope of the reported study

Affinity prediction comparisons cover the completed Davis warm, cold-drug, and cold-protein experiments. Architecture selection used three seeds on official validation fold 0; additional-fold selection stability, independent-dataset tests, and local external-model or classical baseline reruns are outside the reported study. Supporting screening, structural, and molecular-editing analyses retain their declared data sources and endpoint strata. Only completed experiments contribute to the reported evidence.


## Extended numerical results

The following tables transcribe completed result exports. No new fitting, prediction, architecture selection, or statistical testing was performed to assemble them. Full-precision source tables accompany the package.

**Table S5. Architecture selection and trainable prediction parameters.**

| Arm | Architecture | Parameters | Validation MSE | Seed SD | Refit epochs |
|---|---|---|---|---|---|
| A | Protein only | 347843 | 0.71282 | 0.00081 | 28 |
| B | Ligand only | 348035 | 0.62299 | 0.00093 | 28 |
| C | Pooled concatenation | 695365 | 0.23409 | 0.00201 | 72 |
| D | Gated pooled fusion | 697287 | 0.23213 | 0.00383 | 72 |
| E | Attention with FiLM | 2451205 | 0.23490 | 0.00960 | 58 |
| F | Attention without FiLM | 2228293 | 0.22461 | 0.00306 | 69 |
| G | FiLM attention with gates | 2454664 | 0.23569 | 0.00683 | 54 |
| C-matched | Capacity-matched pooled fusion | 2455221 | 0.21528 | 0.00807 | 69 |

Three seeds per arm on official validation fold 0. Parameters exclude the shared frozen encoders. MSE is in squared pKd units. SD is the sample standard deviation across seeds. C-matched is primary BINDRA; F is the retained BINDRA-iBAM companion. Refit durations were derived from validation, not test error.

**Table S6. All executed individual-model Davis test results.**

| Protocol | Arm | Seed | Test pairs | MSE | CI |
|---|---|---|---|---|---|
| warm | A | 20261001 | 5010 | 0.72355 | 0.65771 |
| warm | A | 20261002 | 5010 | 0.73121 | 0.65915 |
| warm | A | 20261003 | 5010 | 0.72248 | 0.65580 |
| warm | B | 20261001 | 5010 | 0.65458 | 0.73980 |
| warm | B | 20261002 | 5010 | 0.66222 | 0.73633 |
| warm | B | 20261003 | 5010 | 0.65300 | 0.74039 |
| warm | C | 20261001 | 5010 | 0.24020 | 0.89175 |
| warm | C | 20261002 | 5010 | 0.25014 | 0.89378 |
| warm | C | 20261003 | 5010 | 0.23215 | 0.89101 |
| warm | D | 20261001 | 5010 | 0.23159 | 0.89003 |
| warm | D | 20261002 | 5010 | 0.23956 | 0.89026 |
| warm | D | 20261003 | 5010 | 0.23320 | 0.88917 |
| warm | E | 20261001 | 5010 | 0.24378 | 0.88854 |
| warm | E | 20261002 | 5010 | 0.23118 | 0.89384 |
| warm | E | 20261003 | 5010 | 0.23543 | 0.89534 |
| warm | F | 20261001 | 5010 | 0.22961 | 0.89878 |
| warm | F | 20261002 | 5010 | 0.22826 | 0.89687 |
| warm | F | 20261003 | 5010 | 0.22264 | 0.89371 |
| warm | G | 20261001 | 5010 | 0.23785 | 0.88522 |
| warm | G | 20261002 | 5010 | 0.22306 | 0.89407 |
| warm | G | 20261003 | 5010 | 0.25514 | 0.86894 |
| warm | C-matched | 20261001 | 5010 | 0.22000 | 0.89649 |
| warm | C-matched | 20261002 | 5010 | 0.21488 | 0.89201 |
| warm | C-matched | 20261003 | 5010 | 0.21180 | 0.89593 |
| cold-drug | C-matched | 20261001 | 2652 | 0.52679 | 0.69474 |
| cold-drug | C-matched | 20261002 | 2652 | 0.55238 | 0.72202 |
| cold-drug | C-matched | 20261003 | 2652 | 0.49643 | 0.70902 |
| cold-drug | F | 20261001 | 2652 | 0.61290 | 0.71847 |
| cold-drug | F | 20261002 | 2652 | 0.66599 | 0.70590 |
| cold-drug | F | 20261003 | 2652 | 0.67410 | 0.73439 |
| cold-drug | E | 20261001 | 2652 | 0.49756 | 0.77016 |
| cold-drug | E | 20261002 | 2652 | 0.58176 | 0.67541 |
| cold-drug | E | 20261003 | 2652 | 0.60025 | 0.71575 |
| cold-drug | G | 20261001 | 2652 | 0.49271 | 0.70013 |
| cold-drug | G | 20261002 | 2652 | 0.52981 | 0.74503 |
| cold-drug | G | 20261003 | 2652 | 0.58273 | 0.73743 |
| cold-protein | C-matched | 20261001 | 2992 | 0.29597 | 0.82792 |
| cold-protein | C-matched | 20261002 | 2992 | 0.30868 | 0.83416 |
| cold-protein | C-matched | 20261003 | 2992 | 0.30421 | 0.82847 |
| cold-protein | F | 20261001 | 2992 | 0.24890 | 0.85646 |
| cold-protein | F | 20261002 | 2992 | 0.25823 | 0.85437 |
| cold-protein | F | 20261003 | 2992 | 0.28171 | 0.84263 |
| cold-protein | E | 20261001 | 2992 | 0.27980 | 0.83855 |
| cold-protein | E | 20261002 | 2992 | 0.28121 | 0.83942 |
| cold-protein | E | 20261003 | 2992 | 0.28466 | 0.83686 |

Each row is one fitted model, not an ensemble. All executed records are retained, including unselected architecture arms. Arm definitions are in Table S5. MSE is in squared pKd units; CI is unitless. Test results were not used to select the primary architecture. Historical test exposure limits confirmatory interpretation. Full RMSE, MAE, correlations, AP, and both r_m-squared definitions remain in individual_model_metrics.csv.

**Table S7. Attention and contact correspondence for every structural complex.**

| PDB | Target | Ligand | Residue AP | Contact prevalence | Gradient AP |
|---|---|---|---|---|---|
| 6YOJ | Q05397 | P4N | 0.17648 | 0.08108 | 0.13307 |
| 8A27 | P00533 | KY9 | 0.11486 | 0.07063 | 0.17400 |
| 6Q4H | P24941 | HGH | 0.10090 | 0.05155 | 0.10325 |
| 8C7X | P15056 | TXV | 0.25938 | 0.06934 | 0.22421 |
| 4ZZN | P28482 | CQ8 | 0.20000 | 0.05310 | 0.12490 |
| 2XIR | P35968 | 00J | 0.09611 | 0.06419 | 0.07710 |
| 1UNQ | P31749 | 4IP | 0.12433 | 0.09244 | 0.09632 |
| 3FX5 | P03367 | KNI | 0.13608 | 0.13131 | 0.17822 |
| 4UDW | P00734 | N6L | 0.13417 | 0.07968 | 0.09478 |
| 1KMV | P00374 | LII | 0.08113 | 0.06989 | 0.07360 |
| 6YT6 | Q05397 | PKE | 0.23330 | 0.07031 | 0.08934 |
| 8A2D | P00533 | KXY | 0.09372 | 0.06762 | 0.09124 |
| 6Q48 | P24941 | HHQ | 0.06580 | 0.01786 | 0.12136 |
| 8C7Y | P15056 | TXV | 0.25712 | 0.06545 | 0.19323 |
| 6SLG | P28482 | LHZ | 0.20996 | 0.06325 | 0.17300 |
| 3VO3 | P35968 | 0KF | 0.07585 | 0.07261 | 0.09888 |
| 1H10 | P31749 | 4IP | 0.12478 | 0.09322 | 0.09684 |
| 3DJK | P03367 | G55 | 0.11187 | 0.11111 | 0.12549 |
| 5AF9 | P00734 | SJR | 0.06228 | 0.05200 | 0.14717 |
| 1KMS | P00374 | LIH | 0.11148 | 0.06486 | 0.07547 |
| 4I4E | Q05397 | 1BQ | 0.16209 | 0.05906 | 0.13655 |
| 8A2A | P00533 | KY9 | 0.10381 | 0.06536 | 0.14485 |
| 6Q4J | P24941 | HHB | 0.13765 | 0.03806 | 0.05589 |
| 7K0V | P15056 | VQP | 0.20471 | 0.07634 | 0.16483 |
| 4QTA | P28482 | 38Z | 0.11835 | 0.06866 | 0.10987 |
| 3WZD | P35968 | LEV | 0.09472 | 0.07194 | 0.09137 |
| 4GV1 | P31749 | 0XZ | 0.08070 | 0.06667 | 0.05293 |
| 3DK1 | P03367 | G05 | 0.13125 | 0.11111 | 0.14443 |
| 5MM6 | P00734 | 32U | 0.06601 | 0.07570 | 0.09629 |
| 3FS6 | P00374 | DH1 | 0.06542 | 0.06989 | 0.08485 |

Thirty complexes in ten target groups. Heavy-atom contacts use 4 angstrom; AP and prevalence are unitless. Gradient refers to gradient-times-input residue scores. Rows are descriptive per-complex values; target-group bootstrap summaries and paired controls are in the main paper and Additional file 2. Residue-atom results, structural eligibility, mappings, and distance masks accompany the numeric exports. Contact correspondence is not evidence of a causal binding mechanism.

**Table S8. Residue attention sensitivity to input order and parameter randomization.**

| Control | Mean map correlation | Mean absolute change in pKd | Target groups |
|---|---|---|---|
| Atom order | 0.98950 | 0.02153 | 10 |
| adapters | -0.00911 | 5.16421 | 10 |
| cross attention | 0.12857 | 5.19797 | 10 |
| score head | 1.00000 | 5.26379 | 10 |

Mean Spearman correlation compares residue maps with their original maps. Parameter-control labels identify the randomized module. A randomized score head can change the prediction without changing upstream attention. Mean absolute prediction differences are pKd units; correlations are unitless. These controls assess model dependence rather than physical causality. Per-complex records are in revision_attention_sensitivity_raw.csv.

**Table S9. Bond attention sensitivity to atom order and ligand-encoder randomization.**

| Control | Entries compared | Ligands | Median correlation | 25th percentile | 75th percentile |
|---|---|---|---|---|---|
| Atom order | Full matrix | 27 | 0.92249 | 0.90028 | 0.95938 |
| Atom order | Supported off-diagonal | 27 | 0.85642 | 0.82678 | 0.91588 |
| Last 1 layers | Full matrix | 27 | 0.99811 | 0.99694 | 0.99857 |
| Last 1 layers | Supported off-diagonal | 27 | 0.96355 | 0.95985 | 0.96695 |
| Last 4 layers | Full matrix | 27 | 0.99814 | 0.99716 | 0.99850 |
| Last 4 layers | Supported off-diagonal | 27 | 0.96404 | 0.95924 | 0.96594 |
| Last 8 layers | Full matrix | 27 | 0.99801 | 0.99684 | 0.99837 |
| Last 8 layers | Supported off-diagonal | 27 | 0.96192 | 0.95602 | 0.96522 |
| Whole encoder | Full matrix | 27 | 0.95174 | 0.92912 | 0.96590 |
| Whole encoder | Supported off-diagonal | 27 | 0.23822 | 0.20005 | 0.26056 |

Spearman correlations are unitless, across 27 unique ligands in 25 chemical components. Supported off-diagonal analysis excludes diagonal and unsupported edges and is secondary. Layer labels reproduce the recorded controls. The ligand encoder is frozen and shared, so member maps are not independent ligand-encoder replications. Full-matrix correlation can reflect matrix structure that persists under randomization. Figure S3 displays supported off-diagonal results.

**Table S10. Generator results separated by measured endpoint.**

| Endpoint | Condition | Queries | Raw validity (%) | Uniqueness (%) | Analog recovery (%) |
|---|---|---|---|---|---|
| Kd | Structured iBAM | 5 | 100.00 | 99.27 | 0.00 |
| Kd | Global only | 5 | 100.00 | 99.00 | 0.00 |
| Kd | Uniform iBAM | 5 | 100.00 | 99.33 | 0.00 |
| Kd | Shuffled iBAM | 5 | 100.00 | 99.20 | 0.00 |
| Kd | Seed only | 5 | 100.00 | 99.20 | 0.00 |
| Kd | Target shuffled | 5 | 100.00 | 99.00 | 0.00 |
| Ki | Structured iBAM | 18 | 87.20 | 94.94 | 2.47 |
| Ki | Global only | 18 | 87.74 | 95.44 | 4.32 |
| Ki | Uniform iBAM | 18 | 87.13 | 95.31 | 2.47 |
| Ki | Shuffled iBAM | 18 | 87.17 | 95.24 | 3.09 |
| Ki | Seed only | 18 | 88.07 | 95.35 | 3.09 |
| Ki | Target shuffled | 18 | 88.39 | 96.31 | 1.85 |
| IC50 | Structured iBAM | 27 | 86.99 | 97.61 | 1.85 |
| IC50 | Global only | 27 | 87.62 | 97.35 | 1.23 |
| IC50 | Uniform iBAM | 27 | 86.84 | 97.65 | 0.62 |
| IC50 | Shuffled iBAM | 27 | 86.69 | 97.65 | 0.62 |
| IC50 | Seed only | 27 | 88.28 | 97.42 | 0.00 |
| IC50 | Target shuffled | 27 | 87.44 | 97.29 | 0.00 |

Equal-query/equal-seed means, three training seeds per condition, 100 draws per query/run. Kd is the primary strict Davis-construct subset: five queries across three series. Ki and IC50 are supporting strata, with 18 and 27 queries. Uniqueness is conditional on valid draws within a query/run. Analog recovery is the mean fraction of eligible held-out analog identities recovered per query/run, not the fraction of generated draws. Repair is none. Endpoint strata do not represent affinity predictions from Ki or IC50 heads. Series-weighted interval centers differ from the equal-query means shown here; all weighting schemes and interval records remain in the CSV.

**Table S11. Ten highest-ranked resolved FAK1 residues by ligand-to-protein attention.**

| Rank | Construct position | PDB residue | Raw attention | Distance (angstrom) | Contact <= 4 angstrom |
|---|---|---|---|---|---|
| 1 | 95 | CYS502 | 0.173233 | 2.87 | True |
| 2 | 125 | THR532 | 0.074561 | 15.95 | False |
| 3 | 148 | SER555 | 0.047539 | 7.66 | False |
| 4 | 135 | PHE542 | 0.041681 | 12.54 | False |
| 5 | 190 | ARG597 | 0.029110 | 21.50 | False |
| 6 | 156 | GLY563 | 0.025189 | 3.40 | True |
| 7 | 83 | ILE490 | 0.023385 | 13.16 | False |
| 8 | 92 | MET499 | 0.022947 | 3.24 | True |
| 9 | 14 | ARG421 | 0.017508 | 15.58 | False |
| 10 | 239 | ASN646 | 0.015157 | 25.20 | False |

Resolved residues only. Ranking averages raw attention over the 37 physical P4N bond tokens. Three of the ten residues contact P4N within 4 angstrom. Full construct-to-author mappings, missing residues, raw scores, and both attention directions accompany the FAK1 case CSVs. Min-max display scaling does not enter this ranking.



## References

1. Davis MI, Hunt JP, Herrgard S, Ciceri P, Wodicka LM, Pallares G, et al. Comprehensive analysis of kinase inhibitor selectivity. Nat Biotechnol. 2011;29(11):1046-1051. doi:10.1038/nbt.1990.

2. Öztürk H, Özgür A, Ozkirimli E. DeepDTA: deep drug-target binding affinity prediction. Bioinformatics. 2018;34(17):i821-i829. doi:10.1093/bioinformatics/bty593.

3. Ozturk H, Ozkirimli E, Ozgur A. WideDTA: prediction of drug-target binding affinity. 2019. doi:10.48550/arXiv.1902.04166. Preprint, arXiv:1902.04166.

4. Nguyen T, Le H, Quinn TP, Nguyen T, Le TD, Venkatesh S. GraphDTA: predicting drug-target binding affinity with graph neural networks. Bioinformatics. 2021;37(8):1140-1147. doi:10.1093/bioinformatics/btaa921.

5. Jiang M, Li Z, Zhang S, Wang S, Wang X, Yuan Q, et al. Drug-target affinity prediction using graph neural network and contact maps. RSC Adv. 2020;10(35):20701-20712. doi:10.1039/d0ra02297g.

6. Yang Z, Zhong W, Zhao L, Yu-Chian Chen C. MGraphDTA: deep multiscale graph neural network for explainable drug-target binding affinity prediction. Chem Sci. 2022;13(3):816-833. doi:10.1039/d1sc05180f.

7. Yuan W, Chen G, Chen CYC. FusionDTA: attention-based feature polymerizer and knowledge distillation for drug-target binding affinity prediction. Brief Bioinform. 2022;23:bbab506. doi:10.1093/bib/bbab506.

8. Pei Q, Wu L, Zhu J, Xia Y, Xie S, Qin T, et al. Breaking the barriers of data scarcity in drug-target affinity prediction. Brief Bioinform. 2023;24:bbad386. doi:10.1093/bib/bbad386.

9. He H, Chen G, Chen CYC. NHGNN-DTA: a node-adaptive hybrid graph neural network for interpretable drug-target binding affinity prediction. Bioinformatics. 2023;39:btad355. doi:10.1093/bioinformatics/btad355.

10. Xiao X, Wang W, Xie J, Zhu L, Chen G, Li Z, et al. HGTDP-DTA: Hybrid Graph-Transformer with Dynamic Prompt for Drug-Target Binding Affinity Prediction. 2024. doi:10.48550/arXiv.2406.17697. Preprint, arXiv:2406.17697.

11. Li J, Yao L. HCAF-DTA: drug-target binding affinity prediction with cross-attention fused hypergraph neural networks. 2025. doi:10.48550/arXiv.2504.02014. Preprint version reviewed, arXiv:2504.02014v1.

12. Ma J, Yang Y, Zhao J, Zhao L. CurvFlow-DTA: dual-graph discrete Ricci curvature flow for drug-target affinity prediction. 2026. doi:10.48550/arXiv.2609.22862. Preprint, arXiv:2609.22862v1.

13. EvolutionaryScale. ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning. 2024. https://www.evolutionaryscale.ai/blog/esm-cambrian. Accessed 24 October 2026.

14. EvolutionaryScale. ESM software repository. 2024. https://github.com/evolutionaryscale/esm. Pinned revision fcc9d36f5fff97d9bb69a36e3252de61cc78c968; accessed 24 October 2026.

15. Li H, Zhang R, Min Y, Ma D, Zhao D, Zeng J. A knowledge-guided pre-training framework for improving molecular representation learning. Nat Commun. 2023;14:7568. doi:10.1038/s41467-023-43214-1.

16. Singh R, Sledzieski S, Bryson B, Cowen L, Berger B. Contrastive learning in protein language space predicts interactions between drugs and protein targets. Proc Natl Acad Sci U S A. 2023;120:e2220778120. doi:10.1073/pnas.2220778120.

17. Jain S, Wallace BC. Attention is not Explanation. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019:3543-3556. doi:10.18653/v1/n19-1357.

18. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity Checks for Saliency Maps. Advances in Neural Information Processing Systems. 2018. https://papers.nips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html.

19. Rogers D, Hahn M. Extended-Connectivity Fingerprints. J Chem Inf Model. 2010;50(5):742-754. doi:10.1021/ci100050t.

20. Liu T, Lin Y, Wen X, Jorissen RN, Gilson MK. BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities. Nucleic Acids Res. 2007;35(Database):D198-D201. doi:10.1093/nar/gkl999.

21. Bemis GW, Murcko MA. The Properties of Known Drugs. 1. Molecular Frameworks. J Med Chem. 1996;39(15):2887-2893. doi:10.1021/jm9602928.

22. Berman HM. The Protein Data Bank. Nucleic Acids Res. 2000;28(1):235-242. doi:10.1093/nar/28.1.235.

23. Worldwide Protein Data Bank. PDB structure 6YOJ. 2020. doi:10.2210/pdb6YOJ/pdb. FAK1-P4N complex; accessed 24 October 2026.

24. Ho J, Jain A, Abbeel P. Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems. 2020. https://arxiv.org/abs/2006.11239.

25. Austin J, Johnson DD, Ho J, Tarlow D, van den Berg R. Structured Denoising Diffusion Models in Discrete State-Spaces. Advances in Neural Information Processing Systems. 2021. https://arxiv.org/abs/2107.03006.

26. Eberhardt J, Santos-Martins D, Tillack AF, Forli S. AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. J Chem Inf Model. 2021;61(8):3891-3898. doi:10.1021/acs.jcim.1c00203.

27. Forli laboratory. Meeko ligand preparation toolkit. 2026. https://github.com/forlilab/Meeko. Version 0.8.0; accessed 24 October 2026.

28. Buttenschoen M, Morris GM, Deane CM. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chem Sci. 2024;15(9):3130-3139. doi:10.1039/d3sc04185a.

29. RDKit contributors. RDKit Open source cheminformatics. 2026. https://www.rdkit.org. Version recorded in the reproducibility supplement; accessed 24 October 2026.

30. Song K, Zhang J, Ye L, Zhang J. BINDRA reference implementation and revision protocol. 2026. https://github.com/kevinmsong/bindra. Accessed 26 October 2026.

31. Hua Y, Song X, Feng Z, Wu X. MFR-DTA: a multi-functional and robust model for predicting drug-target binding affinity and region. Bioinformatics. 2023;39:btad056. doi:10.1093/bioinformatics/btad056.

32. Ranjan A, Bess A, Alvin C, Mukhopadhyay S. MDF-DTA: A Multi-Dimensional Fusion Approach for Drug-Target Binding Affinity Prediction. J Chem Inf Model. 2024;64(13):4980-4990. doi:10.1021/acs.jcim.4c00310.

33. Kroll A, Ranjan S, Lercher MJ. A multimodal Transformer Network for protein-small molecule interactions enhances predictions of kinase inhibition and enzyme-substrate relationships. PLOS Computational Biology. 2024;20(5):e1012100. doi:10.1371/journal.pcbi.1012100.

34. Shah PM, Zhu H, Lu Z, Wang K, Tang J, Li M. DeepDTAGen: a multitask deep learning framework for drug-target affinity prediction and target-aware drugs generation. Nat Commun. 2025;16:5021. doi:10.1038/s41467-025-59917-6.
