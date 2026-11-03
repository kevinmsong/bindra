# BINDRA: pooled fusion and interaction attention for protein-ligand affinity prediction

Kevin Song<sup>1</sup>, John Zhang<sup>1</sup>, Lei Ye<sup>1</sup>, and Jianyi Zhang<sup>*1</sup>

<sup>1</sup>Department of Biomedical Engineering, University of Alabama at Birmingham, Birmingham, Alabama 35294, United States

*Email: jayzhang@uab.edu

## Abstract

### Background

Predicting binding affinity from protein sequences and molecular structures can guide compound prioritization. We developed BINDRA, Binding INference Design Retrieval and Attention, to assess whether protein-ligand attention improves prediction over pooled fusion, which combines separately summarized protein and ligand features. Both approaches used the same fixed encoders and comparable numbers of trainable parameters.

### Results

Across three training seeds on one fixed validation fold, pooled fusion had the lowest mean squared error (MSE) and was selected as the primary predictor. The attention architecture with the lowest validation error was retained as BINDRA-iBAM. On the official 5,010-pair Davis test, the pooled and attention ensembles achieved MSEs of 0.19663 and 0.19586 and concordance indices of 0.90802 and 0.91253, respectively. The attention-minus-pooled MSE difference was -0.00077 (95% bootstrap interval, -0.01464 to 0.01410); the interval included zero. Attention performed better in an exploratory test with protein sequences excluded from affinity training; the difference for held-out drugs remained uncertain. Detection-limit labels accounted for 69.2% of standard-test observations and were examined separately. Across 30 crystal complexes from ten target groups, residue attention achieved average precision of 0.131, compared with contact prevalence of 0.071, shuffled-attention controls of 0.093, and gradient attribution of 0.119. Paired comparisons supported improvement over shuffled controls; the difference from gradient attribution remained uncertain. Randomization and atom-order tests assessed the dependence of attention maps on learned parameters and input representation. The FAK1-P4N case showed both contacting and distant residues among the strongest attention peaks.

### Conclusions

Pooled fusion provided competitive affinity predictions, while attention offered complementary views of model emphasis and modest correspondence with crystal contacts. Historical Davis test inspection makes these findings retrospective; independent replication is needed to assess generalization.

### Scientific Contribution

This study compares pooled fusion and protein-ligand attention using shared molecular and protein features and the same tuning protocol. Mapped bond and residue analyses distinguish attention within the ligand from protein-conditioned emphasis and assess their interpretation using structural and model-based controls.

**Keywords:** cheminformatics; binding affinity prediction; molecular representations; pooled fusion; cross-attention; molecular interpretability; Davis benchmark

## Introduction

Predicting protein-ligand binding affinity can help prioritize compounds for experimental testing. Performance depends on the evaluation setting: held-out pairs of familiar molecules and proteins pose a different problem from new drug identities or protein sequences. The Davis kinase-inhibitor dataset provides a widely used quantitative benchmark, with chemical coverage, repeated protein sequences, and detection-limit labels that shape interpretation [1].

DeepDTA established a sequence-based benchmark, followed by molecular and protein subsequence features in WideDTA and graph representations in GraphDTA, DGraphDTA, and MGraphDTA [2, 3, 4, 5, 6]. Later methods introduced attention, pretrained protein features, knowledge distillation, semisupervised learning, and hybrid graphs [7, 8, 9]. Recent preprints extend graph-transformer, cross-attention, and curvature-based approaches [10, 11, 12]. Their reported results must be considered alongside differences in inputs, partitions, pretraining, and model selection.

Pretrained encoders supply contextual features before an affinity model introduces protein-ligand interactions. ESM Cambrian represents each protein residue in its sequence context [13, 14], while the Line Graph Transformer in KPGT represents molecular connectivity with bond-centered tokens [15]. Contrastive protein-ligand methods likewise demonstrate the value of pretrained features [16]. A controlled comparison with pooled fusion, which combines separately summarized protein and ligand features, can assess the additional value of an interaction module. Attention maps also require evaluation: their patterns may reflect molecular structure and input representation as well as learned parameters [17, 18].

BINDRA addresses two questions: does explicit protein-ligand attention improve affinity prediction over pooled fusion using shared frozen encoders, and how do its maps relate to crystal contacts and model behavior? We compared eight architectures on Davis, examined warm and cold evaluation settings, and assessed residue and bond attention with structural and model-based controls. Screening, molecular editing, and docking were exploratory extensions.

This design separates the predictive comparison from the interpretation study. The pooled model tests how much can be achieved from the encoder features alone, while the retained attention model provides mapped protein and ligand weights. Evaluating both under the same data and tuning protocol links their architectural differences to the settings in which they are used.

## Methods

### Study design and evaluation data

Model selection used mean validation MSE across seeds 20261001, 20261002, and 20261003. The selected pooled architecture is BINDRA; the attention architecture with the lowest validation error is BINDRA-iBAM, the Interpretable Binding Attention Map companion. Each ensemble averages its three members' pKd predictions. Internal features remain member-specific, and prediction disagreement is descriptive rather than a calibrated interval.

Davis test results were inspected during earlier development and exploratory architecture comparisons. Although the present models were selected using validation data, the test set is not historically untouched. These results therefore provide retrospective architectural evidence. Affinity weights remained fixed during supporting analyses; historical experiments and numerical reproduction details are documented in Additional file 1.

We used the official DeepDTA Davis data and fold files [2]: 30056 observations, 68 ligand identifiers, and 442 target identifiers corresponding to 379 unique protein sequences. Labels were pKd, the negative base-10 logarithm of Kd in molar units. The published pKd = 5 detection plateau was retained and analyzed separately. The official warm test contained 5010 pairs; its 25046-pair training pool supplied 5010 validation pairs from fold 0 and 20036 fitting pairs from folds 1 through 4. This was three-seed replication on one fixed validation fold. Final warm refits used the full training pool for a validation-derived duration. Prediction membership, labels, and official pair ordering were verified.

Cold-drug and cold-protein protocols separated normalized chemical identities and exact protein sequences, respectively. Label-independent allocation targeted 70%, 10%, 10%, and 10% fitting, validation, calibration, and test fractions; test sets contained 2652 and 2992 pairs. Each protocol used separately trained ensembles. Chemical separation allowed shared scaffolds, and exact-sequence separation allowed homologous proteins. Nearest-training Morgan similarity and sequence identity described novelty relative to supervised training [19]. Full partition counts, 884 cold-drug audit exclusions, and pretraining exposure strata are provided in Additional files 1 through 3.

### Shared encoders and fusion architectures

The original pretrained ESM-C 600M protein encoder remained frozen and produced 1152-dimensional residue features. Long sequences used overlapping windows with averaged shared-residue vectors; padding and special tokens were excluded from pooling. The local LiGhT-mini ligand encoder represented bonds as atom-bond-atom triplets, retained isolated atoms, and included fingerprint and descriptor tokens [15]. It used width 96, eight layers, and eight heads and was pretrained from scratch for five epochs on 649364 eligible BindingDB molecules before freezing [20]. Chemical identity, normalized-parent, and scaffold audits distinguished exact pretraining exposure from related chemistry. Complete protein-encoder pretraining exposure was unavailable; both architectures shared this provenance.

Trainable adapters projected both encoders' outputs to 192 dimensions. The eight arms were protein-only (A), ligand-only (B), pooled concatenation (C), gated pooled fusion (D), original attention with feature-wise linear modulation (FiLM; E), attention without FiLM (F), FiLM attention with grouped readout gates (G), and capacity-matched concatenation (C-matched). The grouped gates preserved the existing inputs at initialization. Architecture definitions, dimensions, and parameter counts are in Table 1 and Additional file 1.

Selected BINDRA combined pooled protein and ligand features with three global ligand summaries, yielding five readouts and 2455221 trainable parameters per member. BINDRA-iBAM used two bidirectional cross-attention layers, eight heads, and six readouts, with 2228293 trainable parameters. Its global ligand features and pre-attention protein summary bypassed cross-attention before regression. Figure 1 shows the shared inputs and prediction paths. Parameter matching applies to trainable prediction layers; the larger frozen encoders are shared.

These bypasses preserve protein and whole-molecule information alongside the interaction features. The comparison therefore evaluates complete architectures, rather than attributing every prediction difference to the attention operation alone. Separately exported ligand and residue token mappings connect attention weights to chemical bonds and protein positions.

### Training and statistical analysis

Each arm received the same six-configuration grid: learning rates 0.0001, 0.0003, and 0.001 crossed with dropout 0.1 and 0.2. AdamW used weight decay 0.01 and 128 pairs per optimizer update, with at most 200 epochs and patience 15. Validation MSE selected checkpoints. The selected configuration was evaluated across three seeds, including its first-seed tuning run; the rounded-up median epoch of minimum validation MSE determined the full-pool refit duration. BINDRA and BINDRA-iBAM both selected learning rate 0.0003, dropout 0.1, and 69-epoch warm refits. C-matched was within 1% of E and G's trainable parameter counts; comparison with the smaller C arm therefore includes a model-size difference.

We reported mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), concordance index (CI), correlations, and within-target CI. MSE has squared pKd units; RMSE and MAE have pKd units. CI scores unequal-label pairs, giving half credit to prediction ties. Within-target CI averages comparable protein sequences equally. Average precision (AP) used pKd >= 7 actives and was distinguished from trapezoidal precision-recall area; both documented r_m-squared conventions were exported separately. Full definitions are in Additional file 1.

Pooled CI includes comparisons across different proteins, whereas within-target CI asks whether ligands are ranked correctly for the same sequence. Reporting both distinguishes dataset-wide ranking from ligand prioritization for an individual target. Targets without unequal-label pairs contribute no within-target CI.

Seed standard deviations describe training variation. Sampling uncertainty used 1000 crossed bootstrap resamples of exact protein-sequence and Murcko-scaffold groups, with identical weights for paired model comparisons [21]. Structural analyses resampled target groups; generator analyses used chemical series and queries. These groupings account for specified dependencies, while related proteins or chemistry may retain additional dependence. Independent calibration was unavailable for the warm full-pool refits and limited by repeated normalized pairs in cold protocols, so member disagreement was reported without calibrated coverage.

The source-linked literature registry covers 2018 through October 17, 2026 and records all 19 reviewed variants, including split definitions, required inputs, pretraining, selection, publication status, and averaging. External models and classical affinity baselines were not rerun locally. Published values provide descriptive context for the controlled local architectural comparisons.

### Attention and supporting studies

The structural panel comprised 30 noncovalent small-molecule complexes across ten target groups, with at most three complexes per group, resolution <= 2.5 angstrom, and usable sequence and chemical mappings. Selection used structural quality and coverage, independently of attention agreement. Protein Data Bank records supplied coordinates [22]; FAK1-P4N in PDB 6YOJ was the detailed case [23]. Mappings retained author residue numbering, insertion codes, unresolved residues, mutations, alternate conformations, and ligand symmetry.

For each member, iBAMs used last-layer attention averaged across heads; maps were averaged across members only after coordinate alignment. Ligand-to-protein attention defined residue salience. Physical-contact analyses excluded padding and nonphysical knowledge tokens and used minimum heavy-atom distance <= 4 angstrom, with 3.5- and 5-angstrom sensitivity analyses. Controls included uniform and shuffled scores, head/member agreement, atom reordering, prediction swaps, progressive randomization, protein-feature gradient-times-input attribution, and 10% residue-feature masking with ten random controls. Gradients covered protein-dependent paths rather than complete ligand/global attribution. Bond self-attention was examined separately in the shared frozen ligand encoder before target conditioning. Raw weights, excluded mass, mappings, and control definitions accompany the exports.

Three non-kinase screening panels required at least ten measured actives at pKd >= 7 and ten weak compounds at pKd <= 5. Exhaustive frozen-ensemble rankings were compared with seed similarity, property distance, random ranking, and seed-assisted retrieval; seeds and duplicates were excluded. Methods receiving an active seed were identified separately, and presumed decoys were distinguished from measured negatives.

An independent scaffold-preserving diffusion generator tested six matched conditioning approaches across three seeds [24, 25]. Affinity scores were excluded from generator optimization. Docking followed native-redocking gates and identical preparation/search settings [26, 27, 28]. Endpoint-specific cohorts, training, sampling, docking, and every rejection are detailed in Additional file 1.

Experiments used an RTX 5060 Ti with 16 GB memory, cached frozen features, BF16, fused AdamW, and numerically verified acceleration. Checkpoints retained optimizer and random states; RDKit supported chemistry processing [29]. OpenAI Codex assisted with code development, figure preparation, and drafting; the authors retain responsibility for verification and final content. Reference code, environment records, and asset inventories support inspection; complete reproduction requires the encoder revisions, checkpoints, preprocessing, and data identified in the release inventory.

## Results

### Pooled fusion had the lowest validation error

Capacity-matched pooled fusion achieved the lowest mean validation MSE, 0.21528, and was selected as BINDRA. The lowest-error attention arm omitted FiLM and achieved 0.22461; it was retained as BINDRA-iBAM. Protein-only and ligand-only errors were 0.71282 and 0.62299, demonstrating the value of combining both inputs in this experiment. Table 1 and Figure 2 show all eight arms and their seed variation. Comparable parameter counts and tuning budgets support comparison of complete fusion architectures, including their optimization behavior and information paths.

**Table 1. Validation selection across controlled architecture arms.**

| Model | Prediction architecture | Fitted parameters | Mean MSE | Seed SD | Refit epochs |
|---|---|---|---|---|---|
| A | Protein only | 347843 | 0.71282 | 0.00081 | 28 |
| B | Ligand only | 348035 | 0.62299 | 0.00093 | 28 |
| C | Pooled concatenation | 695365 | 0.23409 | 0.00201 | 72 |
| D | Gated pooled fusion | 697287 | 0.23213 | 0.00383 | 72 |
| E | Attention with FiLM | 2451205 | 0.23490 | 0.00960 | 58 |
| F | Attention without FiLM | 2228293 | 0.22461 | 0.00306 | 69 |
| G | FiLM attention and readout gates | 2454664 | 0.23569 | 0.00683 | 54 |
| C-matched | Capacity-matched pooled fusion | 2455221 | 0.21528 | 0.00807 | 69 |

Three seeds per arm; validation is official fold 0. Parameters exclude the shared frozen encoders. MSE is in squared pKd units. Seed SD is the sample standard deviation of validation MSE, not a test confidence interval. C-matched selected the primary predictor; F was retained for iBAM studies.

### Warm prediction and detection-limit diagnostics

On the official warm test, BINDRA achieved MSE 0.19663 and CI 0.90802; BINDRA-iBAM achieved 0.19586 and 0.91253 (Table 2). The paired attention-minus-pooled MSE difference was -0.00077 (95% interval, -0.01464 to 0.01410), and the CI difference was 0.00451 (-0.00372 to 0.01422; Table 4). Both intervals included zero, and the validation-selected primary model was retained.

BINDRA's RMSE and MAE were 0.44343 and 0.23757 pKd, compared with 0.44256 and 0.22683 for BINDRA-iBAM. Individual-model variation and ensemble sampling intervals are shown separately, preserving the distinction between training reproducibility and uncertainty from the observed test sample.

**Table 2. Single-model variation and ensemble prediction uncertainty.**

| Model | Protocol | N | Single MSE mean (SD) | Ensemble MSE [95% interval] | Ensemble CI [95% interval] |
|---|---|---|---|---|---|
| BINDRA | warm | 5010 | 0.2156 (0.0041) | 0.1966 [0.1462, 0.2552] | 0.9080 [0.8904, 0.9237] |
| BINDRA-iBAM | warm | 5010 | 0.2268 (0.0037) | 0.1959 [0.1456, 0.2579] | 0.9125 [0.8957, 0.9274] |
| BINDRA | cold-drug | 2652 | 0.5252 (0.0280) | 0.5038 [0.2970, 0.7778] | 0.7148 [0.6139, 0.8026] |
| BINDRA-iBAM | cold-drug | 2652 | 0.6510 (0.0332) | 0.6360 [0.3430, 1.1367] | 0.7243 [0.6007, 0.8138] |
| BINDRA | cold-protein | 2992 | 0.3030 (0.0064) | 0.2792 [0.1897, 0.3756] | 0.8453 [0.8053, 0.8804] |
| BINDRA-iBAM | cold-protein | 2992 | 0.2629 (0.0169) | 0.2258 [0.1520, 0.3117] | 0.8719 [0.8347, 0.9044] |

Three models per row. SD describes seeds; bracketed intervals describe ensemble sampling uncertainty from 1000 crossed sequence/Murcko resamples. MSE uses squared pKd units. CI is unitless. Warm: 379 sequence/68 chemical groups; cold-drug: 379/6; cold-protein: 44/68. Few chemical groups limit cold-drug precision. Historical test exposure remains disclosed.

The pKd = 5 plateau comprised 3468/5010 observations, or 69.2%. BINDRA's MSE was 0.05498 on plateau observations and 0.51520 above it; the smaller above-plateau stratum contributed most of the total squared error. These strata have different label distributions and serve as diagnostics rather than matched difficulty estimates. Mean within-target CI was 0.88300 for BINDRA and 0.90308 for BINDRA-iBAM across 356 comparable sequences. Table 3 and Figure 3 retain both strata and entity-level errors. The plateau labels tie, so plateau-only CI is undefined. Full correlation, AP, r_m-squared, and per-target/per-ligand diagnostics accompany the numeric exports.

The attention model had slightly lower above-plateau error and slightly higher plateau error. Thus, similar aggregate MSEs summarized different behavior across the label range. Reporting the two strata preserves the standard benchmark while making its detection-limit composition visible.

**Table 3. Detection-plateau diagnostics and within-target ranking.**

| Model | Warm-test stratum | N | MSE | MAE | Mean within-target CI |
|---|---|---|---|---|---|
| BINDRA | all pairs | 5010 | 0.19663 | 0.23757 | 0.88300 |
| BINDRA | detection plateau | 3468 | 0.05498 | 0.11089 | Not reported |
| BINDRA | above plateau | 1542 | 0.51520 | 0.52248 | Not reported |
| BINDRA-iBAM | all pairs | 5010 | 0.19586 | 0.22683 | 0.90308 |
| BINDRA-iBAM | detection plateau | 3468 | 0.06462 | 0.09937 | Not reported |
| BINDRA-iBAM | above plateau | 1542 | 0.49103 | 0.51348 | Not reported |

Plateau is pKd = 5 (3468/5010, 69.2%); above-plateau is pKd > 5. Macro CI averages 356 target sequences with comparable pairs in the full warm test. Plateau labels all tie, so plateau-only CI is undefined. MSE is in squared pKd units and MAE in pKd units. Strata differ in label distribution and are diagnostic, not matched difficulty groups.

Both ensembles fell within a competitive range of author-reported Davis values. BINDRA-iBAM's MSE was lower than reported DeepDTA, GraphDTA-GIN, and MGraphDTA values; HGTDP-DTA, MDF-DTA, ProSmith, and CurvFlow-DTA reported lower MSE, and FusionDTA reported higher CI. Inputs, pretraining, aggregation, selection, and alternate splits qualify these descriptive comparisons; the full registry is in Additional file 1.

### Performance depended on the held-out entity

Cold-drug MSE was 0.50382 for BINDRA and 0.63600 for BINDRA-iBAM, with CIs of 0.71484 and 0.72429. The paired MSE difference was 0.13218 (-0.01051 to 0.39766), with precision limited by six chemical groups. Cold-protein performance favored BINDRA-iBAM: MSE 0.22579 versus 0.27922 and CI 0.87194 versus 0.84526. Its paired MSE interval was -0.11132 to -0.00297 and CI interval 0.00299 to 0.05159 (Table 4; Figure 4). These exploratory contrasts concern one partition with exact-sequence separation and no multiple-comparison correction.

Mean within-target CI was 0.66125 for BINDRA and 0.66306 for BINDRA-iBAM across 307 comparable cold-drug sequences. In cold-protein evaluation, the respective values were 0.86785 and 0.89303 across all 44 held-out sequences.

**Table 4. Paired architectural contrasts across Davis protocols.**

| Protocol | MSE difference | 95% interval | CI difference | 95% interval |
|---|---|---|---|---|
| warm | -0.00077 | [-0.01464, 0.01410] | 0.00451 | [-0.00372, 0.01422] |
| cold-drug | 0.13218 | [-0.01051, 0.39766] | 0.00945 | [-0.07325, 0.10623] |
| cold-protein | -0.05343 | [-0.11132, -0.00297] | 0.02668 | [0.00299, 0.05159] |

Differences are BINDRA-iBAM minus BINDRA: negative MSE and positive CI favor attention. Both models use identical sequence/Murcko resampling weights. CI contrasts are secondary analyses; MSE intervals reproduce the frozen exports. No correction for multiple comparisons was applied; cold results from one partition remain exploratory. Units are squared pKd for MSE and unitless for CI.

Cold-drug ligands had median nearest-training Morgan similarity 0.306. Cold-protein sequences had median nearest-training identity 0.533 (interquartile range, 0.356 to 0.662), with familiar chemistry. These values describe the evaluated supervised novelty, separately from protein-family and pretraining exposure.

### Residue and bond attention addressed different questions

At 4 angstrom, target-group mean residue AP across 30 complexes was 0.13114, compared with contact prevalence 0.07148, shuffled-attention AP 0.09311, and gradient-times-input AP 0.11911. Attention exceeded shuffled scores by 0.03803 (95% interval, 0.00655 to 0.07484); its difference from gradients was 0.01203 (-0.00614 to 0.03359; Table 5; Figure 5). These comparisons assess correspondence with contacts, using distinct information-weighting and prediction-sensitivity measures.

The attention-minus-prevalence increment was 0.05967 (0.03084 to 0.09449). Prevalence is the fraction of contact residues, while empirical shuffled rankings provide a finite-sample control; their AP values need not coincide. Absolute scores and paired comparisons therefore provide complementary descriptions of contact correspondence.

**Table 5. Absolute attention-contact correspondence and paired controls.**

| Metric | Mean | 95% target-group interval | Groups |
|---|---|---|---|
| Residue attention AP | 0.13114 | [0.10251, 0.16419] | 10 |
| Contact prevalence | 0.07148 | [0.06072, 0.08214] | 10 |
| Permutation AP | 0.09311 | [0.07930, 0.10823] | 10 |
| Gradient-times-input AP | 0.11911 | [0.09900, 0.14283] | 10 |
| Attention minus prevalence | 0.05967 | [0.03084, 0.09449] | 10 |
| Attention minus permutation | 0.03803 | [0.00655, 0.07484] | 10 |
| Attention minus gradient | 0.01203 | [-0.00614, 0.03359] | 10 |

Thirty complexes, ten equally weighted target groups, 4-angstrom heavy-atom contacts, 1000 target-group bootstrap resamples. Permutation AP is the per-complex mean over shuffled residue scores. Gradient-times-input scores traverse all protein branches. Paired control increments are secondary analyses of frozen maps. AP and prevalence are unitless; overlapping intervals are not a test of paired differences.

Residue maps were stable after atom reordering (mean Spearman agreement 0.98951; mean absolute prediction change 0.02153 pKd). Agreement fell to 0.12857 after progressive randomization through cross-attention and -0.00911 through the adapters. Randomizing only the downstream regression head left upstream attention unchanged by construction. Target groups, rather than the 90 member-complex entries, defined resampling units.

Across 27 unique ligands, median bond-map atom-order correlation was 0.92249. Full-matrix correlation remained 0.99801 after all eight transformer layers were randomized and 0.95174 after whole-encoder randomization. The corresponding supported off-diagonal correlations were 0.96192 and 0.23822, showing how shared matrix structure contributes to apparent stability. The supported-entry analysis followed inspection of full-matrix behavior and remains secondary. All members share one ligand encoder, so their ligand-only maps are not independent encoder replications. Complete controls are in Additional file 1.

### FAK1-P4N separates residue emphasis from ligand context

For FAK1-P4N (PDB 6YOJ; 1.361-angstrom resolution), BINDRA-iBAM predicted pKd 5.41667 (Figure 6). Ki and IC50 annotations remained assay-specific. The measured active/weak Kd pool contained seven compounds, below screening eligibility, so this case focused on mapped attention and crystal contacts.

P4N contacted 21 of 259 resolved residues within 4 angstrom. Residue AP was 0.17648, against prevalence 0.08108 and shuffled AP 0.09856. Three of the ten highest-ranked resolved residues were contacts. Cys502, the highest-ranked resolved residue, lay 2.87 angstrom from P4N; second-ranked Thr532 lay 15.95 angstrom away. The profile and heatmap therefore distinguished model emphasis from physical proximity (Figure 6A, B).

In the ligand-only map, 83.8% of physical-token attention linked bonds sharing an atom, including self-attention. Separately, physical bond queries directed a mean 29.5% of total outgoing attention to the two knowledge tokens omitted from the heatmap (Figure 6C). Protein-to-ligand attention emphasized methylsulfonyl C-S and selected aromatic C-C and C-N bonds (IDs 0, 35, and 12; Figure 6D). These views describe target-independent ligand context and protein-conditioned weighting, rather than interaction energies or causal contributions to affinity.

### Exploratory screening and design outcomes

The measured-negative panels contained 73, 25, and 38 compounds for galectin-3, beta-2 adrenergic receptor, and KEAP1. BINDRA AP was 0.533, 0.408, and 0.487, versus seed fingerprint AP 0.983, 0.920, and 0.720. Seed methods received an active ligand unavailable to target-only prediction, so the rankings reflect different information as well as different methods.

All six generator conditions achieved 100% raw validity and scaffold retention in five strict Kd queries across three series, but zero measured-analog recovery in 9000 draws. Generated docking covered one eligible query: 55/90 selected instances failed preparation; 35 prepared instances produced 105 poses, of which 91 passed geometry checks. This conditional pose pass rate excludes preparation failures. The results distinguish chemical validity, measured-analog recovery, and docking geometry. Endpoint-specific outcomes and complete failures remain in Additional files 1 through 4.

## Discussion

Pooled fusion had the lowest mean validation error across three seeds at a comparable trainable model size. Its warm-test error was close to that of the retained attention ensemble, with an uncertain paired difference. This supports a well-tuned pooled control when assessing the contribution of interaction attention. The cold-protein comparison also showed that the preferred architecture can depend on the deployment setting: BINDRA-iBAM performed better for withheld exact sequences under the evaluated partition, while the cold-drug comparison remained uncertain.

For an unfamiliar target, cold-protein performance and within-target ligand ranking may be more informative than a small warm-test error difference. When an active compound is already known, seed-based similarity provides an additional reference. These distinctions connect the evaluation protocol to the practical information available at prediction time.

Attention offered complementary views of model behavior. Residue scores corresponded to crystal contacts more closely than shuffled scores, while gradient attribution had similar contact correspondence within the reported uncertainty. Randomization distinguished dependence on trained interaction parameters from patterns retained by ligand representation and matrix structure. The FAK1-P4N example connected these controls to individual residues and bonds, including attention peaks near and far from the ligand. Because global features also reach the prediction through bypasses, attention maps represent part of the model's information use rather than a causal binding mechanism.

The study's scope is defined by Davis coverage, its detection plateau, pretraining exposure, one validation fold, and historical test inspection. Literature values provide context rather than matched external reruns, and classical affinity baselines were not evaluated. Replication across validation folds and independent affinity datasets would assess selection stability and broader generalization. Protein-feature gradients likewise do not cover every ligand/global contribution. The ten structural target groups and single-partition cold tests support conclusions under their declared sampling units.

Supporting studies separated computational outputs from design utility. A known active seed enabled stronger similarity rankings in the non-kinase panels, and valid scaffold-preserving generation did not recover measured analogs in the strict subset. Docking quantified preparation and geometry outcomes for one query. Broader target coverage, synthesis assessment, and prospective affinity measurements would establish the practical settings in which these components are useful.

## Conclusions

BINDRA's capacity-matched pooled architecture had the lowest mean validation error, while BINDRA-iBAM provided complementary attention analyses and better cold-protein performance in the exploratory comparison. Residue-contact and randomization controls distinguished structural correspondence from learned-parameter dependence. The results support strong pooled baselines, evaluation matched to the intended use, and joint structural and model-based assessment of attention. Independent replication would assess generality beyond the declared Davis protocols.

## List of abbreviations

### Key terminology

**Binding affinity (Kd and pKd).** Kd is the equilibrium dissociation constant; lower values indicate tighter binding. pKd is its negative base-ten logarithm in molar units, so higher values indicate tighter binding.

**Prediction metrics (MSE, CI, and AP).** MSE measures numerical error, so lower is better. CI measures correct ranking, so higher is better. AP summarizes how early positives appear in a ranking and depends on how common positives are.

**Training and evaluation.** An epoch is a pass through the training data; a seed controls random choices. Validation data select settings, and test data assess selected models.

**Warm and cold evaluation.** Warm tests allow familiar drugs and proteins in held-out pairs. Cold tests withhold drug identities or protein sequences from affinity training.

**Pretrained, frozen encoder.** An encoder converts an input into numerical features. Pretraining learns these features before affinity training; freezing keeps its weights fixed during that training.

**Pooled fusion and cross-attention.** Pooled fusion combines summarized protein and ligand features. Cross-attention lets each side weight information from the other before prediction.

**iBAM and bond map.** An iBAM shows attention between protein residues and ligand tokens. A bond map shows attention within the ligand before target conditioning. These complementary views describe the information emphasized by the model.

**Bootstrap interval.** Repeated resampling estimates uncertainty. Here, protein and chemical groups are resampled to account for related observations within the specified grouping.

**Scaffold-preserving diffusion.** The scaffold is the retained molecular core. A diffusion generator learns to reverse progressive graph corruption to propose molecular edits while preserving that core.

**Measured-analog recovery.** Recovery requires matching a held-out analog with an experimental measurement. Chemical validity or favorable predicted affinity alone does not establish recovery or experimentally confirmed binding.

### Abbreviations

AP, average precision; BF16, bfloat16 numerical format; CI, concordance index; DTA, drug-target affinity; FiLM, feature-wise linear modulation; iBAM, Interpretable Binding Attention Map; MAE, mean absolute error; MSE, mean squared error; PDB, Protein Data Bank; RMSE, root mean squared error; RMSD, root-mean-square deviation; SMILES, simplified molecular-input line-entry system.

## Declarations

### Ethics approval and consent to participate

Not applicable. This computational study used protein sequences, small-molecule affinity data, and public macromolecular structures and did not recruit human participants or perform animal experiments.

### Consent for publication

Not applicable. The manuscript does not contain identifiable individual participant data.

### Availability of data and materials

Additional files 1 through 5 provide tabular results, frozen prediction exports, every generated-draw record and rejection reason, and reviewed figures. The accompanying materials include an artifact inventory, environment records, mapped structural summaries, and a compact reference implementation. Full checkpoints, chemical structure files, and raw per-head attention matrices are retained in the versioned research package for separate deposit. Public access and redistribution terms for these remaining assets are pending; complete third-party reproduction requires their release. Reference code and the revision protocol are available in the BINDRA repository [30].

### Competing interests

The authors declare that they have no competing interests.

### Funding

This study was supported in part by the National Institutes of Health and National Heart, Lung, and Blood Institute under grant numbers U01HL134764, P01HL160476, R01HL131017, and R01HL149137.

### Authors' contributions

KS (Kevin Song) designed the study and models, implemented the software, and performed model training and evaluations. JZ (John Zhang) and LY (Lei Ye) reviewed the manuscript. JYZ (Jianyi Zhang) reviewed the manuscript and provided project guidance.

### Acknowledgments

No additional acknowledgments are applicable.

## Additional files

**Additional file 1. Supplementary methods and analyses. PDF with editable Word and LaTeX sources.** Detailed protocol and metric definitions, historical exposure, selected FAK1-P4N iBAM and bond-map views, control analyses, and complete supporting screening, generation, and docking methods and results.

**Additional file 2. Full result tables. ZIP containing CSV and JSON files.** Architecture selection, individual models, endpoint-specific generator metrics, all screening rankings summarized by target and method, structural panel results, docking failures and pose checks, literature qualifications, analysis scope, and paired comparisons.

**Additional file 3. Frozen Davis predictions and partitions. ZIP containing Parquet and JSON files.** Primary and companion predictions by protocol and member, official-order identifiers, model manifests, split assignments and exclusions, and nearest-training similarity audits. Parquet files can be read with pandas or Apache Arrow.

**Additional file 4. Complete generator draw records. ZIP containing CSV, Parquet, and JSON files.** All 90000 draws, including invalid outcomes and rejection reasons, query selection, endpoint strata, graph exclusions, and configuration identities.

**Additional file 5. Reviewed figure collection. ZIP containing PDF, SVG, 600-dpi PNG and TIFF, and JSON files.** The current figure collection, source metadata, and review records are identified separately from historical renderings. Large raw arrays, checkpoints, and SDF files are listed in the artifact inventory for repository deposit rather than represented as already public supplementary material.



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

## Figure legends

**Figure 1. Shared encoders and two affinity-prediction paths.** (A) Frozen ESM-C and LiGhT-mini encoders supply the same protein and ligand features; both encoder outputs branch to both architectures. (B) BINDRA summarizes these features separately before combining them for regression. (C) BINDRA-iBAM adds bidirectional cross-attention, with FiLM disabled. The dashed path carries global summaries directly to regression, bypassing attention. Each predictor averages pKd from three independently trained models. iBAMs use last-layer attention averaged across heads; member maps are aligned before averaging, separately from prediction averaging. Parameter counts and layer dimensions are given in Table 1 and Methods; ligand-only bond maps are not shown.

**Figure 2. Validation comparison of eight model architectures.** Bars show mean validation MSE across three training seeds; error bars show sample standard deviations between seeds. Each architecture received the same six-configuration tuning budget. Official folds 1 through 4 supplied 20036 fitting pairs, and fold 0 supplied 5010 validation pairs. Labels A through G and C-matched correspond to Table 1. Lower MSE is better; units are squared pKd. Error bars describe variation between training seeds on this validation fold, separately from test sampling uncertainty.

**Figure 3. Prediction errors for the pooled BINDRA ensemble.** All panels use the official 5010-pair Davis warm test. (A) Predicted versus benchmark pKd labels, with equal axis scales and a square plotting area. Hexagon color indicates pair count on a logarithmic scale; the dashed diagonal indicates perfect agreement. Benchmark labels include detection-limit observations. (B) Error distributions for the pKd = 5 plateau (3468 pairs) and above-plateau observations (1542 pairs), distinguished by color and line style. Error is prediction minus benchmark label; negative values indicate underprediction. Histograms are normalized separately to probability density. (C) Cumulative distributions of mean absolute error for 68 ligands and 379 protein sequences. Error and MAE are expressed in pKd units.

**Figure 4. Paired differences between attention and pooled prediction.** Points show BINDRA-iBAM minus BINDRA for (A) MSE, in squared pKd units, and (B) CI, which is unitless. Horizontal bars are 95% percentile intervals from 1000 crossed resamples of exact protein-sequence and Murcko-scaffold groups, using the same sampled observations for both models. Negative MSE differences and positive CI differences favor attention. Test sets contain 5010 warm, 2652 cold-drug, and 2992 cold-protein pairs. Cold-drug precision is limited by six chemical groups. Cold comparisons are exploratory; no correction for multiple comparisons was applied.

**Figure 5. Attention-contact correspondence and dependence on model parameters.** Analyses cover 30 complexes across ten equally weighted target groups. (A) Mean residue AP for ligand-to-protein attention, contact prevalence, shuffled-attention controls, and protein-feature gradient-times-input attribution. Contacts require a minimum heavy-atom distance of at most 4 angstrom. Bars are target-group means; error bars are 95% target-group bootstrap intervals. Prevalence is the fraction of contacting residues, rather than an empirical random-ranking AP. (B) Mean Spearman agreement between original and perturbed residue maps after atom reordering or progressive parameter randomization. Randomizing only the regression head leaves upstream attention maps unchanged by construction. These controls assess geometric correspondence and dependence on learned parameters.


**Figure 6. FAK1-P4N case study separates interaction attention from ligand context.** PDB 6YOJ has 1.361-angstrom resolution. (A) Ligand-to-protein attention averaged over 37 physical bond tokens across 282 modeled residues, with min-max scaling for display only. Orange points mark 4-angstrom contacts among 259 resolved residues; gray regions lack crystal coordinates. Cys502 and Thr532 use PDB author numbering. (B) Raw ligand-to-protein iBAM, with 37 bond-query rows and 282 residue-key columns; last-layer, mean-head maps are aligned and averaged across three models. (C) Ligand-only self-attention from the shared frozen encoder; rows receive information from source-token columns. (D) Bonds colored by mean protein-to-ligand attention over all 282 modeled residues; callouts identify exported bond indices. Panels B-D show unitless, unrenormalized weights with separate color scales. Nonphysical knowledge tokens are excluded from physical displays and their mass is exported. These maps describe how the model distributes attention across the protein and ligand.
