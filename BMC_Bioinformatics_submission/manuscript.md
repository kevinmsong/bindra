# BINDRA: evaluating pooled fusion and interaction attention for protein-ligand affinity prediction

Kevin Song, John Zhang, Lei Ye, and Jianyi Zhang

Department of Biomedical Engineering, University of Alabama at Birmingham, Birmingham, Alabama, USA

Corresponding author: Jianyi Zhang, jayzhang@uab.edu

Research article draft prepared for BMC Bioinformatics. Author contributions, funder involvement, public repository identifiers, and final author approval require confirmation before submission.

## Abstract

### Background

Features learned from protein sequences and molecular structures can support binding affinity prediction, but whether interaction attention improves accuracy or provides reliable explanations remains unclear. We developed BINDRA, Binding INference Design Retrieval and Attention, to compare pooled fusion, which combines separately summarized protein and ligand features, with attention-based prediction using the same fixed feature encoders and comparable numbers of trainable parameters.

### Results

Validation on one fixed fold across three training seeds selected pooled fusion as the primary predictor. The attention model with the lowest validation error was retained as BINDRA-iBAM. On the official 5,010-pair Davis test, the pooled and attention ensembles achieved mean squared errors (MSEs) of 0.19663 and 0.19586 and concordance indices of 0.90802 and 0.91253, respectively. Their MSE difference was small and uncertain: -0.00077 for attention minus pooled fusion (95% bootstrap interval, -0.01464 to 0.01410). In exploratory cold-start comparisons, attention performed better when test proteins were excluded from affinity training; the difference when test ligands were excluded remained uncertain. Detection-limit labels accounted for 69.2% of the warm test and were examined separately. Across 30 crystal complexes from ten target groups, attention ranked contacting residues with an average precision of 0.131, compared with contact prevalence of 0.071, shuffled-attention controls of 0.093, and gradient-based attribution of 0.119. Paired comparisons supported improvement over shuffled controls but did not establish an advantage over gradient attribution. Parameter randomization and atom-order tests revealed limitations of attention interpretation. Supporting diffusion experiments recovered none of the known measured analogs among 9,000 draws across five queries and three chemical series.

### Conclusions

Pooled fusion provided competitive affinity predictions, while the benefit of interaction attention depended on the evaluation setting. Attention showed modest correspondence with crystal contacts, but this did not establish a causal explanation of binding. Previous Davis test inspection limits confirmatory claims and conclusions about generalization.

**Keywords:** drug-target affinity; pooled fusion; cross-attention; generalization; interpretability

## Background

Predicting how strongly a molecule binds a protein is a central task in computational drug discovery. Affinity models can rank candidates before experimental testing, but their practical value depends on the evaluation setting. A model that predicts unseen combinations of familiar molecules and proteins may behave differently when the molecule or protein is absent from supervised training. The Davis kinase-inhibitor dataset provides a widely used quantitative benchmark, but its limited chemical diversity, repeated protein constructs, and detection-limit measurements require careful interpretation [1].

DeepDTA established a sequence-based neural benchmark using molecular strings and protein sequences [2]. WideDTA added molecular and protein subsequence information [3], while GraphDTA and DGraphDTA introduced molecular graphs and protein contact representations [4, 5]. MGraphDTA developed multiscale molecular representations with visualization analyses [6]. Subsequent approaches have used attention, protein language models, knowledge distillation, semisupervised learning, and hybrid graphs [7, 8, 9]. Recent preprints include graph-transformer and cross-attention methods and curvature-based processing of molecular and predicted protein contact graphs [10, 11, 12]. These methods differ in external resources and evaluation protocols. A lower reported error on a dataset with the same name does not by itself establish a matched comparison.

Pretrained encoders convert protein sequences and molecular structures into numerical features for affinity prediction. ESM Cambrian learns features for each protein residue, or amino acid, using its surrounding sequence [13, 14]. The Line Graph Transformer in KPGT represents molecular connectivity using bond-centered tokens and global chemical information [15]. Contrastive protein-ligand methods also illustrate the value of pretrained representations for interaction prediction [16]. A predictor can also combine separately summarized protein and ligand features. Comparing attention with this simpler pooled approach, at similar model size, helps establish what the interaction module adds.

Attention maps provide accessible summaries of model computation, but an emphasized residue is not necessarily a causal determinant of binding. Attention can depend on molecular connectivity, how inputs are divided into tokens, how features are summarized, or alternate paths to the prediction. Previous work has shown that attractive attribution maps can persist after parameter randomization and that attention weights do not automatically constitute explanations [17, 18]. Crystal structures provide an independent geometric reference for contact agreement, but agreement with a structure and sensitivity of a prediction are distinct questions.

Molecular generation requires additional evidence. A valid chemical graph can be a duplicate, a seed copy, an irrelevant edit, or a molecule without useful affinity. Joint prediction-generation frameworks already exist, including DeepDTAGen [19]; combining these tasks alone is not a novelty claim. Denoising diffusion provides a flexible framework for generative modeling, including discrete molecular edits [20, 21]. Evaluating an independently trained generator against conditioning controls can test whether target information adds value beyond the seed and scaffold constraints.

We designed BINDRA to address two primary questions: does explicit protein-ligand attention improve prediction over pooled fusion with comparable numbers of trainable prediction parameters, and how do its maps correspond to crystal contacts and respond to representation and parameter controls? Warm tests hold out pairs while allowing familiar molecules and proteins; cold tests withhold drug identities or protein sequences from affinity training. Retrospective screening and independent molecular editing provide supporting tests of practical utility, including negative results.

## Methods

### Study design and model roles

Primary model selection used mean validation MSE across seeds 20261001, 20261002, and 20261003. The selected pooled model is called BINDRA. The attention architecture with the lowest mean validation MSE is retained as BINDRA-iBAM, the Interpretable Binding Attention Map companion. Each ensemble takes the arithmetic mean of pKd predictions from its three independently trained members; individual-member performance is reported separately. The models' internal features are not averaged because each model learns its own feature space. Differences between model predictions are reported descriptively and do not provide calibrated prediction intervals.

Davis test results were inspected during earlier model development and exploratory architecture comparisons. Although the present models were selected using validation data, the test set is not historically untouched. The Davis results therefore provide retrospective architectural evidence rather than an independent confirmatory benchmark. No affinity weights were updated for screening, attention assessment, generation, or docking. Development chronology and numerical reproduction details are provided in Additional file 1.

### Davis data and evaluation partitions

We used the official DeepDTA Davis data and fold files [2]. The benchmark contains 30056 observations from 68 ligand identifiers and 442 target identifiers. The latter correspond to 379 unique protein sequences in the released data. Labels were transformed to pKd as minus the base-10 logarithm of Kd in molar units. The published pKd = 5 detection plateau was retained in the primary analysis and examined separately in error diagnostics.

The official warm test contains 5010 pairs, and the training pool contains 25046 pairs. For the controlled experiments, official fold 0 supplied 5010 validation pairs and folds 1 through 4 supplied 20036 fitting pairs. This was three-seed validation replication on one fixed validation fold, not five-fold cross-validation. Final warm models used all five training folds for a fixed duration derived from validation. Exported test rows were sorted by official pair index; their membership, labels, and indexing were checked against the released fold files.

Custom cold-drug and cold-protein assignments targeted 70%, 10%, 10%, and 10% fitting, validation, calibration, and test fractions using label-independent group allocation. Cold-drug separation used normalized chemical identities; it did not require disjoint scaffolds. It retained 20332 fitting, 3094 validation, 3094 calibration, and 2652 test pairs, with 884 quarantined rows retained in an exclusion table. Cold-protein separation used exact protein sequences and retained 21080, 2992, 2992, and 2992 pairs, respectively. These protocols used their own trained ensembles. Nearest-training Morgan similarity and sequence identity were exported to describe the degree of novelty [22]. Exact sequence separation does not establish target-family exclusion, and supervised chemical exclusion does not imply absence from pretraining. These results do not include scaffold or cold-both protocols.

### Fixed feature encoders and prediction architectures

The protein encoder was the original pretrained ESM-C 600M model, frozen throughout the controlled affinity comparisons. It produced a 1152-dimensional representation for each residue. Sequences exceeding 2046 residues were encoded in windows with 256-residue overlap; shared residue vectors were averaged. Padding and special tokens were excluded from residue pooling. Cached upstream representations were bound to encoder and preprocessing identities.

The ligand encoder was a local LiGhT-mini implementation inspired by the bond-token representation of KPGT [15]. Each bond was represented as an atom-bond-atom triplet, with isolated atoms retained. Fingerprint and descriptor tokens supplied global chemical information. The encoder used width 96, eight transformer layers, and eight attention heads. It was trained from scratch for five completed pretraining epochs on 649364 eligible BindingDB molecules, then frozen for the controlled affinity experiments [23]. The exposure audit used the eligible pretraining SMILES, rather than the complete BindingDB download. Chemical identity, normalized-parent, and scaffold keys distinguished exact exposure from related chemistry; failed normalization remained unknown. Supervised cold-drug separation therefore does not establish chemical novelty relative to pretraining. The protein encoder's complete pretraining exposure could not be reconstructed. These exposure limitations apply to both prediction architectures.

Each model used trainable adapters to rescale the encoder features and project them to 192 dimensions. Eight prediction architectures tested protein-only prediction (A), ligand-only prediction (B), pooled concatenation (C), gated pooled fusion (D), the original interaction fusion with feature-wise linear modulation (FiLM; E), the same bidirectional attention with FiLM disabled (F), grouped readout gates added to E (G), and a concatenation control with a comparable number of trainable parameters (C-matched). G retained the existing readouts and applied pair-dependent scalar gates to interaction, ligand-global, and protein-only groups. Gates were initialized to preserve their inputs, allowing the effect of learned gating to be assessed without initially removing information paths.

The selected BINDRA model pooled protein and ligand features and retained three summaries of the whole ligand (global readouts), producing five 192-dimensional feature vectors. Its regression head mapped 960 to 1648 to 256 to one pKd value. It had 2455221 trainable prediction parameters per member. The BINDRA-iBAM model used two bidirectional cross-attention layers, eight heads, feed-forward width 576, and no FiLM. Its six readouts comprised fused protein and ligand features, three global ligand features, and a protein summary computed before cross-attention. The global ligand features and pre-attention protein summary reached the regression head through bypass paths. Its head mapped 1152 to 256 to one value, with 2228293 fitted parameters per member. Figure 1 shows the controlled prediction paths and shared encoder inputs.

### Tuning and validation selection

Each neural arm received the same six-configuration grid: learning rates 0.0001, 0.0003, and 0.001 crossed with dropout 0.1 and 0.2. Training used AdamW with weight decay 0.01 and 128 pairs per optimizer update. Training lasted at most 200 epochs and stopped after 15 epochs without improvement in validation MSE. Checkpoints were selected by validation MSE. The selected configuration for each architecture was evaluated across the three seeds, including the corresponding first-seed tuning run. For each architecture, the median epoch of minimum validation MSE across seeds, rounded up, determined the duration of retraining on the full warm-training pool.

BINDRA and BINDRA-iBAM both selected learning rate 0.0003 and dropout 0.1 and used 69-epoch final warm refits. BINDRA's fitted parameter count was within 1% of E and G, while the compact C control had fewer parameters. Thus, comparing C with C-matched also tests model size; it does not isolate the effect of how features are combined. Final model selection did not use the relative test point estimates. No additional validation folds or local external-model baseline comparisons are reported.

### Prediction metrics and statistical analysis

Validation MSE was the primary selection criterion. Test metrics included MSE, root mean squared error (RMSE), mean absolute error (MAE), concordance index (CI), Pearson and Spearman correlations, and mean within-target CI. CI compares pairs with unequal measured labels, assigning a score of one to correct ordering and one half to tied predictions. MSE has squared pKd units, while RMSE and MAE have pKd units. Within-target CI excludes targets without comparable observations and is not interchangeable with a pooled CI. Pooled CI includes comparisons across different targets, whereas within-target CI asks whether the model orders ligands for the same sequence. Strong pooled ranking can conceal weaker ligand prioritization for individual targets. We therefore report the unweighted mean within-target CI together with the number of target sequences containing comparable pairs.

We exported two documented r_m-squared conventions, including the DeepDTA implementation and the conventional expression, without treating them as identical. Classification diagnostics defined actives as pKd >= 7 and reported average precision (AP) separately from trapezoidal precision-recall area. AP summarizes precision as recall increases; it is influenced by active prevalence. The exact integration and threshold implementation accompany the numeric exports.

Individual-model metrics and the standard deviation across seeds describe variation between training runs. To account for repeated proteins and related chemicals, we used 1000 crossed bootstrap resamples over exact protein sequences and Murcko scaffold groups. Both models used the same resampled observations for paired comparisons [24]. These groups do not fully capture protein homology or chemical dependence. Structural intervals resampled target groups, and generator intervals resampled chemical series. Query-weighted seed means and equally weighted series estimates were labeled separately. We did not test statistical superiority against published scalar metrics or interpret 90000 generated draws as 90000 independent biological observations.

We did not report conformal prediction intervals. Full-pool warm refits lacked independent calibration observations, and normalized-pair replication prevented sufficient calibration isolation in the cold protocols. Ensemble disagreement was reported descriptively and should not be interpreted as calibrated per-compound confidence.

### Literature comparison

The source-linked Davis registry covers 2018 through October 17, 2026 and separates author-reported metrics by split, inputs, pretraining, checkpoint selection, publication status, and aggregation. All 19 reviewed model variants are retained in the supplementary comparison table and machine-readable registry. Literature values provide context rather than controlled reruns or a universal ranking. External DTA models and classical affinity baselines were not locally rerun for the reported Davis study; the controlled comparisons concern BINDRA's downstream architecture arms.

### Supporting screening, generation, and docking

Three coverage-selected measured-negative screening panels required at least ten actives at pKd >= 7 and ten weak compounds at pKd <= 5. Frozen ensembles scored every candidate. Comparators included seed fingerprint similarity, property distance, random ranking, and seed-assisted percentile-rank averaging. Seeds and duplicate identities were excluded. Measured negatives were separated from property-matched presumed decoys, without assigning fabricated affinities to unlabeled molecules.

An independent scaffold-preserving graph-diffusion generator was trained under six matched conditioning conditions with three seeds each. The structured condition used the seed graph, target, endpoint, member-specific features, and mapped iBAM context. Control conditions removed or altered these inputs as specified in Additional file 1. Product-derived information was excluded from conditioning. Connected chemical series remained within partitions. Fifty prespecified held-out queries received 100 draws per condition and seed. Recovery of measured analogs for exact Davis constructs with Kd data was the primary generator outcome; Ki and IC50 strata were analyzed separately. No repair was applied, and affinity predictions were excluded from generator optimization. Bounded Vina docking used native redocking gates and hash-selected generated compounds. Full training, query, preparation, and docking methods are in Additional file 1 [20, 21, 25, 26, 27].

### Assessment of attention-contact correspondence

We retained a fixed panel of up to three complexes per target group, selected by structural quality and coverage rather than attention agreement. The final panel contained 30 noncovalent small-molecule complexes across ten target groups, each with resolution at most 2.5 angstrom and usable chemical and sequence mappings. Experimental structures were obtained from the Protein Data Bank [28]. Mapping exports retained construct-to-author residue numbering, insertion codes, missing residues, mutations, alternate conformations, and ligand symmetry. FAK1-P4N in PDB 6YOJ was the detailed structural example [29].

For each BINDRA-iBAM member, iBAMs used the last cross-attention layer averaged across heads. We aligned residue and atom coordinates before averaging member maps. Both attention directions, individual heads, and nonphysical-token mass were retained. Ligand-to-protein attention defined residue salience. Physical-contact analyses excluded knowledge and padding tokens and used minimum heavy-atom distance <= 4 angstrom, with 3.5- and 5-angstrom sensitivity analyses. We reported residue and residue-atom AP, top-residue enrichment, and informative ligand-atom agreement alongside uniform and permutation controls. Raw weights supported analysis; any display normalization was labeled.

Controls examined agreement across attention heads and ensemble members, atom-order stability, prediction changes after target or ligand swaps, progressive parameter randomization, gradient-times-input attribution of protein features, and 10% residue-feature masking with ten matched random controls. Gradient calculations traversed all protein-using interaction and bypass paths but did not provide complete attribution of ligand and global inputs. Feature masking may be outside the training distribution. The present controls do not include integrated gradients, multiple masking fractions, or coordinate-aligned explanation-map swaps.

Ligand bond maps were evaluated separately because they describe self-attention in the shared frozen ligand encoder before target conditioning. Knowledge-token mass, topology controls, atom-order stability, parameter randomization, and mapped analog edits were exported. Repeated member entries used the same ligand weights and were not independent ligand-encoder replicates. Contact agreement and model sensitivity were treated as distinct evidence rather than proof of a causal binding mechanism.

### Implementation and reproducibility

Experiments used sequential GPU jobs on an NVIDIA RTX 5060 Ti with 16 GB memory. Frozen upstream features were cached with encoder, preprocessing, and schema identities. BF16, fused AdamW, length bucketing, graph trimming, bounded caches, and numerically checked acceleration reduced repeated computation. Checkpoints included optimizer and random states. Software versions, split assignments, failures, numeric maps, source checksums, and resumption contracts accompany the supplementary records. RDKit supported chemical processing [30].

An OpenAI coding assistant was used for implementation, checking, figure preparation, and drafting. It was not assigned authorship. The authors retain responsibility for scientific verification and final approval. The public reference implementation supplies the downstream architectures and validation-only training utilities. The reported prediction evidence is restricted to the completed Davis experiments; separate unfinished benchmark work is excluded. Full encoder and data assets require separate repository identifiers and redistribution terms.

## Results

### Pooled fusion had the lowest mean validation error

The protein-only and ligand-only arms had mean validation MSEs of 0.71282 and 0.62299, respectively. Combining protein and ligand representations substantially reduced error within this experiment. Compact concatenation achieved 0.23409, gated pooled fusion 0.23213, and the original FiLM attention arm 0.23490. Removing FiLM improved the attention arm's point estimate to 0.22461. Adding grouped gates to the FiLM arm yielded 0.23569. The capacity-matched pooled model had the lowest mean validation MSE, 0.21528, and was selected as BINDRA (Table 1; Figure 2).

BINDRA had 2455221 fitted parameters, compared with 2451205 for E and 2454664 for G. The original interaction architecture therefore had a comparable parameter budget. BINDRA was larger than compact concatenation, so their difference reflects both capacity and fusion design. The no-FiLM attention model was retained as BINDRA-iBAM because it had the lowest validation error among the attention arms. Seed variation was appreciable relative to differences among the strongest arms. Parameter matching concerns the trainable prediction layers, not the much larger shared frozen encoders. The architectures also differed in optimization behavior and access to interaction features. The common search budget and comparable parameter counts improve fairness, but the comparison assesses complete architectures rather than isolating the causal contribution of an attention block.

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

### Warm-test differences were small relative to their uncertainty

On the official warm test, BINDRA achieved MSE 0.19663, CI 0.90802, RMSE 0.44343, and MAE 0.23757. BINDRA-iBAM achieved MSE 0.19586, CI 0.91253, RMSE 0.44256, and MAE 0.22683. Table 2 distinguishes single-model variation from three-model ensemble performance and sampling intervals. The paired attention-minus-pooled MSE difference was -0.00077, with a 95% interval of -0.01464 to 0.01410. The corresponding CI difference was 0.00451, with interval -0.00372 to 0.01422 (Table 4). Neither warm-test contrast established a clear advantage; the validation-selected primary model was retained.

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

The detection plateau accounted for 3468 of 5010 observations, or 69.2%. BINDRA's MSE was 0.05498 on plateau observations and 0.51520 above the plateau. These strata have different label distributions, so their errors are diagnostic rather than matched difficulty estimates. BINDRA-iBAM's stratum errors and both models' within-target ranking are shown in Table 3. Mean within-target CI was 0.88300 for BINDRA and 0.90308 for BINDRA-iBAM across 356 comparable target sequences. Plateau-only within-target CI is undefined because the labels tie. Predicting the plateau can reduce global squared error without resolving differences among weak compounds. Above-plateau observations account for most of the pooled model's total squared error despite being the smaller stratum. The attention companion has slightly lower above-plateau MSE and slightly higher plateau MSE, illustrating how similar aggregate errors can arise from different behavior. These observations motivate reporting both strata, not removing plateau observations from the canonical benchmark. Per-target, per-ligand, correlation, AP, and r_m-squared diagnostics remain available in the machine-readable exports (Figure 3).

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

Author-reported values place both ensembles in a competitive Davis range. BINDRA-iBAM's MSE point estimate was lower than reported DeepDTA, GraphDTA-GIN, and MGraphDTA values; HGTDP-DTA, MDF-DTA, ProSmith, and CurvFlow-DTA reported lower MSE, and FusionDTA reported higher CI. Additional inputs, split differences, pretraining, averaging, and selection policies qualify these comparisons. NHGNN-DTA and SSM-DTA used alternate random splits. No significance or universal superiority claim is made from cross-paper scalar metrics.

### Performance on held-out drugs and proteins depended on architecture

Cold-drug MSE was 0.50382 for BINDRA and 0.63600 for BINDRA-iBAM; their CIs were 0.71484 and 0.72429. The paired MSE difference of 0.13218 had a wide interval, -0.01051 to 0.39766. Only six Murcko groups represented the cold-drug test chemistry, limiting precision. Mean within-target CI was 0.66125 for BINDRA and 0.66306 for BINDRA-iBAM across 307 comparable sequences. Thus, the pooled error point estimate and attention ranking point estimate did not establish a reliable overall advantage.

For cold-protein evaluation, BINDRA-iBAM had lower MSE (0.22579 versus 0.27922) and higher CI (0.87194 versus 0.84526). The paired MSE interval was -0.11132 to -0.00297; the paired CI interval was 0.00299 to 0.05159 (Table 4; Figure 4). Mean within-target CI was 0.86785 for BINDRA and 0.89303 for BINDRA-iBAM across all 44 held-out sequences. Both intervals excluded zero under the declared resampling groups. These paired intervals directly estimate model differences using the same observations. They account for dependence within the declared sequence and chemical groups, but not architecture-search history, dataset reuse, or alternative splits. Exact sequence separation also does not imply separation of protein families. The comparison remains exploratory because it uses one partition without correction for multiple comparisons; it supports a protocol-specific difference rather than general superiority of either fusion approach.

**Table 4. Paired architectural contrasts across Davis protocols.**

| Protocol | MSE difference | 95% interval | CI difference | 95% interval |
|---|---|---|---|---|
| warm | -0.00077 | [-0.01464, 0.01410] | 0.00451 | [-0.00372, 0.01422] |
| cold-drug | 0.13218 | [-0.01051, 0.39766] | 0.00945 | [-0.07325, 0.10623] |
| cold-protein | -0.05343 | [-0.11132, -0.00297] | 0.02668 | [0.00299, 0.05159] |

Differences are BINDRA-iBAM minus BINDRA: negative MSE and positive CI favor attention. Both models use identical sequence/Murcko resampling weights. CI contrasts are secondary analyses; MSE intervals reproduce the frozen exports. No correction for multiple comparisons was applied; cold results from one partition remain exploratory. Units are squared pKd for MSE and unitless for CI.

Cold-drug ligands had median nearest-training Morgan similarity 0.306, with familiar target sequences. Cold-protein sequences had median nearest-training identity 0.533, with interquartile range 0.356 to 0.662, and familiar chemicals. These values describe novelty rather than family exclusion. Calibration was withheld, and raw member disagreement does not supply calibrated coverage.

### Attention showed modest agreement with crystal contacts

All 30 complexes across ten target groups met the structural and mapping criteria. At 4 angstrom, equally weighted target-group mean residue AP was 0.13114, compared with contact prevalence 0.07148, permutation AP 0.09311, and gradient-times-input AP 0.11911 (Table 5; Figure 5). Attention exceeded prevalence by 0.05967, with a 95% target-group interval of 0.03084 to 0.09449. Its increment over permutation was 0.03803 (0.00655 to 0.07484), while the increment over gradients was 0.01203 (-0.00614 to 0.03359). These secondary paired contrasts were computed from frozen maps and did not establish an advantage over gradient attribution. Contact prevalence provides a simple reference, but finite random rankings need not have AP equal to prevalence, as the empirical permutation result illustrates. The gradient comparator asks whether local feature sensitivity identifies contact residues, whereas attention measures normalized information weighting. Neither is trained to recover a crystal contact map. Their absolute AP and paired differences therefore measure contact correspondence, not validated detection of physical interactions.

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

Mean residue-map Spearman agreement after atom reordering was 0.98951, and mean absolute prediction change was 0.02153 pKd. Progressive randomization through cross-attention reduced agreement to 0.12857; randomization through the adapters reduced it to -0.00911. Randomizing only the downstream regression head left attention unchanged by construction, despite altering predictions. Member-level controls were averaged within complex and target; they are not 90 independent biological replicates.

Bond maps showed a different sensitivity profile. Across 27 unique ligands, median full-matrix atom-order correlation was 0.92249. Median correlation remained 0.99801 after all eight transformer layers were randomized and 0.95174 after whole-encoder randomization. In the secondary supported-nondiagonal analysis, the corresponding latter values were 0.96192 and 0.23822. The difference between full-matrix and supported-entry results shows how shared matrix structure can inflate apparent stability. High full-matrix correlation alone does not establish a learned chemical explanation. All affinity members share one frozen ligand encoder, so repeated member maps are not independent encoder replications.

FAK1-P4N in 6YOJ provides the detailed 1.361-angstrom example. BINDRA-iBAM predicted pKd 5.41667 for this modeled complex; available Ki and IC50 annotations were retained separately and were not used as equivalent pKd labels. The measured active/weak Kd pool contained seven compounds, below the primary screening coverage rule. Supplementary residue and bond maps illustrate this exploratory case without replacing unfavorable panel results. Physical-contact correspondence, parameter dependence, and prediction sensitivity answer distinct questions. A map can remain stable when a downstream head changes because that head is absent from the map computation. Conversely, changing an upstream adapter can alter both maps and predictions without demonstrating that the highlighted residues cause binding. The supported-edge bond analysis was added after inspecting full-matrix behavior and remains explicitly secondary. Its result must be interpreted alongside the full-matrix analysis because the two analyses evaluate different sets of entries.

### Supporting studies did not establish screening or design superiority

The measured-negative panels contained 73, 25, and 38 compounds for galectin-3, beta-2 adrenergic receptor, and KEAP1. BINDRA AP was 0.533, 0.408, and 0.487, compared with seed fingerprint AP of 0.983, 0.920, and 0.720. Target-only and seed-informed methods had different inputs; neither these results nor the companion rankings established general screening superiority.

Eighteen generator fits produced 90000 draws across six conditions and three seeds. In the five-query strict Kd subset spanning three series, all conditions achieved 100% raw validity and scaffold retention, but none recovered a measured analog in 9000 draws. Across all queries, raw validity under structured conditioning was 88.37%. These results show that the sampling policy often satisfied chemical constraints, but did not establish useful target-conditioned design.

Generated docking covered one qualifying query. Of 90 hash-selected instances, 55 failed preparation; 35 produced 105 search poses, of which 91 passed overall geometry checks. Preparation succeeded for 38.9% of selected instances; 86.7% of the resulting poses passed geometry checks. The pose pass rate is conditional on preparation and does not describe success across all selected compounds. Full generation, preparation failures, docking, and retrieval results are supplementary.

## Discussion

The primary architectural result was that a pooled model with a comparable number of trainable parameters had lower mean validation error than the tested attention architectures across three seeds. Both the pooled and no-FiLM attention ensembles achieved competitive Davis warm-test performance, but their small test MSE difference was uncertain. This finding argues for evaluating interaction modules against strong pooled controls before attributing predictive gains to attention. It does not establish that attention is generally unnecessary: BINDRA-iBAM performed differently across cold protocols and retained a useful interface for structural attention analysis.

The distinction between predictive performance and interpretability was central. Residue attention showed panel-level agreement with crystal contacts, yet individual cases, atom-pair analyses, atom ordering, and randomization controls limited the strength of interpretation. Attention to a contacting residue does not establish that the residue determines the prediction. Global features can also influence affinity estimates through paths that are absent from an attention map. The retained iBAM companion therefore supports questions about model emphasis and structural correspondence, rather than a definitive account of molecular binding mechanisms.

The screening results also constrained the platform's practical claims. Target-only rankings did not consistently outperform simple seed-based controls, and the available panels extended beyond the kinase-focused training domain. These retrospective results support reporting complete rankings and exposure audits, but do not establish general utility for compound screening. Broader screening utility will require larger independent panels with appropriate chemical and target diversity and measured negatives.

The generation study demonstrated a similar separation between valid outputs and useful conditional design. Scaffold constraints and valence-aware sampling produced high validity, but measured-analog recovery remained zero in every strict-Kd condition. Low denoising loss and novelty relative to the recorded training set did not establish measured-analog recovery. The controls did not establish an incremental benefit of structured iBAM context. Further design studies require broader experimentally measured edit data and evidence that outputs depend usefully on the target.

Several limitations affect interpretation. Davis includes only 68 ligand identifiers, a large detection plateau, and repeated exact protein sequences under different target identifiers. Warm, cold-drug, and cold-protein evaluations therefore answer different questions. Cold-protein separation is not family exclusion, and ligand pretraining changes what chemically unseen means. Broad pretraining, three-model averaging, varying literature inputs, and historical test exposure prevent a universal state-of-the-art claim. The affinity comparisons cover one dataset, and architecture selection used three seeds on one fixed validation fold. Selection stability across other folds and performance on an independent affinity dataset were not established. External-model values are author-reported rather than controlled local reruns, and simple classical affinity baselines were not evaluated. These limitations restrict the conclusions to the declared Davis protocols.

The supporting studies have additional boundaries. The structural panel contains ten target groups used as resampling units, the strict generation subset contains three chemical series, and generated docking covers one eligible query/receptor after native gates and preparation failures. Crystal structures can reflect particular constructs and conformational states. Protein-feature gradients do not provide complete attribution of all global and ligand branches. The study did not include expanded attribution controls, an independent affinity dataset, prospective binding assays, synthetic feasibility assessment, or experimental analog testing.

The contribution is a controlled evaluation of predictive fusion and the limits of interaction attention. With the same pretrained representations, the pooled control had the lowest mean validation error, while the retained attention model showed different behavior across warm and cold evaluations. These results support using a well-tuned pooled baseline, selecting evaluation protocols that reflect the intended application, and assessing attention against both structural and model-based controls. For new targets, cold-protein performance and within-target ranking may be more relevant than a small warm-test MSE difference. For compound prioritization, cold-drug performance and seed-based similarity provide complementary reference points. Independent replication is needed to determine whether these findings extend beyond Davis.

## Conclusions

BINDRA's capacity-matched pooled architecture had the lowest mean validation error across three training seeds. Its attention companion showed similar warm-test error and better cold-protein performance in the declared exploratory comparison. Residue attention exceeded contact prevalence and permutation controls but was not clearly superior to gradient attribution. Bond-map randomization exposed substantial representation dependence, and supporting generation did not establish measured recovery or useful target-conditioned design. These findings support testing interaction modules against strong pooled controls while keeping predictive accuracy, structural correspondence, and causal explanation distinct. Broader generalization remains unestablished.

## Glossary of key terms

**Binding affinity (Kd and pKd).** Kd is the equilibrium dissociation constant; lower values indicate tighter binding. pKd is its negative base-ten logarithm in molar units, so higher values indicate tighter binding.

**Prediction metrics (MSE, CI, and AP).** MSE measures numerical error, so lower is better. CI measures correct ranking, so higher is better. AP summarizes how early positives appear in a ranking and depends on how common positives are.

**Training and evaluation.** An epoch is a pass through the training data; a seed controls random choices. Validation data select settings, and test data assess selected models. Earlier Davis test inspection is disclosed separately.

**Warm and cold evaluation.** Warm tests allow familiar drugs and proteins in held-out pairs. Cold tests withhold drug identities or protein sequences from affinity training.

**Pretrained, frozen encoder.** An encoder converts an input into numerical features. Pretraining learns these features before affinity training; freezing keeps its weights fixed during that training.

**Pooled fusion and cross-attention.** Pooled fusion combines summarized protein and ligand features. Cross-attention lets each side weight information from the other before prediction.

**iBAM and bond map.** An iBAM shows attention between protein residues and ligand tokens. A bond map shows attention within the ligand before target conditioning. Neither alone proves a binding mechanism.

**Bootstrap interval.** Repeated resampling estimates uncertainty. Our resampling retains protein and chemical groups to account partly for related observations; it does not correct historical test inspection.

**Scaffold-preserving diffusion.** The scaffold is the retained molecular core. A diffusion generator learns to reverse progressive graph corruption to propose molecular edits while preserving that core.

**Measured-analog recovery.** Recovery requires matching a held-out analog with an experimental measurement. Chemical validity or favorable predicted affinity alone does not establish recovery or experimentally confirmed binding.

## List of abbreviations

AP, average precision; BF16, bfloat16 numerical format; CI, concordance index; DTA, drug-target affinity; FiLM, feature-wise linear modulation; iBAM, Interpretable Binding Attention Map; MAE, mean absolute error; MSE, mean squared error; PDB, Protein Data Bank; RMSE, root mean squared error; RMSD, root-mean-square deviation; SMILES, simplified molecular-input line-entry system.

## Declarations

### Ethics approval and consent to participate

Not applicable. This computational study used protein sequences, small-molecule affinity data, and public macromolecular structures and did not recruit human participants or perform animal experiments.

### Consent for publication

Not applicable. The manuscript does not contain identifiable individual participant data.

### Availability of data and materials

The draft submission package includes the tabular results, frozen prediction exports, all generated-draw records and rejection reasons, and reviewed figures described in Additional files 1 through 5. Full checkpoints, chemical structure files, raw per-head attention matrices, environment records, and an artifact inventory are retained in the versioned research package. The reference code and revision protocol are available at https://github.com/kevinmsong/bindra/tree/main/reference. **AUTHOR ACTION REQUIRED:** deposit the reproducibility assets in an appropriate persistent repository, supply persistent data and full-model identifiers, and document access and redistribution conditions for third-party data and encoder weights. Local workspace paths are not public availability. Replace this draft statement with the final repository-based availability statement before submission.

### Competing interests

The authors declare that they have no competing interests.

### Funding

This study was supported in part by the National Institutes of Health and National Heart, Lung, and Blood Institute under grant numbers U01HL134764, P01HL160476, R01HL131017, and R01HL149137. **AUTHOR CONFIRMATION REQUIRED:** describe any funder role in study design, data collection, analysis, publication decisions, or manuscript preparation.

### Authors' contributions

**AUTHOR INPUT REQUIRED:** specify the contributions of Kevin Song (KS), John Zhang (JZ), Lei Ye (LY), and Jianyi Zhang (JYZ), and confirm that all authors reviewed and approved the final manuscript. Contribution roles and approval have not been inferred from author order.

### Acknowledgments

**AUTHOR CONFIRMATION REQUIRED:** list any additional contributors with their permission, or state that no additional acknowledgments are applicable. Coding and drafting assistance is disclosed in Methods.

## Additional files

**Additional file 1. Supplementary methods and analyses. PDF with editable LaTeX source.** Detailed protocol and metric definitions, historical exposure, selected FAK1-P4N iBAM and bond-map views, control analyses, and native docking results.

**Additional file 2. Full result tables. ZIP containing CSV and JSON files.** Architecture selection, individual models, endpoint-specific generator metrics, all screening rankings summarized by target and method, structural panel results, docking failures and pose checks, literature qualifications, analysis scope, and paired comparisons.

**Additional file 3. Frozen Davis predictions and partitions. ZIP containing Parquet and JSON files.** Primary and companion predictions by protocol and member, official-order identifiers, model manifests, split assignments and exclusions, and nearest-training similarity audits. Parquet files can be read with pandas or Apache Arrow.

**Additional file 4. Complete generator draw records. ZIP containing CSV, Parquet, and JSON files.** All 90000 draws, including invalid outcomes and rejection reasons, query selection, endpoint strata, graph exclusions, and configuration identities.

**Additional file 5. Reviewed figure collection. ZIP containing PDF and JSON files.** The complete figure collection, source metadata, and review records. Large raw arrays, checkpoints, and SDF files are listed in the artifact inventory for repository deposit rather than represented as already public supplementary material.

## Figure legends

**Figure 1. Shared encoders and two affinity-prediction paths.** (A) Frozen ESM-C and LiGhT-mini encoders supply the same protein and ligand features to both architectures. (B) BINDRA summarizes these features separately before combining them for regression. (C) BINDRA-iBAM adds bidirectional cross-attention, with FiLM disabled. The dashed path carries global summaries directly to regression, bypassing attention. Each predictor averages pKd from three independently trained models. iBAMs use last-layer attention averaged across heads; member maps are aligned before averaging, separately from prediction averaging. Parameter counts and layer dimensions are given in Table 1 and Methods; ligand-only bond maps are not shown.

**Figure 2. Validation comparison of eight model architectures.** Bars show mean validation MSE across three training seeds; error bars show sample standard deviations between seeds. Each architecture received the same six-configuration tuning budget. Official folds 1 through 4 supplied 20036 fitting pairs, and fold 0 supplied 5010 validation pairs. Labels A through G and C-matched correspond to Table 1. Lower MSE is better; units are squared pKd. Seed variation is distinct from test sampling uncertainty, and selection stability across additional folds was not assessed.

**Figure 3. Prediction errors for the pooled BINDRA ensemble.** All panels use the official 5010-pair Davis warm test. (A) Predicted versus measured pKd; hexagon color indicates pair count on a logarithmic scale, and the dashed line indicates perfect agreement. (B) Error distributions for the pKd = 5 detection plateau (3468 pairs) and above-plateau observations (1542 pairs). Error is prediction minus measurement, so negative values indicate underprediction. Histograms are normalized separately to probability density. (C) Cumulative distributions of mean absolute error calculated separately for each of 68 ligands and 379 protein sequences. Error and MAE are expressed in pKd units.

**Figure 4. Paired differences between attention and pooled prediction.** Points show BINDRA-iBAM minus BINDRA for (A) MSE, in squared pKd units, and (B) CI, which is unitless. Horizontal bars are 95% percentile intervals from 1000 crossed resamples of exact protein-sequence and Murcko-scaffold groups, using the same sampled observations for both models. Negative MSE differences and positive CI differences favor attention. Test sets contain 5010 warm, 2652 cold-drug, and 2992 cold-protein pairs. Cold-drug precision is limited by six chemical groups. Cold comparisons are exploratory; no correction for multiple comparisons was applied.

**Figure 5. Attention-contact correspondence and dependence on model parameters.** Analyses cover 30 complexes across ten equally weighted target groups. (A) Mean residue AP for ligand-to-protein attention, contact prevalence, shuffled-attention controls, and protein-feature gradient-times-input attribution. Contacts require a minimum heavy-atom distance of at most 4 angstrom. Bars are target-group means; error bars are 95% target-group bootstrap intervals. Prevalence is the fraction of contacting residues, rather than an empirical random-ranking AP. (B) Mean Spearman agreement between original and perturbed residue maps after atom reordering or progressive parameter randomization. Head-only randomization leaves upstream attention maps unchanged by construction. These controls assess geometric correspondence and model dependence, not causal binding mechanisms.


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

19. Shah PM, Zhu H, Lu Z, Wang K, Tang J, Li M. DeepDTAGen: a multitask deep learning framework for drug-target affinity prediction and target-aware drugs generation. Nat Commun. 2025;16:5021. doi:10.1038/s41467-025-59917-6.

20. Ho J, Jain A, Abbeel P. Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems. 2020. https://arxiv.org/abs/2006.11239.

21. Austin J, Johnson DD, Ho J, Tarlow D, van den Berg R. Structured Denoising Diffusion Models in Discrete State-Spaces. Advances in Neural Information Processing Systems. 2021. https://arxiv.org/abs/2107.03006.

22. Rogers D, Hahn M. Extended-Connectivity Fingerprints. J Chem Inf Model. 2010;50(5):742-754. doi:10.1021/ci100050t.

23. Liu T, Lin Y, Wen X, Jorissen RN, Gilson MK. BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities. Nucleic Acids Res. 2007;35(Database):D198-D201. doi:10.1093/nar/gkl999.

24. Bemis GW, Murcko MA. The Properties of Known Drugs. 1. Molecular Frameworks. J Med Chem. 1996;39(15):2887-2893. doi:10.1021/jm9602928.

25. Eberhardt J, Santos-Martins D, Tillack AF, Forli S. AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. J Chem Inf Model. 2021;61(8):3891-3898. doi:10.1021/acs.jcim.1c00203.

26. Forli laboratory. Meeko ligand preparation toolkit. 2026. https://github.com/forlilab/Meeko. Version 0.8.0; accessed 24 October 2026.

27. Buttenschoen M, Morris GM, Deane CM. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chem Sci. 2024;15(9):3130-3139. doi:10.1039/d3sc04185a.

28. Berman HM. The Protein Data Bank. Nucleic Acids Res. 2000;28(1):235-242. doi:10.1093/nar/28.1.235.

29. Worldwide Protein Data Bank. PDB structure 6YOJ. 2020. doi:10.2210/pdb6YOJ/pdb. FAK1-P4N complex; accessed 24 October 2026.

30. RDKit contributors. RDKit Open source cheminformatics. 2026. https://www.rdkit.org. Version recorded in the reproducibility supplement; accessed 24 October 2026.

31. Hua Y, Song X, Feng Z, Wu X. MFR-DTA: a multi-functional and robust model for predicting drug-target binding affinity and region. Bioinformatics. 2023;39:btad056. doi:10.1093/bioinformatics/btad056.

32. Ranjan A, Bess A, Alvin C, Mukhopadhyay S. MDF-DTA: A Multi-Dimensional Fusion Approach for Drug-Target Binding Affinity Prediction. J Chem Inf Model. 2024;64(13):4980-4990. doi:10.1021/acs.jcim.4c00310.

33. Kroll A, Ranjan S, Lercher MJ. A multimodal Transformer Network for protein-small molecule interactions enhances predictions of kinase inhibition and enzyme-substrate relationships. PLOS Computational Biology. 2024;20(5):e1012100. doi:10.1371/journal.pcbi.1012100.
