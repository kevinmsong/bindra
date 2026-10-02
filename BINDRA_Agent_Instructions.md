# BINDRA: retraining, refactoring, and publication-focused instructions

BINDRA = Binding INference, Design, Retrieval, and Attention.

Use BINDRA as the platform name and iBAM as the cross-attention explanation module. Preserve legacy DeepDTA-iBAM configuration/checkpoint compatibility where practical. Revision: 29 September 2026.

## Mission and required capabilities

Refactor the existing DeepDTA-iBAM repository into BINDRA, a reproducible, sequence-based platform for affinity prediction, interpretable attention visualization, target-conditioned ligand/analog retrieval, and diffusion-based analog generation.

Use this backbone:

- Amino acid sequence → pretrained ESMC-600M → residue representations.
- SMILES → official KPGT preprocessing → pretrained KPGT → molecular token representations.
- Trainable shared projections → bidirectional protein–ligand cross-attention → fusion representation.
- Separate pKd, pKi, and pIC50 prediction heads.
- iBAM residue–bond heatmaps, plus explicitly derived residue–atom heatmaps.
- Independent protein and ligand retrieval projections for fast search, followed by cross-attention reranking.
- A target-and-seed-conditioned molecular diffusion branch for generating analogs.

Train on a versioned, filtered BindingDB snapshot. The principal machine is an RTX 5060 Ti with 16 GB VRAM. Begin with frozen pretrained encoders and cached representations; jointly train the downstream projections and task modules. Full encoder fine-tuning is a later, separately measured experiment.

Preserve all four user-facing functions: predict affinity, explain with iBAM, retrieve/fish analogs, and generate analogs through diffusion. A global-embedding MLP is an ablation and development baseline, not the final replacement for cross-attention. A fragment enumerator or random SMILES mutation is not the requested diffusion generator.

These are proposed implementation choices, not established performance improvements. Inspect the actual repository before deciding which existing modules can be reused.

## Publication objective and the three highest-priority upgrades

Treat the original five implementation priorities below as the foundation. Add these three upgrades to turn BINDRA into a focused research contribution. Prioritize a testable mechanism, evidence of chemical/target specificity, and prospective utility over adding more encoders.

Working research question: Can measured analog-pair learning and seed-specific iBAM context improve target-conditioned molecular editing and retrieval on unfamiliar chemistry/targets, and can those improvements survive prospective experimental testing?

The literature already contains overlapping ideas:

| Prior work | Relevant overlap | Required response |
|---|---|---|
| DeepDTAGen [10] | Joint affinity prediction and target-aware generation | Compare shared learning against a credible multitask baseline |
| MT-DiffGen [11] | Sequence/graph prediction plus affinity-guided diffusion | Do not claim that combining affinity and diffusion is new |
| ConPLex [12] | Contrastive protein–ligand representations and scalable screening | Benchmark the retrieval branch against an established sequence-based approach |
| DiffPharma [13] | Interaction-constrained diffusion using 3D structural information | Define BINDRA's sequence-only, seed-analog setting precisely; compare on an input-matched subset if structures are supplied |

This is a targeted literature check, not proof of uniqueness. Maintain a dated related-work matrix before submission. Distinguish published evidence, author-reported results, unavailable code, and genuinely reproduced comparisons. If a baseline must be reimplemented, label it as such. Do not claim superiority over a method whose required inputs or implementation were unavailable.

### Upgrade A: Couple measured affinity changes, iBAM, and diffusion

This is the main methodological hypothesis. Make iBAM useful to analog design through an explicit, ablatable conditioning path, supported by measured analog differences.

Build a matched-pair table using only training observations with compatible target construct, endpoint, and assay context. For seed A and analog B:

~~~text
delta_y(T, A, B, endpoint) = y(T, B, endpoint) - y(T, A, endpoint)
~~~

Positive delta means stronger measured affinity/potency on that endpoint. Keep atom mappings, changed atoms/bonds, retained core, replicate uncertainty, and assay provenance. Begin with exact labels; do not subtract censoring thresholds as though they were measured affinities. Cap the influence of large chemical series.

Implement three linked pieces:

1. A delta-affinity objective. The minimum baseline predicts delta as f(T,B) - f(T,A). Test a dedicated edit-aware delta head using both pair representations and atom-mapped edit features only if it adds value. Enforce zero change for A=A and antisymmetry under A/B reversal. A suitable dedicated-head construction subtracts the outputs of the same ordered-pair network in both directions and divides by two.
2. A target-specific seed edit context. Derive bond/atom-associated residue context from the seed's verified iBAM weights and residue representations. Pass this structured context, the seed graph, endpoint, and any supported requested delta to the denoiser. Compare it with global target–seed fusion alone, using parameter-matched controls.
3. A measured selection objective. Train on observed analog changes and evaluate predicted changes against held-out measurements. Keep the affinity head, delta head, and retrieval scores distinguishable in exported results.

Add delta loss to the shared training objective. If a dedicated head is used, add a small, separately tunable consistency penalty between its prediction and f(T,B)-f(T,A). This consistency term enforces coherence; it supplies no new experimental evidence.

Default iBAM guidance is a conditioning signal, not a claim that high-attention atoms must remain unchanged. Keep the user-selected scaffold as the hard constraint. Any additional attention-based editing restriction must be optional, explicitly mapped, and compared with random/chemistry-based restrictions of equal size. Do not impose a protected fragment on a training pair whose observed product changes that fragment.

Compute generation conditions from the target and clean SEED only. Product embeddings may supervise delta prediction and score completed candidates; they must never enter the denoiser's clean conditioning path. Do not run KPGT on invalid noisy intermediate graphs or claim gradients pass through discrete sampling/RDKit.

At inference, report ensemble delta predictions using each model's paired seed/candidate predictions, preserving their covariance. Calibrate delta intervals on held-out measured pairs only when enough comparable calibration data exist; do not manufacture them by assuming independent seed/candidate errors.

Required controlled comparisons, with identical candidate/sample budgets:

- Base cross-attention/diffusion versus added delta learning.
- Global conditioning versus structured iBAM context.
- Correct iBAM versus shuffled or uniform context, with dimensions and capacity matched.
- Attention-derived context versus a feasible alternative attribution/conditioning baseline.
- Joint versus separate training of the modules.

Measure held-out delta MAE, direction-of-change accuracy outside a prespecified noise margin, and performance on activity cliffs. For generation, report measured analog recovery and prospective seed-to-analog improvement when available. Higher scores from the same predictor used to select molecules cannot establish success.

Explanation evaluation must include sensitivity tests and stability. Neither attention guidance nor model counterfactuals alone establish physical contacts or a causal binding mechanism. Do not force unchanged attention after a chemical edit; a real interaction pattern may change.

### Upgrade B: Establish chemical specificity with a locked stress-test suite

Keep the existing random, scaffold-cold, protein-cold, and double-cold evaluations. Add three focused challenges to determine whether the model understands small chemical/target differences and future data.

Activity-cliff challenge:

- Use MoleculeACE-style evaluation or document a custom definition, such as high fingerprint similarity plus at least one p-unit measured difference. Pin the fingerprint, threshold, endpoint, and assay-comparability policy. [14]
- Keep a strict evaluation panel with both members and their chemical series held out. Do not move one member into training to make the other easier.
- Use test labels only to assign the prespecified diagnostic category and calculate metrics after predictions are frozen.
- Report cliff-specific error, delta error, and change-direction accuracy alongside all-pair performance. Do not equate success on activity cliffs with generalization to novel targets.

Target-selectivity challenge:

- Use compounds measured against related target/countertarget pairs in comparable assay panels.
- Define selectivity as pX(desired target, ligand) - pX(countertarget, ligand), using the same endpoint and compatible conditions.
- Assess selectivity direction and margin error. Include target-swapped controls and a ligand-only baseline.
- Treat missing countertarget measurements as unknown. Do not label computational decoys or untested proteins as negative.
- Preserve the required target-cluster restrictions; related targets can both belong to the held-out cluster.

Temporal challenge:

- Choose a cutoff before modeling from the usable date coverage and save it in the benchmark configuration.
- Use first public experimental availability where supportable. A recent database import of an old measurement is not new evidence.
- Limit training/tuning/calibration to observations available by their designated cutoffs. Exclude unknown-date records from temporal claims.
- The strict future panel contains pairs whose first known evidence occurs after the cutoff. Exclude earlier pair evidence across endpoints/sources, or label a separate repeated-measurement panel explicitly.
- Report temporal and double-cold results separately; a future measurement need not concern an unfamiliar scaffold or protein.
- Document pretrained-encoder exposure and do not claim the entire pretrained system is historically time-clean if that cannot be established.

For drug/analog fishing, emphasize held-out measured weak/inactive compounds and structurally similar hard candidates, while retaining a separately labeled synthetic-decoy panel. Report full-pool EF1%, PR-AUC, Recall@K, and per-target estimates, including misses from ANN/shortlisting. Include chemical-similarity and simple property baselines.

Use an immutable benchmark manifest, prespecified primary metrics, three training seeds for the main comparisons, and confidence intervals that respect target/chemical-series dependence. Define a minimum effect of practical interest before final testing; do not invent a universal percentage improvement requirement. Report small or negative effects.

The minimum publication comparison should include BINDRA's base versus upgraded model, strong fingerprint/sequence baselines, and available close prior methods under matched data access. The benchmark suite tests specificity; it is not a new model component by itself.

### Upgrade C: Support a prospective, measured design–test–learn study

Build the software needed for an actual prospective comparison. Experimental measurements remain work for the research team; manifests and predicted results do not count as completed validation.

Freeze a model version, candidate universe, generation budget, ranking rule, and selection date before inspecting new measurements. Select candidates under matched feasibility and diversity constraints from:

- Target-conditioned retrieval.
- BINDRA diffusion analog generation.
- A credible reference method, initially seed-similarity selection from the same accessible chemistry space.

Export a configurable blinded candidate manifest, with model/arm assignments kept in a separate key. Include seed/parent compounds and appropriate assay controls. A 12-candidate allocation, four per arm plus controls, is a possible feasibility pilot, not a statistically powered default. Choose the actual size with the experimental team based on feasible assays and desired precision.

Match procurement/synthesis feasibility, property restrictions, and assay budgets across arms. Define duplicate handling and failed synthesis/procurement rules before outcomes are known. Report intended, feasible, and tested candidate counts and the reason for every attrition; do not replace failed candidates opportunistically after results arrive.

The result schema must keep separate:

- Verified chemical identity, purity, stereochemistry status, and batch.
- Target construct, assay, concentration-response data, units, relation/censoring, assay QC, and replicate identifiers.
- Direct binding measurements.
- Functional/phenotypic measurements and target-dependence experiments.
- Technical failures, missing data, inactive results, and disagreement.

For DIP2A, prioritize an assay-supported direct-binding measurement and an orthogonal confirmation route where feasible. If activation/proliferation is claimed, establish functional target dependence separately. Do not infer DIP2A binding from cardiomyocyte proliferation, or assume existing compounds are validated binders without direct evidence.

If an appropriate DIP2A binding assay is not yet established, the prospective method benchmark can use a target with a validated assay, while DIP2A remains an explicitly exploratory application. Do not imply that a different target's success validates DIP2A.

Compare confirmed hit rate, measured seed-to-analog changes, scaffold diversity, and uncertainty versus observed errors, with uncertainty intervals appropriate to the small sample. Record negative findings. Novel generated structures are not confirmed hits until they are actually made/obtained and measured.

For a second active-learning round, allow round-one measurements into a newly versioned training set, update calibration appropriately, and select genuinely new candidates before their outcomes are known. Preserve an untouched final audit set and label each prospective round separately. Update comparison baselines with equal access to the same new measurements. Never present a retrospectively refit result as the original prospective prediction.

Deliver candidate manifests, blinding/unblinding support, result ingestion and QC, provenance-aware dataset updates, and an automated report. Separate planned, selected, synthesized/obtained, assayed, and confirmed status. Do not generate fictitious laboratory results.


## First action: inspect and preserve the existing project

Read repository instructions, identify the current encoders, fusion/iBAM implementation, training loop, retrieval path, generator, interfaces, checkpoints, and evaluation scripts. Produce a short migration map from existing components to proposed replacements. Preserve a runnable legacy configuration and its checkpoint compatibility where practical.

Use incremental changes, configuration flags, and resumable stages. Do not silently replace an unavailable pretrained checkpoint with random initialization. Record unavailable weights or data as specific blockers while completing work that does not depend on them.

Implement the following five priorities.

Integrate Upgrades A–C above into these priorities; they are part of the requested BINDRA revision, not disconnected future suggestions. Keep experimental work explicitly pending until measurements exist.

## 1. Build a clean, endpoint-aware BindingDB dataset

### Versioned ingestion and provenance

Download or use an authorized BindingDB snapshot and save its release/date, source URL, checksum, parser version, and filtering configuration. Process large files in chunks. Retain the original raw data and write structured processed tables.

Use an observation table with at least:

- Observation ID, source record IDs, source database, publication/patent and assay identifiers when available.
- Original and standardized SMILES; canonical isomeric identity; a separate parent/connectivity grouping key.
- Actual target sequence and sequence hash; UniProt identifiers, organism, construct/mutation information, and chain count.
- Endpoint, original value/unit/relation, normalized p-value or bound, and exact/censored status.
- Available assay metadata, a conservative comparability-group identifier, and an explicit metadata-confidence flag.
- QC status, exclusion reason, replicate count/spread, and eventual split assignment.

BindingDB can contain repeated chain fields and inequality-valued measurements. Parse these deliberately. Source/publication identity alone does not establish that two measurements came from the same assay. [1]

### Chemical and protein filtering

Use deterministic RDKit standardization. Preserve specified stereochemistry, formal charges, original structures, and a mapping between original and standardized atom indices. Remove recognized counterions using a documented policy. Quarantine ambiguous mixtures, unsupported chemistry, and failed sanitization.

Maintain separate keys for exact stereochemical identity and broader leakage grouping. Do not accidentally merge distinct stereoisomers into one measured molecule. Group salt/tautomer variants conservatively for splitting, with the normalization policy recorded.

The initial supervised dataset should have an unambiguous, valid single-protein sequence and a compatible small molecule. Quarantine multichain complexes and ambiguous targets. Preserve experimental constructs and mutations; do not automatically replace every construct with the full canonical UniProt sequence.

Retain high-confidence Ki and Kd observations. Use IC50 only when the record supports assignment to the intended protein and an appropriate biochemical assay. Keep less-certain IC50 records in a separately flagged tier. Exclude phenotypic EC50, qualitative labels, and unrelated endpoints from the main affinity regression.

Audit missing metadata and report how many observations remain eligible for regression, within-assay ranking, retrieval supervision, and analog-pair training.

### Endpoint conversion and censored measurements

For a positive concentration x in nM:

~~~text
pX = 9 - log10(x)
~~~

Keep pKd, pKi, and pIC50 separate throughout training, evaluation, and output. IC50 remains an assay-dependent potency endpoint. Never relabel it as Kd.

Examples that must become unit tests:

| Original measurement | Correct transformed label |
|---|---|
| Kd = 100 nM | pKd = 7 |
| Kd > 10,000 nM | pKd < 5 |
| Ki < 10 nM | pKi > 8 |
| Missing, zero, negative, or invalid value | Missing/rejected with a reason |

Reverse inequality direction under the negative logarithm. Preserve strict/non-strict relations in the data.

Use Huber loss for exact labels. For a simple explicit treatment of censoring, use a one-sided Huber penalty:

~~~text
Exact y:       Huber(prediction - y)
Lower bound b: Huber(max(0, b - prediction))
Upper bound b: Huber(max(0, prediction - b))
~~~

This is a bound-respecting optimization loss, not a probabilistic censoring model. Treat an interval as the sum of its lower- and upper-bound violations. Do not train against a bound as if it were an exact observation.

Report exact-label metrics separately from censored bound violations. Fit any label scaler only on training observations and preserve its inverse transform. Expose all predictions in original p-units.

### Deduplication and assay compatibility

Remove duplicate database imports of the same underlying measurement before splitting. Keep independent experiments distinguishable. Aggregate only genuinely compatible exact replicates, using a documented robust summary and retaining count/spread. Flag materially discordant replicates; do not average different assays into artificial certainty.

Keep every observation for a given standardized ligand–target pair, across endpoints and sources, in one partition. Related chemical forms must obey the additional grouping rules of the chosen split.

Only construct ranking/analog-improvement pairs when target construct, endpoint, and assay context are compatible. A missing assay identifier must not cause all records for that target to become one giant ranking group.

Deliver a curation report with counts after every filter, endpoint distributions, target-family coverage where known, duplicates, censored observations, disagreements, and exclusions.

## 2. Retain cross-attention and iBAM while jointly learning affinity and retrieval

### Correct pretrained encoder integration

Use a pinned ESMC-600M checkpoint, tokenizer, code revision, and weight checksum. Explicitly distinguish a newer checkpoint from an older checkpoint carrying a similar model name. Start from the verified Biohub ESMC-600M implementation. [2]

Extract final-layer residue embeddings, not vocabulary logits. Strip padding and special tokens using tokenizer metadata. Retain sequence positions and masks. Verify the checkpoint's supported context behavior; never silently truncate long proteins. If overlap-based chunking is needed, document it, merge overlapping residues deterministically, and preserve original residue numbering.

KPGT uses molecular line graphs together with fingerprints/descriptors and knowledge nodes. Use the author's preprocessing and pretrained representation pathway; a generic atom graph is not a drop-in replacement. [3,4]

Pin KPGT weights, preprocessing, descriptor ordering/normalization, graph library, and chemistry-library versions. Verify pretrained weight loading and a reference feature-extraction example. Preserve numerical compatibility when updating dependencies.

Expose BOTH molecular token states before graph pooling and the graph-level readout. Verify the meaning of every token in the actual implementation. Maintain mappings from chemical tokens to original bonds/atoms; identify knowledge, virtual, padding, and any special isolated-atom tokens separately. Do not assume one KPGT token equals one atom.

Audit whether the selected upstream features encode stereochemistry. Preserve input stereochemistry even if an encoder limitation exists, and report that limitation honestly.

Use these conceptual adapter outputs; adapt names to the existing codebase:

~~~text
ProteinFeatures:
  residue_features[L, dp], residue_mask[L], residue_indices[L],
  sequence_hash, encoder_metadata

LigandFeatures:
  token_features[T, dl], token_mask[T], token_types[T],
  graph_features[dg], token_to_bond, token_to_atom,
  atom_index_maps, molecule_id, encoder_metadata
~~~

Discover dimensions from real outputs; do not invent them.

Cache frozen features once per unique sequence/molecule in sharded arrays with indexed metadata. Cache keys must include standardized input, encoder/checkpoint hash, feature schema, precision, and preprocessing policy. Cache base features rather than trainable projections. If an encoder is unfrozen, bypass or invalidate its cached outputs.

### Cross-attention fusion

Project protein residues and ligand tokens into a shared hidden dimension, initially 256. Start with two bidirectional cross-attention blocks and eight heads, preserving residual paths and correct masks.

Keep protein-to-ligand and ligand-to-protein attention distinct: they have different softmax normalization axes and are not interchangeable transposes.

Pool the conditioned token states and combine them with unconditioned global representations to form the pair representation. Feed it to separate pKd/pKi/pIC50 heads. Include endpoint masks so an unmeasured endpoint never receives a fabricated zero label.

The trainable representation adapters must actually be shared by affinity, retrieval, and diffusion conditioning. Log which parameters receive gradients from each loss. Frozen ESMC/KPGT plus jointly trained adapters is a valid design, but describe it accurately.

### iBAM heatmaps are a required output

Retain layer/head-specific cross-attention weights with documented axes and masks. Export the raw numeric matrices and the plotting metadata, not just a rendered image.

Required views:

- Residue × chemical-token/bond heatmap.
- Residue × atom heatmap derived through the verified token-to-atom mapping.
- Per-residue summary and a molecular depiction with atom/bond contributions.
- Layer/head identifiers, model/checkpoint, sequence positions, molecule atom indices, aggregation rule, and ensemble variability where calculated.

For a token corresponding to a bond between two atoms, an explicit initial visualization rule is to divide its contribution equally between its endpoints. Label this as a derived aggregation, not native atom attention. Handle other token types according to their verified semantics.

Do not draw knowledge/virtual nodes as physical atoms. Preserve their attention mass in raw exports and disclose any visualization renormalization. Use a fixed aggregation policy chosen before inspecting test examples.

Use colorblind-safe sequential colors, readable residue/atom labels, vector PDF/SVG exports, and an optional 600-dpi PNG. Provide a full matrix export plus zoomed residue windows for long targets.

Call iBAM an attention-based explanation. Do not label it an experimentally established contact map, binding pose, or mechanism.

Evaluate explanation stability across seeds and model sensitivity to masking high-ranked features versus matched random features. Check contacts only on an appropriately held-out structural subset, with valid residue/atom mappings and a defined contact criterion. Feature masking tests model dependence; it does not establish biological causality.

### Fast retrieval and cross-attention reranking

Implement two operational modes:

1. Target fishing: given a target sequence and endpoint, rank a candidate compound library.
2. Analog fishing: given a target, endpoint, and seed ligand, retrieve similar compounds and rank their predicted suitability for that target.

Create independent protein and ligand retrieval embeddings from the shared, unconditioned adapters. They must be computable without knowing the other member of a pair. Do not index pair-conditioned cross-attention embeddings.

Make the query endpoint-aware. Build a versioned approximate-nearest-neighbor index of ligand vectors. For analog fishing, support a documented chemical-neighborhood filter, such as Morgan-fingerprint similarity or scaffold membership, followed by target scoring. Exclude the seed itself and exact duplicates.

The deployed sequence-plus-SMILES predictor must not depend on publication IDs, source IDs, or assay IDs as shortcuts. Keep these as provenance/grouping fields. Any optional physical assay covariates must have a defined missing-value path and be available for the intended inference use.

Retrieve a configurable shortlist, then rerank it with the full cross-attention affinity model. Compare the two-stage pipeline with exhaustive reranking on a manageable benchmark. Measure losses caused by both ANN approximation and shortlist selection.

Return the chosen endpoint prediction, uncertainty, seed similarity, retrieval score, domain-distance indicators, and candidate provenance. Keep similarity and affinity as separate reported quantities even if a validation-tuned combined score is also offered.

### Joint learning objectives

Use:

~~~text
L_total =
    L_affinity
  + lambda_rank * L_within_assay_ranking
  + lambda_retrieval * L_retrieval
  + lambda_diffusion * L_diffusion
  + lambda_delta * L_measured_analog_delta
  + lambda_consistency * L_delta_consistency_if_separate_head
~~~

The delta consistency term is disabled when delta is defined directly as the difference between affinity predictions. Enable the measured-delta term only for eligible compatible pairs and log its support. Tune auxiliary weights using validation data; no term may consume test or calibration labels.

Warm up the affinity model before increasing auxiliary weights. Normalize each loss over eligible observations/groups and log effective sample counts. Balance endpoints and cap domination by heavily sampled targets.

For ranking, compare exact measurements from the same compatible target/endpoint/assay group. Initially ignore differences smaller than a configurable 0.5 p-unit margin; treat this as a starting hyperparameter, not a universal assay-error threshold.

For retrieval, start with a measured-candidate listwise objective: within a compatible assay group, form soft relevance weights from measured p-values and train the dual-encoder similarity scores to reproduce that ordering. Keep relevance and similarity temperatures configurable.

This avoids treating every unmeasured matrix entry as a nonbinder. If adding binary contrastive training later, use explicitly justified labels, multi-positive handling, and masks for unknown pairs. A weaker measured ligand is not automatically a nonbinder.

Never use calibration/test observations to mine training negatives, select loss weights, or tune shortlist sizes.

## 3. Add genuine target-and-seed-conditioned diffusion analog generation

### Recommended implementation: conditional 2D graph diffusion

Use a DiGress-style discrete molecular graph diffusion model as the initial design. Its categorical atom/bond generation is compatible with sequence-only target conditioning after implementing the conditioning extension. The extension described here is proposed work; it is not a claim that the upstream model already solves this task. [5,6]

Prefer this over adding Gaussian noise to a pooled KPGT embedding without a trained molecular decoder. KPGT is an encoder, and an arbitrary vector cannot be inverted into a valid molecule.

If the existing project already has a functioning latent diffusion implementation, retain it only after demonstrating an explicit graph/SMILES decoder, molecular reconstruction quality, conditional denoising, and valid reverse-process sampling. Otherwise implement the discrete graph approach.

### Inputs and conditioning

The generator must receive:

- Target residue representations and endpoint identity.
- The actual seed ligand's KPGT representations.
- Shared cross-attention features for the target–seed pair.
- Structured iBAM-derived seed edit context, with uniform/shuffled/global-only ablations.
- A preserved scaffold/core with explicit atom mapping and allowed attachment/edit positions.
- An optional requested affinity-change condition, only when appropriate measured training pairs support it.

The denoiser predicts editable atom/bond categories from the noisy graph, timestep, and these conditions. Protein and seed conditioning must enter the denoising network through cross-attention and/or a documented feature-conditioning mechanism.

Never pass the clean desired product's full KPGT representation as a condition: that product is unavailable at inference and would create a reconstruction shortcut.

### Molecular diffusion mechanics

Implement a real categorical forward-noising process and its corresponding reverse sampler. Reuse verified upstream transition mathematics and test it; do not apply a continuous Gaussian scheduler to categorical graphs without a justified model change.

Represent atom attributes, formal charge, bonds including the no-bond category, node masks, and supported stereochemistry explicitly. Keep edges symmetric and prohibit self-bonds.

Sample a permitted atom count conditioned on the scaffold/seed, or use a clearly specified count model. Padding must be distinct from chemical atoms. Fit graph-category marginals and size distributions on training data only.

Keep the protected scaffold's atom identities, internal bonds, and stereochemistry fixed throughout both training corruption and sampling. Let permitted attachment edges and substituents change. Verify the final preserved substructure by atom mapping.

Operate on noisy graph tensors directly. Intermediate noisy graphs can violate chemical valence; do not require RDKit sanitization or KPGT encoding at every diffusion step.

Decode final atom/bond predictions explicitly into an RDKit molecule, sanitize, verify connectivity/scaffold constraints, and produce canonical isomeric SMILES. Preserve known stereocenters; identify newly created unspecified stereochemistry instead of inventing it. Report unsupported chemical classes.

Any repair step must be deterministic, logged, and distinguished from raw model output. Report validity before and after repair/filtering.

Use a small denoiser initially and profile it on the actual GPU. Upstream DiGress documents implementation fixes and checkpoint limitations; inspect them, pin a tested revision, and verify that configuration values actually control the constructed model. Do not assume a molecular checkpoint supports protein conditioning. [6]

### Training data and shared learning

Build train-only matched analog pairs with compatible target, endpoint, and assay context. Record parent/product identities, atom mappings, common core, measured labels, and the measured change when known.

Only training partitions may supply analog pairs, transformation statistics, vocabularies, chemistry pretraining, or task-specific fine-tuning. Split related pairs together to prevent a compound appearing as a training product and an evaluation seed.

A chemistry-denoising warmup on training molecules can precede conditional analog training. Masked reconstruction supports chemical learning but must not be reported as demonstrated lead optimization.

Train denoising loss on parent-to-analog examples, then enable joint training through the shared conditioning adapters. Make gradient paths explicit; discrete molecule sampling and RDKit conversion are not differentiable.

If comparable analog pairs are scarce, implement and report target/scaffold-conditioned generation with the available support, and state that improved-affinity generation is unestablished. Do not synthesize experimental improvement labels from the model's own predictions.

### Sampling and assessing generated analogs

Provide an API accepting target sequence, seed SMILES, endpoint, sample count, scaffold policy, and random seed. Return a reproducible candidate set with generation parameters and rejection reasons.

After generation:

1. Sanitize and enforce scaffold/edit constraints.
2. Deduplicate candidates and exclude exact seed copies from the novel-analog count.
3. Re-encode valid candidates with KPGT.
4. Score them with the affinity ensemble.
5. Report target relevance, uncertainty, seed similarity, diversity, and basic molecular-property/feasibility flags.
6. Export SMILES and SDF, and produce iBAM plots for selected candidates.

Treat predicted improvement as a selection hypothesis. Keep a different-architecture evaluator out of the optimization loop; use it as a secondary check, with prospective measurements as the stronger evidence. A different model trained on similar data is not independent experimental validation.

Compare target-conditioned diffusion against seed/scaffold-only diffusion, target-shuffled conditioning, and a simple analog-enumeration baseline under the same sampling budget. Measure validity, uniqueness, novelty relative to training, scaffold retention, diversity, similarity, measured held-out analog recovery, and target-specific performance.

Do not let a scoring loop repeatedly optimize against the supposed independent evaluator.

## 4. Add calibrated uncertainty and applicability-domain reporting

Train three independent downstream models with different initialization/data-order seeds, using the same fixed partitions and final selected training protocol. Train them sequentially on the 16-GB GPU. Do not represent three dropout passes as three independently trained models.

For each endpoint, report the ensemble mean and standard deviation in p-units. Ensemble disagreement is a heuristic uncertainty signal, not a calibrated prediction interval or a guarantee of correctness.

Reserve a calibration partition separate from both training and hyperparameter validation. Calibrate only after the final shared representation and model checkpoints are frozen.

A concrete initial option is split conformal calibration on exact labels:

~~~text
score_i = abs(y_i - ensemble_mean_i)
k = ceil((n_calibration + 1) * (1 - alpha))
q = kth smallest calibration score
interval = [ensemble_mean - q, ensemble_mean + q]
~~~

Start with 90% intervals, separately by endpoint. If the required order statistic is unsupported, return an insufficient-calibration status rather than a made-up finite interval.

A later adaptive version may divide residuals by max(ensemble_SD, a fixed floor), then multiply interval width by that scale. Select the floor and method on validation data before calibration.

Evaluate empirical coverage and interval width for each split regime and by target similarity/chemical novelty. Conformal guarantees rely on appropriate exchangeability; correlated assays and novel protein families can violate it. Do not claim guaranteed coverage on a new target such as DIP2A. [7]

Calculate target similarity and ligand similarity against TRAINING entities only. Store the methods and thresholds. Return an OOD/domain-status flag independently of ensemble spread.

For conservative selection, support a configurable mean-minus-uncertainty ranking or an appropriate calibrated lower bound, followed by chemical diversity selection. Report which rule was used. If converting a pKd interval to nM, reverse its endpoints under the inverse logarithm.

Affinity predictions alone do not identify DIP2A activators, target dependence, or cardiomyocyte proliferation.

## 5. Prove generalization, drug fishing, and the benefit of unification

### Create partitions before any task-specific training

Maintain immutable train, tuning-validation, calibration, and test manifests. Apply restrictions across all supervised and auxiliary data sources.

Required regimes:

| Regime | Required separation |
|---|---|
| Grouped random-pair | All records/endpoints of a ligand–target pair stay together |
| Cold drug | Standardized chemical identities and scaffold groups are disjoint |
| Cold target | Protein sequence/homology groups are disjoint |
| Double cold | Both chemical scaffold groups and protein groups are disjoint |

Use a documented Bemis–Murcko policy, with an explicit fallback for molecules lacking a ring scaffold. Audit nearest-neighbor fingerprint similarity: different scaffolds alone do not guarantee chemical dissimilarity.

For proteins, begin with a prespecified sequence-identity/coverage rule, for example 40% identity with 80% coverage of the shorter sequence, then audit cross-partition alignments. This is an operational threshold, not proof that all shared domains or families are absent. Handle fragments and constructs explicitly and quarantine offending overlaps.

Clustering labels alone are insufficient evidence of separation. Verify the actual forbidden cross-partition relationships. Use a similarity-aware splitter or graph-component constraints as appropriate. [8]

For double-cold evaluation, assign protein groups and drug groups independently, then retain an interaction only when both entity assignments belong to that partition. Quarantine cross-partition interactions and report attrition. Grouping by a combined protein/drug cluster ID is not double-cold splitting.

Do not select a split seed based on favorable model performance. Record actual partition sizes and exclusions.

Off-the-shelf encoders may have seen unlabeled test molecules/sequences during their original pretraining. Document known overlap and distinguish supervised cold-start evaluation from a claim that the entities were never seen during pretraining.

Keep KIBA/Davis as separate, optional benchmark configurations. KIBA is not a pKd label. If external training data are used, audit and remove prohibited pair/entity/family overlap before making benchmark claims.

### Drug-fishing and analog-fishing studies against decoys

Create a locked per-target challenge set with held-out actives and candidate controls. Keep targets, actives, seeds, candidate library, decoy source, labels, thresholds, and query conditions in a benchmark manifest.

Use two control tiers:

- Preferred: experimentally measured weaker/inactive compounds under appropriate assays and prespecified endpoint thresholds.
- Supplementary: property-matched computational decoys, explicitly labeled as presumed controls with unknown binding status.

Computational decoys are not experimental nonbinders and must not acquire fabricated affinity labels. DUD-E-style construction can favor 2D similarity methods, so decoy enrichment must be accompanied by similarity/property baselines and bias checks. [9]

For analog fishing, choose the seed without consulting held-out performance, exclude it from the candidate positives, and apply the same similarity filters and candidate universe to every compared method. Do not filter only the model's candidate list and compare it with an easier/different baseline pool.

Report target-only fishing separately from seed-assisted fishing. A known binder supplied as the seed is useful chemical information; describe it as seed-assisted evaluation rather than zero-shot discovery.

Compare fingerprint similarity alone, KPGT similarity alone, target-only retrieval, cross-attention-only exhaustive scoring on tractable pools, and the combined two-stage analog-fishing workflow. Include target-shuffled and ligand-only controls to detect shortcuts.

Use experimental held-out candidate sets for quantitative affinity metrics. Treat unmeasured library members as unlabeled, not known false positives.

Calculate end-to-end enrichment and active recovery against the full locked challenge, counting actives missed by the retrieval stage. Do not report metrics only on the easier reranked shortlist as full-library performance. If DIP2A lacks validated binding positives and controls, present its output as a prospective shortlist; phenotypic activity alone does not supply a binding benchmark.

### Metrics and ablations

Report every metric separately by endpoint and split:

| Component | Required assessment |
|---|---|
| Affinity | RMSE, MAE, Pearson/Spearman where defined; within-target ranking/concordance |
| Analog changes and selectivity | Delta MAE, change-direction accuracy, activity-cliff error, target/countertarget selectivity margin |
| Fishing/retrieval | Recall@K, PR-AUC, ROC-AUC, EF1%, BEDROC with declared parameter; candidate counts/prevalence |
| Retrieval efficiency | ANN recall versus exact search, end-to-end active recovery, latency, memory, index size |
| Uncertainty | Empirical coverage, interval width, error versus uncertainty and domain distance |
| iBAM | Mapping correctness, seed stability, perturbation comparison, held-out contact comparison where available |
| Diffusion | Raw/filtered validity, uniqueness, training novelty, scaffold retention, diversity, target conditioning, analog recovery |
| Prospective study | Confirmed hit rate, measured seed-to-analog change, attrition, calibration/error, per-round outcomes |

Use per-target/macro summaries alongside pooled metrics. Report when a metric is undefined or a target lacks enough measurements. Provide uncertainty across training seeds and, where appropriate, bootstrap at the target/cluster level rather than pretending all observations are independent.

Minimum staged ablations:

1. Existing ESMC + GAT model, retrained on the same curated data where feasible.
2. ESMC + KPGT pooled fusion.
3. ESMC + KPGT cross-attention with iBAM.
4. Cross-attention plus ranking and retrieval objectives.
5. Above plus shared diffusion conditioning.
6. Jointly trained modules versus separately trained counterparts.
7. Target-conditioned versus target-agnostic/shuffled retrieval and diffusion.
8. Single model versus ensemble/calibration.
9. Measured delta learning on/off and iBAM guidance versus global, shuffled, uniform, and feasible attribution controls.

Add the locked activity-cliff, target-selectivity, and temporal panels described in Upgrade B. Predefine which comparisons are primary, and use shared cached features and identical split manifests to keep the experiment budget feasible.

Match data access, split, tuning budget, candidate sets, and approximate compute for the comparison being made. Do not compare published random-split numbers to new cold-split results as if they were equivalent.

Treat joint training as a hypothesis. If it damages a task, report negative transfer and retain a configurable independent branch. Affinity plus target-aware generation already has prior work; demonstrate what the particular integration contributes. [10]

## Engineering plan, practical defaults, and acceptance criteria

### Implementation order

1. Repository audit, legacy baseline, dependency/GPU smoke tests.
2. Curation, immutable splits, leakage checks, and data reports.
   Lock the activity-cliff, selectivity, and temporal benchmark definitions at this stage.
3. Verified frozen encoder adapters and caches.
4. Cross-attention affinity heads and correct iBAM exports.
5. Joint ranking/retrieval and the locked decoy fishing benchmark.
6. Molecular diffusion mechanics, chemical validation, then target/seed conditioning.
   Add measured-delta learning and structured iBAM conditioning as separate ablations.
7. Shared multitask training, independent training ablations, and three model seeds.
8. Final calibration, held-out evaluation, and example end-to-end outputs.
9. Freeze a prospective candidate-selection manifest, then ingest actual experimental results when supplied.
10. Optionally run a separately versioned second design–test–learn round while preserving an untouched audit set.

Calibration must follow the last changes to the shared model. Keep a pilot run distinct from a completed full-data experiment.

### Initial settings to expose in configuration

| Setting | Starting choice |
|---|---|
| Pretrained encoders | Frozen ESMC-600M and pretrained KPGT |
| Shared token width | 256 |
| Cross-attention | 2 blocks, 8 heads, dropout 0.1 |
| Optimizer for new modules | AdamW; initial learning rate 3e-4 |
| Affinity objective | Huber plus one-sided censored penalties |
| Analog delta | Difference-of-affinity baseline; dedicated antisymmetric head only as an ablation |
| iBAM diffusion context | Explicit seed-derived conditioning; user scaffold remains the hard constraint |
| Auxiliary weights | Start at zero; trial rank 0.1, retrieval 0.1, diffusion 0.05 after normalization/warmup |
| Ensemble | 3 independent seeds |
| Conformal level | 90%, endpoint-specific |
| Precision | BF16/FP16 where supported; numerically sensitive reductions in FP32 |
| Batch size | Profile token/graph budgets; use gradient accumulation |
| Generation | Small conditional discrete graph denoiser; explicit scaffold constraints |
| Retrieval shortlist | Validation-tuned; report shortlist and ANN recall |

These are starting configurations, not tuned recommendations or performance guarantees. Inspect loss scales and gradient magnitudes before adopting numerical weights.

Use current GPU-compatible PyTorch/CUDA packages and verify the actual DGL/PyG operations used. Do not blindly copy legacy CUDA versions from upstream environment files. If encoder environments conflict, isolate feature extraction and exchange versioned cached arrays.

Estimate storage and runtime from a representative pilot before encoding the entire corpus. Keep large caches sharded and stream batches. Compute attention plots only for selected pairs; do not save every head's matrices for millions of screening pairs.

### Required project deliverables

Provide clear configuration files and commands for preparing data, making splits, caching both encoders, training, calibrating, predicting, retrieving, generating, plotting iBAM, and evaluating. Use existing project naming where possible.

Deliver:

- Migration notes and a concise reproducibility README.
- Curated-data schema, provenance/filters report, and immutable split manifests.
- Encoder adapters with verified pretrained loading and index mappings.
- Cross-attention model, endpoint heads, iBAM plotting/export code.
- Retrieval index builder, target/analog-fishing APIs, and benchmark manifests.
- A genuine conditional diffusion trainer/sampler and graph-to-molecule decoder.
- Ensemble inference, calibration artifacts, and domain diagnostics.
- Machine-readable metrics, ablations, example predictions, fishing results, analogs, and heatmaps.
- Matched-pair data and delta training/evaluation, iBAM-guidance ablations, and a dated related-work comparison matrix.
- Locked activity-cliff/selectivity/temporal manifests and reports.
- Prospective selection/blinding manifests, result-ingestion/QC code, and versioned round reports.
- Checkpoint metadata containing source code/config/data/split/encoder hashes.

### Tests that must pass

Test concrete failure risks rather than duplicating implementation:

- Units, log conversion, inequality reversal, missing labels, and endpoint masks.
- Duplicate imports, conflicting assays, and pair/scaffold/protein separation.
- Protein special-token stripping, long-sequence mapping, and padding invariance.
- KPGT reference extraction and verified bond/atom/special-token mappings.
- Atom-renumbering consistency of predictions and mapped iBAM displays within numerical tolerance.
- Correct pretrained loading; cached/live feature agreement; stale-cache rejection.
- No held-out labels entering training losses, analog pairs, negative mining, scalers, or calibration.
- Tiny-data overfit and nonzero shared-module gradients for each enabled task.
- Indexed versus exact retrieval on a small library, with identical candidate pools.
- Diffusion forward/reverse consistency, graph symmetry, scaffold invariance, decoding, and reproducibility.
- Generated-product features cannot enter the conditioning path.
- Analog delta sign, antisymmetry, identity-pair zero, compatible-assay eligibility, and series separation.
- iBAM context can be replaced by controlled uniform/shuffled inputs without changing parameter or sample budgets.
- Temporal splits reject old measurements reimported after cutoff and exclude future labels from training.
- Prospective predictions remain immutable; assay failures/missing results cannot become fabricated negative labels.
- Calibration order statistics and exclusion of tuning/test labels from calibration fitting.
- Saving/reloading a model reproduces inference within declared tolerance.

The final system must run from target sequence and SMILES through affinity prediction plus iBAM, retrieve target-relevant analogs against a documented candidate/decoy set, and generate valid scaffold-constrained analogs through actual diffusion. It must report failures and unsupported cases explicitly.

Do not mark all requested functionality complete merely because interfaces exist. Separate implemented, smoke-tested, trained, benchmarked, and experimentally validated status. Do not invent performance gains, biological mechanisms, or validation results.

The publication claim must follow the evidence: component integration establishes engineering utility; controlled gains support the proposed mechanism; prospective binding and seed-to-analog measurements support discovery utility. Preserve useful negative findings. Do not claim BINDRA is the first affinity–diffusion framework or that it has demonstrated activation from affinity predictions.

## Primary references and implementation starting points

These sources support component behavior and methodological cautions. The integration, defaults, phases, and acceptance criteria above are design recommendations.

1. [BindingDB TSV format](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/BindingDB-TSV-Format.pdf)
2. [Biohub ESMC-600M model card](https://huggingface.co/biohub/ESMC-600M) and [official ESM code](https://github.com/Biohub/esm)
3. [Official KPGT repository](https://github.com/lihan97/KPGT)
4. [KPGT paper, Nature Communications](https://www.nature.com/articles/s41467-023-43214-1)
5. [DiGress paper](https://arxiv.org/abs/2209.14734)
6. [Official DiGress repository](https://github.com/cvignac/DiGress)
7. [Angelopoulos and Bates: A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)
8. [DataSAIL: data splitting to avoid information leakage](https://www.nature.com/articles/s41467-025-58606-8)
9. [Hidden bias in DUD-E virtual-screening benchmarks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220113)
10. [DeepDTAGen: affinity prediction and target-aware generation](https://www.nature.com/articles/s41467-025-59917-6)
11. [MT-DiffGen: affinity prediction and target-aware generation with multitask diffusion](https://www.sciencedirect.com/science/article/abs/pii/S095070512600345X)
12. [ConPLex: contrastive protein–ligand representations and screening](https://cb.csail.mit.edu/conplex/)
13. [DiffPharma: interaction-constrained molecular diffusion](https://www.nature.com/articles/s44386-026-00040-x)
14. [MoleculeACE: activity-cliff evaluation](https://github.com/molML/MoleculeACE) and [primary paper](https://doi.org/10.1021/acs.jcim.2c01073)
