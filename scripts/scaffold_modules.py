"""Create the remaining BINDRA package tree as honest stubs.

Each generated module states the specification clause it owes and raises
``NotImplementedError`` when called. Nothing returns a plausible-looking value
it has not computed, because a silent stub is worse than a missing module: it
produces numbers that look like results.

Idempotent — existing files are never overwritten.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "bindra"

# module path -> (summary, spec clause it owes)
MODULES: dict[str, tuple[str, str]] = {
    "config/profiles.py": (
        "Named experiment profiles and config hashing.",
        "Initial settings to expose in configuration; checkpoint metadata "
        "containing source code/config/data/split/encoder hashes.",
    ),
    "data/bindingdb/ingest.py": (
        "Chunked, versioned BindingDB TSV ingestion with provenance.",
        "Priority 1: save release/date, source URL, checksum, parser version "
        "and filtering configuration; process large files in chunks.",
    ),
    "data/bindingdb/standardize.py": (
        "Deterministic RDKit standardization with atom-index mapping.",
        "Priority 1: preserve stereochemistry and formal charges; keep a "
        "mapping between original and standardized atom indices; quarantine "
        "ambiguous mixtures rather than coercing them.",
    ),
    "data/bindingdb/qc.py": (
        "Deduplication, replicate aggregation, assay-comparability grouping.",
        "Priority 1: remove duplicate imports before splitting; a missing "
        "assay identifier must not collapse a target into one ranking group.",
    ),
    "data/splits.py": (
        "Immutable split manifests: random-pair, cold-drug, cold-target, "
        "double-cold, temporal.",
        "Priority 5: assign protein and drug groups independently for "
        "double-cold, then keep an interaction only when both sides agree; "
        "verify forbidden cross-partition relationships, do not trust cluster "
        "labels alone.",
    ),
    "data/pairs.py": (
        "Matched analog pairs and measured delta_y, train partitions only.",
        "Upgrade A: compatible target construct, endpoint and assay context; "
        "exact labels only; do not subtract censoring thresholds as if they "
        "were measured affinities; cap large-series influence.",
    ),
    "data/batching.py": (
        "Datasets, token-budget batch samplers, endpoint-aware collation.",
        "Port of the legacy TokenBudgetBatchSampler, extended with endpoint "
        "masks and censoring relations.",
    ),
    "features/cache.py": (
        "Sharded, memory-mapped feature cache with staleness rejection.",
        "Priority 2: cache keys must include standardized input, encoder "
        "checkpoint hash, feature schema, precision and preprocessing policy; "
        "cache base features, not trainable projections.",
    ),
    "encoders/ligand_kpgt.py": (
        "KPGT molecular encoder adapter via the isolated feature environment.",
        "Priority 2: use the author's preprocessing and pretrained pathway; "
        "expose token states AND the graph readout; do not assume one token "
        "equals one atom.",
    ),
    "encoders/ligand_gat.py": (
        "Legacy 78-feature GAT ligand encoder, retained as an ablation arm.",
        "Ablation 1: existing ESMC + GAT model retrained on the same curated "
        "data.",
    ),
    "fusion/adapters.py": (
        "Shared trainable projections over frozen encoder features.",
        "Priority 2: the adapters must actually be shared by affinity, "
        "retrieval and diffusion conditioning; log which parameters receive "
        "gradients from each loss.",
    ),
    "fusion/cross_attention.py": (
        "Bidirectional protein-ligand cross-attention with retained weights.",
        "Priority 2: protein-to-ligand and ligand-to-protein are not "
        "interchangeable transposes; keep both, with documented axes/masks.",
    ),
    "heads/affinity.py": (
        "Separate pKd / pKi / pIC50 heads with endpoint masks.",
        "Priority 2: an unmeasured endpoint must never receive a fabricated "
        "zero label.",
    ),
    "heads/delta.py": (
        "Analog delta prediction: difference baseline and antisymmetric head.",
        "Upgrade A: enforce zero change for A=A and antisymmetry under "
        "reversal; the dedicated head is an ablation, not the default.",
    ),
    "heads/retrieval.py": (
        "Independent protein and ligand retrieval projections.",
        "Priority 2: computable without knowing the other member of a pair; "
        "do not index pair-conditioned cross-attention embeddings.",
    ),
    "ibam/export.py": (
        "Raw attention matrices plus plotting metadata.",
        "Priority 2: export numeric matrices, not just images; residue x "
        "token/bond and derived residue x atom views; bond tokens split "
        "equally between endpoints and labelled as a derived aggregation.",
    ),
    "ibam/plots.py": (
        "Colorblind-safe heatmaps, vector export, zoomed residue windows.",
        "Priority 2: do not draw knowledge/virtual nodes as physical atoms; "
        "disclose any visualization renormalization.",
    ),
    "retrieval/index.py": (
        "Versioned approximate-nearest-neighbour index over ligand vectors.",
        "Priority 2: measure ANN recall against exact search; report losses "
        "from both approximation and shortlisting.",
    ),
    "retrieval/fishing.py": (
        "Target fishing and seed-assisted analog fishing APIs.",
        "Priority 5: report target-only and seed-assisted fishing separately; "
        "seed-assisted evaluation is not zero-shot discovery.",
    ),
    "generation/digress/noise.py": (
        "Categorical forward noising over atom and bond categories.",
        "Priority 3: a real discrete process, not a Gaussian scheduler "
        "applied to categorical graphs.",
    ),
    "generation/digress/denoiser.py": (
        "Conditional denoiser over noisy graph tensors.",
        "Priority 3: target and seed conditioning enter via cross-attention; "
        "the clean product representation must never be a condition.",
    ),
    "generation/digress/sampler.py": (
        "Reverse sampler with hard scaffold constraints.",
        "Priority 3: scaffold atom identities, internal bonds and "
        "stereochemistry stay fixed through corruption and sampling.",
    ),
    "generation/digress/decode.py": (
        "Graph tensors to a sanitized RDKit molecule and canonical SMILES.",
        "Priority 3: report validity before and after repair; identify newly "
        "created unspecified stereochemistry instead of inventing it.",
    ),
    "generation/legacy_latent.py": (
        "Quarantined legacy Gaussian DDPM over atom feature vectors.",
        "Priority 3: retained only as a labelled ablation. It has no "
        "molecular decoder and holds bond structure fixed, so it cannot be "
        "presented as analog generation.",
    ),
    "uncertainty/ensemble.py": (
        "Independently seeded ensemble inference.",
        "Priority 4: three independently trained models; dropout passes are "
        "not an ensemble.",
    ),
    "uncertainty/conformal.py": (
        "Split-conformal calibration, endpoint-specific.",
        "Priority 4: return an insufficient-calibration status rather than a "
        "fabricated finite interval when the order statistic is unsupported.",
    ),
    "uncertainty/domain.py": (
        "Applicability-domain distance against training entities only.",
        "Priority 4: return an OOD flag independently of ensemble spread.",
    ),
    "training/engine.py": (
        "Training loop: EMA, AMP, ensembling, resumable checkpoints.",
        "Port of the legacy engine, single-GPU only, with composed losses.",
    ),
    "training/losses/affinity.py": (
        "Huber for exact labels plus one-sided penalties for bounds.",
        "Priority 1: do not train against a bound as if it were exact; report "
        "exact-label metrics separately from bound violations.",
    ),
    "training/losses/ranking.py": (
        "Within-assay pairwise ranking.",
        "Priority 2: compare only within a compatible target/endpoint/assay "
        "group, with a configurable p-unit margin.",
    ),
    "training/losses/retrieval.py": (
        "Listwise objective over measured candidates.",
        "Priority 2: do not treat every unmeasured pair as a nonbinder; a "
        "weaker measured ligand is not automatically a nonbinder.",
    ),
    "training/checkpoints.py": (
        "Checkpoint metadata with code/config/data/split/encoder hashes.",
        "Deliverables: checkpoint metadata containing all provenance hashes.",
    ),
    "eval/metrics.py": (
        "Affinity, ranking, retrieval and calibration metrics.",
        "Priority 5: report per-endpoint and per-split; say when a metric is "
        "undefined rather than emitting a placeholder.",
    ),
    "eval/benchmarks/activity_cliff.py": (
        "Locked activity-cliff panel.",
        "Upgrade B: hold out both members and their chemical series; assign "
        "diagnostic categories only after predictions are frozen.",
    ),
    "eval/benchmarks/selectivity.py": (
        "Target / countertarget selectivity panel.",
        "Upgrade B: missing countertarget measurements are unknown, not "
        "negative.",
    ),
    "eval/benchmarks/temporal.py": (
        "Temporal holdout panel.",
        "Upgrade B: a recent database import of an old measurement is not new "
        "evidence; exclude unknown-date records from temporal claims.",
    ),
    "eval/benchmarks/fishing.py": (
        "Locked drug-fishing and analog-fishing challenge.",
        "Priority 5: computational decoys are presumed controls with unknown "
        "binding status and must never acquire fabricated affinity labels.",
    ),
    "prospective/manifest.py": (
        "Blinded candidate manifests with a separate assignment key.",
        "Upgrade C: freeze model version, candidate universe, budget, ranking "
        "rule and selection date before inspecting new measurements.",
    ),
    "prospective/ingest.py": (
        "Experimental result ingestion and QC.",
        "Upgrade C: keep identity, assay, binding, functional and failure "
        "records separate; a technical failure is not a negative result.",
    ),
    "prospective/report.py": (
        "Per-round prospective reports.",
        "Upgrade C: separate planned, selected, obtained, assayed and "
        "confirmed status. Never present a retrospective refit as the original "
        "prospective prediction.",
    ),
}

TEMPLATE = '''"""{summary}

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    {clause}
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "{flat_clause}"
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{{what}} is scaffolded but not implemented.\\n"
        f"Specification requirement: {{_SPEC_CLAUSE}}\\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
'''


def wrap_clause(clause: str, width: int = 76, indent: str = "    ") -> str:
    """Wrap a clause for the docstring body."""
    words = clause.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) + len(indent) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return f"\n{indent}".join(lines)


def main() -> int:
    created: list[str] = []
    skipped: list[str] = []

    for relative, (summary, clause) in sorted(MODULES.items()):
        path = PACKAGE_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)

        # Make every intermediate directory a package.
        for parent in list(path.parents):
            if parent == PACKAGE_ROOT.parent:
                break
            init = parent / "__init__.py"
            if parent.is_relative_to(PACKAGE_ROOT) and not init.exists():
                init.write_text("", encoding="utf-8")

        if path.exists():
            skipped.append(relative)
            continue

        path.write_text(
            TEMPLATE.format(
                summary=summary,
                clause=wrap_clause(clause),
                flat_clause=clause.replace('"', "'"),
            ),
            encoding="utf-8",
        )
        created.append(relative)

    print(f"created {len(created)} module(s), left {len(skipped)} existing file(s) alone")
    for relative in created:
        print(f"  + bindra/{relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
