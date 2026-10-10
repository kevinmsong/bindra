"""``bindra generate``: train the analog generator, or sample from it.

Two subcommands:

``train``
    Fit the conditional discrete diffusion on analog pairs drawn from the
    **training partitions only**. A pair is two observations, so a pair touching
    val or test would carry a held-out molecule into the generator.

``sample``
    Draw scaffold-constrained analogs from a trained checkpoint and report what
    came back, with validity split into what the model produced and what RDKit
    repaired. A generator whose headline validity is mostly sanitiser rescues
    has not learned chemistry.

The generator is a **separate checkpoint from the affinity model**. It does not
need one to train, and conditioning on the affinity model's attention is a
later addition rather than a requirement, so the two can be trained in either
order.
"""

from __future__ import annotations

import json
from pathlib import Path

OK = "  ok   "
INFO = " info  "
WARN = " warn  "
FAIL = " FAIL  "


def cmd_train(args) -> int:
    import torch

    from bindra.training.generation import (
        GenerationConfig,
        GenerationTrainingError,
        build_triples,
        train,
    )

    split_directory = Path(args.splits) / args.regime
    if not (split_directory / "manifest.json").exists():
        print(f"[{FAIL}] no split at {split_directory}")
        return 1

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[{OK}] device {device}", flush=True)
    print(
        f"[{INFO}] building analog pairs from the {args.regime} training "
        "partition (one pass over the corpus)...",
        flush=True,
    )
    try:
        triples = build_triples(
            args.processed, split_directory,
            max_per_group=args.max_pairs_per_group,
            limit=args.limit, seed=args.seed,
        )
    except GenerationTrainingError as exc:
        print(f"[{FAIL}] {exc}")
        return 2
    print(f"[{OK}] {len(triples):,} analog pairs, train partition only", flush=True)

    config = GenerationConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        timesteps=args.timesteps, max_atoms=args.max_atoms,
        model_dim=args.model_dim, n_layers=args.layers, n_heads=args.heads,
        dropout=args.dropout, seed=args.seed, bond_weight=args.bond_weight,
    )
    out = Path(args.out) / f"{args.regime}-{args.tag}" if args.tag else Path(args.out) / args.regime
    try:
        payload = train(triples, config=config, out_directory=out, device=device)
    except GenerationTrainingError as exc:
        print(f"[{FAIL}] {exc}")
        return 2

    attrition = payload["attrition"]
    print(
        f"[{OK}] {attrition['kept']:,} of {attrition['input']:,} pairs featurized "
        f"({attrition['unencodable']:,} outside the atom vocabulary or size cap)",
        flush=True,
    )
    print(f"[{OK}] {payload['trainable_parameters']:,} trainable parameters")
    print(f"[{OK}] wrote {out / 'best.pt'} and {out / 'history.json'}")
    return 0


def cmd_sample(args) -> int:
    import numpy as np
    import torch

    from bindra.generation.digress.decode import decode_graph, validity_report
    from bindra.generation.digress.denoiser import DenoiserConfig, GraphDenoiser
    from bindra.generation.digress.graphs import (
        ATOM_VOCABULARY, BOND_VOCABULARY, N_ATOM_TYPES, N_BOND_TYPES,
        encode_molecule, scaffold_mask,
    )
    from bindra.generation.digress.noise import CategoricalMarginals, DiscreteGraphNoise
    from bindra.generation.digress.sampler import ScaffoldConstraint, sample, verify_scaffold

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"[{FAIL}] no checkpoint at {checkpoint}")
        return 1
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "bindra-analog-generator-v1":
        print(f"[{FAIL}] {checkpoint} is a {payload.get('kind')!r}, not an analog generator")
        return 2

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    config = payload["config"]
    model = GraphDenoiser(DenoiserConfig(
        n_atom_types=N_ATOM_TYPES, n_bond_types=N_BOND_TYPES,
        condition_dim=payload["condition_dim"], model_dim=config["model_dim"],
        n_layers=config["n_layers"], n_heads=config["n_heads"], dropout=0.0,
    ))
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    marginals = CategoricalMarginals(
        atom=payload["marginals"]["atom"], bond=payload["marginals"]["bond"]
    )
    noise = DiscreteGraphNoise(marginals, timesteps=config["timesteps"])
    max_atoms = config["max_atoms"]

    seed_graph = encode_molecule(args.seed, max_atoms=max_atoms)
    keep = scaffold_mask(args.seed, max_atoms=max_atoms)
    print(
        f"[{OK}] seed {args.seed}: {seed_graph.n_atoms} atoms, "
        f"{int(keep.sum())} frozen as scaffold",
        flush=True,
    )
    print(
        f"[{INFO}] requesting {args.delta:+.2f} log units on {args.endpoint}; "
        "this conditions the sampler, it does not verify the result",
        flush=True,
    )
    if not keep.any():
        print(f"[{WARN}] the seed has no Murcko scaffold, so nothing is held fixed")

    n = args.n
    atoms0 = torch.nn.functional.one_hot(
        torch.from_numpy(seed_graph.atoms), N_ATOM_TYPES
    ).float().unsqueeze(0).repeat(n, 1, 1).to(device)
    bonds0 = torch.nn.functional.one_hot(
        torch.from_numpy(seed_graph.bonds), N_BOND_TYPES
    ).float().unsqueeze(0).repeat(n, 1, 1, 1).to(device)
    keep_t = torch.from_numpy(keep).unsqueeze(0).repeat(n, 1).to(device)
    mask = torch.from_numpy(seed_graph.mask).unsqueeze(0).repeat(n, 1).to(device)
    constraint = ScaffoldConstraint(keep=keep_t, atoms=atoms0, bonds=bonds0)

    from bindra.heads.affinity import ENDPOINT_INDEX, ENDPOINT_ORDER

    endpoint = torch.zeros(n, len(ENDPOINT_ORDER), device=device)
    endpoint[:, ENDPOINT_INDEX.get(args.endpoint, 0)] = 1.0
    summary = torch.from_numpy(np.concatenate([
        np.bincount(seed_graph.atoms[seed_graph.mask], minlength=11).astype(np.float32),
        np.array([seed_graph.n_atoms], dtype=np.float32),
    ])).unsqueeze(0).repeat(n, 1).to(device)
    # The REQUESTED potency change. The model was trained on observed changes,
    # so this asks for an analog of that kind; it does not promise one. What
    # comes back is bounded by what the training distribution supports for this
    # scaffold and target, and the affinity model is what checks the claim.
    from bindra.training.generation import encode_delta

    delta = encode_delta(
        [args.delta] * n, [ENDPOINT_INDEX.get(args.endpoint, 0)] * n, device=device
    )
    conditions = torch.cat((endpoint, summary, delta), dim=-1).unsqueeze(1)

    generator = torch.Generator(device=device).manual_seed(args.seed_value)
    print(f"[{INFO}] sampling {n} analogs over {config['timesteps']} steps...", flush=True)
    out_atoms, out_bonds = sample(
        model, noise, conditions=conditions, atom_mask=mask,
        constraint=constraint, generator=generator, temperature=args.temperature,
    )
    print(f"[{OK}] scaffold check: {verify_scaffold(out_atoms, out_bonds, constraint)}")

    from rdkit import Chem

    bond_types = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                  Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    frozen = {int(i) for i in np.flatnonzero(keep)}
    decoded = [
        decode_graph(
            out_atoms[i].argmax(-1).cpu().numpy(),
            out_bonds[i].argmax(-1).cpu().numpy(),
            list(ATOM_VOCABULARY), bond_types,
            scaffold_indices=frozen,
        )
        for i in range(n)
    ]
    report = validity_report(decoded)
    print(f"\n  +-- {n} samples from {checkpoint.name}")
    print(f"  |   model's own validity   {report['valid_before_sanitize']:.1%}")
    print(f"  |   after RDKit sanitising {report['valid_after_sanitize']:.1%}"
          f"  (rescued {report['sanitizer_rescue_rate']:.1%})")
    print(f"  |   unique molecules       {report['unique_smiles']}")
    if report.get("scaffold_preserved") is not None:
        print(f"  |   scaffold preserved     {report['scaffold_preserved']:.1%}")
    for molecule in [d for d in decoded if d.usable][: args.show]:
        print(f"  |     {molecule.smiles}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": str(checkpoint), "seed": args.seed, "endpoint": args.endpoint,
        "n": n, "temperature": args.temperature, "requested_delta": args.delta,
        "report": report,
        "smiles": [d.smiles for d in decoded if d.usable],
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n[{OK}] wrote {out_path}")
    return 0


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="bindra generate",
        description="Train the analog generator, or sample scaffold-constrained analogs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Fit the conditional discrete diffusion.")
    t.add_argument("--processed", default="data/processed/202610")
    t.add_argument("--splits", default="data/splits/202610")
    t.add_argument("--regime", default="random_pair")
    t.add_argument("--out", default="runs/generator")
    t.add_argument("--tag", default="")
    t.add_argument("--device", default=None)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=1e-2)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--timesteps", type=int, default=500)
    t.add_argument("--max-atoms", type=int, default=64,
                   help="Molecules above this are skipped, never truncated: a "
                        "truncated molecule is a different molecule.")
    t.add_argument("--model-dim", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--bond-weight", type=float, default=0.5,
                   help="Bonds outnumber atoms by N, so an unweighted sum lets "
                        "the bond matrix dominate the objective.")
    t.add_argument("--max-pairs-per-group", type=int, default=32,
                   help="Cap per scaffold series, so one campaign cannot "
                        "dominate the gradient with a single chemotype.")
    t.add_argument("--limit", type=int, default=None, help="Cap pairs, for a smoke run.")
    t.add_argument("--seed", type=int, default=20261001)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("sample", help="Draw analogs from a trained generator.")
    s.add_argument("checkpoint")
    s.add_argument("--seed", required=True, help="Seed molecule, as SMILES.")
    s.add_argument("--endpoint", default="IC50", choices=("Kd", "Ki", "IC50"))
    s.add_argument(
        "--delta", type=float, default=1.0,
        help="Requested potency change in log units, positive for more potent. "
             "Bucketed against that endpoint's replicate-noise floor (0.945 Kd, "
             "0.641 Ki, 0.671 IC50), so a change under the floor reads as no "
             "real change. A request, not a guarantee: score what comes back "
             "with the affinity model.",
    )
    s.add_argument("-n", type=int, default=32)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--show", type=int, default=10)
    s.add_argument("--device", default=None)
    s.add_argument("--seed-value", type=int, default=20261001)
    s.add_argument("--out", default="runs/generated.json")
    s.set_defaults(func=cmd_sample)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


__all__ = ["build_parser", "cmd_sample", "cmd_train", "main"]
