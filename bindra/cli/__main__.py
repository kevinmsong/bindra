"""``bindra`` command dispatcher.

Subcommands map onto the stages in the specification's implementation order.
Commands whose stage is not built yet exit with a clear message naming the
blocker, rather than appearing to work.
"""

from __future__ import annotations

import argparse
import importlib
import sys

#: name -> (module path, one-line help). Modules are imported lazily so that
#: `bindra doctor` works even when optional dependencies are missing.
COMMANDS: dict[str, tuple[str, str]] = {
    "doctor": ("bindra.cli.doctor", "Check that this machine can train on the GPU."),
    "prepare-data": ("bindra.cli.prepare_data", "Ingest and curate a BindingDB snapshot."),
    "make-splits": ("bindra.cli.make_splits", "Build immutable split manifests."),
    "cache": ("bindra.cli.cache", "Populate frozen encoder feature caches."),
    "train": ("bindra.cli.train", "Train the affinity / retrieval / diffusion stack."),
    "calibrate": ("bindra.cli.calibrate", "Fit conformal calibration on the held-out partition."),
    "predict": ("bindra.cli.predict", "Predict affinity for sequence/SMILES pairs."),
    "retrieve": ("bindra.cli.retrieve", "Target fishing and analog fishing."),
    "generate": ("bindra.cli.generate", "Generate scaffold-constrained analogs."),
    "plot-ibam": ("bindra.cli.plot_ibam", "Export iBAM matrices and heatmaps."),
    "evaluate": ("bindra.cli.evaluate", "Run the locked evaluation panels."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bindra",
        description="BINDRA — Binding INference, Design, Retrieval, and Attention.",
        epilog="See docs/STATUS.md for what is actually implemented.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="subcommand to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the subcommand")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        print("\nsubcommands:")
        for name, (_, description) in sorted(COMMANDS.items()):
            print(f"  {name:<14} {description}")
        return 0

    command, *rest = argv
    if command not in COMMANDS:
        print(f"bindra: unknown command {command!r}", file=sys.stderr)
        print(f"available: {', '.join(sorted(COMMANDS))}", file=sys.stderr)
        return 2

    module_path, _ = COMMANDS[command]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        print(
            f"bindra {command}: not implemented yet.\n"
            "See docs/STATUS.md for the current stage and its blockers.",
            file=sys.stderr,
        )
        return 3

    return int(module.main(rest))


if __name__ == "__main__":
    sys.exit(main())
