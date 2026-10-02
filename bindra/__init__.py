"""BINDRA — Binding INference, Design, Retrieval, and Attention.

A sequence-based platform for affinity prediction, attention-based explanation
(iBAM), target-conditioned retrieval, and diffusion-based analog generation.

The specification this package implements is ``BINDRA_Agent_Instructions.md``
at the repository root. ``docs/STATUS.md`` records which parts are actually
built; interfaces existing is not the same as functionality working.
"""

from __future__ import annotations

import os

# Must happen before anything creates a CUDA context, so that device indices
# follow PCI bus order rather than CUDA's own heuristic ordering. See
# bindra.runtime.device for why this machine in particular cares.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
