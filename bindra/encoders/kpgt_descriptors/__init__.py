"""Vendored RDKit 2D normalized descriptors: KPGT's 200-dim knowledge input.

**This is third-party code, copied rather than written.** Source:
``lihan97/KPGT`` at ``src/data/descriptors/``, Apache License 2.0, whose text is
in ``LICENSE`` beside this file. KPGT in turn derives it from
``bp-kelley/descriptastorus``.

## Why vendored rather than depended on

KPGT's second knowledge node receives these 200 numbers, and the pretrained
weights were fitted against *this* implementation. The normalization is a CDF
against distribution parameters fitted on ChEMBL and frozen in ``dists.py``, so a
different version of descriptastorus -- or a differently fitted set -- would feed
the pretrained encoder inputs it has never seen, with no error and no obvious
symptom. Pinning the exact code is the only way to keep that honest.

``tests/test_ligand_kpgt.py::TestVendoredDescriptors`` checks the vendored copy
against the original running in the CPU-only DGL environment, so "exact" is
measured rather than asserted.

## Changes made to the upstream files

Three, all mechanical, none touching a computed value:

1. ``DescriptorGenerator.py`` imported ``pandas_flavor`` solely to register one
   dataframe convenience method. Replaced with a no-op decorator, dropping the
   dependency; nothing in the descriptor path used it.
2. ``scipy.stats.gilbrat`` was renamed ``gibrat`` in SciPy 1.11. Four entries in
   ``dists.py`` name the old spelling. Aliased below rather than edited, so the
   data file stays byte-identical to upstream.
3. ``np.bool`` / ``np.float`` / ``np.int`` were removed in NumPy 1.24 and are
   used by ``GetColumns``. Aliased below to the builtins they were aliases for.

The shims are installed here, before the submodules import, because the broken
names are read at *module import time*.

## The seventeen unnormalized descriptors

``rdNormalizedDescriptors`` logs one root-logger warning per RDKit descriptor
that has no fitted CDF in the frozen ``dists.py``, at import time. There are
seventeen, they are expected, and nothing downstream can act on them -- so as
warnings they were pure noise, emitted every time anything imported the KPGT
encoder, and read by nobody.

They are captured here into :data:`UNNORMALIZED_DESCRIPTORS` instead of being
printed. That keeps the information and makes it testable: the set is a
property of the installed RDKit crossed with a data file that is pinned, so an
RDKit upgrade that adds a descriptor changes it. ``TestUnnormalizedDescriptorsArePinnedNotPrinted`` asserts the expected set,
which turns a warning nobody reads into a failure somebody must look at.
The capture is scoped to the vendored import and removed in a ``finally``, so
no other warning is suppressed.
"""

from __future__ import annotations

import logging as _logging

import numpy as _numpy
import scipy.stats as _stats

# -- compatibility shims, installed before the vendored modules load ----------
if not hasattr(_stats, "gilbrat"):  # SciPy >= 1.11
    _stats.gilbrat = _stats.gibrat
# `np.object` is excluded on purpose: NumPy still resolves it through a
# __getattr__ that emits a FutureWarning, so probing it would fire the warning
# this shim exists to avoid, and nothing in the vendored code uses it.
for _name, _builtin in (("bool", bool), ("float", float), ("int", int)):
    if _name not in _numpy.__dict__:  # NumPy >= 1.24; avoids the deprecation hook
        setattr(_numpy, _name, _builtin)


class _CaptureUnnormalized(_logging.Filter):
    """Collect the import-time "No normalization for X" records, emit none."""

    PREFIX = "No normalization for"

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def filter(self, record: _logging.LogRecord) -> bool:
        if str(record.msg).startswith(self.PREFIX) and record.args:
            self.names.append(str(record.args[0]))
            return False  # swallow it; it is recorded below instead
        return True


_capture = _CaptureUnnormalized()
_logging.root.addFilter(_capture)
try:
    from bindra.encoders.kpgt_descriptors.rdNormalizedDescriptors import (  # noqa: E402
        RDKit2DNormalized,
    )
finally:
    _logging.root.removeFilter(_capture)

#: RDKit descriptors with no fitted normalization in the pinned ``dists.py``.
#: They return 0.0 from ``applyNormalizedFunc``, which is upstream behaviour and
#: is what the pretrained KPGT weights were fitted against -- so this is a fact
#: to pin, not a defect to fix.
UNNORMALIZED_DESCRIPTORS: tuple[str, ...] = tuple(_capture.names)

#: Width of the descriptor vector KPGT consumes, excluding the leading
#: "was it calculable" flag that :meth:`RDKit2DNormalized.process` prepends.
N_DESCRIPTORS = 200

__all__ = ["N_DESCRIPTORS", "UNNORMALIZED_DESCRIPTORS", "RDKit2DNormalized"]
