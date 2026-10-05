# The KPGT / DGL isolated environment

**Status: built, CPU only, and it has now done its job.** It served as the
numerical oracle for the pure-PyTorch port in `bindra/encoders/kpgt_light.py`
and `bindra/encoders/kpgt_graph.py`, which is verified against it stage by stage
(`tests/test_kpgt_port.py`) and runs on the RTX 5060 Ti. **KPGT is no longer
blocked.** This environment is kept only for re-verifying the port if either
side changes.

> **Correction (2026-10-02).** An earlier version of this file claimed "this
> machine's GPU is compute capability 12.0" and concluded DGL could not run
> here at all. That was wrong in one respect: the machine has **two** CUDA
> devices, and DGL's cu121 build *would* run on the second one.
> See "What was measured" below.

This directory holds the recipe, not the environment. The environment itself
lives at `C:\bindra\envs\kpgt`, deliberately outside OneDrive (see
"Why it lives outside the repo" below).

## What was measured, and what it rules out

The blocker previously recorded in `docs/STATUS.md` was that DGL "stops at
Python 3.12 / no longer ships Windows CUDA wheels". That was wrong in its
specifics and right in its conclusion. The measured situation:

| Fact | Evidence |
|---|---|
| DGL's final release is **2.2.1**, uploaded 2024-05-13 | PyPI release index; nothing newer exists as of 2026-10-02 |
| 2.2.1 **does** ship `cp312-cp312-win_amd64` | it is the only 2.x release that does; 2.0.0/2.1.0/2.2.0 do not |
| 2.2.1's `graphbolt` extension is compiled **per torch version** | the wheel ships `graphbolt_pytorch_{2.1.0,2.1.1,2.1.2,2.2.0,2.2.1,2.2.2,2.3.0}.dll` and loads by exact match |
| So the newest torch DGL can import under is **2.3.0** | torch 2.4.1 fails with `FileNotFoundError: ... graphbolt_pytorch_2.4.1.dll` |
| torch 2.3.0 is built against **CUDA 12.1** | no `sm_120` in its arch list |
**This machine has two CUDA devices**, which the first version of this file
missed, and the two tools order them *oppositely*:

| | torch index | nvidia-smi index | capability | memory |
|---|---|---|---|---|
| RTX 5060 Ti (Blackwell) | 1 | 0 | **12.0** | 16.0 GiB |
| T1200 Laptop (Turing) | 0 | 1 | **7.5** | 4.0 GiB |

The index order is also **not stable between runs**: two invocations in one
session reported the 5060 Ti at `cuda:0` and then at `cuda:1`. Selecting a
device by index is therefore unsafe here. `bindra/runtime/device.py` already
selects by requirement rather than index, and callers must go through it — an
ad-hoc benchmark written during this work bypassed it and silently measured the
4 GiB card.

So the accurate conclusion is narrower than the one first recorded:

* DGL **cannot** run on the RTX 5060 Ti. torch 2.3.0+cu121 has no `sm_120`
  kernels, DGL's last release predates Blackwell, and no future wheel changes that.
* DGL **could** run on the T1200, whose `sm_75` is supported by CUDA 12.1.

The second option was rejected on its merits rather than on impossibility: it
would confine a 92M-parameter encoder to a 4 GiB card a generation and a half
older than the one the project is built around, while permanently binding the
codebase to a dependency whose final release was May 2024. Porting cost one
session and removed the dependency entirely.

This is why `bindra/encoders/*` must not import DGL, and why the ligand encoder
sits behind the abstraction in `bindra/encoders/ligand.py`.

## What this environment is for

Exactly one thing: running reference DGL code on CPU to produce **expected
output tensors** that a pure-PyTorch port is then checked against. A port of a
published architecture that is not numerically verified against its reference
is a guess, and this environment is what turns that guess into a test.

That job is done. `tools/kpgt_oracle/make_reference.py` runs here and dumps the
graph, the weights and ten staged intermediates; `tests/test_kpgt_port.py`
checks the port against them. Measured agreement, max absolute deviation:

| | small config | base config (768d, 12 layers, 12 heads) |
|---|---|---|
| end to end, port's own featurizer | 4.8e-08 | 6.9e-06 (relative 1.1e-06) |
| on the RTX 5060 Ti | 7.2e-06 | — |

That is float32 reassociation, not a logic difference: the port replaces DGL's
fused kernels with segment operations, which sum in a different order. The base
state dict loads with `strict=True` and zero missing or unexpected keys, so the
published checkpoint will load unchanged.

Regenerate with:

```bash
# KPGT_REF points at a clone of https://github.com/lihan97/KPGT, a separate
# repository under its own licence, so it is not vendored here. Run from the
# repository root.
export KPGT_REF=C:/bindra/kpgt-ref
/c/bindra/envs/kpgt/Scripts/python.exe tools/kpgt_oracle/make_reference.py \
    C:/bindra/oracle/reference.npz small
/c/bindra/envs/kpgt/Scripts/python.exe tools/kpgt_oracle/make_reference.py \
    C:/bindra/oracle/reference_base.npz base
```

A second copy of this script lived at `C:/bindra/oracle/make_reference.py` with
the reference checkout hardcoded, and these instructions used to point at it.
The committed script takes `KPGT_REF` and exits with an explanation when the
clone is missing, so it is the only one to use.

The base artifact is 327 MB, so neither is committed and the tests skip when
they are absent.

It is never imported by `bindra` itself. Nothing in the main environment depends
on it, and it holds no CUDA build.

## Dependency pins, and why each is forced

DGL 2.2.1 predates several breaking changes in its own dependencies, so a plain
`uv pip install dgl==2.2.1` installs a broken environment. Each pin below was
added in response to an actual import failure:

| Pin | Failure it fixes |
|---|---|
| `torch==2.3.0` | newest torch with a matching `graphbolt_pytorch_*.dll` |
| `setuptools<72` | `ModuleNotFoundError: No module named 'setuptools.extern'` — DGL imports a private path removed in setuptools 72 |
| `torchdata==0.7.1` | `ModuleNotFoundError: No module named 'torchdata.datapipes'` — the datapipes API was removed in torchdata 0.8 |
| `pyyaml` | `ModuleNotFoundError: No module named 'yaml'` — an undeclared dependency of `graphbolt.impl.ondisk_dataset` |
| `pydantic` | undeclared, same path |
| `numpy<2` | torch 2.3.0 is a pre-NumPy-2 build |
| `pandas<2.3`, `scipy<1.14` | contemporaneous with torch 2.3.0; the resolver otherwise pulls pandas 3.x |

The exact resolved set is in `requirements.lock.txt`.

## Why it lives outside the repo

`envs/kpgt/.venv` under OneDrive failed three separate ways:

1. `uv` hardlinking into a OneDrive-backed path fails with `os error 396`.
   `UV_LINK_MODE=copy` is required and is set in the build command below.
2. A partially-synced install left `psutil==7.2.2` with a missing `METADATA`,
   which made the environment un-repairable in place.
3. Deleting it was not possible — both `rm -rf` and
   `Remove-Item -Recurse -Force` failed on
   `torch/include/ATen/ops` ("directory is not empty"), because OneDrive holds
   handles on files inside deep trees. Recovery required moving the tree out of
   OneDrive first and deleting it there.

A multi-gigabyte torch install also has no business being synced to the cloud.

## Build

```bash
export UV_LINK_MODE=copy
uv venv --python 3.12 /c/bindra/envs/kpgt
uv pip install --python /c/bindra/envs/kpgt/Scripts/python.exe \
    -r envs/kpgt/requirements.lock.txt
```

## Verify

```bash
/c/bindra/envs/kpgt/Scripts/python.exe -W ignore -c "
import dgl, torch, dgl.function as fn
g = dgl.graph(([0,1,2],[1,2,0]), num_nodes=3)
g.ndata['h'] = torch.arange(12, dtype=torch.float32).reshape(3,4)
g.update_all(fn.copy_u('h','m'), fn.sum('m','out'))
assert g.ndata['out'].tolist() == [[8,9,10,11],[0,1,2,3],[4,5,6,7]]
assert dgl.batch([g, g]).num_nodes() == 6
print('ok', dgl.__version__, torch.__version__)
"
```

Expected: `ok 2.2.1 2.3.0+cpu`. A `SyntaxWarning: invalid escape sequence '\P'`
from `dgl/backend/backend.py` is harmless and comes from a docstring in DGL
itself; `-W ignore` suppresses it.
