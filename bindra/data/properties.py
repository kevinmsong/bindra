"""Cached molecular properties for the curated ligand set.

Used for decoy property matching in the fishing panel, and later for the
feasibility flags the specification asks for on generated analogs. Built once and
cached for the same reason fingerprints are: the work is per molecule, not per
observation, and recomputing it in three places invites three slightly different
answers.

The descriptor set is deliberately small and conventional -- molecular weight,
cLogP, topological polar surface area, formal charge, rotatable bonds, heavy atoms,
and hydrogen-bond donors/acceptors. These are the axes DUD-E-style decoy matching
uses, and pinning them matters because "property-matched" means nothing without
saying matched on what.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Columns the store provides, in a fixed order. Part of the cache identity.
PROPERTY_COLUMNS = (
    "molecular_weight",
    "clogp",
    "tpsa",
    "formal_charge",
    "rotatable_bonds",
    "heavy_atoms",
    "hbd",
    "hba",
    "rings",
)


class PropertyError(RuntimeError):
    """Properties could not be computed or loaded."""


def compute_properties(smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(values, valid)`` for a chunk of SMILES.

    An unparseable molecule yields a row of NaN and ``valid=False`` rather than an
    exception, so one bad entry cannot lose a batch -- but it is never treated as a
    molecule with real properties.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    values = np.full((len(smiles), len(PROPERTY_COLUMNS)), np.nan, dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)

    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is None:
            continue
        try:
            values[row] = (
                Descriptors.MolWt(molecule),
                Crippen.MolLogP(molecule),
                rdMolDescriptors.CalcTPSA(molecule),
                Chem.GetFormalCharge(molecule),
                rdMolDescriptors.CalcNumRotatableBonds(molecule),
                molecule.GetNumHeavyAtoms(),
                rdMolDescriptors.CalcNumHBD(molecule),
                rdMolDescriptors.CalcNumHBA(molecule),
                rdMolDescriptors.CalcNumRings(molecule),
            )
            valid[row] = True
        except Exception:  # noqa: BLE001 - a descriptor failure is a skipped molecule
            continue
    return values, valid


@dataclass(frozen=True)
class PropertyStore:
    identity_keys: tuple[str, ...]
    values: np.ndarray  # [n, len(PROPERTY_COLUMNS)] float32
    valid: np.ndarray  # [n] bool
    columns: tuple[str, ...] = PROPERTY_COLUMNS

    @property
    def index(self) -> dict[str, int]:
        """Key -> row, built once and memoised. See FingerprintStore.index."""
        cached = self.__dict__.get("_index_cache")
        if cached is None:
            cached = {key: position for position, key in enumerate(self.identity_keys)}
            object.__setattr__(self, "_index_cache", cached)
        return cached

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.columns.index(name)]

    def to_frame(self):
        import pandas as pd

        frame = pd.DataFrame(self.values, columns=list(self.columns))
        frame.insert(0, "identity_key", list(self.identity_keys))
        frame["valid"] = self.valid
        return frame

    def to_meta(self) -> dict[str, Any]:
        return {
            "n_molecules": len(self.identity_keys),
            "n_valid": int(self.valid.sum()),
            "columns": list(self.columns),
        }


def build_property_store(
    processed_dir: Path | str,
    *,
    workers: int = 14,
    chunk_size: int = 5_000,
    rebuild: bool = False,
) -> PropertyStore:
    """Build (or load) molecular properties for every standardized ligand."""
    import pandas as pd

    directory = Path(processed_dir) / "properties"
    meta_path = directory / "meta.json"

    if meta_path.exists() and not rebuild:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if tuple(meta.get("columns", ())) == PROPERTY_COLUMNS:
            keys = json.loads((directory / "keys.json").read_text(encoding="utf-8"))
            return PropertyStore(
                identity_keys=tuple(keys),
                values=np.load(directory / "values.npy", mmap_mode="r"),
                valid=np.load(directory / "valid.npy"),
            )
        # The descriptor set changed, so stored values mean something different.

    ligands = pd.read_parquet(
        Path(processed_dir) / "ligands", columns=["identity_key", "standardized_smiles"]
    )
    ligands = ligands.dropna(subset=["identity_key", "standardized_smiles"])
    ligands = ligands.drop_duplicates("identity_key").reset_index(drop=True)
    keys = ligands["identity_key"].tolist()
    smiles = ligands["standardized_smiles"].tolist()

    chunks = [smiles[i : i + chunk_size] for i in range(0, len(smiles), chunk_size)]
    if workers <= 1:
        results = [compute_properties(c) for c in chunks]
    else:
        from joblib import Parallel, delayed

        results = list(
            Parallel(n_jobs=workers, batch_size=1)(
                delayed(compute_properties)(c) for c in chunks
            )
        )

    values = np.concatenate([r[0] for r in results], axis=0)
    valid = np.concatenate([r[1] for r in results], axis=0)

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "values.npy", values)
    np.save(directory / "valid.npy", valid)
    (directory / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    store = PropertyStore(identity_keys=tuple(keys), values=values, valid=valid)
    meta_path.write_text(
        json.dumps(store.to_meta(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return store


__all__ = [
    "PROPERTY_COLUMNS",
    "PropertyError",
    "PropertyStore",
    "build_property_store",
    "compute_properties",
]
