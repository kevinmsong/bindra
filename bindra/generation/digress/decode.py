"""Back to a real molecule: RDKit decode, sanitise, scaffold re-check.

**Validity is reported before and after sanitisation, separately.** A generator
that emits 40% chemically valid graphs and is rescued to 95% by RDKit's
sanitiser has not learned chemistry; the sanitiser has. Collapsing the two into
one "validity" number is the most common way a discrete graph generator is made
to look better than it is, so both are returned and neither is called "validity"
on its own.

The scaffold check is by **atom mapping**, not by substructure match. A
substructure match asks whether the scaffold appears somewhere; the constraint
is that *these specific atoms* were never touched. Those differ exactly when the
model rebuilds an equivalent scaffold elsewhere in the molecule, which is a
failure wearing a success's face.

Owed by the specification:
    Priority 3: RDKit decode, sanitise, scaffold re-checked by atom mapping;
    validity reported before and after.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Category index 0 means "no bond" on the bond axis, and "no atom" on the atom
#: axis for a padded position. Both are a convention the featurizer and this
#: module must share; stated here so a change breaks loudly.
NO_BOND = 0


class DecodeError(ValueError):
    """A generated graph could not be turned into a molecule at all."""


@dataclass
class DecodedMolecule:
    """One decoded graph and the full story of how it decoded."""

    smiles: str | None
    valid_before_sanitize: bool
    valid_after_sanitize: bool
    sanitize_error: str | None = None
    scaffold_preserved: bool | None = None
    n_atoms: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.smiles is not None and self.valid_after_sanitize


def decode_graph(
    atom_types: "list[int] | object",
    bond_types: "object",
    atom_vocabulary: "list[str]",
    bond_vocabulary: "list[object]",
    *,
    scaffold_indices: "set[int] | None" = None,
    scaffold_elements: "dict[int, str] | None" = None,
) -> DecodedMolecule:
    """One graph to a molecule. Never raises for a bad molecule, only reports.

    A generator producing unparseable graphs is the normal case early in
    training, so an exception per bad sample would make the validity rate
    impossible to measure.
    """
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:  # pragma: no cover - rdkit is a hard dependency
        raise DecodeError("RDKit is required to decode generated graphs") from exc

    RDLogger.DisableLog("rdApp.*")
    import numpy as np

    atoms = np.asarray(atom_types).tolist()
    bonds = np.asarray(bond_types)
    notes: list[str] = []

    mol = Chem.RWMol()
    index_map: dict[int, int] = {}
    for position, category in enumerate(atoms):
        symbol = atom_vocabulary[int(category)] if int(category) < len(atom_vocabulary) else None
        if not symbol or symbol == "*":
            continue
        atom = Chem.Atom(symbol)
        # Carry the generated position through as an atom map number so the
        # scaffold check can be done by identity rather than by matching.
        atom.SetAtomMapNum(position + 1)
        index_map[position] = mol.AddAtom(atom)

    if not index_map:
        return DecodedMolecule(None, False, False, "no decodable atoms", notes=["empty graph"])

    for i in index_map:
        for j in index_map:
            if j <= i:
                continue
            category = int(bonds[i, j])
            if category == NO_BOND or category >= len(bond_vocabulary):
                continue
            mol.AddBond(index_map[i], index_map[j], bond_vocabulary[category])

    molecule = mol.GetMol()
    # "Before" means: does it parse as a molecule at all, without repair.
    before = True
    try:
        Chem.SanitizeMol(
            Chem.Mol(molecule), sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    except Exception as exc:  # noqa: BLE001 - an invalid molecule is data, not a crash
        before = False
        notes.append(f"pre-sanitize: {type(exc).__name__}")

    after, error, smiles = True, None, None
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:  # noqa: BLE001
        after, error = False, f"{type(exc).__name__}: {exc}"

    # The scaffold check reads the atom map numbers, so it runs BEFORE they are
    # stripped. Order matters in both directions.
    preserved = None
    if scaffold_indices is not None and after:
        preserved = _scaffold_by_atom_map(molecule, scaffold_indices, scaffold_elements)

    if after:
        # Strip the map numbers before emitting SMILES. They are internal
        # bookkeeping: left in, every molecule renders as "[CH3:1][CH2:2][OH:3]"
        # instead of "CCO", and worse, two copies of one molecule whose atoms
        # were generated in a different order get different strings -- which
        # would make the uniqueness rate in validity_report meaningless.
        emitted = Chem.Mol(molecule)
        for atom in emitted.GetAtoms():
            atom.SetAtomMapNum(0)
        try:
            smiles = Chem.MolToSmiles(emitted)
        except Exception as exc:  # noqa: BLE001
            after, error = False, f"{type(exc).__name__}: {exc}"

    return DecodedMolecule(
        smiles=smiles,
        valid_before_sanitize=before,
        valid_after_sanitize=after,
        sanitize_error=error,
        scaffold_preserved=preserved,
        n_atoms=len(index_map),
        notes=notes,
    )


def _scaffold_by_atom_map(molecule, scaffold_indices, scaffold_elements) -> bool:
    """Are the scaffold's original positions still present, and still themselves?

    By atom map number, set at construction. A rebuilt-elsewhere scaffold fails
    this and passes a substructure match, which is why this is not a match.
    """
    found = {}
    for atom in molecule.GetAtoms():
        number = atom.GetAtomMapNum()
        if number:
            found[number - 1] = atom.GetSymbol()
    for position in scaffold_indices:
        if position not in found:
            return False
        if scaffold_elements and found[position] != scaffold_elements.get(position, found[position]):
            return False
    return True


def validity_report(decoded: "list[DecodedMolecule]") -> dict[str, object]:
    """Aggregate, keeping the two validity rates apart.

    ``sanitizer_rescue_rate`` is the share that only became valid because RDKit
    repaired them. It is the number that says how much of the headline validity
    belongs to the model.
    """
    n = len(decoded)
    if not n:
        return {"n": 0, "status": "nothing decoded"}
    before = sum(d.valid_before_sanitize for d in decoded)
    after = sum(d.valid_after_sanitize for d in decoded)
    rescued = sum(d.valid_after_sanitize and not d.valid_before_sanitize for d in decoded)
    unique = {d.smiles for d in decoded if d.smiles}
    checked = [d for d in decoded if d.scaffold_preserved is not None]
    report: dict[str, object] = {
        "n": n,
        "valid_before_sanitize": before / n,
        "valid_after_sanitize": after / n,
        "sanitizer_rescue_rate": rescued / n,
        "note": (
            "valid_after_sanitize includes molecules RDKit repaired; "
            "sanitizer_rescue_rate is how many. The model's own validity is "
            "valid_before_sanitize."
        ),
        "unique_smiles": len(unique),
        "uniqueness": len(unique) / after if after else 0.0,
    }
    if checked:
        report["scaffold_preserved"] = sum(d.scaffold_preserved for d in checked) / len(checked)
        report["n_scaffold_checked"] = len(checked)
    else:
        report["scaffold_preserved"] = None
        report["n_scaffold_checked"] = 0
    return report


__all__ = [
    "NO_BOND",
    "DecodeError",
    "DecodedMolecule",
    "decode_graph",
    "validity_report",
]
