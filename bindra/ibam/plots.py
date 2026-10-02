"""Colorblind-safe iBAM figures over an exported :class:`IBAMExport`.

Owed by the specification:
    Colorblind-safe sequential colors, readable residue/atom labels, vector
    PDF/SVG exports, and an optional 600-dpi PNG. Provide a full matrix export
    plus zoomed residue windows for long targets.

Every figure is drawn from an :class:`~bindra.ibam.export.IBAMExport`, never from
a live model. That is deliberate: the numeric export is the artifact of record, so
any figure in a paper can be re-derived from the file that produced it, and a
figure can never show something the export does not contain.

## Colour

``viridis`` only, and sequential only. Attention is a non-negative magnitude with
no meaningful midpoint, so a diverging map (``coolwarm``, ``RdBu``) would invent
one and make an arbitrary value look like a neutral baseline. Viridis is
perceptually uniform and legible in deuteranopia, protanopia and greyscale. The
colormap is not configurable, because "configurable" in practice means someone
eventually picks ``jet``, whose luminance is non-monotonic -- it creates visible
bands where the data is smooth, which is a way of seeing structure that is not there.

## Labelling and the disclaimer

Every figure carries the export's disclaimer in its caption. The specification
forbids presenting iBAM as a contact map or a mechanism, and a figure separated
from its caption is how that happens, so the text is burned into the figure
itself rather than left to the surrounding document.

The aggregation rule and its hash also appear on every figure, so two figures
made under different policies cannot be confused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from bindra.ibam.export import IBAMExport

#: Not configurable. See the module docstring.
COLORMAP = "viridis"

#: Above this many residues a full matrix is unreadable and a window is required.
READABLE_RESIDUE_LIMIT = 200
#: Above this many residues tick labels are thinned rather than overlapping.
LABEL_LIMIT = 60


class IBAMPlotError(ValueError):
    """The figure could not be drawn from this export."""


@dataclass(frozen=True)
class FigureStyle:
    """Sizes and output formats. Colour is deliberately absent."""

    width: float = 10.0
    height: float = 6.0
    #: Vector formats are the deliverable; PNG is for slides.
    vector_formats: tuple[str, ...] = ("pdf", "svg")
    png_dpi: int = 600
    write_png: bool = False
    font_size: int = 8

    def __post_init__(self) -> None:
        for fmt in self.vector_formats:
            if fmt not in ("pdf", "svg", "eps"):
                raise IBAMPlotError(f"{fmt!r} is not a vector format")
        if self.png_dpi < 72:
            raise IBAMPlotError(f"png_dpi {self.png_dpi} is below screen resolution")


def _caption(export: IBAMExport, extra: str = "") -> str:
    """The text that must not be separated from the figure."""
    parts = [
        f"{export.molecule_id} | {export.ligand_encoder} + {export.protein_encoder}"
        f" | checkpoint {export.model_checkpoint}",
        f"aggregation: {export.policy.describe()} [{export.policy.policy_hash}]",
        export.disclaimer,
    ]
    if export.renormalized:
        parts.insert(
            2,
            f"physical tokens renormalised; {export.nonphysical_mass.sum():.4f} "
            "attention mass on non-physical tokens was set aside",
        )
    if extra:
        parts.insert(1, extra)
    return "\n".join(parts)


def _save(figure, stem: Path, style: FigureStyle) -> dict[str, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in style.vector_formats:
        path = stem.with_suffix(f".{fmt}")
        figure.savefig(path, bbox_inches="tight")
        written[fmt] = path
    if style.write_png:
        path = stem.with_suffix(".png")
        figure.savefig(path, dpi=style.png_dpi, bbox_inches="tight")
        written["png"] = path
    return written


def _thin(values: np.ndarray, limit: int) -> tuple[list[int], list[str]]:
    """Tick positions and labels, thinned so they stay readable."""
    count = len(values)
    if count <= limit:
        step = 1
    else:
        step = int(np.ceil(count / limit))
    positions = list(range(0, count, step))
    return positions, [str(int(values[i])) for i in positions]


def plot_residue_atom_heatmap(
    export: IBAMExport,
    path: Path | str,
    *,
    style: FigureStyle | None = None,
    view: str = "atom",
) -> dict[str, Path]:
    """Residue x atom (derived) or residue x token (native) heatmap.

    ``view="atom"`` is the derived aggregation and is labelled as such on the
    figure, not only in the metadata.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    style = style or FigureStyle()
    if view == "atom":
        matrix = export.residue_atom
        columns = export.atom_indices
        column_label = "atom index (standardized molecule)"
        title = "iBAM residue x atom - DERIVED AGGREGATION"
    elif view == "token":
        matrix = export.residue_token
        columns = np.arange(export.residue_token.shape[1])
        column_label = "ligand token index"
        title = "iBAM residue x token - native attention"
    else:
        raise IBAMPlotError(f"unknown view {view!r}; use 'atom' or 'token'")

    if matrix.shape[0] > READABLE_RESIDUE_LIMIT:
        raise IBAMPlotError(
            f"{matrix.shape[0]} residues will not be readable in one figure; use "
            f"IBAMExport.residue_window() to zoom, or plot_residue_profile() for an "
            f"overview of the whole sequence"
        )

    figure, axes = plt.subplots(figsize=(style.width, style.height))
    # vmin=0 rather than the data minimum, and this matters more than it looks.
    # Autoscaling stretches whatever range the data happens to occupy across the
    # full colormap, so a near-uniform attention map -- which is what an untrained
    # or uninformative model produces -- is rendered as vivid structure. Anchoring
    # at zero makes a flat map look flat, which is the honest reading.
    image = axes.imshow(matrix, aspect="auto", cmap=COLORMAP, vmin=0.0, origin="upper")
    bar = figure.colorbar(image, ax=axes)
    bar.set_label("attention weight", fontsize=style.font_size)

    row_positions, row_labels = _thin(export.residue_positions, LABEL_LIMIT)
    axes.set_yticks(row_positions)
    axes.set_yticklabels(row_labels, fontsize=style.font_size)
    column_positions, column_labels = _thin(columns, LABEL_LIMIT)
    axes.set_xticks(column_positions)
    axes.set_xticklabels(column_labels, fontsize=style.font_size, rotation=90)

    axes.set_ylabel("residue position in the original sequence", fontsize=style.font_size)
    axes.set_xlabel(column_label, fontsize=style.font_size)
    axes.set_title(title, fontsize=style.font_size + 2)
    figure.text(
        0.0, -0.08, _caption(export), fontsize=style.font_size - 1, va="top", wrap=True
    )
    try:
        return _save(figure, Path(path), style)
    finally:
        plt.close(figure)


def plot_residue_profile(
    export: IBAMExport,
    path: Path | str,
    *,
    style: FigureStyle | None = None,
    top: int = 10,
) -> dict[str, Path]:
    """Per-residue summary across the whole sequence, with the top residues named.

    The overview that remains readable for a 34,000-residue construct, where a
    full matrix cannot be.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    style = style or FigureStyle()
    figure, axes = plt.subplots(figsize=(style.width, style.height * 0.5))
    axes.fill_between(
        export.residue_positions, export.per_residue, color="#440154", alpha=0.85
    )
    if export.ensemble_std is not None:
        axes.fill_between(
            export.residue_positions,
            np.clip(export.per_residue - export.ensemble_std, 0.0, None),
            export.per_residue + export.ensemble_std,
            color="#21918c",
            alpha=0.35,
            label="ensemble +/- 1 SD",
        )
        axes.legend(fontsize=style.font_size)
    for position, score in export.top_residues(top):
        axes.annotate(
            str(position),
            (position, score),
            fontsize=style.font_size - 1,
            ha="center",
            va="bottom",
        )
    axes.set_xlabel("residue position in the original sequence", fontsize=style.font_size)
    axes.set_ylabel("summed attention", fontsize=style.font_size)
    axes.set_title("iBAM per-residue summary", fontsize=style.font_size + 2)
    axes.margins(x=0.01)
    figure.text(
        0.0, -0.18, _caption(export), fontsize=style.font_size - 1, va="top", wrap=True
    )
    try:
        return _save(figure, Path(path), style)
    finally:
        plt.close(figure)


def plot_molecule_contributions(
    export: IBAMExport,
    smiles: str,
    path: Path | str,
    *,
    size: tuple[int, int] = (600, 600),
) -> dict[str, Path]:
    """A molecular depiction with per-atom contributions, as SVG.

    Atom indices are mapped through ``atom_index_maps["std_to_output"]`` when it is
    present, because the depiction RDKit draws is ordered by the canonical SMILES
    output order, not by the standardized atom order the export uses. Drawing
    without that map puts the right numbers on the wrong atoms -- a figure that is
    confidently, invisibly wrong.

    Non-physical tokens are never drawn. Their mass appears in the caption instead.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise IBAMPlotError(f"RDKit could not parse {smiles!r}")
    if molecule.GetNumAtoms() != export.num_atoms:
        raise IBAMPlotError(
            f"{smiles!r} has {molecule.GetNumAtoms()} atoms but the export describes "
            f"{export.num_atoms}; these are not the same molecule"
        )

    contributions = export.per_atom.astype(float)
    highest = float(contributions.max()) if contributions.size else 0.0
    scaled = contributions / highest if highest > 0 else contributions

    colors: dict[int, tuple[float, float, float]] = {}
    radii: dict[int, float] = {}
    for standardized_index, weight in enumerate(scaled):
        drawn_index = standardized_index
        output_order = export.atom_index_maps.get("std_to_output")
        if output_order:
            if standardized_index >= len(output_order):
                raise IBAMPlotError(
                    "std_to_output is shorter than the molecule; the mapping does "
                    "not describe this export"
                )
            drawn_index = int(output_order[standardized_index])
        # Viridis endpoints, interpolated: low is dark purple, high is yellow.
        colors[drawn_index] = (
            0.267 + (0.993 - 0.267) * float(weight),
            0.005 + (0.906 - 0.005) * float(weight),
            0.329 + (0.144 - 0.329) * float(weight),
        )
        radii[drawn_index] = 0.2 + 0.4 * float(weight)

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=list(colors),
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
        highlightBonds=[],
    )
    drawer.FinishDrawing()

    target = Path(path).with_suffix(".svg")
    target.parent.mkdir(parents=True, exist_ok=True)
    svg = drawer.GetDrawingText()
    # The disclaimer travels inside the file, not only in a sidecar.
    comment = _caption(export, extra=f"max per-atom contribution {highest:.4f}")
    svg = svg.replace("<svg", f"<!--\n{comment}\n-->\n<svg", 1)
    target.write_text(svg, encoding="utf-8")
    return {"svg": target}


def plot_all(
    export: IBAMExport,
    directory: Path | str,
    *,
    smiles: str | None = None,
    style: FigureStyle | None = None,
    windows: Sequence[tuple[int, int]] = (),
) -> dict[str, dict[str, Path]]:
    """The full figure set, plus any requested zoomed residue windows.

    The matrix heatmap is skipped rather than drawn illegibly when the sequence is
    long; the per-residue profile always works, and windows are how a long target
    is inspected.
    """
    directory = Path(directory)
    style = style or FigureStyle()
    produced: dict[str, dict[str, Path]] = {
        "residue_profile": plot_residue_profile(export, directory / "residue_profile", style=style)
    }
    if export.sequence_length <= READABLE_RESIDUE_LIMIT:
        produced["residue_atom"] = plot_residue_atom_heatmap(
            export, directory / "residue_atom", style=style, view="atom"
        )
        produced["residue_token"] = plot_residue_atom_heatmap(
            export, directory / "residue_token", style=style, view="token"
        )
    for start, stop in windows:
        window = export.residue_window(start, stop)
        produced[f"window_{start}_{stop}"] = plot_residue_atom_heatmap(
            window, directory / f"window_{start}_{stop}", style=style, view="atom"
        )
    if smiles is not None:
        produced["molecule"] = plot_molecule_contributions(
            export, smiles, directory / "molecule"
        )
    return produced


__all__ = [
    "COLORMAP",
    "LABEL_LIMIT",
    "READABLE_RESIDUE_LIMIT",
    "FigureStyle",
    "IBAMPlotError",
    "plot_all",
    "plot_molecule_contributions",
    "plot_residue_atom_heatmap",
    "plot_residue_profile",
]
