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

## Readability

Tick labels carry chemical and biological identity when the caller supplies the
sequence and the SMILES: ``D381`` rather than ``380``, ``N7`` rather than ``7``.
Identity is not in the export because it is not model output -- it comes from the
inputs -- so it is passed alongside, exactly as ``plot_molecule_contributions``
already takes the SMILES separately. Residue labels are numbered **1-based** for
the biological convention while ``residue_positions`` is 0-based indexing into
the sequence; the axis label says which, and a position outside the sequence
raises rather than mislabelling, because an off-by-one in a residue number is
invisible in the figure and fatal in the interpretation.

## Head agreement

:func:`plot_head_agreement` exists because every other figure here shows a
head-mean. A mean over eight heads that disagree looks exactly like a mean over
eight heads that agree, and only one of those supports reading a peak as
"the model attends here". The figure shows the per-head distributions so the
reader can see which case they are in.
"""

from __future__ import annotations

import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, replace
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

#: Viridis endpoints, for hand-drawn elements that must match the colormap.
_DARK = "#440154"
_MID = "#21918c"
_LIGHT = "#5ec962"


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
    #: Sans-serif by default: it stays legible when a figure is scaled to a
    #: single column, where a serif face at 7 pt does not.
    font_family: str = "DejaVu Sans"
    #: How many top residues/atoms to name on summary figures.
    annotate_top: int = 12
    #: Marginal summary bars beside the heatmap. The matrix shows where the mass
    #: is jointly; the margins show each axis's total, which is what a reader
    #: tries to estimate by eye from the matrix and estimates badly.
    marginals: bool = True

    def __post_init__(self) -> None:
        for fmt in self.vector_formats:
            if fmt not in ("pdf", "svg", "eps"):
                raise IBAMPlotError(f"{fmt!r} is not a vector format")
        if self.png_dpi < 72:
            raise IBAMPlotError(f"png_dpi {self.png_dpi} is below screen resolution")


#: The deliverable preset: larger type, 600-dpi raster beside the vectors.
PUBLICATION = FigureStyle(
    width=11.0, height=7.0, font_size=9, write_png=True, png_dpi=600
)


def _rc(style: FigureStyle) -> dict[str, object]:
    """Typography and axis furniture for a figure that goes into a paper."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [style.font_family, "Arial", "Helvetica"],
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "axes.titlesize": style.font_size + 1,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": style.font_size - 1,
        "ytick.labelsize": style.font_size - 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.fontsize": style.font_size - 1,
        "lines.linewidth": 0.9,
        "figure.dpi": 150,
        # Keep text as text in the vector output: the disclaimer has to be
        # selectable and searchable, not a picture of words.
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }


@contextmanager
def _drawing(style: FigureStyle):
    """Matplotlib, configured, with the Agg backend and the rc applied."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(_rc(style)):
        yield plt


#: One-letter codes that are not a residue. Anything else is passed through, so
#: an unexpected character shows up in the label rather than being hidden.
_NON_RESIDUE = {"X", "-", "*"}


def residue_tick_labels(
    positions: np.ndarray, sequence: str | None, *, one_based: bool = True
) -> list[str]:
    """``[380, 381]`` -> ``['D381', 'L382']``, given the original sequence.

    Without a sequence the numbers are returned alone, so a caller that does not
    have the sequence still gets a correct figure rather than a guess. A position
    outside the sequence raises: a residue number that is quietly wrong by one is
    invisible in the figure and changes the interpretation completely.
    """
    offset = 1 if one_based else 0
    if sequence is None:
        return [str(int(p) + offset) for p in positions]
    labels: list[str] = []
    for position in positions:
        index = int(position)
        if not 0 <= index < len(sequence):
            raise IBAMPlotError(
                f"residue position {index} is outside a sequence of length "
                f"{len(sequence)}; these do not describe the same protein"
            )
        code = sequence[index].upper()
        labels.append(f"{'' if code in _NON_RESIDUE else code}{index + offset}")
    return labels


def atom_tick_labels(atom_indices: np.ndarray, smiles: str | None) -> list[str]:
    """``[0, 6]`` -> ``['C1', 'N7']``, given the standardized SMILES.

    Numbering is 1-based for display while ``atom_indices`` is 0-based, matching
    how the residue axis is labelled. Element symbols come from the molecule in
    *standardized* atom order, which is the order the export uses -- not the
    canonical output order the depiction is drawn in.
    """
    if smiles is None:
        return [str(int(i) + 1) for i in atom_indices]
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise IBAMPlotError(f"RDKit could not parse {smiles!r}")
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    labels: list[str] = []
    for index in atom_indices:
        position = int(index)
        if not 0 <= position < len(symbols):
            raise IBAMPlotError(
                f"atom index {position} is outside a molecule of {len(symbols)} "
                "atoms; these are not the same molecule"
            )
        labels.append(f"{symbols[position]}{position + 1}")
    return labels


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


def _draw_caption(figure, export: IBAMExport, style: FigureStyle, extra: str = "") -> None:
    """Lay the caption into a reserved strip under the axes.

    Pre-wrapped rather than left to ``wrap=True``, which measures against the
    figure box and clips under ``bbox_inches="tight"`` -- losing the end of the
    disclaimer, which is the one line that must survive.
    """
    width = max(60, int(style.width * 12))
    lines: list[str] = []
    for paragraph in _caption(export, extra).split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    # Placed *below* the figure box, not inside it. ``bbox_inches="tight"``
    # grows the canvas to include it, so the caption costs exactly the height
    # it needs; reserving space with ``subplots_adjust`` instead leaves a band
    # of white between the last panel and the text that scales with the figure.
    figure.text(
        0.01,
        -0.012,
        "\n".join(lines),
        fontsize=style.font_size - 1.5,
        va="top",
        ha="left",
        color="#333333",
        linespacing=1.35,
    )


def _save(
    figure, stem: Path, style: FigureStyle, export: IBAMExport | None = None
) -> dict[str, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    title = "BINDRA iBAM"
    description = "" if export is None else _caption(export)
    written: dict[str, Path] = {}
    for fmt in style.vector_formats:
        path = stem.with_suffix(f".{fmt}")
        if fmt == "pdf":
            metadata = {"Title": title, "Subject": description, "Creator": "BINDRA"}
        elif fmt == "svg":
            metadata = {"Title": title, "Description": description, "Creator": "BINDRA"}
        else:
            metadata = None
        # Provenance travels inside the file as well as on its face, so a figure
        # lifted into a slide deck still carries the aggregation rule.
        figure.savefig(path, bbox_inches="tight", metadata=metadata)
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


def _score_label(export: IBAMExport) -> str:
    """Axis label naming which direction the per-residue score came from."""
    if export.residue_scores_are_informative:
        return "ligand attention onto residue"
    return "summed attention (CONSTANT by construction)"


def _score_title(export: IBAMExport) -> str:
    if export.residue_scores_are_informative:
        return "ligand-to-protein attention per residue"
    return "per-residue attention - UNINFORMATIVE"


def _warn_uninformative(axes, export: IBAMExport, style: FigureStyle) -> None:
    """Say it on the figure when the per-residue score cannot distinguish residues.

    An export made without the ligand->protein direction has only
    ``per_residue``, whose every entry is 1.0 because the protein->ligand rows
    are softmax-normalised over atoms. Plotted, that is a flat band and a
    top-residue list that is argsort over ties -- which looks exactly like a
    model that attends evenly, rather than like a quantity that cannot vary.
    The difference matters, so the figure states it rather than letting the
    reader interpret a structural artifact.
    """
    if export.residue_scores_are_informative:
        return
    axes.text(
        0.5, 0.5,
        "per-residue totals are constant by construction\n"
        "(protein-to-ligand rows each sum to 1); this panel\n"
        "ranks ties. Re-export with the ligand-to-protein\n"
        "direction for a residue score that varies.",
        transform=axes.transAxes, ha="center", va="center",
        fontsize=style.font_size, color="#7f1d1d", linespacing=1.5,
        bbox={"facecolor": "#fef2f2", "edgecolor": "#dc2626", "linewidth": 0.8},
    )


def _annotate_peaks(axes, xs, ys, labels, style, span: float) -> int:
    """Name the peaks without letting the names overprint each other.

    Top residues cluster -- that is the interesting case, and it is exactly when
    a fixed two-row stagger produces ``S396N400`` as one unreadable word. Labels
    are placed into whichever stagger row is free at that x, and a label with no
    free row is dropped rather than drawn on top of another. Returns how many
    were actually drawn, so the caller can say so.
    """
    rows = 4
    # A label is roughly this wide in data units at this font size.
    clearance = span * 0.035 * max(1, style.font_size - 2) / 7.0
    occupied: list[list[float]] = [[] for _ in range(rows)]
    drawn = 0
    for x, y, label in zip(xs, ys, labels):
        for row in range(rows):
            if all(abs(x - taken) > clearance for taken in occupied[row]):
                occupied[row].append(x)
                axes.annotate(
                    label,
                    (x, y),
                    xytext=(0, 3 + 9 * row),
                    textcoords="offset points",
                    fontsize=style.font_size - 2,
                    ha="center",
                    va="bottom",
                    color=_DARK,
                )
                drawn += 1
                break
    return drawn


def _thin_labels(labels: Sequence[str], limit: int) -> tuple[list[int], list[str]]:
    """As :func:`_thin`, but over labels that are already formatted."""
    count = len(labels)
    step = 1 if count <= limit else int(np.ceil(count / limit))
    positions = list(range(0, count, step))
    return positions, [labels[i] for i in positions]


def plot_residue_atom_heatmap(
    export: IBAMExport,
    path: Path | str,
    *,
    style: FigureStyle | None = None,
    view: str = "atom",
    sequence: str | None = None,
    smiles: str | None = None,
) -> dict[str, Path]:
    """Residue x atom (derived) or residue x token (native) heatmap.

    ``view="atom"`` is the derived aggregation and is labelled as such on the
    figure, not only in the metadata.

    ``sequence`` and ``smiles`` are optional and only add identity to the tick
    labels -- ``D381`` instead of ``380``. They never change the matrix.
    """
    style = style or FigureStyle()
    if view == "atom":
        matrix = export.residue_atom
        columns = export.atom_indices
        column_label = "atom (standardized molecule, 1-based)"
        column_labels = atom_tick_labels(export.atom_indices, smiles)
        title = "iBAM residue x atom - DERIVED AGGREGATION"
    elif view == "token":
        matrix = export.residue_token
        columns = np.arange(export.residue_token.shape[1])
        column_label = "ligand token index"
        column_labels = [str(int(c)) for c in columns]
        title = "iBAM residue x token - native attention"
    else:
        raise IBAMPlotError(f"unknown view {view!r}; use 'atom' or 'token'")

    if matrix.shape[0] > READABLE_RESIDUE_LIMIT:
        raise IBAMPlotError(
            f"{matrix.shape[0]} residues will not be readable in one figure; use "
            f"IBAMExport.residue_window() to zoom, or plot_residue_profile() for an "
            f"overview of the whole sequence"
        )

    row_labels_all = residue_tick_labels(export.residue_positions, sequence)

    with _drawing(style) as plt:
        figure = plt.figure(figsize=(style.width, style.height))
        if style.marginals:
            grid = figure.add_gridspec(
                2, 3,
                width_ratios=(1.0, 0.13, 0.035),
                height_ratios=(0.13, 1.0),
                wspace=0.04, hspace=0.04,
            )
            axes = figure.add_subplot(grid[1, 0])
            top = figure.add_subplot(grid[0, 0], sharex=axes)
            side = figure.add_subplot(grid[1, 1], sharey=axes)
            cax = figure.add_subplot(grid[1, 2])
        else:
            axes = figure.add_subplot(1, 1, 1)
            top = side = cax = None

        # vmin=0 rather than the data minimum, and this matters more than it looks.
        # Autoscaling stretches whatever range the data happens to occupy across the
        # full colormap, so a near-uniform attention map -- which is what an untrained
        # or uninformative model produces -- is rendered as vivid structure. Anchoring
        # at zero makes a flat map look flat, which is the honest reading.
        image = axes.imshow(
            matrix, aspect="auto", cmap=COLORMAP, vmin=0.0, origin="upper",
            interpolation="nearest",
        )
        bar = figure.colorbar(image, cax=cax) if cax is not None else figure.colorbar(
            image, ax=axes
        )
        bar.set_label("attention weight", fontsize=style.font_size)
        bar.ax.tick_params(labelsize=style.font_size - 2)
        bar.outline.set_linewidth(0.5)

        if top is not None:
            column_total = matrix.sum(axis=0)
            top.bar(np.arange(len(column_total)), column_total, color=_DARK,
                    width=1.0, linewidth=0)
            top.set_ylabel("sum", fontsize=style.font_size - 2, rotation=0,
                           ha="right", va="center")
            top.tick_params(labelbottom=False, labelsize=style.font_size - 2)
            top.margins(x=0)
            top.set_title(title, fontsize=style.font_size + 1, pad=6)
            row_total = matrix.sum(axis=1)
            side.barh(np.arange(len(row_total)), row_total, color=_DARK,
                      height=1.0, linewidth=0)
            side.set_xlabel("sum", fontsize=style.font_size - 2)
            side.tick_params(labelleft=False, labelsize=style.font_size - 2)
            side.margins(y=0)
            side.invert_yaxis()
        else:
            axes.set_title(title, fontsize=style.font_size + 2)

        row_positions, row_tick_labels = _thin_labels(row_labels_all, LABEL_LIMIT)
        axes.set_yticks(row_positions)
        axes.set_yticklabels(row_tick_labels, fontsize=style.font_size - 1)
        column_positions, column_tick_labels = _thin_labels(column_labels, LABEL_LIMIT)
        axes.set_xticks(column_positions)
        axes.set_xticklabels(
            column_tick_labels, fontsize=style.font_size - 1, rotation=90
        )

        axes.set_ylabel(
            "residue in the original sequence (1-based)", fontsize=style.font_size
        )
        axes.set_xlabel(column_label, fontsize=style.font_size)
        _draw_caption(figure, export, style)
        try:
            return _save(figure, Path(path), style, export)
        finally:
            plt.close(figure)


def plot_residue_profile(
    export: IBAMExport,
    path: Path | str,
    *,
    style: FigureStyle | None = None,
    top: int | None = None,
    sequence: str | None = None,
) -> dict[str, Path]:
    """Per-residue summary across the whole sequence, with the top residues named.

    The overview that remains readable for a 34,000-residue construct, where a
    full matrix cannot be.
    """
    style = style or FigureStyle()
    count = style.annotate_top if top is None else top

    with _drawing(style) as plt:
        figure, axes = plt.subplots(figsize=(style.width, style.height * 0.5))
        scores = export.residue_scores
        axes.fill_between(
            export.residue_positions + 1, scores, color=_DARK, alpha=0.85,
            linewidth=0,
        )
        if export.ensemble_std is not None:
            axes.fill_between(
                export.residue_positions + 1,
                np.clip(scores - export.ensemble_std, 0.0, None),
                scores + export.ensemble_std,
                color=_MID,
                alpha=0.35,
                linewidth=0,
                label="ensemble +/- 1 SD",
            )
            axes.legend(fontsize=style.font_size)

        # A horizontal reference for "no structure at all": if every residue got
        # the same share, this is the height it would sit at. Peaks are only
        # interesting relative to it.
        uniform = float(scores.sum()) / max(1, len(scores))
        axes.axhline(uniform, color="#999999", linestyle=(0, (4, 3)), linewidth=0.7)
        axes.annotate(
            "uniform attention",
            (export.residue_positions[-1] + 1, uniform),
            fontsize=style.font_size - 2, color="#666666", ha="right", va="bottom",
        )

        ranked = export.top_residues(count)
        names = residue_tick_labels(
            np.array([p for p, _ in ranked], dtype=np.int64), sequence
        )
        highest = max((score for _, score in ranked), default=0.0)
        span = float(export.residue_positions[-1] - export.residue_positions[0]) or 1.0
        shown = _annotate_peaks(
            axes,
            [p + 1 for p, _ in ranked],
            [s for _, s in ranked],
            names,
            style,
            span,
        )
        if highest > 0:
            axes.set_ylim(0, highest * 1.34)
        axes.set_xlabel(
            "residue in the original sequence (1-based)", fontsize=style.font_size
        )
        axes.set_ylabel(_score_label(export), fontsize=style.font_size)
        suffix = "" if shown == len(ranked) else f" ({len(ranked) - shown} too close to label)"
        axes.set_title(
            f"iBAM per-residue summary - top {len(ranked)} named{suffix}",
            fontsize=style.font_size + 2,
        )
        _warn_uninformative(axes, export, style)
        axes.margins(x=0.01)
        _draw_caption(figure, export, style)
        try:
            return _save(figure, Path(path), style, export)
        finally:
            plt.close(figure)


def plot_head_agreement(
    export: IBAMExport,
    path: Path | str,
    *,
    style: FigureStyle | None = None,
    sequence: str | None = None,
) -> dict[str, Path]:
    """Where each individual head puts its mass, against the head-mean.

    Every other figure in this module shows an aggregate. A mean over heads that
    disagree is indistinguishable from a mean over heads that agree, and only the
    second licenses reading a peak as "the model attends here". Three panels:

    * per-head residue profiles, one line each, all layers;
    * the fraction of each head's mass on its own top-10 residues, which is a
      concentration measure -- a head spread uniformly scores near 10/L;
    * the rank correlation between every pair of heads in the reduced layer, so
      agreement is a number rather than an impression of overlapping lines.
    """
    style = style or FigureStyle()
    attention = export.attention_by_layer_head  # [n_layers, H, L, T]
    if attention.ndim != 4:
        raise IBAMPlotError(
            f"attention_by_layer_head has {attention.ndim} dimensions, expected 4"
        )
    n_layers, n_heads, length, _ = attention.shape
    # [n_layers, H, L]: each head's per-residue mass.
    profiles = attention.sum(axis=3)
    positions = export.residue_positions + 1

    with _drawing(style) as plt:
        figure = plt.figure(figsize=(style.width, style.height))
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.9), hspace=0.42, wspace=0.26)
        axes = figure.add_subplot(grid[0, :])
        colors = plt.get_cmap(COLORMAP)(np.linspace(0.08, 0.92, n_heads))
        for layer in range(n_layers):
            for head in range(n_heads):
                axes.plot(
                    positions,
                    profiles[layer, head],
                    color=colors[head],
                    alpha=0.55,
                    linewidth=0.7,
                    linestyle="-" if layer == n_layers - 1 else (0, (2, 2)),
                )
        axes.plot(
            positions, export.per_residue, color="#000000", linewidth=1.4,
            label="head-mean (what the other figures show)",
        )
        axes.legend(fontsize=style.font_size - 1, loc="upper right")
        axes.set_xlabel(
            "residue in the original sequence (1-based)", fontsize=style.font_size
        )
        axes.set_ylabel("attention mass", fontsize=style.font_size)
        axes.set_title(
            f"per-head profiles - {n_heads} heads x {n_layers} layers "
            f"(solid = reduced layer, dashed = earlier)",
            fontsize=style.font_size + 1,
        )
        axes.margins(x=0.01)
        # Anchored at zero and with the offset notation off, for the same reason
        # the heatmaps use vmin=0: autoscaling onto a narrow band turns float
        # noise around a flat profile into a dramatic-looking signal, and an
        # axis labelled "1e-7 + 1" is read as structure by almost everyone.
        axes.set_ylim(bottom=0.0)
        axes.ticklabel_format(axis="y", useOffset=False, style="plain")

        # Concentration. A head spread evenly over L residues puts 10/L of its
        # mass on any 10 of them, which is the line to read peaks against.
        concentration = figure.add_subplot(grid[1, 0])
        keep = min(10, length)
        fractions = np.array(
            [
                [
                    np.sort(profiles[layer, head])[::-1][:keep].sum()
                    / max(profiles[layer, head].sum(), 1e-12)
                    for head in range(n_heads)
                ]
                for layer in range(n_layers)
            ]
        )
        width = 0.8 / n_layers
        for layer in range(n_layers):
            concentration.bar(
                np.arange(n_heads) + layer * width - 0.4 + width / 2,
                fractions[layer],
                width=width,
                color=plt.get_cmap(COLORMAP)(0.15 + 0.6 * layer / max(1, n_layers - 1)),
                label=f"layer {layer}",
                linewidth=0,
            )
        baseline = keep / length
        concentration.axhline(baseline, color="#999999", linestyle=(0, (4, 3)), linewidth=0.7)
        concentration.annotate(
            f"uniform = {baseline:.3f}",
            (n_heads - 0.5, baseline),
            fontsize=style.font_size - 2, color="#666666", ha="right", va="bottom",
        )
        concentration.set_xticks(np.arange(n_heads))
        concentration.set_xticklabels([str(h) for h in range(n_heads)])
        concentration.set_xlabel("head", fontsize=style.font_size)
        concentration.set_ylabel(f"mass on own top {keep}", fontsize=style.font_size)
        concentration.set_title("concentration per head", fontsize=style.font_size + 1)
        if n_layers > 1:
            concentration.legend(fontsize=style.font_size - 2)

        # Pairwise agreement in the reduced layer.
        agreement = figure.add_subplot(grid[1, 1])
        from scipy.stats import spearmanr

        last = profiles[-1]
        matrix = np.eye(n_heads)
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                value = spearmanr(last[i], last[j]).statistic
                matrix[i, j] = matrix[j, i] = 0.0 if np.isnan(value) else value
        # Agreement is a correlation and does have a meaningful zero, so this is
        # the one place a symmetric scale is correct; it stays within viridis.
        image = agreement.imshow(matrix, cmap=COLORMAP, vmin=-1.0, vmax=1.0)
        figure.colorbar(image, ax=agreement, fraction=0.046).ax.tick_params(
            labelsize=style.font_size - 2
        )
        agreement.set_xticks(np.arange(n_heads))
        agreement.set_yticks(np.arange(n_heads))
        agreement.set_xlabel("head", fontsize=style.font_size)
        agreement.set_ylabel("head", fontsize=style.font_size)
        offdiag = matrix[~np.eye(n_heads, dtype=bool)]
        agreement.set_title(
            f"Spearman between heads - median {np.median(offdiag):.2f}",
            fontsize=style.font_size + 1,
        )

        _draw_caption(
            figure,
            export,
            style,
            extra=(
                "head-mean aggregation hides disagreement; these panels show the "
                "individual heads so a peak can be judged against how much the "
                "heads actually agree"
            ),
        )
        try:
            return _save(figure, Path(path), style, export)
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


def _molecule_raster(
    export: IBAMExport, smiles: str, size: tuple[int, int] = (1100, 850)
) -> np.ndarray:
    """The same depiction as :func:`plot_molecule_contributions`, as pixels.

    Needed only to place the molecule inside a composite matplotlib figure. The
    colouring and the index mapping are identical; this renders through Cairo
    instead of writing SVG.
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
    output_order = export.atom_index_maps.get("std_to_output")

    colors: dict[int, tuple[float, float, float]] = {}
    radii: dict[int, float] = {}
    for index, weight in enumerate(scaled):
        drawn = index
        if output_order:
            if index >= len(output_order):
                raise IBAMPlotError(
                    "std_to_output is shorter than the molecule; the mapping does "
                    "not describe this export"
                )
            drawn = int(output_order[index])
        colors[drawn] = (
            0.267 + (0.993 - 0.267) * float(weight),
            0.005 + (0.906 - 0.005) * float(weight),
            0.329 + (0.144 - 0.329) * float(weight),
        )
        radii[drawn] = 0.2 + 0.4 * float(weight)

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, molecule,
        highlightAtoms=list(colors), highlightAtomColors=colors,
        highlightAtomRadii=radii, highlightBonds=[],
    )
    drawer.FinishDrawing()

    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB"))


def plot_interaction_panel(
    export: IBAMExport,
    path: Path | str,
    *,
    smiles: str | None = None,
    sequence: str | None = None,
    style: FigureStyle | None = None,
    window: tuple[int, int] | None = None,
    prediction: str = "",
) -> dict[str, Path]:
    """One composite figure: profile, zoomed matrix, molecule, and ranked tables.

    The figure a paper actually uses. Panel B zooms on the highest-scoring region
    rather than the whole sequence, because a 1,000-residue matrix compressed into
    four inches shows nothing; the chosen window is stated on the panel, so the
    reader knows they are looking at a selected region and which one.
    """
    style = style or FigureStyle()
    positions = export.residue_positions

    if window is None:
        # Centre on the strongest residue, which is a stated selection rule and
        # not an eye-chosen region.
        if len(positions) <= READABLE_RESIDUE_LIMIT:
            window = (int(positions[0]), int(positions[-1]) + 1)
        else:
            peak = int(positions[int(np.argmax(export.per_residue))])
            half = READABLE_RESIDUE_LIMIT // 2
            low = max(int(positions[0]), peak - half)
            window = (low, low + READABLE_RESIDUE_LIMIT)
    zoom = export.residue_window(*window)

    with _drawing(style) as plt:
        figure = plt.figure(figsize=(style.width, style.height * 1.45))
        grid = figure.add_gridspec(
            3, 2,
            height_ratios=(0.60, 1.0, 0.40),
            width_ratios=(1.0, 0.62),
            hspace=0.34, wspace=0.20,
        )

        # --- A: whole-sequence profile, with the window marked ---------------
        profile = figure.add_subplot(grid[0, :])
        scores = export.residue_scores
        profile.fill_between(
            positions + 1, scores, color=_DARK, alpha=0.85, linewidth=0
        )
        profile.axvspan(window[0] + 1, window[1], color=_LIGHT, alpha=0.25, linewidth=0)
        uniform = float(scores.sum()) / max(1, len(scores))
        profile.axhline(uniform, color="#999999", linestyle=(0, (4, 3)), linewidth=0.7)
        ranked = export.top_residues(style.annotate_top)
        named = residue_tick_labels(
            np.array([p for p, _ in ranked], dtype=np.int64), sequence
        )
        _annotate_peaks(
            profile,
            [p + 1 for p, _ in ranked],
            [s for _, s in ranked],
            named,
            style,
            float(positions[-1] - positions[0]) or 1.0,
        )
        profile.set_title(
            f"A   {_score_title(export)} over all {export.sequence_length:,} residues "
            f"(shaded: panel B window {window[0] + 1}-{window[1]})",
            fontsize=style.font_size + 1, loc="left",
        )
        profile.set_xlabel("residue (1-based)", fontsize=style.font_size)
        profile.set_ylabel(_score_label(export), fontsize=style.font_size)
        _warn_uninformative(profile, export, style)
        profile.margins(x=0.01)
        if ranked:
            profile.set_ylim(0, max(score for _, score in ranked) * 1.42)

        # --- B: zoomed residue x atom matrix --------------------------------
        heat = figure.add_subplot(grid[1, 0])
        image = heat.imshow(
            zoom.residue_atom, aspect="auto", cmap=COLORMAP, vmin=0.0,
            origin="upper", interpolation="nearest",
        )
        bar = figure.colorbar(image, ax=heat, fraction=0.035, pad=0.02)
        bar.set_label("attention weight", fontsize=style.font_size - 1)
        bar.ax.tick_params(labelsize=style.font_size - 2)
        row_positions, row_labels = _thin_labels(
            residue_tick_labels(zoom.residue_positions, sequence), 26
        )
        heat.set_yticks(row_positions)
        heat.set_yticklabels(row_labels, fontsize=style.font_size - 2)
        column_positions, column_labels = _thin_labels(
            atom_tick_labels(zoom.atom_indices, smiles), 40
        )
        heat.set_xticks(column_positions)
        heat.set_xticklabels(column_labels, fontsize=style.font_size - 2, rotation=90)
        heat.set_title(
            "B   residue x atom - DERIVED AGGREGATION",
            fontsize=style.font_size + 1, loc="left",
        )
        heat.set_ylabel("residue (1-based)", fontsize=style.font_size)
        heat.set_xlabel("atom (1-based)", fontsize=style.font_size)

        # --- C: molecule depiction ------------------------------------------
        depiction = figure.add_subplot(grid[1, 1])
        depiction.set_axis_off()
        if smiles is not None:
            try:
                depiction.imshow(_molecule_raster(export, smiles), interpolation="lanczos")
            except Exception as error:  # pragma: no cover - renderer-dependent
                depiction.text(
                    0.5, 0.5, f"depiction unavailable\n{error}",
                    ha="center", va="center", fontsize=style.font_size - 1,
                )
        else:
            depiction.text(
                0.5, 0.5, "no SMILES supplied", ha="center", va="center",
                fontsize=style.font_size - 1, color="#888888",
            )
        depiction.set_title(
            "C   per-atom contribution (dark = low, yellow = high)",
            fontsize=style.font_size + 1, loc="left",
        )

        # --- D: ranked tables ------------------------------------------------
        table = figure.add_subplot(grid[2, :])
        table.set_axis_off()
        residue_rows = export.top_residues(10)
        atom_rows = export.top_atoms(10)
        atom_names = atom_tick_labels(
            np.array([a for a, _ in atom_rows], dtype=np.int64), smiles
        )
        residue_names = residue_tick_labels(
            np.array([p for p, _ in residue_rows], dtype=np.int64), sequence
        )
        residue_total = float(export.residue_scores.sum()) or 1.0
        total = float(export.per_residue.sum()) or 1.0
        lines = [
            f"{'rank':<5}{'residue':<11}{'share':<10}{'atom':<9}{'share':<9}",
            f"{'':<5}{'(ligand->protein)':<21}{'(protein->ligand)':<18}"
            if export.residue_scores_are_informative
            else f"{'':<5}{'(TIED - see note)':<21}{'(protein->ligand)':<18}",
        ]
        for rank in range(max(len(residue_rows), len(atom_rows))):
            left = (
                f"{residue_names[rank]:<11}{residue_rows[rank][1] / residue_total:<10.4f}"
                if rank < len(residue_rows) else " " * 21
            )
            right = (
                f"{atom_names[rank]:<9}{atom_rows[rank][1] / total:<9.4f}"
                if rank < len(atom_rows) else ""
            )
            lines.append(f"{rank + 1:<5}{left}{right}")
        table.text(
            0.0, 1.0, "\n".join(lines), family="monospace",
            fontsize=style.font_size - 1, va="top", ha="left",
        )
        nonphysical = float(export.nonphysical_mass.sum())
        notes = [
            f"non-physical token mass: {nonphysical:.4f}"
            f" ({nonphysical / (nonphysical + total):.2%} of the total)"
            "  - knowledge and virtual nodes map to no atom and are excluded"
            " from panels B-C",
            f"shares are fractions of the summed physical attention ({total:.4f})",
        ]
        if prediction:
            notes.insert(0, prediction)
        table.text(
            0.52, 1.0, "\n".join(textwrap.fill(n, 62) for n in notes),
            fontsize=style.font_size - 1, va="top", ha="left", color="#333333",
        )
        table.set_title("D   ranked contributions", fontsize=style.font_size + 1, loc="left")

        _draw_caption(figure, export, style)
        try:
            return _save(figure, Path(path), style, export)
        finally:
            plt.close(figure)


def plot_all(
    export: IBAMExport,
    directory: Path | str,
    *,
    smiles: str | None = None,
    sequence: str | None = None,
    style: FigureStyle | None = None,
    windows: Sequence[tuple[int, int]] = (),
    prediction: str = "",
) -> dict[str, dict[str, Path]]:
    """The full figure set, plus any requested zoomed residue windows.

    The matrix heatmap is skipped rather than drawn illegibly when the sequence is
    long; the per-residue profile always works, and windows are how a long target
    is inspected.
    """
    directory = Path(directory)
    style = style or FigureStyle()
    produced: dict[str, dict[str, Path]] = {
        "residue_profile": plot_residue_profile(
            export, directory / "residue_profile", style=style, sequence=sequence
        ),
        "head_agreement": plot_head_agreement(
            export, directory / "head_agreement", style=style, sequence=sequence
        ),
        "interaction_panel": plot_interaction_panel(
            export,
            directory / "interaction_panel",
            smiles=smiles,
            sequence=sequence,
            # The composite is the deliverable; give it room even when the
            # caller's style is sized for a single panel.
            style=replace(style, width=max(style.width, 11.0)),
            prediction=prediction,
        ),
    }
    if export.sequence_length <= READABLE_RESIDUE_LIMIT:
        produced["residue_atom"] = plot_residue_atom_heatmap(
            export, directory / "residue_atom", style=style, view="atom",
            sequence=sequence, smiles=smiles,
        )
        produced["residue_token"] = plot_residue_atom_heatmap(
            export, directory / "residue_token", style=style, view="token",
            sequence=sequence,
        )
    for start, stop in windows:
        window = export.residue_window(start, stop)
        produced[f"window_{start}_{stop}"] = plot_residue_atom_heatmap(
            window, directory / f"window_{start}_{stop}", style=style, view="atom",
            sequence=sequence, smiles=smiles,
        )
    if smiles is not None:
        produced["molecule"] = plot_molecule_contributions(
            export, smiles, directory / "molecule"
        )
    return produced


__all__ = [
    "COLORMAP",
    "LABEL_LIMIT",
    "PUBLICATION",
    "READABLE_RESIDUE_LIMIT",
    "FigureStyle",
    "IBAMPlotError",
    "atom_tick_labels",
    "plot_all",
    "plot_head_agreement",
    "plot_interaction_panel",
    "plot_molecule_contributions",
    "plot_residue_atom_heatmap",
    "plot_residue_profile",
    "residue_tick_labels",
]
