"""Shared trainable projections over frozen encoder features.

Owed by the specification:
    Priority 2: the adapters must actually be shared by affinity, retrieval
    and diffusion conditioning; log which parameters receive gradients from
    each loss.

Both halves of that clause are implemented here, and the second is the
interesting one. "Shared" is easy to claim and easy to get wrong: a refactor that
constructs a second adapter for the retrieval branch, or a ``detach()`` left in a
conditioning path, produces a model that trains, converges and reports plausible
numbers while the representation is no longer shared at all. The multi-task claim
would then be false in a way no loss curve reveals.

:func:`attribute_gradients` turns that into a measurement. It runs each loss's
backward in isolation and reports which named parameters actually received
gradient, so "the adapters are shared by affinity, retrieval and diffusion" is a
test assertion rather than a design intention.

**Dimensions are discovered, never assumed.** Nothing here hardcodes 1152 or 256;
:meth:`AdapterConfig.from_encoders` reads the widths off the encoder metadata that
produced the features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from bindra.features.types import EncoderMetadata


class AdapterError(ValueError):
    """An adapter was built or fed inconsistently with its declared widths."""


@dataclass(frozen=True)
class AdapterConfig:
    """Widths in, width out. Every input dimension is supplied, never guessed."""

    protein_dim: int
    ligand_dim: int
    ligand_graph_dim: int
    model_dim: int = 256
    dropout: float = 0.1
    #: A bottleneck narrower than ``model_dim`` keeps the adapter small relative
    #: to the frozen encoder it sits on. ``None`` means a single linear map.
    bottleneck_dim: int | None = None
    #: Widths of the graph readout's parts, in order, summing to
    #: ``ligand_graph_dim``. Each gets its own projection and its own block of
    #: the pair vector.
    #:
    #: ``None`` keeps the old behaviour: one projection over the whole
    #: concatenation into one block. That was a real bottleneck for a readout
    #: built from distinct things -- for LiGhT, a fingerprint node, a descriptor
    #: node and a mean, two thirds of it knowledge, all compressed into a single
    #: ``model_dim`` block beside three others.
    graph_components: tuple[int, ...] | None = None
    #: How many ESM-C layers the protein cache holds, concatenated along the
    #: feature axis. ``1`` is the historical single-layer cache. Above 1,
    #: :class:`LayerMix` collapses them with learned weights before the adapter,
    #: so ``protein_dim`` here remains the width of **one** layer.
    protein_layers: int = 1
    #: Width of the per-observation assay-condition vector, or 0 for none.
    assay_dim: int = 0
    #: Hidden width inside :class:`KnowledgeFiLM`'s projection, or None for a
    #: single map. The single map is the largest non-fusion block in the
    #: model; a bottleneck here costs little because the conditioner emits a
    #: scale and a shift, not a representation.
    film_bottleneck: int | None = None

    @property
    def protein_cache_dim(self) -> int:
        """Width the cache actually stores, which is what the model is handed."""
        return self.protein_dim * self.protein_layers

    @property
    def graph_blocks(self) -> int:
        """How many ``model_dim`` blocks the graph readout contributes."""
        return 1 if self.graph_components is None else len(self.graph_components)

    def __post_init__(self) -> None:
        for name in ("protein_dim", "ligand_dim", "ligand_graph_dim", "model_dim"):
            if getattr(self, name) < 1:
                raise AdapterError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.bottleneck_dim is not None and self.bottleneck_dim < 1:
            raise AdapterError(f"bottleneck_dim must be >= 1, got {self.bottleneck_dim}")
        if self.graph_components is not None:
            if not self.graph_components:
                raise AdapterError("graph_components must not be empty; use None")
            total = sum(self.graph_components)
            if total != self.ligand_graph_dim:
                raise AdapterError(
                    f"graph_components sum to {total} but ligand_graph_dim is "
                    f"{self.ligand_graph_dim}; a split that does not cover the "
                    "readout would silently drop or duplicate part of it"
                )

    @classmethod
    def from_encoders(
        cls,
        protein_metadata: EncoderMetadata,
        ligand_metadata: EncoderMetadata,
        **kwargs: object,
    ) -> "AdapterConfig":
        """Read the input widths off the metadata of the encoders in use.

        The ligand graph width lives in ``extra["graph_dim"]`` because the pooled
        readout is generally wider than one token; its absence is an error rather
        than a default, since silently assuming ``ligand_dim`` would mis-shape the
        conditioning path.
        """
        graph_dim = ligand_metadata.extra.get("graph_dim")
        if graph_dim is None:
            raise AdapterError(
                f"ligand encoder {ligand_metadata.name!r} does not declare "
                "extra['graph_dim']; the pooled readout width cannot be inferred"
            )
        return cls(
            protein_dim=protein_metadata.output_dim,
            ligand_dim=ligand_metadata.output_dim,
            ligand_graph_dim=int(graph_dim),
            **kwargs,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "protein_dim": self.protein_dim,
            "ligand_dim": self.ligand_dim,
            "ligand_graph_dim": self.ligand_graph_dim,
            "model_dim": self.model_dim,
            "dropout": self.dropout,
            "bottleneck_dim": self.bottleneck_dim,
            "protein_layers": self.protein_layers,
            "assay_dim": self.assay_dim,
            "film_bottleneck": self.film_bottleneck,
        }


class LayerMix(nn.Module):
    """Collapse several cached ESM-C layers into one, with learned weights.

    The cache held **only the final layer** -- ``last_hidden_state``, after the
    final LayerNorm. That is the representation most specialised for ESM-C's own
    masked-token objective, and it is not generally the one that carries the most
    structural signal. A scalar mix over several layers lets the model choose,
    and costs one weight per layer.

    **Identity at initialisation**, like :class:`KnowledgeFiLM`: the logits are
    set so the softmax puts essentially all mass on the last layer, so a model
    built with a multi-layer cache starts out seeing exactly what the
    single-layer model saw. Switching this on is then an ablation rather than a
    different model with different weights.

    Each layer is normalised separately first. Layer norms across a transformer
    stack differ by a large factor, and mixing raw states would make the weights
    a statement about magnitude rather than usefulness.
    """

    #: Logit given to the last layer at init. At 8.0 a four-layer mix puts
    #: 0.9990 of the softmax mass on the last layer; measured deviation from
    #: last-layer-only is ~5e-3 on unit-scale inputs. Close enough that the
    #: first optimiser steps see today's model, not an exact identity -- unlike
    #: AttentionPool and KnowledgeFiLM, which are exact.
    LAST_LAYER_LOGIT = 8.0

    def __init__(self, layer_dim: int, n_layers: int) -> None:
        super().__init__()
        if n_layers < 1:
            raise AdapterError(f"n_layers must be >= 1, got {n_layers}")
        self.layer_dim = layer_dim
        self.n_layers = n_layers
        self.norms = nn.ModuleList(nn.LayerNorm(layer_dim) for _ in range(n_layers))
        logits = torch.zeros(n_layers)
        logits[-1] = self.LAST_LAYER_LOGIT
        self.logits = nn.Parameter(logits)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected = self.layer_dim * self.n_layers
        if features.shape[-1] != expected:
            raise AdapterError(
                f"expected {self.n_layers} layers of {self.layer_dim} "
                f"({expected} wide), got {features.shape[-1]}. The cache and the "
                "model disagree about how many layers were stored."
            )
        if self.n_layers == 1:
            return self.norms[0](features)
        parts = torch.split(features, self.layer_dim, dim=-1)
        weights = self.weights()
        # Accumulated in place of a stack. `torch.stack` would materialise all
        # n_layers normalised copies at once -- at a 65,536-residue budget and
        # four layers that is an extra ~900 MiB of activation on top of the
        # 1.2 GiB input, purely to be summed away on the next line. Summing as
        # we go holds one layer at a time.
        mixed = None
        for weight, norm, part in zip(weights, self.norms, parts):
            term = weight * norm(part)
            mixed = term if mixed is None else mixed + term
        return mixed


class AttentionPool(nn.Module):
    """Pool a padded sequence by learned relevance instead of averaging it.

    A binding site is roughly 20-30 residues of a median 454, so a plain mean
    over residues dilutes whatever the fusion stack found by more than an order
    of magnitude before it reaches the heads. This scores each position and
    takes a weighted sum instead.

    **Exactly equal to** :func:`masked_mean` **at initialisation.** The scorer is
    zero-initialised, so every real position gets the same logit and the softmax
    is uniform over the mask; the head-combining projection is initialised to
    average its heads. Both properties together mean a model built with pooling
    on and one built with it off are bit-identical until the scorer learns
    something -- so the ablation measures the pooling, not a different draw.

    Multiple heads because a pocket is not one thing: separate heads can settle
    on the catalytic residues, the hydrophobic wall and the solvent-exposed rim
    without competing for one softmax.
    """

    def __init__(self, model_dim: int, *, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if num_heads < 1:
            raise AdapterError(f"num_heads must be >= 1, got {num_heads}")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.scorer = nn.Linear(model_dim, num_heads)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)
        self.dropout = nn.Dropout(dropout)
        self.combine = nn.Linear(num_heads * model_dim, model_dim, bias=False)
        # Average the heads at init. With a uniform softmax above, every head
        # returns the same masked mean, so this reproduces masked_mean exactly.
        with torch.no_grad():
            eye = torch.eye(model_dim).repeat(1, num_heads) / num_heads
            self.combine.weight.copy_(eye)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype is not torch.bool:
            raise AdapterError(f"mask must be bool, got {mask.dtype}")
        if features.shape[-1] != self.model_dim:
            raise AdapterError(
                f"expected last dimension {self.model_dim}, got {features.shape[-1]}"
            )
        # [B, L, H]; padding gets -inf so it cannot take softmax mass. Computed
        # in float32: a bf16 softmax over 34,350 residues loses the tail.
        logits = self.scorer(features).float()
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(logits, dim=-2).to(features.dtype)
        weights = self.dropout(weights)
        # [B, H, L] x [B, L, D] -> [B, H, D] -> [B, H * D]
        pooled = torch.einsum("blh,bld->bhd", weights, features)
        return self.combine(pooled.flatten(start_dim=-2))

    def attention(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The pooling weights themselves, ``[B, L, H]``, for iBAM.

        Where the model decided the binding site is, which is a claim worth
        being able to inspect rather than only act on.
        """
        logits = self.scorer(features).float()
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return torch.softmax(logits, dim=-2)


class AssayEncoder(nn.Module):
    """Embed the recorded assay conditions of one measurement.

    An IC50 is not a constant of a protein-ligand pair: it moves with substrate
    and enzyme concentration, with pH and temperature, and with how carefully the
    source curated it. IC50 is 72% of this corpus, and the conditions are partly
    *recorded* -- so some of what the replicate-noise floor attributes to
    irreducible scatter is in columns the model was never shown.

    **Only descriptive conditions are encoded here, never group identity.**
    ``comparability_group_id`` and ``replicate_group_id`` would let the model
    recognise *which assay* a row came from, and since replicates of one
    measurement share a group, that is a route to the label rather than a
    property of the experiment. They are excluded deliberately; see
    ``ASSAY_FIELDS`` in :mod:`bindra.data.assay`.
    """

    def __init__(self, assay_dim: int, config: AdapterConfig) -> None:
        super().__init__()
        self.assay_dim = assay_dim
        self.norm = nn.LayerNorm(assay_dim)
        self.project = nn.Sequential(
            nn.Linear(assay_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.output_norm = nn.LayerNorm(config.model_dim)

    def forward(self, assay: torch.Tensor) -> torch.Tensor:
        if assay.shape[-1] != self.assay_dim:
            raise AdapterError(
                f"expected assay width {self.assay_dim}, got {assay.shape[-1]}"
            )
        return self.output_norm(self.project(self.norm(assay)))


class FeatureAdapter(nn.Module):
    """Project one frozen feature stream into the shared model width.

    Pre-norm on the input: the encoders feed in at very different scales (ESM-C
    post-LayerNorm residue states against a freshly initialised GAT), and without
    normalising first the larger stream dominates the fused representation for
    the first few thousand steps purely by magnitude.
    """

    def __init__(self, input_dim: int, config: AdapterConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.input_norm = nn.LayerNorm(input_dim)
        inner = config.bottleneck_dim
        if inner is None:
            self.projection: nn.Module = nn.Linear(input_dim, config.model_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, inner),
                nn.GELU(),
                nn.Linear(inner, config.model_dim),
            )
        self.dropout = nn.Dropout(config.dropout)
        self.output_norm = nn.LayerNorm(config.model_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise AdapterError(
                f"expected last dimension {self.input_dim}, got {features.shape[-1]}"
            )
        return self.output_norm(self.dropout(self.projection(self.input_norm(features))))


class SharedAdapters(nn.Module):
    """The one adapter instance that every downstream task must receive.

    Constructed once and passed by reference. The affinity head, the retrieval
    projections and the diffusion conditioner all call *this* object; building a
    second one for any of them is the failure
    :func:`attribute_gradients` exists to catch.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        # Several cached layers collapse to one before the adapter sees them,
        # so the adapter's input width is one layer regardless of how many the
        # cache holds. None when the cache is single-layer, so a checkpoint
        # trained against the old cache still loads.
        self.layer_mix = (
            LayerMix(config.protein_dim, config.protein_layers)
            if config.protein_layers > 1
            else None
        )
        self.protein = FeatureAdapter(config.protein_dim, config)
        self.ligand = FeatureAdapter(config.ligand_dim, config)
        self.assay = (
            AssayEncoder(config.assay_dim, config) if config.assay_dim else None
        )
        # One adapter per readout part when the arm declares a split, so each
        # part reaches the pair vector at full model width instead of sharing a
        # single compressed block. `ligand_graph` keeps its name for the
        # unsplit case, because trained checkpoints hold that key.
        if config.graph_components is None:
            self.ligand_graph = FeatureAdapter(config.ligand_graph_dim, config)
            self.ligand_graph_parts = None
        else:
            self.ligand_graph = None  # type: ignore[assignment]
            self.ligand_graph_parts = nn.ModuleList(
                FeatureAdapter(width, config) for width in config.graph_components
            )

    @property
    def model_dim(self) -> int:
        return self.config.model_dim

    def forward(
        self,
        protein_features: torch.Tensor,
        ligand_features: torch.Tensor,
        ligand_graph_features: torch.Tensor | None = None,
        assay_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Project all streams in one call, so no caller adapts only one."""
        if self.layer_mix is not None:
            protein_features = self.layer_mix(protein_features)
        result = {
            "protein": self.protein(protein_features),
            "ligand": self.ligand(ligand_features),
        }
        if ligand_graph_features is not None:
            result["ligand_graph"] = self._adapt_graph(ligand_graph_features)
        if assay_features is not None:
            if self.assay is None:
                raise AdapterError(
                    "assay features were supplied but these adapters were built "
                    "with assay_dim=0; they would be silently dropped"
                )
            result["assay"] = self.assay(assay_features)
        elif self.assay is not None:
            raise AdapterError(
                "these adapters were built with an assay encoder but no assay "
                "features were supplied"
            )
        return result

    def _adapt_graph(self, features: torch.Tensor) -> torch.Tensor:
        """``[B, graph_blocks * model_dim]``, split by component when declared."""
        if self.ligand_graph_parts is None:
            return self.ligand_graph(features)
        widths = self.config.graph_components or ()
        if features.shape[-1] != sum(widths):
            raise AdapterError(
                f"graph readout is {features.shape[-1]} wide but the declared "
                f"components sum to {sum(widths)}"
            )
        pieces = torch.split(features, list(widths), dim=-1)
        return torch.cat(
            [adapter(piece) for adapter, piece in zip(self.ligand_graph_parts, pieces)],
            dim=-1,
        )


class KnowledgeFiLM(nn.Module):
    """Let the molecule's knowledge vector *modulate* the fused streams.

    Before this, knowledge reached the output three ways and all of them were
    **additive**: two tokens the protein may attend to, three blocks of the pair
    vector, and LiGhT's own mixing into every bond. An additive path can be
    ignored -- the model is free to learn a zero weight on it and never consult
    the fingerprint again. FiLM cannot be ignored in the same way: it scales and
    shifts every residue and every bond representation, so the molecule's global
    character changes what the protein sees rather than being appended to it.

    The conditioner reads only the **fingerprint and descriptor** blocks of the
    adapted readout, not the pooled-mean block. The mean is a summary of the
    bonds that are already being modulated, so conditioning on it would feed the
    stream back into its own scale.

    **Identity at initialisation.** The final projection starts at zero and the
    scale is ``1 + delta``, so a freshly built model with FiLM on is
    bit-identical to one with it off. Without that, switching it on would change
    the architecture and reinitialise the weights at the same time, and any
    comparison between the two would measure both at once.

    One consequence of the zero init looks like a bug and is not: ``norm``
    receives **no gradient on the first step**, because the chain rule runs it
    through a zero ``project.weight``. Measured: zero at step 1, nonzero at
    step 2 once ``project`` has moved off zero. Do not "fix" it by initialising
    ``project`` randomly -- that buys a first-step gradient and loses the
    identity property, which is the more valuable of the two.
    """

    def __init__(
        self,
        config: AdapterConfig,
        *,
        knowledge_blocks: int = 2,
        assay_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.knowledge_blocks = knowledge_blocks
        width = knowledge_blocks * config.model_dim
        # The assay context joins the molecule's knowledge as a second thing
        # that modulates rather than is appended. Same reasoning: an appended
        # assay vector can be ignored, a multiplicative one cannot.
        self.assay_conditioning = assay_conditioning
        if assay_conditioning:
            width += config.model_dim
        self.norm = nn.LayerNorm(width)
        # Four outputs per feature: scale and shift, for the protein and the
        # ligand stream. A full width -> 4*model_dim map is the single largest
        # non-fusion block in the model (788,992 at model_dim 256, 10.7% of
        # the total) for what is a conditioning signal, not a representation.
        # A bottleneck cuts it to roughly a quarter.
        #
        # The ZERO INIT moves to the last layer either way: that is what makes
        # conditioning an exact no-op at initialisation, and therefore an
        # ablation rather than a different model.
        inner = config.film_bottleneck
        if inner:
            first = nn.Linear(width, inner)
            last = nn.Linear(inner, 4 * config.model_dim)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.project: nn.Module = nn.Sequential(first, nn.GELU(), last)
        else:
            self.project = nn.Linear(width, 4 * config.model_dim)
            nn.init.zeros_(self.project.weight)
            nn.init.zeros_(self.project.bias)

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        ligand_graph: torch.Tensor,
        assay: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_dim = self.config.model_dim
        needed = self.knowledge_blocks * model_dim
        if ligand_graph.shape[-1] < needed:
            raise AdapterError(
                f"FiLM needs the first {self.knowledge_blocks} readout blocks "
                f"({needed} wide) but the adapted readout is "
                f"{ligand_graph.shape[-1]}; this arm declares fewer components "
                "than the conditioner expects"
            )
        context = ligand_graph[..., :needed]
        if self.assay_conditioning:
            if assay is None:
                raise AdapterError(
                    "FiLM was built with assay conditioning but no assay vector "
                    "was supplied. Silently conditioning on zeros would make "
                    "every measurement look like the same experiment."
                )
            context = torch.cat((context, assay), dim=-1)
        elif assay is not None:
            raise AdapterError(
                "an assay vector was supplied to a FiLM built without assay "
                "conditioning; it would be silently ignored"
            )
        knowledge = self.norm(context)
        scale_protein, shift_protein, scale_ligand, shift_ligand = torch.split(
            self.project(knowledge), model_dim, dim=-1
        )
        # [B, D] -> [B, 1, D], broadcasting over residues and over tokens.
        protein_out = protein * (1.0 + scale_protein.unsqueeze(-2)) + shift_protein.unsqueeze(-2)
        ligand_out = ligand * (1.0 + scale_ligand.unsqueeze(-2)) + shift_ligand.unsqueeze(-2)
        return protein_out, ligand_out


def masked_mean(
    features: torch.Tensor, mask: torch.Tensor, *, dim: int = -2
) -> torch.Tensor:
    """Mean over ``dim`` counting only masked-in positions.

    Padding is the whole point. A plain ``.mean()`` over a padded protein batch
    divides by the padded length, so every sequence shorter than the batch
    maximum is scaled down by a factor that depends on which other proteins
    shared its batch -- a silent, batch-composition-dependent corruption.
    """
    if mask.dtype is not torch.bool:
        raise AdapterError(f"mask must be bool, got {mask.dtype}")
    weights = mask.unsqueeze(-1).to(features.dtype)
    total = (features * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return total / count


def attribute_gradients(
    module: nn.Module,
    losses: Mapping[str, torch.Tensor],
    *,
    atol: float = 0.0,
) -> dict[str, tuple[str, ...]]:
    """Which parameters of ``module`` receive gradient from each loss.

    Discharges the specification's "log which parameters receive gradients from
    each loss". Each loss is backpropagated in isolation with gradients zeroed in
    between, so the result attributes parameters to losses rather than to the sum
    of them.

    Returns ``{loss_name: (parameter names with gradient, ...)}``, sorted.

    The graph is retained across losses, so the caller must pass losses computed
    from a single forward pass -- which is the realistic case, and also the only
    case where the question is meaningful.
    """
    names = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if not names:
        return {name: () for name in losses}

    attribution: dict[str, tuple[str, ...]] = {}
    ordered = list(losses.items())
    for position, (loss_name, loss) in enumerate(ordered):
        module.zero_grad(set_to_none=True)
        # The last backward may release the graph; earlier ones must not.
        loss.backward(retain_graph=position < len(ordered) - 1)
        touched = tuple(
            sorted(
                name
                for name, parameter in module.named_parameters()
                if parameter.grad is not None and parameter.grad.abs().sum().item() > atol
            )
        )
        attribution[loss_name] = touched
    module.zero_grad(set_to_none=True)
    return attribution


def assert_shared(
    attribution: Mapping[str, Sequence[str]], *, prefix: str, losses: Sequence[str]
) -> None:
    """Raise unless every named loss reaches at least one parameter under ``prefix``.

    The check the specification's "must actually be shared" reduces to. Called
    from tests and usable as a training-time assertion.
    """
    missing = [
        name
        for name in losses
        if not any(parameter.startswith(prefix) for parameter in attribution.get(name, ()))
    ]
    if missing:
        raise AdapterError(
            f"losses {missing} send no gradient to any parameter under {prefix!r}; "
            "the module is not shared by them"
        )


__all__ = [
    "AdapterConfig",
    "AdapterError",
    "FeatureAdapter",
    "SharedAdapters",
    "assert_shared",
    "attribute_gradients",
    "masked_mean",
]
