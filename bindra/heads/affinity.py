"""Separate pKd / pKi / pIC50 heads with endpoint masks.

Owed by the specification:
    Priority 2: an unmeasured endpoint must never receive a fabricated zero
    label.

## Why three heads rather than one output with an endpoint feature

pKd, pKi and pIC50 are not three measurements of one quantity. Kd and Ki are
binding constants; IC50 depends on substrate concentration, enzyme concentration
and mechanism, so the same complex can give IC50 values differing by orders of
magnitude between assays. The curation layer already refuses to merge them, and a
single regression output would re-merge them in the model after the data layer
took care not to.

The measured replicate noise says the same thing: sigma_rep is 0.945 for pKd,
0.641 for pKi and 0.671 for pIC50. One shared output would be fitted against
three different noise floors at once.

## How a fabricated zero is made structurally impossible

The head emits ``[B, 3]`` -- one value per endpoint for every pair -- because a
prediction for an unmeasured endpoint is a legitimate *output* (it is what
prospective use asks for). What must never exist is a *label* for it.

So no target ever lives in this module. :class:`AffinityPrediction` cannot be
compared to anything without an explicit endpoint index, and
:meth:`AffinityPrediction.observed` is the only way to extract the values a loss
may use. Reading the whole ``[B, 3]`` tensor into a loss is possible but requires
writing code that visibly ignores the endpoint index, rather than happening by
default through a zero-filled target tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bindra.data.endpoints import Endpoint

#: Column order of the head's output. Fixed, and recorded in every prediction, so
#: a checkpoint cannot be read with the columns permuted.
ENDPOINT_ORDER: tuple[Endpoint, ...] = (Endpoint.KD, Endpoint.KI, Endpoint.IC50)

#: ``endpoint value -> column``, for turning curated strings into indices.
ENDPOINT_INDEX: dict[str, int] = {
    endpoint.value: column for column, endpoint in enumerate(ENDPOINT_ORDER)
}


class AffinityHeadError(ValueError):
    """Shapes or endpoint indices inconsistent with the declared order."""


def endpoint_indices(endpoints: "list[str] | tuple[str, ...]") -> torch.Tensor:
    """Turn curated endpoint strings into column indices.

    An unknown endpoint raises rather than mapping to 0. Mapping it to 0 would
    train an unrelated head against it, which is the quiet version of the error
    this module exists to prevent.
    """
    try:
        return torch.tensor([ENDPOINT_INDEX[str(name)] for name in endpoints], dtype=torch.long)
    except KeyError as exc:
        raise AffinityHeadError(
            f"endpoint {exc.args[0]!r} is not one of {sorted(ENDPOINT_INDEX)}"
        ) from None


@dataclass(frozen=True)
class AffinityPrediction:
    """Predictions for all three endpoints, with no label anywhere in sight.

    ``values`` is ``[B, 3]`` in :data:`ENDPOINT_ORDER`. There is deliberately no
    ``__eq__`` against a tensor and no implicit flattening: a loss must name the
    endpoint it is scoring.
    """

    values: torch.Tensor  # [B, 3]
    endpoint_order: tuple[Endpoint, ...] = ENDPOINT_ORDER

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise AffinityHeadError(
                f"values must be [B, E], got {tuple(self.values.shape)}"
            )
        if self.values.shape[1] != len(self.endpoint_order):
            raise AffinityHeadError(
                f"values has {self.values.shape[1]} columns but endpoint_order names "
                f"{len(self.endpoint_order)}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    def observed(self, endpoint_index: torch.Tensor) -> torch.Tensor:
        """The single prediction per row that an actual measurement exists for.

        This is the only sanctioned path from a prediction to a loss. ``[B]`` out.
        """
        if endpoint_index.shape != (self.batch_size,):
            raise AffinityHeadError(
                f"endpoint_index must be [{self.batch_size}], got "
                f"{tuple(endpoint_index.shape)}"
            )
        if endpoint_index.dtype not in (torch.long, torch.int32, torch.int64):
            raise AffinityHeadError(
                f"endpoint_index must be an integer tensor, got {endpoint_index.dtype}"
            )
        # Range-checked only on CPU. On an accelerator ``int(tensor.max())`` is a
        # device-to-host synchronisation, and this runs once per training step --
        # measured at ~6 ms a call, which is more than the rest of the loss. The
        # check is not lost: targets are built on the CPU by
        # ``AffinityTargets.from_frame`` and ``endpoint_indices`` already refuses
        # an unknown endpoint there, and a bad index on device still faults in
        # ``gather`` rather than reading the wrong head.
        if (
            self.batch_size
            and endpoint_index.device.type == "cpu"
            and (
                int(endpoint_index.min()) < 0
                or int(endpoint_index.max()) >= len(self.endpoint_order)
            )
        ):
            raise AffinityHeadError(
                f"endpoint_index out of range for {len(self.endpoint_order)} endpoints"
            )
        return self.values.gather(1, endpoint_index.unsqueeze(1).to(torch.long)).squeeze(1)

    def for_endpoint(self, endpoint: Endpoint | str) -> torch.Tensor:
        """The whole column for one endpoint, for prospective scoring."""
        name = endpoint.value if isinstance(endpoint, Endpoint) else str(endpoint)
        try:
            column = ENDPOINT_INDEX[name]
        except KeyError:
            raise AffinityHeadError(
                f"endpoint {name!r} is not one of {sorted(ENDPOINT_INDEX)}"
            ) from None
        return self.values[:, column]


@dataclass(frozen=True)
class AffinityHeadConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    #: A separate trunk per endpoint. Sharing one trunk and splitting only at the
    #: final linear layer is cheaper, and is offered, but it ties the three
    #: endpoints' representations together -- the thing three heads exist to avoid.
    share_trunk: bool = False
    num_layers: int = 2

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise AffinityHeadError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.hidden_dim < 1:
            raise AffinityHeadError(f"hidden_dim must be >= 1, got {self.hidden_dim}")

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "share_trunk": self.share_trunk,
            "num_layers": self.num_layers,
            "endpoint_order": [endpoint.value for endpoint in ENDPOINT_ORDER],
        }


def _trunk(input_dim: int, config: AffinityHeadConfig) -> nn.Module:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(config.num_layers):
        layers += [nn.Linear(width, config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout)]
        width = config.hidden_dim
    return nn.Sequential(*layers)


class AffinityHeads(nn.Module):
    """One regression head per endpoint over a pooled pair representation.

    Output is in p-units (``9 - log10(nM)``) and unbounded. No sigmoid, no
    rescaling to a fixed range: the corpus spans roughly pX 2 to 12, and squashing
    into an assumed interval would make the extremes -- which is where selectivity
    and potency claims live -- unreachable by construction.
    """

    def __init__(self, input_dim: int, config: AffinityHeadConfig | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config or AffinityHeadConfig()
        if self.config.share_trunk:
            self.trunk: nn.Module = _trunk(input_dim, self.config)
            self.outputs = nn.ModuleList(
                nn.Linear(self.config.hidden_dim, 1) for _ in ENDPOINT_ORDER
            )
        else:
            self.trunk = nn.ModuleList(
                _trunk(input_dim, self.config) for _ in ENDPOINT_ORDER
            )
            self.outputs = nn.ModuleList(
                nn.Linear(self.config.hidden_dim, 1) for _ in ENDPOINT_ORDER
            )

    def forward(self, pair_representation: torch.Tensor) -> AffinityPrediction:
        if pair_representation.ndim != 2:
            raise AffinityHeadError(
                f"expected [B, D], got {tuple(pair_representation.shape)}"
            )
        if pair_representation.shape[1] != self.input_dim:
            raise AffinityHeadError(
                f"expected width {self.input_dim}, got {pair_representation.shape[1]}"
            )
        if self.config.share_trunk:
            shared = self.trunk(pair_representation)
            columns = [output(shared) for output in self.outputs]
        else:
            columns = [
                output(trunk(pair_representation))
                for trunk, output in zip(self.trunk, self.outputs)  # type: ignore[arg-type]
            ]
        return AffinityPrediction(values=torch.cat(columns, dim=1))


__all__ = [
    "ENDPOINT_INDEX",
    "ENDPOINT_ORDER",
    "AffinityHeadConfig",
    "AffinityHeadError",
    "AffinityHeads",
    "AffinityPrediction",
    "endpoint_indices",
]
