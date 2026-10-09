"""Training loop: AMP, EMA, resumable checkpoints, composed losses.

Owed by the specification:
    Port of the legacy engine, single-GPU only, with composed losses.

Single-GPU on purpose. The legacy engine wrapped the model in ``DataParallel``
over every visible device, and this machine has a 16 GiB RTX 5060 Ti alongside a
4 GiB T1200: sharding a batch across those shards it onto a card that cannot
hold it, at the speed of the slower one. `bindra/runtime/device.py` selects by
requirement instead, and nothing here iterates devices.

## What this loop refuses to do

**It never reads the test partition.** :class:`Trainer` takes a training loader
and an optional validation loader, and
:meth:`Trainer.fit` asserts both come from partitions named ``train`` and
``val``. Model selection, early stopping and every logged number come from
validation only. The locked benchmark panels are sealed behind
``bindra.eval.benchmarks.lock`` and this module does not import it.

**It does not quietly substitute random weights.** A requested checkpoint that
is absent is an error, per the specification's "do not silently replace an
unavailable pretrained checkpoint with random initialization".

## Resumability

A full epoch is ~97 minutes. Checkpoints therefore carry everything needed to
continue a run rather than restart it: model, optimizer, scheduler, EMA shadow,
epoch, global step, the best metric so far, and the torch/numpy RNG states. The
loader's shuffle is a pure function of ``(seed, epoch)``, so it needs no state
of its own. Writing is atomic -- a temp file and a rename -- so an interruption
mid-write leaves the previous checkpoint intact rather than a truncated one.

## Metrics

Exact-label error and censored-bound violations are reported separately, never
summed, because they answer different questions. Every RMSE is logged next to
the measured replicate noise floor for its endpoint (0.945 pKd, 0.641 pKi,
0.671 pIC50): a model at 0.95 pKd RMSE is at the noise floor, not skilful, and a
number without that comparison invites the wrong reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
import warnings
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from bindra.data.batching import PairBatch, PairLoader
from bindra.heads.affinity import ENDPOINT_ORDER
from bindra.training.losses.affinity import (
    AffinityLossConfig,
    affinity_loss,
    noise_floor_table,
)


NEWLINE = chr(10)


def _duration(seconds: float) -> str:
    """``1h02m``, ``7m13s``, ``42s`` -- short enough to sit in a progress line."""
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds:.0f}s"


def _ranking_config():
    from bindra.training.losses.ranking import RankingLossConfig

    return RankingLossConfig()


class TrainingError(RuntimeError):
    """The run could not proceed as configured."""


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that affects the result, declared in one place and hashed."""

    #: Peak rate after warmup. 5e-4 by choice, not by sweep -- no tuning run has
    #: been done, and the 3e-4 and 1e-3 that preceded it were equally
    #: unvalidated. It sits between them, which is the honest description.
    #:
    #: Read it together with ``accumulate_steps``: the physical batch averages
    #: 87 pairs, which is small enough that step-to-step gradient noise, rather
    #: than genuine outliers, is what makes `grad_clip` fire. Accumulation is
    #: what makes a rate in this range reasonable.
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    epochs: int = 10
    #: Fraction of total steps spent warming the learning rate from zero.
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    #: Micro-batches accumulated before each optimizer step. The *physical*
    #: batch is capped by VRAM -- the residue budget of 65,536 peaks at 6.66 GiB
    #: and doubling it exhausts the card, because fusion activations scale with
    #: ``pairs x padded_residues``. This raises the *optimizer* batch instead,
    #: at no memory cost: at the measured mean of 87 pairs per micro-batch, 4
    #: gives an effective ~348 and 8 gives ~700. Worth more than it looks at
    #: lr 1e-3, where a batch of 87 is small enough that step-to-step gradient
    #: noise is what triggers clipping.
    accumulate_steps: int = 1
    #: bf16 on this GPU, not fp16: bf16 has the same exponent range as fp32, so
    #: no loss scaler is needed and a large censored penalty cannot overflow to
    #: inf the way it can in fp16.
    amp_dtype: str = "bfloat16"
    use_amp: bool = True
    #: Exponential moving average of the weights, evaluated instead of the raw
    #: ones. 0 disables.
    ema_decay: float = 0.999
    #: Stop when validation has not improved for this many epochs. 0 disables.
    patience: int = 3
    #: What early stopping and checkpoint selection watch. Exact RMSE, because
    #: it is the only metric comparable to the measured noise floor.
    selection_metric: str = "exact_rmse"
    selection_mode: str = "min"
    seed: int = 20261001
    log_every: int = 50
    #: Per-step detail: the three loss terms separately, the pre-clip gradient
    #: norm, batch shape, and how much of each step was spent waiting for data
    #: rather than computing. Costs one extra host-device sync per logged step,
    #: so it is off by default and on when you are watching.
    verbose: bool = False
    #: Write ``last.pt`` every N **micro-batches**, not optimizer steps, so the
    #: cadence is how much *work* is at risk rather than something that
    #: stretches by ``accumulate_steps``. At the measured ~2.3 batches/s this is
    #: roughly two minutes. The write itself lands only on an accumulation
    #: boundary, so the saved optimizer state is never mid-group. 0 disables.
    #:
    #: An epoch here is hours: the first real run was interrupted 42 minutes in
    #: and lost all of it, because nothing had been written yet.
    checkpoint_every_batches: int = 250
    loss: AffinityLossConfig = field(default_factory=AffinityLossConfig)
    #: Within-assay pairwise ranking. **Weight 0 by default**, which makes this
    #: a no-op: adding a term changes the objective, and the comparison to a run
    #: without it is only meaningful if turning it on is deliberate.
    ranking: "RankingLossConfig" = field(default_factory=lambda: _ranking_config())

    def __post_init__(self) -> None:
        if self.amp_dtype not in ("bfloat16", "float16"):
            raise TrainingError(f"unknown amp_dtype {self.amp_dtype!r}")
        if self.selection_mode not in ("min", "max"):
            raise TrainingError(f"selection_mode must be min or max")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise TrainingError(f"warmup_fraction must be in [0, 1)")
        if not 0.0 <= self.ema_decay < 1.0:
            raise TrainingError(f"ema_decay must be in [0, 1)")
        if self.accumulate_steps < 1:
            raise TrainingError(
                f"accumulate_steps must be >= 1, got {self.accumulate_steps}"
            )

    @property
    def torch_amp_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "checkpoint_every_batches": self.checkpoint_every_batches,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "warmup_fraction": self.warmup_fraction,
            "grad_clip": self.grad_clip,
            "accumulate_steps": self.accumulate_steps,
            "amp_dtype": self.amp_dtype,
            "verbose": self.verbose,
            "use_amp": self.use_amp,
            "ema_decay": self.ema_decay,
            "patience": self.patience,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "seed": self.seed,
            "loss": self.loss.to_dict(),
            # Recorded even at weight 0, so a checkpoint states what objective
            # produced it rather than leaving a reader to assume the default.
            "ranking": self.ranking.to_dict(),
        }
        return payload


class ExponentialMovingAverage:
    """Shadow weights, kept in fp32 regardless of the model's dtype.

    Averaged weights usually generalise slightly better than the final step's,
    and cost one extra copy of the parameters. Buffers are copied rather than
    averaged: a running normalisation statistic is already an average, and
    averaging it again would lag the data twice.
    """

    def __init__(self, module: nn.Module, decay: float, *, warmup: int = 10) -> None:
        self.decay = decay
        #: Steps over which the effective decay ramps up. Without this the
        #: shadow is dominated by the random initialisation for thousands of
        #: steps: at decay 0.999, after 66 steps it is still 94% initial
        #: weights, so validation scores an untrained model and reports a
        #: constant-offset RMSE of ~6.5 against a train RMSE of ~1.8. Model
        #: selection over the first epochs would be meaningless.
        self.warmup = warmup
        self.steps = 0
        self.shadow = {
            name: parameter.detach().clone().float()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    def effective_decay(self) -> float:
        """Ramps from ~0 to ``decay``, so early averages track the live weights.

        The same correction Adam applies to its moments, in the form used by
        most EMA implementations: ``(1 + t) / (warmup + t)`` capped at ``decay``.
        """
        return min(self.decay, (1.0 + self.steps) / (self.warmup + self.steps))

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        decay = self.effective_decay()
        self.steps += 1
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    parameter.detach().float(), alpha=1.0 - decay
                )

    @torch.no_grad()
    def swap_in(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Install the shadow weights, returning the originals to restore."""
        backup: dict[str, torch.Tensor] = {}
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name].to(parameter.dtype))
        return backup

    @torch.no_grad()
    def restore(self, module: nn.Module, backup: Mapping[str, torch.Tensor]) -> None:
        for name, parameter in module.named_parameters():
            if name in backup:
                parameter.copy_(backup[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": {name: tensor.cpu() for name, tensor in self.shadow.items()},
            "steps": self.steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Resuming must restore the step count too, or the effective decay
        # restarts its warmup and the shadow is pulled back toward the live
        # weights it had already averaged away from.
        shadow = state.get("shadow", state)
        self.steps = int(state.get("steps", 0)) if isinstance(state, Mapping) else 0
        for name, tensor in shadow.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@dataclass
class EpochReport:
    """What one epoch produced. Serialised verbatim into the run log."""

    epoch: int
    split: str
    n_pairs: int
    n_exact: int
    n_censored: int
    loss: float
    exact_rmse: float
    exact_mae: float
    bound_violation_rate: float
    per_endpoint: Mapping[str, Mapping[str, float]]
    seconds: float
    pairs_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "split": self.split,
            "n_pairs": self.n_pairs,
            "n_exact": self.n_exact,
            "n_censored": self.n_censored,
            "loss": self.loss,
            "exact_rmse": self.exact_rmse,
            "exact_mae": self.exact_mae,
            "bound_violation_rate": self.bound_violation_rate,
            "per_endpoint": {k: dict(v) for k, v in self.per_endpoint.items()},
            "noise_floor": noise_floor_table(),
            "seconds": round(self.seconds, 1),
            "pairs_per_second": round(self.pairs_per_second, 1),
        }

    def summary(self) -> str:
        """A short line, for a tail -f."""
        floors = noise_floor_table()
        parts = []
        for endpoint in ENDPOINT_ORDER:
            entry = self.per_endpoint.get(endpoint.value, {})
            if entry.get("n_exact"):
                ratio = entry["rmse"] / floors[endpoint.value]
                parts.append(f"{endpoint.value} {entry['rmse']:.3f} ({ratio:.2f}x)")
        return (
            f"epoch {self.epoch} {self.split:<5}: loss {self.loss:.4f} "
            f"rmse {self.exact_rmse:.3f} viol {self.bound_violation_rate:.3f} "
            f"| {self.n_pairs:,} pairs in {self.seconds:.0f}s "
            f"({self.pairs_per_second:.0f}/s) | " + " ".join(parts)
        )

    def table(self) -> str:
        """The full per-endpoint breakdown, printed at the end of each epoch.

        Every RMSE sits next to the measured replicate noise floor for its
        endpoint. A ratio at or below 1.0 means the model is at the limit of
        what repeated measurements of the same quantity agree on, which is the
        point past which a smaller number is not a better model.
        """
        floors = noise_floor_table()
        lines = [
            f"  +-- epoch {self.epoch} {self.split} "
            f"({self.n_pairs:,} pairs, {self.n_exact:,} exact, "
            f"{self.n_censored:,} censored, {self.seconds:.0f}s "
            f"@ {self.pairs_per_second:.0f} pairs/s)",
            f"  |   loss {self.loss:.4f}   exact RMSE {self.exact_rmse:.4f}   "
            f"MAE {self.exact_mae:.4f}   bound violations {self.bound_violation_rate:.4f}",
            f"  |   {'endpoint':<8} {'n exact':>9} {'RMSE':>8} {'floor':>7} "
            f"{'x floor':>8} {'MAE':>7} {'n cens':>8} {'viol':>7}",
        ]
        for endpoint in ENDPOINT_ORDER:
            entry = self.per_endpoint.get(endpoint.value, {})
            exact = int(entry.get("n_exact", 0))
            censored = int(entry.get("n_censored", 0))
            if not exact and not censored:
                continue
            floor = floors[endpoint.value]
            rmse = entry.get("rmse")
            lines.append(
                f"  |   {endpoint.value:<8} {exact:>9,} "
                + (f"{rmse:>8.4f} {floor:>7.3f} {rmse / floor:>8.2f} " if rmse else
                   f"{'-':>8} {floor:>7.3f} {'-':>8} ")
                + (f"{entry['mae']:>7.4f} " if entry.get("mae") else f"{'-':>7} ")
                + f"{censored:>8,} "
                + (f"{entry['bound_violation_rate']:>7.3f}"
                   if entry.get("bound_violation_rate") is not None and censored
                   else f"{'-':>7}")
            )
        return NEWLINE.join(lines)


class _MetricAccumulator:
    """Sums over an epoch rather than averaging per-batch averages.

    Batches here differ in size by more than an order of magnitude, because the
    residue budget gives short proteins far more pairs. Averaging batch means
    would weight a 1-pair batch holding a 34,350-residue protein the same as a
    256-pair batch, which silently reweights the metric toward long proteins.
    """

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.n_pairs = 0
        self.n_exact = 0
        self.n_censored = 0
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.violations = 0
        self.per_endpoint: dict[str, dict[str, float]] = {
            endpoint.value: {"n_exact": 0.0, "sq": 0.0, "abs": 0.0,
                             "n_censored": 0.0, "violations": 0.0}
            for endpoint in ENDPOINT_ORDER
        }

    def update(self, output, batch: PairBatch) -> None:
        self.loss_sum += float(output.total.detach()) * batch.size
        self.n_pairs += batch.size
        self.n_exact += output.n_exact
        self.n_censored += output.n_censored
        self.squared_error += float(output.exact_rmse.detach()) ** 2 * output.n_exact
        self.absolute_error += float(output.exact_mae.detach()) * output.n_exact
        self.violations += int(
            round(float(output.bound_violation_rate.detach()) * output.n_censored)
        )
        for name, entry in output.per_endpoint.items():
            target = self.per_endpoint[name]
            exact = entry.get("n_exact", 0)
            target["n_exact"] += exact
            if exact:
                target["sq"] += entry["rmse"] ** 2 * exact
                target["abs"] += entry["mae"] * exact
            censored = entry.get("n_censored", 0)
            target["n_censored"] += censored
            if censored:
                target["violations"] += entry.get("bound_violation_rate", 0.0) * censored

    def snapshot(self) -> dict[str, float]:
        """Epoch-to-date figures for a progress line.

        Running totals, not the last batch: a single batch can be 1 pair or 256,
        so its loss says very little about where the epoch is.
        """
        return {
            "loss": self.loss_sum / max(1, self.n_pairs),
            "exact_rmse": math.sqrt(self.squared_error / max(1, self.n_exact)),
            "bound_violation_rate": self.violations / max(1, self.n_censored),
            "n_pairs": self.n_pairs,
            "n_exact": self.n_exact,
            "n_censored": self.n_censored,
        }

    def report(self, epoch: int, split: str, seconds: float) -> EpochReport:
        per_endpoint = {}
        for name, entry in self.per_endpoint.items():
            summary: dict[str, float] = {
                "n_exact": int(entry["n_exact"]),
                "n_censored": int(entry["n_censored"]),
            }
            if entry["n_exact"]:
                summary["rmse"] = math.sqrt(entry["sq"] / entry["n_exact"])
                summary["mae"] = entry["abs"] / entry["n_exact"]
            if entry["n_censored"]:
                summary["bound_violation_rate"] = entry["violations"] / entry["n_censored"]
            per_endpoint[name] = summary
        return EpochReport(
            epoch=epoch,
            split=split,
            n_pairs=self.n_pairs,
            n_exact=self.n_exact,
            n_censored=self.n_censored,
            loss=self.loss_sum / max(1, self.n_pairs),
            exact_rmse=math.sqrt(self.squared_error / max(1, self.n_exact)),
            exact_mae=self.absolute_error / max(1, self.n_exact),
            bound_violation_rate=self.violations / max(1, self.n_censored),
            per_endpoint=per_endpoint,
            seconds=seconds,
            pairs_per_second=self.n_pairs / max(1e-9, seconds),
        )


class Trainer:
    """Single-GPU training over :class:`PairLoader` batches."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        run_directory: Path | str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.run_directory = Path(run_directory) if run_directory else None
        if self.run_directory:
            self.run_directory.mkdir(parents=True, exist_ok=True)
            for stale in self.run_directory.glob("*.pt.tmp"):
                print(
                    f"  [ckpt] found {stale.name} from an interrupted write. It is a "
                    f"complete checkpoint whose rename failed; rename it by hand if it "
                    f"is newer than {stale.name[:-4]}.",
                    flush=True,
                )
        #: Hashes identifying the corpus, splits and encoders, written into every
        #: checkpoint so a result can be traced back to what produced it.
        self.provenance = dict(provenance or {})

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.ema = (
            ExponentialMovingAverage(model, self.config.ema_decay)
            if self.config.ema_decay
            else None
        )
        self.epoch = 0
        self.global_step = 0
        #: Batches already consumed in the current epoch, for a mid-epoch resume.
        self.batch_in_epoch = 0
        #: Batches dropped to OOM in the last epoch. Reported, never hidden.
        self.skipped_batches = 0
        self.best_metric: float | None = None
        self.history: list[dict[str, Any]] = []
        #: Last batch's ranking diagnostics, for the step logger. None until the
        #: term runs, which at weight 0 is never.
        self.last_ranking = None

    # -- the loop ---------------------------------------------------------

    def fit(
        self,
        train_loader: PairLoader,
        val_loader: PairLoader | None = None,
        *,
        on_epoch: Callable[[EpochReport], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Train, validating after each epoch. Never touches the test partition."""
        self._assert_partitions(train_loader, val_loader)
        total_steps = max(
            1,
            math.ceil(len(train_loader) / self.config.accumulate_steps)
            * self.config.epochs,
        )
        # Rebuilt every time, then fast-forwarded to where the run actually is,
        # rather than built only when absent.
        #
        # A resumed run restores the scheduler from its checkpoint, so a
        # schedule sized for the *old* epoch count would survive a change to
        # --epochs: ask for 5 after training 3 and the rate would anneal to zero
        # at the end of epoch 2 and stay there, with nothing to show it. Since
        # the cosine depends only on (step, total_steps), reconstructing and
        # replaying is exact -- resuming with an unchanged budget reproduces the
        # loaded state precisely, so this costs nothing when nothing changed.
        rebuilt = cosine_schedule_with_warmup(
            self.optimizer,
            total_steps,
            int(total_steps * self.config.warmup_fraction),
        )
        # Replaying the schedule without a matching optimizer.step() trips
        # torch's "called step() before optimizer.step()" warning. Here that is
        # exactly the intent -- the schedule is being fast-forwarded, not used
        # to train -- so the warning is noise on every resumed run.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
            for _ in range(self.global_step):
                rebuilt.step()
        if self.scheduler is not None and self.global_step:
            previous = self.optimizer.param_groups[0]["lr"]
            print(
                f"  schedule spans {total_steps:,} optimizer steps for "
                f"{self.config.epochs} epochs; resuming at step "
                f"{self.global_step:,} with lr {previous:.2e}",
                flush=True,
            )
        self.scheduler = rebuilt

        stale = 0
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config.epochs):
            self.epoch = epoch
            train_loader.set_epoch(epoch)
            report = self._train_epoch(train_loader)
            self.history.append(report.to_dict())
            print(report.table(), flush=True)

            if val_loader is not None:
                validation = self.evaluate(val_loader, split="val")
                self.history.append(validation.to_dict())
                print(validation.table(), flush=True)
                if on_epoch:
                    on_epoch(validation)
                improved = self._record_best(validation)
                stale = 0 if improved else stale + 1
                marker = "BEST" if improved else f"no improvement ({stale}/{self.config.patience})"
                print(
                    f"  +-- val {self.config.selection_metric} "
                    f"{getattr(validation, self.config.selection_metric):.4f}"
                    f" | best {self.best_metric:.4f} | {marker}",
                    flush=True,
                )
                if self.config.patience and stale >= self.config.patience:
                    print(
                        f"early stop: {self.config.selection_metric} has not improved "
                        f"for {stale} epochs",
                        flush=True,
                    )
                    break
            self.save_checkpoint("last.pt")
            self._write_history()
        return self.history

    def _train_epoch(self, loader: PairLoader) -> EpochReport:
        self.model.train()
        metrics = _MetricAccumulator()
        started = time.perf_counter()
        n_batches = len(loader)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        waiting = 0.0
        micro = 0
        skipped = 0
        skipped_pairs = 0
        accumulate = self.config.accumulate_steps
        last_tick = time.perf_counter()
        skip = self.batch_in_epoch
        # Told to the loader rather than filtered here. Iterating and dropping
        # the first `skip` batches would *build* every one of them first --
        # 25,504 batches is about three hours of assembling work to throw away,
        # and silently, since the progress line sits after the filter.
        loader.start_at = skip
        if skip:
            print(
                f"  resuming epoch {self.epoch} at batch {skip:,} of "
                f"{n_batches:,} ({100 * skip / max(1, n_batches):.1f}% done); "
                f"{n_batches - skip:,} batches remain this epoch",
                flush=True,
            )
        for offset, batch in enumerate(loader):
            index = offset + skip
            # Everything between the end of the last step and here was spent
            # waiting on the loader. The split between this and compute is the
            # whole question when throughput disappoints.
            waiting += time.perf_counter() - last_tick
            if micro == 0:
                self.optimizer.zero_grad(set_to_none=True)
            try:
                batch = batch.to(self.device)
                with torch.autocast(
                    self.device.type,
                    dtype=self.config.torch_amp_dtype,
                    enabled=self.config.use_amp and self.device.type == "cuda",
                ):
                    output = self.model(
                        batch.protein_features, batch.protein_mask, batch.ligands
                    )
                    loss = affinity_loss(
                        output.prediction, batch.targets, config=self.config.loss
                    )
                    loss = self._add_ranking(loss, output, batch)
                # Scaled so the accumulated gradient is the mean over
                # micro-batches rather than their sum, which would scale the
                # effective rate by accumulate_steps and undo the point of
                # raising the batch.
                (loss.total / accumulate).backward()
            except torch.OutOfMemoryError:
                # One batch is 1/22,044 of an epoch; the run is hours. Dying
                # here to save a single batch is the wrong trade, so the batch
                # is dropped, counted and reported -- never silently swallowed.
                # The partial gradient from this accumulation group is discarded
                # with it, so no half-batch contributes a skewed step.
                skipped += 1
                skipped_pairs += batch.size
                del batch
                self.optimizer.zero_grad(set_to_none=True)
                micro = 0
                torch.cuda.empty_cache()
                print(
                    f"  [oom] skipped batch {index} ({skipped} so far, "
                    f"{skipped_pairs:,} pairs); emptied the cache and continued",
                    flush=True,
                )
                last_tick = time.perf_counter()
                continue
            micro += 1

            grad_norm = None
            stepped = micro == accumulate or index == n_batches - 1
            if stepped:
                if self.config.grad_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.global_step += 1
                micro = 0
            metrics.update(loss, batch)
            if self.config.log_every and (
                index % self.config.log_every == 0 or index == n_batches - 1
            ):
                self._log_step(
                    index, n_batches, metrics, started,
                    loss=loss, grad_norm=grad_norm, batch=batch, waiting=waiting,
                )
            if (
                stepped
                and self.config.checkpoint_every_batches
                and (index + 1) % self.config.checkpoint_every_batches < accumulate
            ):
                # Mid-epoch, so an interruption costs minutes rather than hours.
                # `batch_in_epoch` lets a resume skip what it already saw instead
                # of silently replaying it, which would weight those pairs twice.
                self.batch_in_epoch = index + 1
                self.save_checkpoint("last.pt")
            last_tick = time.perf_counter()
        self.batch_in_epoch = 0
        loader.start_at = 0
        if skipped:
            print(
                f"  [oom] {skipped} batch(es), {skipped_pairs:,} pairs, were "
                f"dropped this epoch for lack of memory",
                flush=True,
            )
        self.skipped_batches = skipped
        return metrics.report(self.epoch, "train", time.perf_counter() - started)

    def _log_step(
        self,
        index: int,
        n_batches: int,
        metrics: "_MetricAccumulator",
        started: float,
        *,
        loss=None,
        grad_norm=None,
        batch=None,
        waiting: float = 0.0,
    ) -> None:
        """One progress line: where the epoch is, how fast, and how it is doing.

        Reports epoch-to-date running metrics rather than the current batch,
        and an ETA from the rate actually achieved so far rather than from the
        first few steps, which are always slower.
        """
        elapsed = time.perf_counter() - started
        done = index + 1
        rate = metrics.n_pairs / max(1e-9, elapsed)
        remaining = (n_batches - done) * (elapsed / done)
        running = metrics.snapshot()
        floors = noise_floor_table()

        per_endpoint = []
        for endpoint in ENDPOINT_ORDER:
            entry = metrics.per_endpoint[endpoint.value]
            if entry["n_exact"]:
                rmse = math.sqrt(entry["sq"] / entry["n_exact"])
                per_endpoint.append(
                    f"{endpoint.value} {rmse:.3f}/{rmse / floors[endpoint.value]:.2f}x"
                )

        memory = ""
        if self.device.type == "cuda":
            memory = f" | {torch.cuda.max_memory_allocated(self.device) / 2**30:.1f}G"

        detail = ""
        if self.config.verbose and loss is not None:
            share = 100 * waiting / max(1e-9, elapsed)
            parts = [
                f"ex {loss.exact_loss.detach().item():.4f}",
                f"cen {loss.censored_loss.detach().item():.4f}",
                f"aux {loss.auxiliary_loss.detach().item():.4f}",
            ]
            if grad_norm is not None:
                norm = float(grad_norm.detach() if hasattr(grad_norm, "detach") else grad_norm)
                clipped = "*" if norm > self.config.grad_clip else " "
                parts.append(f"|g| {norm:.3f}{clipped}")
            if batch is not None:
                # Asked of the batch rather than assumed: the arm owns its own
                # batch type, and a diagnostic must never be the thing that
                # ends a training run.
                tokens = getattr(batch.ligands, "total_tokens", None)
                size = "?" if tokens is None else f"{tokens}"
                parts.append(
                    f"b {batch.size}x{batch.padded_residues}"
                    f" pad {batch.padding_fraction:.2f}"
                    f" tok {size}"
                )
            if self.last_ranking is not None:
                # Pairs and concordance, not just the term: a ranking loss that
                # never finds a comparable pair reports 0.0000 and is
                # indistinguishable from one the model has already satisfied.
                ranked = self.last_ranking
                parts.append(
                    f"rank {float(ranked.loss.detach()):.4f}"
                    f" ({ranked.n_pairs}/{ranked.n_pairs_available} pairs,"
                    f" {ranked.n_groups} grp, c-idx {ranked.concordance:.3f})"
                )
            parts.append(f"wait {share:.0f}%")
            detail = NEWLINE + "        " + "  ".join(parts)

        print(
            f"  e{self.epoch} {done:>5}/{n_batches} ({100 * done / n_batches:5.1f}%)"
            f" step {self.global_step:>6}"
            f" | loss {running['loss']:.4f}"
            f" rmse {running['exact_rmse']:.3f}"
            f" viol {running['bound_violation_rate']:.3f}"
            f" | {running['n_exact']:,}ex {running['n_censored']:,}cen"
            f" | {rate:5.0f} pair/s"
            f" | {_duration(elapsed)} elapsed, {_duration(remaining)} left"
            f" | lr {self.optimizer.param_groups[0]['lr']:.2e}{memory}"
            + (" | " + " ".join(per_endpoint) if per_endpoint else "")
            + detail,
            flush=True,
        )

    @torch.no_grad()
    def evaluate(
        self, loader: PairLoader, *, split: str = "val", use_ema: bool = True
    ) -> EpochReport:
        """Evaluate, under the EMA weights when they exist."""
        self.model.eval()
        backup = None
        if use_ema and self.ema is not None:
            backup = self.ema.swap_in(self.model)
        metrics = _MetricAccumulator()
        started = time.perf_counter()
        try:
            for batch in loader:
                batch = batch.to(self.device)
                with torch.autocast(
                    self.device.type,
                    dtype=self.config.torch_amp_dtype,
                    enabled=self.config.use_amp and self.device.type == "cuda",
                ):
                    output = self.model(
                        batch.protein_features, batch.protein_mask, batch.ligands
                    )
                    loss = affinity_loss(
                        output.prediction, batch.targets, config=self.config.loss
                    )
                metrics.update(loss, batch)
        finally:
            if backup is not None and self.ema is not None:
                self.ema.restore(self.model, backup)
        return metrics.report(self.epoch, split, time.perf_counter() - started)

    def _add_ranking(self, loss, output, batch):
        """Add the within-assay ranking term, when it is switched on.

        Returns the loss object with ``total`` increased and the ranking
        diagnostics attached. At weight 0 -- the default -- this returns the
        input untouched and costs one float comparison, so a run without the
        term is bit-identical to one from before the term existed.

        The term is added to the **training** loss only. Validation keeps
        reporting the affinity loss alone, because early stopping selects on
        ``exact_rmse`` and a validation number that silently changed meaning
        when a flag was flipped would make two runs incomparable.
        """
        import dataclasses

        from bindra.training.losses.ranking import ranking_loss

        if not self.config.ranking.weight:
            return loss
        ranked = ranking_loss(
            output.prediction.observed(batch.targets.endpoint_index),
            batch.targets,
            self.config.ranking,
        )
        self.last_ranking = ranked
        if not ranked.n_pairs:
            return loss
        return dataclasses.replace(
            loss, total=loss.total + self.config.ranking.weight * ranked.loss
        )

    # -- discipline -------------------------------------------------------

    @staticmethod
    def _assert_partitions(train_loader: PairLoader, val_loader: PairLoader | None) -> None:
        """Refuse to train on, or select against, anything but train and val.

        The cheapest possible guard against the most expensive possible mistake.
        Held-out labels entering a training loss or a model-selection decision
        invalidates every number the run produces, and nothing downstream would
        reveal it.
        """
        if train_loader.dataset.partition != "train":
            raise TrainingError(
                f"training loader is on partition "
                f"{train_loader.dataset.partition!r}, expected 'train'"
            )
        if val_loader is not None and val_loader.dataset.partition != "val":
            raise TrainingError(
                f"validation loader is on partition "
                f"{val_loader.dataset.partition!r}, expected 'val'. Test and "
                "calibration partitions must not be read during training."
            )

    def _record_best(self, report: EpochReport) -> bool:
        value = getattr(report, self.config.selection_metric, None)
        if value is None:
            raise TrainingError(
                f"selection_metric {self.config.selection_metric!r} is not on EpochReport"
            )
        better = (
            self.best_metric is None
            or (self.config.selection_mode == "min" and value < self.best_metric)
            or (self.config.selection_mode == "max" and value > self.best_metric)
        )
        if better:
            self.best_metric = float(value)
            self.save_checkpoint("best.pt")
        return better

    # -- checkpoints ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "ema": None if self.ema is None else self.ema.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "batch_in_epoch": self.batch_in_epoch,
            "best_metric": self.best_metric,
            "history": self.history,
            "config": self.config.to_dict(),
            "provenance": self.provenance,
            # Resuming must continue the same stream, not restart it.
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }

    def save_checkpoint(self, name: str = "last.pt", *, attempts: int = 6) -> Path | None:
        """Write to a temp file, then rename. Retries, and never kills the run.

        The rename is the fragile part on Windows. A cloud-sync client -- the
        project's own tree lives under OneDrive -- notices an 87 MB file appear
        and opens it to upload, and ``replace`` then fails with "the process
        cannot access the file because it is being used by another process".
        That is exactly how the second real run died: ``last.pt`` written at
        14:03:15, a complete ``last.pt.tmp`` at 14:04:20, and an unhandled
        PermissionError that ended the run an hour and a half into epoch 1.

        The lock is transient, so the rename is retried with backoff. If it
        still fails, the run **continues** and says so: losing a checkpoint
        costs minutes, while raising here costs hours of training. Keeping the
        run alive is worth more than any single save.
        """
        if self.run_directory is None:
            return None
        path = self.run_directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(self.state_dict(), temporary)
        except OSError as exc:
            print(f"  [ckpt] could not write {temporary.name}: {exc}", flush=True)
            return None

        for attempt in range(attempts):
            try:
                temporary.replace(path)
                return path
            except PermissionError:
                if attempt == attempts - 1:
                    break
                time.sleep(0.5 * 2**attempt)
        print(
            f"  [ckpt] {path.name} is locked by another process (cloud sync?); "
            f"kept {temporary.name} and continued. The previous checkpoint is intact.",
            flush=True,
        )
        return None

    def load_checkpoint(self, path: Path | str, *, strict: bool = True) -> None:
        """Resume a run, including its RNG stream."""
        path = Path(path)
        if not path.exists():
            raise TrainingError(
                f"no checkpoint at {path}. Refusing to start from random weights "
                "when a checkpoint was requested."
            )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"], strict=strict)
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            if self.scheduler is None:
                self.scheduler = cosine_schedule_with_warmup(self.optimizer, 1, 0)
            self.scheduler.load_state_dict(payload["scheduler"])
        if payload.get("ema") is not None and self.ema is not None:
            self.ema.load_state_dict(payload["ema"])
        # A checkpoint written mid-epoch resumes *inside* that epoch; one
        # written at an epoch boundary (batch_in_epoch == 0) moves to the next.
        self.batch_in_epoch = int(payload.get("batch_in_epoch", 0))
        self.epoch = int(payload["epoch"]) + (0 if self.batch_in_epoch else 1)
        self.global_step = int(payload["global_step"])
        self.best_metric = payload.get("best_metric")
        self.history = list(payload.get("history", []))
        if payload.get("torch_rng") is not None:
            torch.set_rng_state(payload["torch_rng"].cpu().to(torch.uint8))
        if payload.get("numpy_rng") is not None:
            np.random.set_state(payload["numpy_rng"])

    def _write_history(self) -> None:
        if self.run_directory is None:
            return
        path = self.run_directory / "history.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "config": self.config.to_dict(),
                        "provenance": self.provenance,
                        "noise_floor": noise_floor_table(),
                        "epochs": self.history,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            # Same cloud-sync lock as the checkpoint rename, and the same
            # judgement: a run log is not worth ending a run over. The history
            # also lives in the checkpoints, so nothing is lost permanently.
            for attempt in range(4):
                try:
                    temporary.replace(path)
                    return
                except PermissionError:
                    if attempt == 3:
                        raise
                    time.sleep(0.5 * 2**attempt)
        except OSError as exc:
            print(f"  [hist] could not update history.json ({exc}); continuing", flush=True)


__all__ = [
    "EpochReport",
    "ExponentialMovingAverage",
    "Trainer",
    "TrainingConfig",
    "TrainingError",
    "cosine_schedule_with_warmup",
]
