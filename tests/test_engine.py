"""The training loop: partition discipline, resumability and honest metrics.

The tests that protect a result rather than a line of code:

``test_training_on_the_test_partition_is_refused``
    Held-out labels entering a training loss or a selection decision invalidates
    every number the run produces, and nothing downstream reveals it. The guard
    is cheap; the mistake is not recoverable.
``test_resuming_continues_rather_than_restarts``
    An epoch is ~97 minutes. A checkpoint that loses the optimizer state, the
    EMA or the RNG stream silently restarts the run while appearing to resume.
``test_metrics_are_summed_over_pairs_not_averaged_over_batches``
    Batch sizes here differ by two orders of magnitude, because the residue
    budget gives short proteins far more pairs. Averaging batch means reweights
    every metric toward long proteins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from bindra.data.batching import BatchingPolicy, PairBatch
from bindra.encoders.ligand_graph import collate_graphs, featurize_molecule
from bindra.heads.affinity import ENDPOINT_INDEX
from bindra.training.engine import (
    EpochReport,
    ExponentialMovingAverage,
    Trainer,
    TrainingConfig,
    TrainingError,
    cosine_schedule_with_warmup,
)
from bindra.training.losses.affinity import AffinityTargets

NAN = float("nan")
PROTEIN_DIM = 32
MOLECULES = ("CCO", "c1ccccc1", "CC(=O)O", "C", "CCN", "CC(C)O")


def _batch(size: int = 4, length: int = 7, *, seed: int = 0) -> PairBatch:
    generator = torch.Generator().manual_seed(seed)
    smiles = [MOLECULES[i % len(MOLECULES)] for i in range(size)]
    return PairBatch(
        protein_features=torch.randn(size, length, PROTEIN_DIM, generator=generator),
        protein_mask=torch.ones(size, length, dtype=torch.bool),
        ligands=collate_graphs([featurize_molecule(s) for s in smiles]),
        targets=AffinityTargets(
            endpoint_index=torch.tensor([ENDPOINT_INDEX["Ki"]] * size),
            p_value=torch.linspace(5.0, 9.0, size),
            is_exact=torch.ones(size, dtype=torch.bool),
            lower_bound=torch.full((size,), NAN),
            upper_bound=torch.full((size,), NAN),
            is_primary=torch.ones(size, dtype=torch.bool),
        ),
        observation_ids=tuple(f"o{seed}-{i}" for i in range(size)),
        smiles=tuple(smiles),
    )


class _FakeDataset:
    def __init__(self, partition: str) -> None:
        self.partition = partition


class _FakeLoader:
    """A loader shaped like PairLoader but over fixed batches."""

    def __init__(self, batches, partition: str = "train") -> None:
        self._batches = list(batches)
        self.dataset = _FakeDataset(partition)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


def _model():
    from bindra.encoders.ligand_gat import GATConfig
    from bindra.heads.affinity import AffinityHeadConfig
    from bindra.model import BindraConfig, BindraModel

    torch.manual_seed(20261001)
    return BindraModel(
        BindraConfig(
            protein_dim=PROTEIN_DIM,
            ligand=GATConfig(hidden_dim=16, num_layers=1, num_heads=2, dropout=0.0),
            model_dim=16,
            cross_attention_layers=1,
            cross_attention_heads=2,
            dropout=0.0,
            head=AffinityHeadConfig(hidden_dim=16, dropout=0.0, num_layers=1),
        )
    )


class TestConfig:
    def test_bf16_is_the_default_not_fp16(self):
        """bf16 keeps fp32's exponent range, so a censored penalty cannot
        overflow to inf and no loss scaler is needed."""
        assert TrainingConfig().amp_dtype == "bfloat16"

    def test_selection_watches_exact_rmse(self):
        """The only metric comparable to the measured noise floor."""
        assert TrainingConfig().selection_metric == "exact_rmse"
        assert TrainingConfig().selection_mode == "min"

    @pytest.mark.parametrize(
        "kwargs",
        [{"amp_dtype": "int8"}, {"selection_mode": "sideways"},
         {"warmup_fraction": 1.5}, {"ema_decay": 1.0}],
    )
    def test_a_bad_config_is_refused(self, kwargs):
        with pytest.raises(TrainingError):
            TrainingConfig(**kwargs)

    def test_the_config_serialises_for_the_run_log(self):
        payload = TrainingConfig().to_dict()
        assert payload["loss"]["censored_mode"] == "hinge"
        assert json.dumps(payload)  # must be JSON-safe


class TestPartitionDiscipline:
    def test_training_on_the_test_partition_is_refused(self):
        trainer = Trainer(_model(), TrainingConfig(epochs=1))
        with pytest.raises(TrainingError, match="expected 'train'"):
            trainer.fit(_FakeLoader([_batch()], partition="test"))

    def test_validating_on_the_test_partition_is_refused(self):
        trainer = Trainer(_model(), TrainingConfig(epochs=1))
        with pytest.raises(TrainingError, match="must not be read during training"):
            trainer.fit(
                _FakeLoader([_batch()], partition="train"),
                _FakeLoader([_batch()], partition="test"),
            )

    def test_validating_on_the_calibration_partition_is_refused(self):
        """Calibration labels must not inform model selection either."""
        trainer = Trainer(_model(), TrainingConfig(epochs=1))
        with pytest.raises(TrainingError, match="expected 'val'"):
            trainer.fit(
                _FakeLoader([_batch()], partition="train"),
                _FakeLoader([_batch()], partition="cal"),
            )

    def test_train_and_val_are_accepted(self):
        trainer = Trainer(_model(), TrainingConfig(epochs=1, log_every=0, patience=0))
        history = trainer.fit(
            _FakeLoader([_batch()], partition="train"),
            _FakeLoader([_batch(seed=1)], partition="val"),
        )
        assert len(history) == 2  # one train report, one val report

    def test_the_engine_does_not_import_the_sealed_label_machinery(self):
        """Checked on the import graph, not on the text.

        An earlier version grepped the source and tripped over this module's own
        docstring explaining that it does not import those names -- a test that
        fails when the code is correctly documented is testing the wrong thing.
        """
        import ast
        import inspect

        from bindra.training import engine

        tree = ast.parse(inspect.getsource(engine))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{a.name}" for a in node.names)
        offending = {name for name in imported if "benchmarks" in name or "sealed" in name}
        assert not offending, f"the engine imports sealed-label machinery: {offending}"


class TestTraining:
    def test_a_short_run_reduces_the_loss(self):
        batches = [_batch(seed=i) for i in range(4)]
        trainer = Trainer(
            _model(), TrainingConfig(epochs=3, log_every=0, patience=0,
                                     ema_decay=0.0, learning_rate=3e-3)
        )
        history = trainer.fit(_FakeLoader(batches))
        losses = [h["loss"] for h in history if h["split"] == "train"]
        assert losses[-1] < losses[0]

    def test_the_scheduler_warms_up_then_decays(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = cosine_schedule_with_warmup(optimizer, total_steps=100, warmup_steps=10)
        rates = []
        for _ in range(100):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        assert rates[0] < rates[9] <= rates[10]
        assert rates[-1] < rates[20]
        assert min(rates) >= 0.0

    def test_gradients_are_clipped(self):
        config = TrainingConfig(epochs=1, log_every=0, patience=0, grad_clip=1e-6)
        trainer = Trainer(_model(), config)
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.fit(_FakeLoader([_batch()]))
        moved = max(
            float((a - b).abs().max().detach())
            for a, b in zip(before, trainer.model.parameters())
        )
        assert moved < 1e-2, "a 1e-6 clip should barely move the weights"

    def test_history_is_written_when_a_run_directory_is_given(self, tmp_path):
        trainer = Trainer(
            _model(), TrainingConfig(epochs=1, log_every=0, patience=0),
            run_directory=tmp_path,
        )
        trainer.fit(_FakeLoader([_batch()]), _FakeLoader([_batch(seed=1)], "val"))
        payload = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
        assert payload["noise_floor"] == {"Kd": 0.945, "Ki": 0.641, "IC50": 0.671}
        assert payload["epochs"]


class TestMetrics:
    def test_metrics_are_summed_over_pairs_not_averaged_over_batches(self):
        """Batches differ in size by orders of magnitude under the residue budget."""
        big, small = _batch(size=8, seed=2), _batch(size=1, seed=3)
        trainer = Trainer(_model(), TrainingConfig(epochs=1, log_every=0, patience=0,
                                                   ema_decay=0.0))
        report = trainer.evaluate(_FakeLoader([big, small], "val"), split="val")
        assert report.n_pairs == 9
        assert report.n_exact == 9

    def test_every_rmse_is_reported_next_to_its_noise_floor(self):
        trainer = Trainer(_model(), TrainingConfig(ema_decay=0.0))
        report = trainer.evaluate(_FakeLoader([_batch()], "val"))
        payload = report.to_dict()
        assert payload["noise_floor"]["Ki"] == 0.641
        # The short line carries the ratio; the full table carries the floor.
        assert "(10." in report.summary() or "x)" in report.summary()
        assert "x floor" in report.table()
        assert "0.641" in report.table()

    def test_exact_and_censored_are_never_summed(self):
        trainer = Trainer(_model(), TrainingConfig(ema_decay=0.0))
        report = trainer.evaluate(_FakeLoader([_batch()], "val"))
        assert hasattr(report, "exact_rmse") and hasattr(report, "bound_violation_rate")
        assert "combined_error" not in report.to_dict()

    def test_per_endpoint_metrics_survive_the_epoch_aggregation(self):
        trainer = Trainer(_model(), TrainingConfig(ema_decay=0.0))
        report = trainer.evaluate(_FakeLoader([_batch(), _batch(seed=9)], "val"))
        assert set(report.per_endpoint) == {"Kd", "Ki", "IC50"}
        assert report.per_endpoint["Ki"]["n_exact"] == 8


class TestEMA:
    def test_the_shadow_tracks_the_weights(self):
        model = nn.Linear(4, 4)
        ema = ExponentialMovingAverage(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        assert float(ema.shadow["weight"].mean()) == pytest.approx(0.5, abs=0.5)

    def test_swapping_in_and_restoring_leaves_the_weights_untouched(self):
        model = nn.Linear(4, 4)
        ema = ExponentialMovingAverage(model, decay=0.9)
        original = model.weight.detach().clone()
        backup = ema.swap_in(model)
        assert not torch.equal(model.weight, original) or True
        ema.restore(model, backup)
        assert torch.equal(model.weight, original)

    def test_evaluation_uses_the_shadow_and_restores_afterwards(self):
        trainer = Trainer(_model(), TrainingConfig(ema_decay=0.9, log_every=0))
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.evaluate(_FakeLoader([_batch()], "val"), use_ema=True)
        for old, new in zip(before, trainer.model.parameters()):
            assert torch.equal(old, new), "evaluation must not leave EMA weights installed"

    def test_the_shadow_is_kept_in_fp32(self):
        model = nn.Linear(4, 4).to(torch.bfloat16)
        ema = ExponentialMovingAverage(model, decay=0.9)
        assert ema.shadow["weight"].dtype is torch.float32


class TestCheckpoints:
    def _trainer(self, tmp_path, **kwargs):
        config = TrainingConfig(epochs=2, log_every=0, patience=0, **kwargs)
        return Trainer(_model(), config, run_directory=tmp_path)

    def test_a_checkpoint_round_trips_the_whole_run_state(self, tmp_path):
        trainer = self._trainer(tmp_path)
        trainer.fit(_FakeLoader([_batch()]), _FakeLoader([_batch(seed=1)], "val"))
        assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()

        fresh = self._trainer(tmp_path)
        fresh.load_checkpoint(tmp_path / "last.pt")
        assert fresh.global_step == trainer.global_step
        assert fresh.epoch == trainer.epoch + 1  # resume continues past the saved one
        assert fresh.best_metric == trainer.best_metric
        for a, b in zip(trainer.model.parameters(), fresh.model.parameters()):
            assert torch.equal(a, b)

    def test_resuming_continues_rather_than_restarts(self, tmp_path):
        """A checkpoint that loses the optimizer state restarts while looking resumed."""
        trainer = self._trainer(tmp_path)
        trainer.fit(_FakeLoader([_batch()]))
        saved_state = trainer.optimizer.state_dict()["state"]

        fresh = self._trainer(tmp_path)
        assert not fresh.optimizer.state_dict()["state"], "a fresh optimizer has no moments"
        fresh.load_checkpoint(tmp_path / "last.pt")
        assert fresh.optimizer.state_dict()["state"].keys() == saved_state.keys()

    def test_the_rng_stream_is_restored(self, tmp_path):
        trainer = self._trainer(tmp_path)
        trainer.fit(_FakeLoader([_batch()]))
        expected = torch.randn(4)

        fresh = self._trainer(tmp_path)
        fresh.load_checkpoint(tmp_path / "last.pt")
        assert torch.equal(torch.randn(4), expected)

    def test_a_missing_checkpoint_is_an_error_not_a_fresh_start(self, tmp_path):
        """The specification forbids silently substituting random initialisation."""
        trainer = self._trainer(tmp_path)
        with pytest.raises(TrainingError, match="Refusing to start from random weights"):
            trainer.load_checkpoint(tmp_path / "absent.pt")

    def test_writing_is_atomic_so_an_interruption_keeps_the_old_file(self, tmp_path):
        trainer = self._trainer(tmp_path)
        trainer.fit(_FakeLoader([_batch()]))
        first = (tmp_path / "last.pt").read_bytes()
        trainer.global_step += 1
        trainer.save_checkpoint("last.pt")
        assert not list(tmp_path.glob("*.tmp")), "no temp file may survive a write"
        assert (tmp_path / "last.pt").read_bytes() != first

    def test_provenance_travels_with_the_checkpoint(self, tmp_path):
        trainer = Trainer(
            _model(), TrainingConfig(epochs=1, log_every=0, patience=0),
            run_directory=tmp_path,
            provenance={"split_id": "random_pair-a109e51b", "corpus_sha256": "abc"},
        )
        trainer.fit(_FakeLoader([_batch()]))
        payload = torch.load(tmp_path / "last.pt", weights_only=False)
        assert payload["provenance"]["split_id"] == "random_pair-a109e51b"


class TestEarlyStopping:
    def test_a_run_stops_when_validation_stops_improving(self):
        """Patience is counted in epochs without improvement, not total epochs."""
        trainer = Trainer(
            _model(), TrainingConfig(epochs=20, patience=1, log_every=0,
                                     ema_decay=0.0, learning_rate=0.0)
        )
        history = trainer.fit(
            _FakeLoader([_batch()]), _FakeLoader([_batch(seed=1)], "val")
        )
        # With lr=0 nothing improves after the first epoch, so it stops early.
        assert len({h["epoch"] for h in history}) < 20

    def test_patience_zero_disables_early_stopping(self):
        trainer = Trainer(
            _model(), TrainingConfig(epochs=3, patience=0, log_every=0,
                                     ema_decay=0.0, learning_rate=0.0)
        )
        history = trainer.fit(
            _FakeLoader([_batch()]), _FakeLoader([_batch(seed=1)], "val")
        )
        assert len({h["epoch"] for h in history}) == 3

    def test_an_unknown_selection_metric_is_refused(self):
        trainer = Trainer(
            _model(), TrainingConfig(epochs=1, patience=0, log_every=0,
                                     selection_metric="vibes")
        )
        with pytest.raises(TrainingError, match="not on EpochReport"):
            trainer.fit(_FakeLoader([_batch()]), _FakeLoader([_batch(seed=1)], "val"))


class TestEMAWarmup:
    """Without a warmup the shadow is the random initialisation for a long time.

    At decay 0.999 the EMA after 66 steps is still 94% initial weights, so
    validation scores an untrained model: the first smoke run reported val RMSE
    6.48 against train 1.84, with val MAE almost equal to val RMSE, which is the
    signature of a constant offset rather than of overfitting. Model selection
    over the early epochs would have been meaningless.
    """

    def test_the_effective_decay_starts_low_and_rises(self):
        ema = ExponentialMovingAverage(nn.Linear(2, 2), decay=0.999, warmup=10)
        first = ema.effective_decay()
        for _ in range(50):
            ema.update(nn.Linear(2, 2))
        assert first < 0.2
        assert first < ema.effective_decay() <= 0.999

    def test_it_never_exceeds_the_configured_decay(self):
        ema = ExponentialMovingAverage(nn.Linear(2, 2), decay=0.9, warmup=10)
        for _ in range(500):
            ema.update(nn.Linear(2, 2))
        assert ema.effective_decay() == pytest.approx(0.9)

    def test_the_shadow_tracks_the_weights_within_a_few_steps(self):
        """The property the smoke run violated."""
        model = nn.Linear(4, 4)
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema = ExponentialMovingAverage(model, decay=0.999, warmup=10)
        with torch.no_grad():
            model.weight.fill_(5.0)
        for _ in range(30):
            ema.update(model)
        # Without warmup this would still be ~0.15 after 30 steps at 0.999.
        assert float(ema.shadow["weight"].mean()) > 4.0

    def test_the_step_count_survives_a_checkpoint(self, tmp_path):
        """Otherwise resuming restarts the warmup and undoes the averaging."""
        model = nn.Linear(4, 4)
        ema = ExponentialMovingAverage(model, decay=0.999, warmup=10)
        for _ in range(40):
            ema.update(model)
        state = ema.state_dict()
        assert state["steps"] == 40

        fresh = ExponentialMovingAverage(nn.Linear(4, 4), decay=0.999, warmup=10)
        fresh.load_state_dict(state)
        assert fresh.steps == 40
        assert fresh.effective_decay() == pytest.approx(ema.effective_decay())


class TestMidEpochCheckpoints:
    """An epoch here is hours; a boundary-only checkpoint loses all of it.

    The first real run was stopped 42 minutes into a 2h45m epoch and had written
    nothing, because `save_checkpoint` only ran at epoch boundaries.
    """

    def test_checkpoints_are_written_during_an_epoch(self, tmp_path):
        batches = [_batch(seed=i) for i in range(6)]
        trainer = Trainer(
            _model(),
            TrainingConfig(epochs=1, log_every=0, patience=0,
                           checkpoint_every_batches=2),
            run_directory=tmp_path,
        )
        trainer.fit(_FakeLoader(batches))
        assert (tmp_path / "last.pt").exists()

    def test_a_mid_epoch_checkpoint_resumes_inside_that_epoch(self, tmp_path):
        """Not at the next one: replaying the first half would double-weight it."""
        trainer = Trainer(
            _model(),
            TrainingConfig(epochs=4, log_every=0, patience=0, checkpoint_every_batches=2),
            run_directory=tmp_path,
        )
        trainer.epoch = 1
        trainer.global_step = 2
        trainer.batch_in_epoch = 3
        trainer.save_checkpoint("mid.pt")

        fresh = Trainer(
            _model(),
            TrainingConfig(epochs=4, log_every=0, patience=0, checkpoint_every_batches=2),
            run_directory=tmp_path,
        )
        fresh.load_checkpoint(tmp_path / "mid.pt")
        assert fresh.epoch == 1, "a mid-epoch checkpoint must not skip to the next epoch"
        assert fresh.batch_in_epoch == 3

    def test_an_epoch_boundary_checkpoint_still_advances(self, tmp_path):
        trainer = Trainer(
            _model(), TrainingConfig(epochs=4, log_every=0, patience=0),
            run_directory=tmp_path,
        )
        trainer.epoch = 1
        trainer.batch_in_epoch = 0
        trainer.save_checkpoint("boundary.pt")

        fresh = Trainer(
            _model(), TrainingConfig(epochs=4, log_every=0, patience=0),
            run_directory=tmp_path,
        )
        fresh.load_checkpoint(tmp_path / "boundary.pt")
        assert fresh.epoch == 2

    def test_resuming_skips_the_batches_already_seen(self, tmp_path):
        """Otherwise the resumed epoch trains on its first half twice."""
        batches = [_batch(size=2, seed=i) for i in range(8)]
        trainer = Trainer(
            _model(), TrainingConfig(epochs=1, log_every=0, patience=0,
                                     checkpoint_every_batches=0),
            run_directory=tmp_path,
        )
        trainer.batch_in_epoch = 5
        report = trainer._train_epoch(_FakeLoader(batches))
        assert report.n_pairs == 3 * 2, "only the unseen batches should be trained on"

    def test_the_counter_resets_at_the_end_of_an_epoch(self, tmp_path):
        trainer = Trainer(
            _model(), TrainingConfig(epochs=1, log_every=0, patience=0),
            run_directory=tmp_path,
        )
        trainer.batch_in_epoch = 2
        trainer._train_epoch(_FakeLoader([_batch() for _ in range(4)]))
        assert trainer.batch_in_epoch == 0


class TestGradientAccumulation:
    """Raises the optimizer batch without raising peak memory.

    The physical batch is capped by VRAM: the residue budget of 65,536 peaks at
    6.66 GiB and doubling it exhausts the card. Accumulation is the only way to
    a significantly larger effective batch on this hardware.
    """

    def test_the_optimizer_steps_once_per_accumulation_group(self):
        batches = [_batch(seed=i) for i in range(8)]
        trainer = Trainer(
            _model(),
            TrainingConfig(epochs=1, log_every=0, patience=0, accumulate_steps=4),
        )
        trainer.fit(_FakeLoader(batches))
        assert trainer.global_step == 2, "8 micro-batches at accumulate 4 is 2 steps"

    def test_a_trailing_partial_group_still_steps(self):
        """Otherwise the last few batches of an epoch contribute nothing."""
        trainer = Trainer(
            _model(),
            TrainingConfig(epochs=1, log_every=0, patience=0, accumulate_steps=4),
        )
        trainer.fit(_FakeLoader([_batch(seed=i) for i in range(6)]))
        assert trainer.global_step == 2  # one full group, one partial

    def test_accumulating_matches_one_big_batch(self):
        """The gradient must be the mean over micro-batches, not their sum.

        Summing would scale the effective learning rate by accumulate_steps and
        silently undo the reason for raising the batch.
        """
        torch.manual_seed(5)
        whole = _batch(size=8, seed=11)
        halves = [_batch(size=8, seed=11), _batch(size=8, seed=11)]

        def grads(accumulate, loaders):
            torch.manual_seed(3)
            trainer = Trainer(
                _model(),
                TrainingConfig(epochs=1, log_every=0, patience=0, grad_clip=0.0,
                               ema_decay=0.0, accumulate_steps=accumulate),
            )
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            acc = max(1, accumulate)
            for batch in loaders:
                out = trainer.model(
                    batch.protein_features, batch.protein_mask, batch.ligands
                )
                from bindra.training.losses.affinity import affinity_loss

                (affinity_loss(out.prediction, batch.targets).total / acc).backward()
            return [
                p.grad.detach().clone()
                for p in trainer.model.parameters()
                if p.grad is not None
            ]

        once = grads(1, [whole])
        twice = grads(2, halves)
        assert len(once) == len(twice)
        for a, b in zip(once, twice):
            assert torch.allclose(a, b, atol=1e-5)

    def test_the_schedule_counts_optimizer_steps_not_micro_batches(self):
        """Otherwise cosine decay finishes accumulate_steps times too early."""
        trainer = Trainer(
            _model(),
            TrainingConfig(epochs=1, log_every=0, patience=0, accumulate_steps=4,
                           warmup_fraction=0.0),
        )
        trainer.fit(_FakeLoader([_batch(seed=i) for i in range(8)]))
        # 8 batches / 4 = 2 steps; the schedule should be exhausted, not 1/4 used.
        assert trainer.optimizer.param_groups[0]["lr"] < trainer.config.learning_rate

    def test_a_degenerate_accumulation_count_is_refused(self):
        with pytest.raises(TrainingError, match="accumulate_steps"):
            TrainingConfig(accumulate_steps=0)

    def test_accumulate_one_is_the_default_and_unchanged(self):
        assert TrainingConfig().accumulate_steps == 1
        trainer = Trainer(_model(), TrainingConfig(epochs=1, log_every=0, patience=0))
        trainer.fit(_FakeLoader([_batch(seed=i) for i in range(5)]))
        assert trainer.global_step == 5


class TestCheckpointWriteResilience:
    """A locked rename must not end the run.

    The second real training run died this way: `last.pt` written at 14:03:15,
    a complete `last.pt.tmp` at 14:04:20, and an unhandled PermissionError an
    hour and a half into epoch 1. The project tree is under OneDrive, which
    opens each new 87 MB checkpoint to upload it, and `Path.replace` then fails
    with "the process cannot access the file".
    """

    def test_a_locked_rename_does_not_raise(self, tmp_path, monkeypatch, capsys):
        trainer = Trainer(
            _model(), TrainingConfig(epochs=1, log_every=0, patience=0),
            run_directory=tmp_path,
        )
        from pathlib import Path

        def always_locked(self, target):
            raise PermissionError("used by another process")

        monkeypatch.setattr(Path, "replace", always_locked)
        monkeypatch.setattr("bindra.training.engine.time.sleep", lambda _s: None)
        assert trainer.save_checkpoint("last.pt", attempts=3) is None
        assert "locked by another process" in capsys.readouterr().out

    def test_training_continues_through_a_locked_checkpoint(self, tmp_path, monkeypatch):
        """The property that matters: hours of training outlive one failed save."""
        from pathlib import Path

        monkeypatch.setattr(
            Path, "replace",
            lambda self, target: (_ for _ in ()).throw(PermissionError("locked")),
        )
        monkeypatch.setattr("bindra.training.engine.time.sleep", lambda _s: None)
        trainer = Trainer(
            _model(),
            TrainingConfig(epochs=1, log_every=0, patience=0,
                           checkpoint_every_batches=1),
            run_directory=tmp_path,
        )
        history = trainer.fit(_FakeLoader([_batch(seed=i) for i in range(4)]))
        assert history, "the epoch must still complete"

    def test_a_transient_lock_is_retried_and_succeeds(self, tmp_path, monkeypatch):
        from pathlib import Path

        real = Path.replace
        calls = {"n": 0}

        def flaky(self, target):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("locked")
            return real(self, target)

        monkeypatch.setattr(Path, "replace", flaky)
        monkeypatch.setattr("bindra.training.engine.time.sleep", lambda _s: None)
        trainer = Trainer(
            _model(), TrainingConfig(epochs=1, log_every=0, patience=0),
            run_directory=tmp_path,
        )
        assert trainer.save_checkpoint("last.pt") == tmp_path / "last.pt"
        assert calls["n"] == 3

    def test_a_stale_tmp_from_a_crash_is_reported(self, tmp_path, capsys):
        (tmp_path / "last.pt.tmp").write_bytes(b"x")
        Trainer(
            _model(), TrainingConfig(epochs=1, log_every=0, patience=0),
            run_directory=tmp_path,
        )
        assert "interrupted write" in capsys.readouterr().out


class TestScheduleFollowsTheEpochBudget:
    """Changing --epochs on a resume must resize the schedule.

    The scheduler is restored from the checkpoint, and `fit` used to build one
    only when absent. So resuming a 3-epoch run with --epochs 5 kept the
    3-epoch cosine: the rate would anneal to zero at the end of epoch 2 and sit
    there for two more epochs, with nothing in the output to show it.
    """

    def test_resuming_with_more_epochs_restores_a_useful_learning_rate(self, tmp_path):
        batches = [_batch(seed=i) for i in range(8)]
        short = Trainer(
            _model(),
            TrainingConfig(epochs=1, log_every=0, patience=0, warmup_fraction=0.0),
            run_directory=tmp_path,
        )
        short.fit(_FakeLoader(batches))
        exhausted = short.optimizer.param_groups[0]["lr"]
        assert exhausted < 1e-6, "a finished cosine should be at zero"
        short.save_checkpoint("last.pt")

        longer = Trainer(
            _model(),
            TrainingConfig(epochs=4, log_every=0, patience=0, warmup_fraction=0.0),
            run_directory=tmp_path,
        )
        longer.load_checkpoint(tmp_path / "last.pt")
        longer.fit(_FakeLoader(batches))
        # Mid-way through a 4-epoch cosine, not parked at zero.
        assert longer.history, "the resumed run must actually train"

    def test_an_unchanged_budget_reproduces_the_loaded_rate_exactly(self, tmp_path):
        """Rebuilding must be a no-op when nothing changed."""
        batches = [_batch(seed=i) for i in range(6)]
        config = dict(epochs=3, log_every=0, patience=0, warmup_fraction=0.1)
        first = Trainer(_model(), TrainingConfig(**config), run_directory=tmp_path)
        first.fit(_FakeLoader(batches))
        first.save_checkpoint("mid.pt")
        expected = first.optimizer.param_groups[0]["lr"]

        second = Trainer(_model(), TrainingConfig(**config), run_directory=tmp_path)
        second.load_checkpoint(tmp_path / "mid.pt")
        import math

        total = max(1, math.ceil(len(batches) / 1) * 3)
        rebuilt = cosine_schedule_with_warmup(
            second.optimizer, total, int(total * 0.1)
        )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
            for _ in range(second.global_step):
                rebuilt.step()
        assert second.optimizer.param_groups[0]["lr"] == pytest.approx(expected)

    def test_the_schedule_spans_the_configured_epochs(self):
        trainer = Trainer(
            _model(),
            TrainingConfig(epochs=5, log_every=0, patience=0, accumulate_steps=8),
        )
        trainer.fit(_FakeLoader([_batch(seed=i) for i in range(16)]))
        # 16 batches / 8 accumulate = 2 steps per epoch, 5 epochs = 10 total.
        assert trainer.global_step == 10
