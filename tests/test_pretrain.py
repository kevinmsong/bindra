"""Knowledge-guided pre-training: the objective, the loop, and the leak guards.

Four tests carry the weight of this file, and each of them stands for a failure
that already happened once or would be invisible if it happened:

``test_both_knowledge_projections_receive_gradient``
    ``triplet_emb.fp_proj`` and ``md_proj`` are how knowledge enters the model
    during affinity training. Masking both knowledge nodes, or masking a
    fraction of molecules, leaves both at their initialisation while the loss
    falls anyway -- the stack underneath compensates. This was found by
    checking which parameters received gradient, not by reading the paper.
``test_the_masked_node_cannot_be_seen_by_what_predicts_it``
    The objective is reconstruction, so a path from the target to the
    prediction turns it into copying. The loss curve looks *better* when it
    leaks.
``test_the_mask_embedding_and_both_predictors_are_trained``
    The arm's own freezing policy freezes all three, because no *affinity*
    forward pass reaches them. Running the loop without
    :func:`prepare_for_pretraining` trains the encoder against three heads
    pinned at random.
``test_a_checkpoint_from_another_split_build_is_refused``
    Pre-training leaks no labels but it does leak structure, and the leak is
    undetectable downstream: the run trains normally and reports held-out
    numbers that are quietly inflated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bindra.encoders.ligand_arm import LigandArmError
from bindra.encoders.ligand_featurizers import LigandBatchBuilder
from bindra.encoders.ligand_kpgt import PRETRAIN_CHECKPOINT_KIND, KPGTConfig
from bindra.encoders.ligand_kpgt_arm import KPGTLigandArmModule
from bindra.training.pretrain import (
    MoleculeLoader,
    PretrainConfig,
    PretrainError,
    PretrainRunner,
    freeze_unused_heads,
    knowledge_loss,
    masked_knowledge_forward,
    prepare_for_pretraining,
    sample_mask,
    split_molecules,
    train_only_molecules,
)

#: Real curated-style molecules, including a salt so the unbonded-atom path
#: (``vavn == -1``) is exercised rather than assumed absent.
MOLECULES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCO",
    "c1ccccc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Clc1ccccc1",
    "CCN(CC)CC",
    "OC(=O)c1ccccc1O",
    "C1CCNCC1",
    "CC(=O)Nc1ccc(O)cc1",
    "[Na+].[Cl-]",
    "CCOC(=O)c1ccccc1",
    "c1ccc2ccccc2c1",
    "NC(=O)c1ccncc1",
    "CSCC[C@H](N)C(=O)O",
    "CC(C)(C)c1ccccc1",
]

#: A shape small enough to build many of, and the same regime the CLI uses:
#: ``from_scratch``, which is what makes the freezing question live.
SMALL = dict(d_g_feats=32, n_heads=4, n_mol_layers=2, d_hpath_ratio=2,
             n_trainable_layers=2)


def small_config(**overrides) -> KPGTConfig:
    return KPGTConfig.mini(**{**SMALL, **overrides})


@pytest.fixture(scope="module")
def builder() -> LigandBatchBuilder:
    return LigandBatchBuilder.for_config(small_config())


@pytest.fixture(scope="module")
def batch(builder):
    """One collated batch of all the molecules, built once."""
    loader = MoleculeLoader(
        MOLECULES, builder, batch_size=len(MOLECULES), shuffle=False,
        prefetch_workers=0,
    )
    return next(iter(loader))


@pytest.fixture
def arm() -> KPGTLigandArmModule:
    torch.manual_seed(20261008)
    return KPGTLigandArmModule(small_config())


def nonzero_gradient(predictor, prefix: str) -> bool:
    """True when every parameter under ``prefix`` got a non-zero gradient."""
    found = [
        parameter
        for name, parameter in predictor.named_parameters()
        if name.startswith(prefix)
    ]
    assert found, f"no parameter named {prefix!r}"
    return all(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in found
    )


class TestTheObjective:
    def test_both_knowledge_projections_receive_gradient(self, arm, batch):
        """The failure the module docstring records, as a test.

        Masking both knowledge nodes overwrites the output of both projections
        before it reaches the stack, so neither receives gradient at all.
        Cross-prediction -- mask one, keep the other -- is what keeps them live,
        and they are exactly how knowledge enters the model during affinity
        training.
        """
        prepare_for_pretraining(arm.module)
        mask = sample_mask(batch.num_graphs, PretrainConfig())
        fp_logits, md_pred = masked_knowledge_forward(
            arm.module, batch.graphs, batch.fingerprints, batch.descriptors,
            mask_fp=mask,
        )
        loss = knowledge_loss(
            fp_logits, md_pred, batch.fingerprints[mask],
            batch.descriptors[~mask], PretrainConfig(),
        )
        loss.total.backward()
        assert nonzero_gradient(arm.module, "triplet_emb.fp_proj")
        assert nonzero_gradient(arm.module, "triplet_emb.md_proj")

    def test_the_mask_embedding_and_both_predictors_are_trained(self, arm, batch):
        """All three are frozen by the arm's policy and must be unfrozen here."""
        prepare_for_pretraining(arm.module)
        mask = sample_mask(batch.num_graphs, PretrainConfig())
        fp_logits, md_pred = masked_knowledge_forward(
            arm.module, batch.graphs, batch.fingerprints, batch.descriptors,
            mask_fp=mask,
        )
        knowledge_loss(
            fp_logits, md_pred, batch.fingerprints[mask],
            batch.descriptors[~mask], PretrainConfig(),
        ).total.backward()
        assert nonzero_gradient(arm.module, "mask_emb.weight")
        assert nonzero_gradient(arm.module, "fp_predictor")
        assert nonzero_gradient(arm.module, "md_predictor")

    def test_the_arm_alone_would_leave_those_three_frozen(self, arm):
        """Why :func:`prepare_for_pretraining` exists rather than being implied.

        The arm freezes them because no affinity forward pass reaches them. A
        pre-training loop that trusted the arm's state would optimise an
        encoder against three heads pinned at random initialisation, and would
        still report a falling loss.
        """
        pinned = {
            name
            for name, parameter in arm.module.named_parameters()
            if not parameter.requires_grad
        }
        assert "mask_emb.weight" in pinned
        assert any(n.startswith("fp_predictor") for n in pinned)
        assert any(n.startswith("md_predictor") for n in pinned)

    def test_what_this_objective_cannot_reach_stays_frozen(self, arm):
        """``node_predictor`` and LiGhT's internal ``mask_emb``, named.

        Left trainable they sit in the optimizer accruing state that never
        updates, and inflate every reported trainable-parameter count.
        """
        frozen = prepare_for_pretraining(arm.module)
        assert "model.mask_emb.weight" in frozen
        assert any(name.startswith("node_predictor") for name in frozen)
        for name in frozen:
            assert not dict(arm.module.named_parameters())[name].requires_grad
        # And nothing else was caught by accident.
        assert all(
            name.startswith("node_predictor") or name == "model.mask_emb.weight"
            for name in frozen
        )

    def test_freeze_unused_heads_is_idempotent(self, arm):
        first = freeze_unused_heads(arm.module)
        assert freeze_unused_heads(arm.module) == first

    def test_the_masked_node_cannot_be_seen_by_what_predicts_it(self, arm, batch):
        """Reconstruction with a path to the target is copying, not learning.

        Molecule 0 has its fingerprint masked. Changing that fingerprint must
        change nothing at all; changing its *descriptors*, which stay visible,
        must change the fingerprint prediction -- otherwise the "plus the other"
        half of the objective is not happening either.
        """
        arm.module.eval()
        mask = torch.zeros(batch.num_graphs, dtype=torch.bool)
        mask[0] = True  # fingerprint masked for molecule 0 only

        def predict(fingerprints, descriptors):
            with torch.no_grad():
                logits, _ = masked_knowledge_forward(
                    arm.module, batch.graphs, fingerprints, descriptors,
                    mask_fp=mask,
                )
            return logits

        baseline = predict(batch.fingerprints, batch.descriptors)

        hidden = batch.fingerprints.clone()
        hidden[0] = 1.0 - hidden[0]
        assert torch.equal(predict(hidden, batch.descriptors), baseline), (
            "the masked fingerprint reached its own prediction"
        )

        visible = batch.descriptors.clone()
        visible[0] = visible[0] + 1.0
        assert not torch.equal(predict(batch.fingerprints, visible), baseline), (
            "the visible descriptor node does not inform the prediction, so "
            "nothing is being predicted *from the other* node"
        )

    def test_the_knowledge_nodes_align_with_the_molecule_axis(self, batch):
        """What makes matching predictions to targets by boolean index correct.

        ``masked_knowledge_forward`` indexes nodes and the caller indexes
        molecules. The two agree only because each knowledge node appears
        exactly once per molecule in ascending molecule order; if it did not,
        every molecule would be trained against another molecule's knowledge
        and the loss would still fall.
        """
        graphs = batch.graphs
        expected = torch.arange(graphs.num_graphs)
        assert torch.equal(graphs.graph_index[graphs.vavn == 1], expected)
        assert torch.equal(graphs.graph_index[graphs.vavn == 2], expected)

    def test_the_row_counts_are_checked_rather_than_trusted(
        self, arm, batch, monkeypatch
    ):
        """A mismatch trains on another molecule's knowledge and still converges.

        Predictions are selected by a boolean index over *nodes* and targets by
        one over *molecules*. The two agree today; the check is here because
        nothing in either index would complain if they stopped.
        """
        import bindra.training.pretrain as module

        runner = PretrainRunner(arm, PretrainConfig())

        def short_fingerprint(*args, **kwargs):
            full = original(*args, **kwargs)
            return full[0][:-1], full[1]

        original = module.masked_knowledge_forward
        monkeypatch.setattr(module, "masked_knowledge_forward", short_fingerprint)
        with pytest.raises(PretrainError, match="masked fingerprint nodes against"):
            runner.batch_loss(batch, 0)

        def short_descriptors(*args, **kwargs):
            full = original(*args, **kwargs)
            return full[0], full[1][:-1]

        monkeypatch.setattr(module, "masked_knowledge_forward", short_descriptors)
        with pytest.raises(PretrainError, match="masked descriptor nodes against"):
            runner.batch_loss(batch, 0)

    def test_sample_mask_always_contains_both_kinds(self):
        """An all-one-way draw gives a mean over an empty tensor, which is NaN."""
        for share in (0.001, 0.5, 0.999):
            config = PretrainConfig(fingerprint_mask_share=share)
            for seed in range(25):
                generator = torch.Generator().manual_seed(seed)
                mask = sample_mask(8, config, generator=generator)
                assert bool(mask.any()) and not bool(mask.all())

    def test_a_degenerate_mask_share_is_refused_at_construction(self):
        for share in (0.0, 1.0):
            with pytest.raises(ValueError, match="fingerprint_mask_share"):
                PretrainConfig(fingerprint_mask_share=share)

    def test_the_baseline_is_the_majority_bit_not_the_all_zeros_rate(self):
        """The correction that matters most to reading a pre-training curve.

        These fingerprints are dense: 83.9% of the 512 bits are set on real
        curated molecules. Scored against all-zeros, a model at 0.582 looks far
        ahead of 0.161; scored against the constant predictor actually
        available to it, it is well behind 0.839.
        """
        dense = torch.zeros(4, 512)
        dense[:, :440] = 1.0  # 85.9% set, like the real corpus
        all_zeros = knowledge_loss(
            torch.full((4, 512), -5.0), torch.zeros(4, 200),
            dense, torch.zeros(4, 200), PretrainConfig(),
        )
        assert all_zeros.fingerprint_accuracy == pytest.approx(1.0 - 440 / 512)
        assert all_zeros.fingerprint_baseline == pytest.approx(440 / 512)
        assert all_zeros.fingerprint_accuracy < all_zeros.fingerprint_baseline

        sparse = torch.zeros(4, 512)
        sparse[:, :20] = 1.0
        either_way = knowledge_loss(
            torch.full((4, 512), -5.0), torch.zeros(4, 200),
            sparse, torch.zeros(4, 200), PretrainConfig(),
        )
        # max(p, 1 - p) is the right floor whichever way the density falls.
        assert either_way.fingerprint_baseline == pytest.approx(1.0 - 20 / 512)


class TestMoleculeLoader:
    def test_every_molecule_is_seen_once_an_epoch(self, builder):
        loader = MoleculeLoader(
            MOLECULES, builder, batch_size=5, seed=3, prefetch_workers=0
        )
        seen = sum(b.num_graphs for b in loader)
        assert seen == len(MOLECULES)

    def test_a_trailing_singleton_is_merged_rather_than_dropped(self, builder):
        """A batch of one cannot be masked, and must not vanish silently.

        ``sample_mask`` needs two rows to place both mask kinds, so a trailing
        chunk of one would be skipped -- and it would not appear in ``dropped``
        either, which records featurization failures.
        """
        thirteen = MOLECULES[:13]
        loader = MoleculeLoader(
            thirteen, builder, batch_size=3, shuffle=False, prefetch_workers=0
        )
        sizes = [b.num_graphs for b in loader]
        assert sum(sizes) == 13
        assert min(sizes) >= 2
        assert sizes == [3, 3, 3, 4]

    def test_len_agrees_with_what_the_iterator_yields(self, builder):
        """The schedule is built from ``len``; disagreement mis-sizes warmup."""
        for size in (3, 4, 5, 7):
            loader = MoleculeLoader(
                MOLECULES[:13], builder, batch_size=size, shuffle=False,
                prefetch_workers=0,
            )
            assert len(loader) == sum(1 for _ in loader), size

    def test_the_order_is_reproducible_from_seed_and_epoch(self, builder):
        a = MoleculeLoader(MOLECULES, builder, batch_size=4, seed=11)
        b = MoleculeLoader(MOLECULES, builder, batch_size=4, seed=11)
        a.set_epoch(2)
        b.set_epoch(2)
        assert a._order() == b._order()

    def test_a_later_epoch_sees_a_different_order(self, builder):
        loader = MoleculeLoader(MOLECULES, builder, batch_size=4, seed=11)
        loader.set_epoch(0)
        first = loader._order()
        loader.set_epoch(1)
        assert loader._order() != first

    def test_an_unfeaturizable_molecule_is_dropped_and_counted(self, builder):
        """A thirty-epoch run must not die at epoch 12 over one molecule.

        Dropped rather than raised on, and counted rather than dropped quietly:
        the epoch report prints the number.
        """
        loader = MoleculeLoader(
            ["CCO", "not a molecule", "c1ccccc1", "CCN"], builder,
            batch_size=4, shuffle=False, prefetch_workers=0,
        )
        sizes = [b.num_graphs for b in loader]
        assert sizes == [3]
        assert loader.dropped == ["not a molecule"]

    def test_a_batch_of_one_is_refused_at_construction(self, builder):
        with pytest.raises(PretrainError, match="batch_size must be at least 2"):
            MoleculeLoader(MOLECULES, builder, batch_size=1)

    def test_the_batch_carries_its_own_knowledge_inputs(self, batch):
        """They are both the input and the target here, so they must be aligned."""
        assert batch.fingerprints.shape == (len(MOLECULES), 512)
        assert batch.descriptors.shape[0] == len(MOLECULES)
        assert torch.isfinite(batch.descriptors).all()


class TestSplitMolecules:
    def test_the_held_out_slice_comes_out_of_the_training_molecules(self):
        """Never from the affinity split's val partition.

        Selecting a checkpoint on val molecules puts their structures inside
        the encoder through the selection rather than the gradient, which is the
        same leak by a quieter route.
        """
        train, held = split_molecules(MOLECULES, val_fraction=0.25, seed=5)
        assert set(train).isdisjoint(held)
        assert set(train) | set(held) == set(MOLECULES)
        assert held

    def test_it_is_reproducible_from_the_seed(self):
        first = split_molecules(MOLECULES, val_fraction=0.25, seed=5)
        assert split_molecules(MOLECULES, val_fraction=0.25, seed=5) == first
        assert split_molecules(MOLECULES, val_fraction=0.25, seed=6) != first

    def test_input_order_does_not_change_the_split(self):
        """Sorted first, so a reordered molecule list is the same split."""
        forward = split_molecules(MOLECULES, val_fraction=0.25, seed=5)
        backward = split_molecules(
            list(reversed(MOLECULES)), val_fraction=0.25, seed=5
        )
        assert forward == backward

    def test_one_molecule_is_not_a_pretraining_set(self):
        with pytest.raises(PretrainError, match="nothing to hold back"):
            split_molecules(["CCO"], val_fraction=0.1, seed=1)

    def test_at_least_one_molecule_is_always_held_back(self):
        train, held = split_molecules(["CCO", "CCN"], val_fraction=0.001, seed=1)
        assert len(held) == 1 and len(train) == 1

    def test_an_out_of_range_val_fraction_is_refused(self):
        for fraction in (0.0, 0.5, 0.9):
            with pytest.raises(ValueError, match="val_fraction"):
                PretrainConfig(val_fraction=fraction)


class TestTrainOnlyMolecules:
    """The rule that decides what may be pretrained on at all.

    Built against a synthetic split rather than the real corpus so the
    partitions can be made to overlap deliberately. The real-corpus numbers it
    stands for are in the module docstring.
    """

    @pytest.fixture
    def corpus(self, tmp_path):
        import gzip

        import pandas as pd

        processed = tmp_path / "processed"
        (processed / "observations").mkdir(parents=True)
        (processed / "ligands").mkdir(parents=True)

        # L1 train only; L2 train and val; L3 train and quarantine;
        # L4 quarantine only; L5 train but unparseable upstream (status != OK).
        observations = pd.DataFrame(
            {
                "observation_id": ["o1", "o2", "o3", "o4", "o5", "o6", "o7"],
                "ligand_ref": ["L1", "L2", "L2", "L3", "L3", "L4", "L5"],
            }
        )
        observations.to_parquet(processed / "observations" / "part-0.parquet")
        pd.DataFrame(
            {
                "ligand_ref": ["L1", "L2", "L3", "L4", "L5"],
                "standardized_smiles": ["CCO", "CCN", "CCC", "CCCC", "CCCCC"],
                "status": ["OK", "OK", "OK", "OK", "REJECTED"],
            }
        ).to_parquet(processed / "ligands" / "part-0.parquet")

        ids = tmp_path / "splits" / "random_pair" / "ids"
        ids.mkdir(parents=True)
        partitions = {
            "train": ["o1", "o2", "o4", "o7"],
            "val": ["o3"],
            "cal": [],
            "test": [],
            "quarantine": ["o5", "o6"],
        }
        for name, members in partitions.items():
            with gzip.open(ids / f"{name}.txt.gz", "wt") as handle:
                handle.write("\n".join(members))
        return tmp_path / "processed", tmp_path / "splits"

    def test_a_molecule_in_val_as_well_as_train_is_excluded(self, corpus):
        processed, splits = corpus
        kept = train_only_molecules(processed, splits, "random_pair")
        assert "CCN" not in kept  # L2, in train and val

    def test_a_quarantined_panel_molecule_is_excluded(self, corpus):
        """The exclusion that was missing, and the one that matters most.

        ``quarantine`` is where the locked benchmark panels' observations are
        reserved -- 337,191 of them on the real split. Leaving it out of the
        held-out list put 6,212 sealed-panel molecules inside the encoder: 1.1%
        of the pre-training set, and every one of them in a panel the project
        scores once and cannot unscore.
        """
        processed, splits = corpus
        kept = train_only_molecules(processed, splits, "random_pair")
        assert "CCC" not in kept, "L3 is in train and in a sealed panel"
        assert "CCCC" not in kept, "L4 is quarantine-only and was never train"

    def test_a_molecule_in_train_alone_survives(self, corpus):
        processed, splits = corpus
        assert "CCO" in train_only_molecules(processed, splits, "random_pair")

    def test_a_rejected_ligand_is_not_pretrained_on(self, corpus):
        """status != OK means the standardization did not produce a molecule."""
        processed, splits = corpus
        assert "CCCCC" not in train_only_molecules(processed, splits, "random_pair")

    def test_the_result_is_sorted_and_unique(self, corpus):
        processed, splits = corpus
        kept = train_only_molecules(processed, splits, "random_pair")
        assert kept == sorted(set(kept))


class TestTheLoop:
    @pytest.fixture
    def run(self, arm, builder, tmp_path):
        config = PretrainConfig(
            epochs=3, batch_size=4, learning_rate=3e-3, patience=0, seed=99
        )
        runner = PretrainRunner(
            arm, config, run_directory=tmp_path,
            provenance={"split_regime": "random_pair", "split_id": "split-abc123"},
        )
        train, held = split_molecules(MOLECULES, val_fraction=0.25, seed=99)
        loaders = (
            MoleculeLoader(train, builder, batch_size=4, seed=99, prefetch_workers=0),
            MoleculeLoader(
                held, builder, batch_size=4, shuffle=False, prefetch_workers=0
            ),
        )
        return runner, loaders

    def test_the_loss_falls(self, run):
        runner, (train_loader, val_loader) = run
        history = runner.fit(train_loader, val_loader)
        train_losses = [e["loss"] for e in history if e["split"] == "train"]
        assert len(train_losses) == 3
        assert train_losses[-1] < train_losses[0], train_losses

    def test_both_checkpoints_and_a_history_are_written(self, run, tmp_path):
        runner, loaders = run
        runner.fit(*loaders)
        assert (tmp_path / "last.pt").exists()
        assert (tmp_path / "best.pt").exists()
        import json

        payload = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
        assert payload["kind"] == PRETRAIN_CHECKPOINT_KIND
        assert payload["provenance"]["split_id"] == "split-abc123"
        assert payload["trainable_parameters"] > 0
        assert payload["frozen_parameters"]

    def test_the_held_out_number_is_what_selects_the_checkpoint(self, run):
        runner, loaders = run
        history = runner.fit(*loaders)
        val_losses = [e["loss"] for e in history if e["split"] == "val"]
        assert runner.best_metric == pytest.approx(min(val_losses))

    def test_a_resumed_run_continues_the_step_count(self, run, tmp_path, builder):
        runner, loaders = run
        runner.fit(*loaders)
        steps, epochs = runner.global_step, runner.epoch

        fresh = PretrainRunner(
            KPGTLigandArmModule(small_config()), runner.config,
            run_directory=tmp_path,
        )
        fresh.load_checkpoint(tmp_path / "last.pt")
        assert fresh.global_step == steps
        assert fresh.epoch == epochs
        assert fresh.best_metric == pytest.approx(runner.best_metric)
        assert len(fresh.history) == len(runner.history)

    def test_a_resume_re_unfreezes_what_the_objective_needs(self, run, tmp_path):
        """A state-dict load does not restore ``requires_grad``.

        Without re-applying it, a resumed run silently stops training the two
        predictors and the mask embedding.
        """
        runner, loaders = run
        runner.fit(*loaders)
        fresh = PretrainRunner(
            KPGTLigandArmModule(small_config()), runner.config,
            run_directory=tmp_path,
        )
        fresh.load_checkpoint(tmp_path / "last.pt")
        named = dict(fresh.predictor.named_parameters())
        assert named["mask_emb.weight"].requires_grad
        assert named["fp_predictor.0.weight"].requires_grad
        assert named["md_predictor.0.weight"].requires_grad

    def test_patience_stops_the_run(self, arm, builder, tmp_path):
        config = PretrainConfig(
            epochs=20, batch_size=4, learning_rate=0.0, patience=1, seed=99
        )
        runner = PretrainRunner(arm, config, run_directory=tmp_path)
        train = MoleculeLoader(
            MOLECULES[:12], builder, batch_size=4, seed=1, prefetch_workers=0
        )
        held = MoleculeLoader(
            MOLECULES[12:], builder, batch_size=4, shuffle=False, prefetch_workers=0
        )
        runner.fit(train, held)
        # A zero learning rate cannot improve, so the second epoch is stale.
        assert runner.epoch < 20

    def test_an_empty_loader_is_refused_rather_than_reported_as_an_epoch(
        self, arm, builder, tmp_path
    ):
        runner = PretrainRunner(arm, PretrainConfig(epochs=1), run_directory=tmp_path)
        empty = MoleculeLoader([], builder, batch_size=4, prefetch_workers=0)
        with pytest.raises(PretrainError, match="produced no batch"):
            runner.fit(empty)

    def test_the_reported_trainable_count_excludes_the_frozen_heads(self, arm):
        runner = PretrainRunner(arm, PretrainConfig())
        frozen_total = sum(
            dict(runner.predictor.named_parameters())[name].numel()
            for name in runner.frozen
        )
        everything = sum(p.numel() for p in runner.predictor.parameters())
        assert runner.n_trainable == everything - frozen_total


class TestLoadingIntoTheArm:
    """The ``bindra train --ligand-checkpoint`` contract."""

    @pytest.fixture
    def checkpoint(self, arm, builder, tmp_path) -> Path:
        runner = PretrainRunner(
            arm, PretrainConfig(epochs=1, batch_size=4, patience=0),
            run_directory=tmp_path,
            provenance={"split_regime": "random_pair", "split_id": "split-abc123"},
        )
        runner.fit(
            MoleculeLoader(MOLECULES, builder, batch_size=4, seed=1, prefetch_workers=0)
        )
        return tmp_path / "best.pt"

    def test_a_round_trip_reproduces_the_weights_exactly(self, checkpoint, arm):
        fresh = KPGTLigandArmModule(small_config())
        identity = fresh.load_ligand_checkpoint(
            checkpoint, regime="random_pair", split_id="split-abc123"
        )
        assert identity.startswith("best.pt@")
        before, after = arm.module.state_dict(), fresh.module.state_dict()
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key]), key

    def test_the_identity_reaches_provenance(self, checkpoint):
        """Otherwise a results table cannot say the run was pretrained."""
        fresh = KPGTLigandArmModule(small_config())
        fresh.load_ligand_checkpoint(checkpoint)
        assert fresh.pretrained_checkpoint is not None
        assert "pretrain epoch" in fresh.pretrained_checkpoint

    def test_a_checkpoint_from_another_regime_is_refused(self, checkpoint):
        """The regimes hold out different molecules; one encoder is not both."""
        fresh = KPGTLigandArmModule(small_config())
        with pytest.raises(LigandArmError, match="cold_drug"):
            fresh.load_ligand_checkpoint(
                checkpoint, regime="cold_drug", split_id="split-abc123"
            )

    def test_a_checkpoint_from_another_split_build_is_refused(self, checkpoint):
        """Same regime, rebuilt split: the held-out molecules moved.

        No flag overrides this. The run it would produce looks entirely normal.
        """
        fresh = KPGTLigandArmModule(small_config())
        with pytest.raises(LigandArmError, match="split_id"):
            fresh.load_ligand_checkpoint(
                checkpoint, regime="random_pair", split_id="split-rebuilt"
            )

    def test_a_shape_mismatch_is_named_rather_than_left_to_torch(self, checkpoint):
        """Which shape differs, instead of a wall of tensor-size errors."""
        other = KPGTLigandArmModule(small_config(d_g_feats=64))
        with pytest.raises(LigandArmError, match="d_g_feats"):
            other.load_ligand_checkpoint(checkpoint, regime="random_pair")

    def test_a_depth_mismatch_is_also_named(self, checkpoint):
        other = KPGTLigandArmModule(small_config(n_mol_layers=4, n_trainable_layers=4))
        with pytest.raises(LigandArmError, match="n_mol_layers"):
            other.load_ligand_checkpoint(checkpoint, regime="random_pair")

    def test_freezing_is_re_applied_after_the_load(self, checkpoint):
        """A load does not restore ``requires_grad``, and the arm's policy
        differs from pre-training's: the predictors go back to frozen."""
        fresh = KPGTLigandArmModule(small_config())
        fresh.load_ligand_checkpoint(checkpoint)
        named = dict(fresh.module.named_parameters())
        assert not named["fp_predictor.0.weight"].requires_grad
        assert not named["mask_emb.weight"].requires_grad
        assert named["triplet_emb.fp_proj.in_proj.weight"].requires_grad

    def test_a_loaded_arm_is_no_longer_partially_frozen_at_random(self, checkpoint):
        fresh = KPGTLigandArmModule(small_config())
        fresh.load_ligand_checkpoint(checkpoint)
        assert fresh.partially_frozen_at_random() == 0
        assert not fresh.frozen_and_untrained()

    def test_a_missing_file_is_refused_rather_than_silently_random(self, tmp_path):
        fresh = KPGTLigandArmModule(small_config())
        with pytest.raises(LigandArmError, match="Refusing to train from random"):
            fresh.load_ligand_checkpoint(tmp_path / "absent.pt")

    def test_a_file_that_is_neither_is_refused_by_the_strict_loader(self, tmp_path):
        """Anything without our marker is treated as a published base.pth."""
        path = tmp_path / "junk.pt"
        torch.save({"something": "else"}, path)
        fresh = KPGTLigandArmModule(small_config())
        with pytest.raises(Exception):
            fresh.load_ligand_checkpoint(path)

    def test_resume_refuses_a_checkpoint_that_is_not_a_pretraining_one(
        self, arm, tmp_path
    ):
        path = tmp_path / "affinity.pt"
        torch.save({"model": {}, "epoch": 1, "global_step": 1}, path)
        runner = PretrainRunner(arm, PretrainConfig())
        with pytest.raises(PretrainError, match="is not a"):
            runner.load_checkpoint(path)

    def test_the_checkpoint_records_what_it_was_pretrained_on(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["kind"] == PRETRAIN_CHECKPOINT_KIND
        assert payload["provenance"]["split_regime"] == "random_pair"
        assert payload["model_kwargs"]["d_g_feats"] == SMALL["d_g_feats"]
        assert payload["pretrain_config"]["objective"].startswith("cross-prediction")
        # The published checkpoint's own key, so one loader reads either file.
        assert "model_state_dict" in payload


class TestTheCliIsReachable:
    def test_pretrain_is_registered(self):
        from bindra.cli.__main__ import COMMANDS

        assert "pretrain" in COMMANDS

    def test_the_parser_defaults_match_the_measured_configuration(self):
        from bindra.cli.pretrain import build_parser

        args = build_parser().parse_args([])
        assert args.batch_size == 512
        # 14 workers exhausted memory on this machine and killed the pool.
        assert args.featurize_workers == 8
        assert args.knowledge_cache == "data/cache/knowledge"

    def test_train_accepts_a_ligand_checkpoint(self):
        from bindra.cli.train import build_parser

        args = build_parser().parse_args(["--ligand-checkpoint", "runs/x/best.pt"])
        assert args.ligand_checkpoint == "runs/x/best.pt"
        assert build_parser().parse_args([]).ligand_checkpoint is None
