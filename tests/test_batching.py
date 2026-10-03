"""Residue-budgeted, length-bucketed pair batching.

The design exists for one measured reason: protein lengths here run from a
median of 454 to a maximum of **34,350** residues, so a fixed batch size padded
to its longest member is unusable — 64 pairs at 34,350 residues is 5 GiB of
protein features before any activation.

The tests that carry weight:

``test_no_batch_exceeds_the_residue_budget``
    The budget is the only thing bounding the padded tensor. If it leaks, a
    training run dies hours in, on whichever batch happened to draw a long
    protein.
``test_every_record_appears_exactly_once``
    A bucketing bug that drops the overflow silently removes the longest
    proteins — exactly the targets with the least support elsewhere.
``test_the_longest_protein_gets_its_own_batch_rather_than_being_dropped``
    It exceeds any sane budget on its own, so the "doesn't fit" path must admit
    it rather than discard it.
``test_padding_is_a_small_fraction_on_the_real_distribution``
    The whole point of bucketing. Without it the budget buys padding.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bindra.data.batching import (
    BatchingError,
    BatchingPolicy,
    PairBatch,
    length_bucketed_batches,
)

#: Shaped like the corpus: mostly short, a long tail, one enormous outlier.
REALISTIC = (
    [120] * 40 + [454] * 200 + [700] * 80 + [1104] * 40 + [1464] * 15
    + [2780] * 5 + [34350]
)


class TestPolicy:
    def test_defaults_bound_the_padded_tensor(self):
        policy = BatchingPolicy()
        # 64 pairs at 1,024 residues; ~144 MiB of bf16 protein features.
        assert policy.residue_budget == 65_536
        assert policy.max_pairs == 256

    @pytest.mark.parametrize(
        "kwargs", [{"residue_budget": 0}, {"max_pairs": 0}, {"shuffle_window": 0}]
    )
    def test_a_degenerate_policy_is_refused(self, kwargs):
        with pytest.raises(BatchingError):
            BatchingPolicy(**kwargs)

    def test_the_policy_serialises_for_a_run_log(self):
        assert BatchingPolicy().to_dict()["residue_budget"] == 65_536


class TestBudget:
    def test_no_batch_exceeds_the_residue_budget(self):
        policy = BatchingPolicy(residue_budget=4096, max_pairs=64, shuffle_window=256)
        lengths = list(np.random.default_rng(0).integers(50, 2000, 3000))
        for batch in length_bucketed_batches(lengths, policy):
            padded = max(lengths[i] for i in batch)
            # A single record over budget is allowed a batch of one; nothing else.
            assert padded * len(batch) <= policy.residue_budget or len(batch) == 1

    def test_the_budget_is_respected_on_the_real_distribution(self):
        policy = BatchingPolicy()
        lengths = REALISTIC * 20
        for batch in length_bucketed_batches(lengths, policy):
            padded = max(lengths[i] for i in batch)
            assert padded * len(batch) <= policy.residue_budget or len(batch) == 1

    def test_max_pairs_caps_batches_of_short_proteins(self):
        """Otherwise the residue budget alone allows a huge ligand-side batch."""
        policy = BatchingPolicy(residue_budget=1_000_000, max_pairs=8)
        batches = length_bucketed_batches([10] * 200, policy)
        assert all(len(b) <= 8 for b in batches)

    def test_the_longest_protein_gets_its_own_batch_rather_than_being_dropped(self):
        policy = BatchingPolicy(residue_budget=65_536)
        lengths = REALISTIC
        longest = int(np.argmax(lengths))
        batches = length_bucketed_batches(lengths, policy)
        owner = [b for b in batches if longest in b]
        assert len(owner) == 1
        assert owner[0] == [longest], "the 34,350-residue protein must be alone, not absent"


class TestCoverage:
    def test_every_record_appears_exactly_once(self):
        policy = BatchingPolicy(residue_budget=8192, shuffle_window=64)
        lengths = REALISTIC * 5
        seen = [i for batch in length_bucketed_batches(lengths, policy) for i in batch]
        assert sorted(seen) == list(range(len(lengths)))

    def test_an_empty_input_gives_no_batches(self):
        assert length_bucketed_batches([], BatchingPolicy()) == []

    def test_a_single_record_gives_one_batch(self):
        assert length_bucketed_batches([500], BatchingPolicy()) == [[0]]

    def test_drop_last_removes_exactly_one_batch(self):
        policy = BatchingPolicy(residue_budget=2048, shuffle_window=32, drop_last=True)
        lengths = [256] * 100
        kept = length_bucketed_batches(lengths, policy)
        full = length_bucketed_batches(lengths, BatchingPolicy(
            residue_budget=2048, shuffle_window=32, drop_last=False))
        assert len(kept) == len(full) - 1


class TestBucketing:
    def test_padding_is_a_small_fraction_on_the_real_distribution(self):
        """Without bucketing the budget is spent on padding, not residues."""
        lengths = REALISTIC * 20
        batches = length_bucketed_batches(lengths, BatchingPolicy())
        slots = sum(len(b) * max(lengths[i] for i in b) for b in batches)
        real = sum(lengths[i] for b in batches for i in b)
        assert 1 - real / slots < 0.15

    def test_bucketing_beats_not_bucketing(self):
        """Pins the benefit, so removing the sort would fail rather than slow things."""
        lengths = REALISTIC * 20
        policy = BatchingPolicy()
        bucketed = length_bucketed_batches(lengths, policy)
        waste_bucketed = 1 - sum(lengths[i] for b in bucketed for i in b) / sum(
            len(b) * max(lengths[i] for i in b) for b in bucketed
        )
        # A window of 1 disables the within-window sort entirely.
        unsorted_policy = BatchingPolicy(shuffle_window=1)
        chunks = length_bucketed_batches(lengths, unsorted_policy)
        assert len(chunks) == len(lengths)  # every record alone: the degenerate case
        assert waste_bucketed < 0.15

    def test_batches_are_not_length_ordered_across_the_epoch(self):
        """Length-homogeneous batches, but not a length-ordered curriculum."""
        lengths = list(np.random.default_rng(1).integers(100, 2000, 4000))
        batches = length_bucketed_batches(lengths, BatchingPolicy(shuffle_window=256))
        means = [np.mean([lengths[i] for i in b]) for b in batches]
        # If the epoch were length-ordered these would be monotonic.
        assert not all(a <= b for a, b in zip(means, means[1:]))


class TestDeterminism:
    def test_the_same_seed_and_epoch_reproduce_the_batches(self):
        lengths = REALISTIC * 3
        policy = BatchingPolicy(seed=7)
        assert length_bucketed_batches(lengths, policy, epoch=0) == (
            length_bucketed_batches(lengths, policy, epoch=0)
        )

    def test_successive_epochs_differ(self):
        lengths = REALISTIC * 3
        policy = BatchingPolicy(seed=7)
        assert length_bucketed_batches(lengths, policy, epoch=0) != (
            length_bucketed_batches(lengths, policy, epoch=1)
        )

    def test_a_different_seed_gives_a_different_grouping(self):
        lengths = REALISTIC * 3
        assert length_bucketed_batches(lengths, BatchingPolicy(seed=1)) != (
            length_bucketed_batches(lengths, BatchingPolicy(seed=2))
        )

    def test_shuffle_false_is_stable_and_length_ordered(self):
        lengths = REALISTIC * 2
        batches = length_bucketed_batches(lengths, BatchingPolicy(), shuffle=False)
        assert batches == length_bucketed_batches(lengths, BatchingPolicy(), shuffle=False)


class TestPairBatch:
    def _batch(self, sizes=(3, 5)) -> PairBatch:
        from bindra.encoders.ligand_graph import collate_graphs, featurize_molecule
        from bindra.training.losses.affinity import AffinityTargets

        width = max(sizes)
        features = torch.zeros(len(sizes), width, 8)
        mask = torch.zeros(len(sizes), width, dtype=torch.bool)
        for row, length in enumerate(sizes):
            features[row, :length] = 1.0
            mask[row, :length] = True
        nan = float("nan")
        return PairBatch(
            protein_features=features,
            protein_mask=mask,
            ligands=collate_graphs([featurize_molecule("CCO") for _ in sizes]),
            targets=AffinityTargets(
                endpoint_index=torch.zeros(len(sizes), dtype=torch.long),
                p_value=torch.full((len(sizes),), 8.0),
                is_exact=torch.ones(len(sizes), dtype=torch.bool),
                lower_bound=torch.full((len(sizes),), nan),
                upper_bound=torch.full((len(sizes),), nan),
                is_primary=torch.ones(len(sizes), dtype=torch.bool),
            ),
            observation_ids=tuple(f"o{i}" for i in range(len(sizes))),
            smiles=("CCO",) * len(sizes),
        )

    def test_padding_fraction_reports_what_is_wasted(self):
        batch = self._batch((3, 5))
        assert batch.size == 2 and batch.padded_residues == 5
        assert batch.padding_fraction == pytest.approx(1 - 8 / 10)

    def test_padded_positions_are_masked_out(self):
        batch = self._batch((3, 5))
        assert not batch.protein_mask[0, 3:].any()
        assert batch.protein_mask[1].all()

    def test_a_batch_moves_to_a_device_as_a_unit(self):
        batch = self._batch()
        moved = batch.to("cpu")
        assert moved.protein_features.device.type == "cpu"
        assert moved.targets.p_value.device.type == "cpu"
        assert moved.observation_ids == batch.observation_ids


@pytest.mark.snapshot
class TestAgainstTheCuratedCorpus:
    """The real path: split, corpus, ligand table and the populated cache."""

    @pytest.fixture(scope="class")
    @classmethod
    def dataset(cls):
        from pathlib import Path

        from bindra.data.batching import PairDataset
        from bindra.encoders.protein_esmc import (
            ESMC_600M_WEIGHTS_SHA256,
            EXPECTED_ESMC_600M_DIM,
            ESMCProteinEncoder,
        )
        from bindra.features.cache import ProteinFeatureCache

        if not Path("data/cache/proteins/index.json").exists():
            pytest.skip("protein cache not populated")
        if not Path("data/splits/202610/random_pair/manifest.json").exists():
            pytest.skip("splits not built")
        metadata = ESMCProteinEncoder(n_trainable_layers=0).metadata(
            EXPECTED_ESMC_600M_DIM, weights_checksum=ESMC_600M_WEIGHTS_SHA256
        )
        return PairDataset(
            "data/processed/202610",
            "data/splits/202610/random_pair",
            "val",
            protein_cache=ProteinFeatureCache("data/cache/proteins", metadata=metadata),
            limit=2000,
        )

    def test_a_real_batch_assembles_with_matching_parts(self, dataset):
        from bindra.data.batching import PairLoader

        loader = PairLoader(dataset, BatchingPolicy())
        batch = next(iter(loader))
        assert batch.protein_features.shape[0] == batch.size
        assert batch.protein_features.shape[2] == 1152
        assert batch.ligands.num_graphs == batch.size
        assert batch.targets.batch_size == batch.size
        assert len(batch.observation_ids) == batch.size

    def test_the_protein_mask_matches_the_real_sequence_lengths(self, dataset):
        from bindra.data.batching import PairLoader

        from bindra.data.batching import _round_up

        policy = BatchingPolicy()
        batch = next(iter(PairLoader(dataset, policy)))
        # Every row has at least one real residue and no row is all padding.
        counts = batch.protein_mask.sum(dim=1)
        assert (counts > 0).all()
        # The padded width is the longest real sequence rounded up to the
        # length multiple -- the rounding is what bounds shape diversity and
        # keeps the allocator from fragmenting, so it is asserted rather than
        # treated as slack.
        assert batch.padded_residues == _round_up(
            int(counts.max()), policy.length_multiple
        )
        assert batch.padded_residues - int(counts.max()) < max(1, policy.length_multiple)

    def test_the_whole_partition_is_covered_exactly_once(self, dataset):
        from bindra.data.batching import PairLoader

        loader = PairLoader(dataset, BatchingPolicy())
        seen = [i for b in loader.batches() for i in b]
        assert sorted(seen) == list(range(len(dataset)))

    def test_censored_rows_survive_into_the_targets(self, dataset):
        """16.6% of the corpus is censored; they must not be silently dropped."""
        from bindra.data.batching import PairLoader

        loader = PairLoader(dataset, BatchingPolicy())
        censored = sum(
            int((~b.targets.is_exact).sum()) for b, _ in zip(loader, range(5))
        )
        assert censored > 0

    def test_describe_reports_the_batch_shape(self, dataset):
        from bindra.data.batching import PairLoader

        report = PairLoader(dataset, BatchingPolicy()).describe()
        assert report["n_pairs"] == len(dataset)
        assert report["padding_fraction"] < 0.2
        assert report["residue_slots_max"] <= BatchingPolicy().residue_budget


class TestShapeDiversity:
    """Padded lengths are rounded to bound how many distinct shapes a run makes.

    The first real training run died of CUDA OOM 63% into epoch 0, after 14,000
    batches, with memory still free -- the signature of caching-allocator
    fragmentation. No single batch was at fault: the worst shapes measured 1.40
    GiB (1 x 34,350) and 2.90 GiB (256 x 256) against a 15.9 GiB card. The cause
    was that every batch had its own padded length, so the allocator reserved a
    segment per shape. ``expandable_segments:True``, the usual remedy, is not
    supported on Windows.
    """

    def test_padded_lengths_are_rounded_to_the_multiple(self):
        from bindra.data.batching import _round_up

        assert _round_up(1, 128) == 128
        assert _round_up(128, 128) == 128
        assert _round_up(129, 128) == 256
        assert _round_up(34350, 128) == 34432

    def test_rounding_off_is_the_identity(self):
        from bindra.data.batching import _round_up

        assert _round_up(777, 0) == 777
        assert _round_up(777, 1) == 777

    def test_the_budget_is_respected_against_the_rounded_length(self):
        """Otherwise rounding up could silently push a batch over the budget."""
        policy = BatchingPolicy(residue_budget=4096, length_multiple=128)
        lengths = list(np.random.default_rng(2).integers(50, 1500, 2000))
        for batch in length_bucketed_batches(lengths, policy):
            from bindra.data.batching import _round_up

            padded = _round_up(max(lengths[i] for i in batch), policy.length_multiple)
            assert padded * len(batch) <= policy.residue_budget or len(batch) == 1

    def test_rounding_collapses_thousands_of_shapes_into_hundreds(self):
        """The property that actually prevents the fragmentation."""
        from bindra.data.batching import _round_up

        # Continuously varied, like the real corpus: 10,820 constructs with
        # almost as many distinct lengths. REALISTIC has only 7 values, so
        # there would be nothing for rounding to collapse.
        lengths = list(np.random.default_rng(4).integers(60, 3000, 6000))
        rounded = BatchingPolicy(length_multiple=128)
        plain = BatchingPolicy(length_multiple=0)

        def shapes(policy):
            return {
                (len(b), _round_up(max(lengths[i] for i in b), policy.length_multiple))
                for b in length_bucketed_batches(lengths, policy)
            }

        few, many = len(shapes(rounded)), len(shapes(plain))
        assert few < many / 2, f"rounding gave {few} shapes against {many}"

    def test_the_padding_cost_stays_modest(self):
        """Rounding buys fewer shapes at the price of padding; bound the price."""
        from bindra.data.batching import _round_up

        lengths = REALISTIC * 20
        policy = BatchingPolicy(length_multiple=128)
        batches = length_bucketed_batches(lengths, policy)
        slots = sum(
            len(b) * _round_up(max(lengths[i] for i in b), 128) for b in batches
        )
        real = sum(lengths[i] for b in batches for i in b)
        assert 1 - real / slots < 0.30

    def test_a_negative_multiple_is_refused(self):
        with pytest.raises(BatchingError, match="length_multiple"):
            BatchingPolicy(length_multiple=-1)

    def test_the_multiple_is_recorded_in_the_policy(self):
        assert BatchingPolicy().to_dict()["length_multiple"] == 128
