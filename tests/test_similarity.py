"""Fingerprints, folding, and the exactness of the two-stage neighbour search.

The headline test is ``test_folding_is_not_a_guaranteed_upper_bound``. It is
tempting to argue that folding can only over-report similarity, and that argument
is wrong; the screen is a heuristic and is treated as one, with the provably sound
popcount bound applied alongside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from bindra.data.similarity import (
    DEFAULT_BITS,
    fold,
    measure_fold_shortfall,
    morgan_fingerprints,
    nearest_neighbour,
    popcount_upper_bound,
    tanimoto_matrix,
    unpack,
)

MOLECULES = [
    "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)Nc1ccc(OC)cc1",
    "CC(=O)Nc1ccc(Cl)cc1",
    "c1ccc2ccccc2c1CCN",
    "CCN(CC)CC",
    "OC(=O)c1ccccc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "C1CCNCC1",
    "Cc1ccccc1",
    "CCO",
]


@pytest.fixture
def packed():
    fingerprints, valid = morgan_fingerprints(MOLECULES)
    assert valid.all()
    return fingerprints


class TestFingerprints:
    def test_shape_and_validity(self, packed):
        assert packed.shape == (len(MOLECULES), DEFAULT_BITS // 8)
        assert packed.dtype == np.uint8

    def test_invalid_smiles_is_flagged_not_raised(self):
        fingerprints, valid = morgan_fingerprints(["CCO", "not-a-molecule", ""])
        assert list(valid) == [True, False, False]
        # An unparseable molecule gets an all-zero row and is never treated as real.
        assert fingerprints[1].sum() == 0

    def test_identical_molecules_have_identical_fingerprints(self):
        fingerprints, _ = morgan_fingerprints(["CCO", "OCC"])
        assert np.array_equal(fingerprints[0], fingerprints[1])

    def test_self_similarity_is_one(self, packed):
        bits = unpack(packed)
        matrix = tanimoto_matrix(bits, bits)
        assert np.allclose(np.diag(matrix), 1.0)

    def test_similar_analogues_score_above_unrelated_molecules(self, packed):
        bits = unpack(packed)
        matrix = tanimoto_matrix(bits, bits)
        # Paracetamol-like pair against an unrelated amine.
        assert matrix[0, 1] > matrix[0, 4]


class TestFolding:
    def test_folding_reduces_width(self, packed):
        assert fold(packed, 512).shape == (len(MOLECULES), 512)

    def test_folding_to_a_non_divisor_is_refused(self, packed):
        with pytest.raises(Exception):
            fold(packed, 300)

    def test_folding_never_sets_a_bit_that_was_unset(self, packed):
        """Each folded bit is an OR over its block, so it cannot invent evidence."""
        bits = unpack(packed)
        folded = fold(packed, 512)
        blocks = bits.reshape(bits.shape[0], DEFAULT_BITS // 512, 512)
        assert np.array_equal(folded, blocks.max(axis=1))

    def test_folding_is_not_a_guaranteed_upper_bound(self, packed):
        """The inequality people assume here is false, and the code says so.

        Two bits set in the same molecule can collide into one folded position,
        shrinking that molecule's popcount and with it the intersection. A
        hand-built case makes it concrete.
        """
        # A and B both carry bits 0 and 512; with to_bits=512 they collide.
        width = 1024
        a = np.zeros(width, dtype=np.uint8)
        b = np.zeros(width, dtype=np.uint8)
        a[0] = a[512] = 1
        b[0] = b[512] = b[1] = 1

        packed_pair = np.packbits(np.stack([a, b]), axis=1)
        true_value = tanimoto_matrix(np.stack([a]), np.stack([b]))[0, 0]
        folded = fold(packed_pair, 512)
        folded_value = tanimoto_matrix(folded[:1], folded[1:])[0, 0]
        assert folded_value < true_value

    def test_shortfall_is_measured_and_reported(self, packed):
        summary = measure_fold_shortfall(packed, n_pairs=200)
        assert set(summary) >= {
            "n_pairs",
            "n_below_true",
            "fraction_below_true",
            "worst_shortfall",
        }
        assert "NOT a guaranteed upper bound" in summary["note"]


class TestPopcountBound:
    def test_the_bound_is_never_below_the_true_similarity(self, packed):
        """This one really is sound, which is why it gates the screen."""
        bits = unpack(packed)
        bound = popcount_upper_bound(bits, bits)
        true_value = tanimoto_matrix(bits, bits)
        assert bool((bound + 1e-6 >= true_value).all())

    def test_the_bound_holds_on_random_fingerprints(self):
        rng = np.random.default_rng(5)
        bits = (rng.random((40, 256)) < 0.1).astype(np.uint8)
        bound = popcount_upper_bound(bits, bits)
        assert bool((bound + 1e-6 >= tanimoto_matrix(bits, bits)).all())


class TestNearestNeighbour:
    def test_top_k_matches_brute_force(self, packed):
        query, reference = packed[:4], packed[4:]
        result = nearest_neighbour(query, reference, top_k=3, rescore_top=len(reference))
        brute = tanimoto_matrix(unpack(query), unpack(reference))
        expected = np.sort(np.sort(brute, axis=1)[:, -3:], axis=1)
        assert np.allclose(np.sort(result.similarity, axis=1), expected, atol=1e-6)

    def test_returned_indices_point_at_the_reported_similarity(self, packed):
        query, reference = packed[:3], packed[3:]
        result = nearest_neighbour(query, reference, top_k=2, rescore_top=len(reference))
        brute = tanimoto_matrix(unpack(query), unpack(reference))
        for row in range(query.shape[0]):
            for column in range(2):
                index = result.index[row, column]
                assert brute[row, index] == pytest.approx(
                    result.similarity[row, column], abs=1e-6
                )

    def test_a_molecule_finds_itself_first(self, packed):
        result = nearest_neighbour(packed, packed, top_k=1, rescore_top=len(packed))
        assert np.allclose(result.similarity[:, 0], 1.0)
        assert list(result.index[:, 0]) == list(range(len(packed)))

    def test_empty_inputs_are_handled(self, packed):
        empty = packed[:0]
        assert nearest_neighbour(empty, packed, top_k=1).similarity.shape[0] == 0
        assert nearest_neighbour(packed, empty, top_k=1).similarity.shape[0] == len(packed)

    @pytest.mark.gpu
    def test_gpu_agrees_with_cpu(self, packed):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from bindra.runtime.device import DeviceRequirement, select_device

        device = str(select_device(DeviceRequirement()))
        query, reference = packed[:5], packed[5:]
        on_cpu = nearest_neighbour(query, reference, top_k=2, rescore_top=len(reference))
        on_gpu = nearest_neighbour(
            query, reference, top_k=2, rescore_top=len(reference), device=device
        )
        assert np.allclose(on_cpu.similarity, on_gpu.similarity, atol=1e-5)


class TestCrossPartitionAudit:
    def test_audit_reports_a_distribution_and_its_own_assumption(self, packed):
        from bindra.data.similarity import cross_partition_audit

        payload = cross_partition_audit(packed[:4], packed[4:], thresholds=(0.5, 0.9))
        assert payload["n_query"] == 4
        assert set(payload["max_tanimoto"]) == {"median", "p90", "max"}
        assert set(payload["fraction_at_or_above"]) == {"0.5", "0.9"}
        # The screen's measured shortfall travels with the audit.
        assert "screen_shortfall" in payload
        assert "Reported, not a failure condition" in payload["interpretation"]


class TestFingerprintStore:
    """The store exists so three consumers build fingerprints once, not thrice."""

    def _store(self, n: int = 50_000):
        from bindra.data.similarity import FingerprintStore

        return FingerprintStore(
            identity_keys=tuple(f"L{i}" for i in range(n)),
            packed=np.zeros((n, 256), dtype=np.uint8),
            valid=np.ones(n, dtype=bool),
            radius=2,
            n_bits=2048,
            use_chirality=False,
            rdkit_version="test",
        )

    def test_the_index_is_memoised_not_rebuilt(self):
        """A rebuilt index made the cliff scan O(n) per comparability group.

        With 1.43M molecules and 165,020 groups that was ~2e11 insertions and the
        scan never progressed. Identity of the returned dict is the cheap,
        unambiguous way to assert it is built once.
        """
        store = self._store()
        assert store.index is store.index

    def test_subset_is_fast_enough_to_call_per_group(self):
        """A loose wall-clock bound, because the failure mode was a hang."""
        import time

        store = self._store()
        store.index  # warm the memo, as the real caller does
        start = time.perf_counter()
        for _ in range(500):
            store.subset(["L1", "L2", "L3"])
        elapsed = time.perf_counter() - start
        # Rebuilding a 50k dict each time would be ~5 ms per call, i.e. ~2.5 s.
        assert elapsed < 0.5, f"subset() took {elapsed:.3f}s for 500 calls"

    def test_subset_returns_only_valid_entries(self):
        from bindra.data.similarity import FingerprintStore

        store = FingerprintStore(
            identity_keys=("a", "b", "c"),
            packed=np.zeros((3, 256), dtype=np.uint8),
            valid=np.array([True, False, True]),
            radius=2,
            n_bits=2048,
            use_chirality=False,
            rdkit_version="test",
        )
        keys, packed = store.subset(["a", "b", "c", "missing"])
        assert keys == ["a", "c"]
        assert packed.shape == (2, 256)

    def test_subset_of_nothing_returns_an_empty_block(self):
        keys, packed = self._store(10).subset(["nope"])
        assert keys == []
        assert packed.shape == (0, 256)

    def test_mismatched_lengths_are_refused(self):
        from bindra.data.similarity import FingerprintStore, SimilarityError

        with pytest.raises(SimilarityError):
            FingerprintStore(
                identity_keys=("a", "b"),
                packed=np.zeros((1, 256), dtype=np.uint8),
                valid=np.ones(1, dtype=bool),
                radius=2,
                n_bits=2048,
                use_chirality=False,
                rdkit_version="test",
            )

    def test_a_parameter_mismatch_is_refused_by_the_cliff_builder(self):
        """A store at a different radius would redefine the panel's similarity."""
        from bindra.eval.benchmarks.activity_cliff import CliffDefinition, build_cliff_candidates

        store = self._store(10)
        object.__setattr__(store, "radius", 3)
        with pytest.raises(ValueError, match="does not match this panel definition"):
            build_cliff_candidates(
                ".", definition=CliffDefinition(radius=2), fingerprints=store
            )
