"""Retrieval, the delta head, analog pairs, and prospective bookkeeping.

The properties here are all ones that are invisible in a loss curve and
catastrophic in a result:

- a retrieval tower that is pair-conditioned gives spectacular offline numbers
  and an index that cannot be built
- an unmeasured pair used as a negative teaches that absence of a measurement
  is evidence of non-binding
- an analog pair built from a held-out observation carries its label into
  training
- a technical failure counted as an inactive charges a handling problem to the
  model
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bindra.data.pairs import AnalogPairError, PairPolicy, assert_no_held_out, build_pairs
from bindra.heads.delta import (
    DeltaHead,
    DeltaHeadConfig,
    assert_delta_properties,
    difference_baseline,
)
from bindra.heads.retrieval import (
    RetrievalHeadConfig,
    RetrievalHeads,
    assert_not_pair_conditioned,
)
from bindra.retrieval.fishing import combined_report, enrichment, fish
from bindra.retrieval.index import IndexContract, VectorIndex
from bindra.training.losses.retrieval import listwise_loss


class TestRetrievalTowers:
    def test_the_towers_are_not_pair_conditioned(self):
        """An index over a pair-conditioned vector would have to be rebuilt per query."""
        head = RetrievalHeads(RetrievalHeadConfig(model_dim=48, embedding_dim=32))
        assert_not_pair_conditioned(head, 48)

    def test_each_side_encodes_without_the_other(self):
        head = RetrievalHeads(RetrievalHeadConfig(model_dim=48, embedding_dim=32))
        protein = head.encode_protein(torch.randn(3, 20, 48), torch.ones(3, 20, dtype=torch.bool))
        ligand = head.encode_ligand(torch.randn(5, 11, 48), torch.ones(5, 11, dtype=torch.bool))
        assert protein.shape == (3, 32) and ligand.shape == (5, 32)
        assert head.similarity(protein, ligand).shape == (3, 5)

    def test_outputs_are_unit_norm_when_the_contract_says_so(self):
        head = RetrievalHeads(RetrievalHeadConfig(model_dim=48, embedding_dim=32, normalize=True))
        vectors = head.encode_ligand(torch.randn(4, 9, 48), torch.ones(4, 9, dtype=torch.bool))
        assert torch.allclose(vectors.norm(dim=-1), torch.ones(4), atol=1e-5)


class TestIndex:
    def _index(self, n=500, dim=16, seed=0):
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(n, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = VectorIndex(IndexContract(checkpoint="t", embedding_dim=dim, normalized=True))
        index.add([f"L{i}" for i in range(n)], vectors)
        return index, vectors

    def test_exact_search_returns_the_query_itself_first(self):
        index, vectors = self._index()
        ids, _ = index.search_exact(vectors[7:8], 3)
        assert int(ids[0][0]) == 7

    def test_approximation_loss_is_measured_not_assumed(self):
        index, vectors = self._index()
        index.train_coarse(16)
        report = index.recall_against_exact(vectors[:20], 10)
        assert 0.0 <= report["approximation_recall"] <= 1.0
        assert report["approximation_loss"] == pytest.approx(1 - report["approximation_recall"])

    def test_shortlisting_loss_is_reported_separately(self):
        """Different failure, different fix; one number conflates them."""
        report = VectorIndex.shortlist_loss([1, 5, 80, 200], 50)
        assert report["shortlisting_loss"] == pytest.approx(0.5)
        assert "independent of approximation loss" in report["note"]

    def test_an_incompatible_contract_is_refused(self):
        index, _ = self._index()
        other = IndexContract(checkpoint="different", embedding_dim=16, normalized=True)
        with pytest.raises(Exception, match="contract mismatch"):
            index.contract.assert_compatible(other)

    def test_unnormalized_vectors_under_a_normalized_contract_are_refused(self):
        index = VectorIndex(IndexContract(checkpoint="t", embedding_dim=4, normalized=True))
        with pytest.raises(Exception, match="not unit length"):
            index.add(["a"], np.array([[3.0, 0, 0, 0]], dtype=np.float32))


class TestFishingModes:
    def _index(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=(200, 8)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        index = VectorIndex(IndexContract(checkpoint="t", embedding_dim=8, normalized=True))
        index.add([f"L{i}" for i in range(200)], v)
        return index, v

    def test_seed_assisted_requires_a_seed(self):
        """Otherwise it silently degrades into the easier label's opposite."""
        index, v = self._index()
        with pytest.raises(Exception, match="seed"):
            fish(index, v[0], mode="seed_assisted")

    def test_the_mode_travels_with_the_result(self):
        index, v = self._index()
        result = fish(index, v[0], mode="seed_assisted", seed_id="L5", seed_vector=v[5])
        payload = result.to_dict()
        assert payload["mode"] == "seed_assisted"
        assert "not zero-shot discovery" in payload["caveat"]

    def test_target_only_carries_no_caveat(self):
        index, v = self._index()
        assert "caveat" not in fish(index, v[0]).to_dict()

    def test_the_combined_report_keeps_them_apart(self):
        report = combined_report()
        assert "not comparable" in report["warning"]
        assert report["target_only"]["status"] == "not run"

    def test_a_random_ranker_enriches_by_about_one(self):
        actives = {f"L{i}" for i in range(20)}
        ranked = [f"L{i}" for i in range(200)]
        rng = np.random.default_rng(0)
        rng.shuffle(ranked)
        result = enrichment(ranked, actives, 200, 0.5)
        assert result["enrichment_factor"] == pytest.approx(1.0, abs=0.6)


class TestListwiseLoss:
    def test_unmeasured_pairs_are_never_negatives(self):
        """Absence of a measurement is not evidence of non-binding."""
        scores = torch.randn(2, 10)
        potency = torch.full((2, 10), float("nan"))
        potency[:, :3] = torch.tensor([9.0, 7.0, 4.0])
        out = listwise_loss(scores, potency, torch.zeros(2, 10, dtype=torch.long))
        # Only the 3 measured entries can pair: 3 ordered pairs per query.
        assert out.n_pairs <= 2 * 3

    def test_pairs_inside_the_noise_floor_are_dropped(self):
        """A gap under the replicate scatter says nothing about ordering."""
        scores = torch.randn(1, 4)
        # Kd floor is 0.945; a 0.1 gap is noise.
        potency = torch.tensor([[8.0, 8.1, 8.05, 7.95]])
        out = listwise_loss(scores, potency, torch.zeros(1, 4, dtype=torch.long))
        assert out.n_pairs == 0
        assert out.n_dropped_below_margin > 0

    def test_a_clear_gap_produces_pairs(self):
        scores = torch.randn(1, 4)
        potency = torch.tensor([[10.0, 8.0, 6.0, 4.0]])
        out = listwise_loss(scores, potency, torch.zeros(1, 4, dtype=torch.long))
        assert out.n_pairs > 0

    def test_too_few_measured_rows_yields_no_loss(self):
        scores = torch.randn(1, 5)
        potency = torch.full((1, 5), float("nan"))
        potency[0, 0] = 9.0
        out = listwise_loss(scores, potency, torch.zeros(1, 5, dtype=torch.long))
        assert out.n_queries == 0


class TestDeltaHead:
    def test_both_structural_properties_hold(self):
        head = DeltaHead(DeltaHeadConfig(pair_dim=64, hidden_dim=32))
        assert_delta_properties(head, 64)

    def test_they_survive_random_weights(self):
        """Structural means at every parameter value, not just at init."""
        head = DeltaHead(DeltaHeadConfig(pair_dim=48, hidden_dim=24))
        for parameter in head.parameters():
            with torch.no_grad():
                parameter.normal_(0, 0.4)
        assert_delta_properties(head, 48)

    def test_the_baseline_is_exactly_the_difference(self):
        a, b = torch.randn(5, 3), torch.randn(5, 3)
        assert torch.equal(difference_baseline(a, b), b - a)

    def test_the_baseline_is_the_default(self):
        """The dedicated head is the ablation, not the default."""
        assert DeltaHeadConfig(pair_dim=8).use_dedicated_head is False
        assert DeltaHeadConfig(pair_dim=8).to_dict()["default"] == "difference_baseline"

    def test_mismatched_shapes_are_refused(self):
        head = DeltaHead(DeltaHeadConfig(pair_dim=16, hidden_dim=8))
        with pytest.raises(Exception, match="shapes differ"):
            head(torch.randn(2, 16), torch.randn(2, 8))


class TestAnalogPairs:
    def _frame(self):
        import pandas as pd

        return pd.DataFrame({
            "observation_id": ["o1", "o2", "o3", "o4"],
            "sequence_sha256": ["t1"] * 4,
            "endpoint": ["Ki"] * 4,
            "assay_condition_bucket": ["pH7.4|T1"] * 4,
            "p_value": [9.0, 7.0, 5.0, 8.0],
            "is_exact": [True, True, False, True],
        })

    def test_censored_rows_are_never_differenced(self):
        """Subtracting a detection limit invents a delta."""
        frame = self._frame()
        pairs = build_pairs(frame, partition_of={f"o{i}": "train" for i in range(1, 5)})
        assert pairs.attrition["dropped_censored"] == 1
        assert "o3" not in set(pairs.left) | set(pairs.right)

    def test_held_out_observations_are_excluded(self):
        frame = self._frame()
        partition = {"o1": "train", "o2": "val", "o3": "train", "o4": "train"}
        pairs = build_pairs(frame, partition_of=partition)
        assert pairs.attrition["dropped_wrong_partition"] == 1
        assert_no_held_out(pairs, partition)

    def test_building_without_partitions_is_refused(self):
        with pytest.raises(AnalogPairError, match="partition_of is required"):
            build_pairs(self._frame())

    def test_the_series_cap_bites(self):
        import pandas as pd

        n = 40
        frame = pd.DataFrame({
            "observation_id": [f"o{i}" for i in range(n)],
            "sequence_sha256": ["t1"] * n,
            "endpoint": ["Ki"] * n,
            "assay_condition_bucket": ["pH7.4|T1"] * n,
            "p_value": np.linspace(4.0, 11.0, n),
            "is_exact": [True] * n,
        })
        pairs = build_pairs(
            frame,
            partition_of={f"o{i}": "train" for i in range(n)},
            policy=PairPolicy(max_pairs_per_group=20),
        )
        assert len(pairs) <= 20
        assert pairs.attrition["dropped_by_series_cap"] > 0

    def test_incompatible_endpoints_never_pair(self):
        import pandas as pd

        frame = pd.DataFrame({
            "observation_id": ["o1", "o2"],
            "sequence_sha256": ["t1", "t1"],
            "endpoint": ["Ki", "IC50"],
            "assay_condition_bucket": ["pH7.4|T1"] * 2,
            "p_value": [9.0, 5.0],
            "is_exact": [True, True],
        })
        pairs = build_pairs(frame, partition_of={"o1": "train", "o2": "train"})
        assert len(pairs) == 0


class TestProspective:
    def test_a_technical_failure_is_not_a_negative_result(self):
        from bindra.prospective.ingest import IngestedRound, Record, RecordKind, hit_rate

        round_ = IngestedRound("r1")
        round_.add(Record("C1", RecordKind.BINDING, {"value": 8.0}))
        round_.add(Record("C2", RecordKind.BINDING, {"value": 5.0}))
        round_.add(Record("C3", RecordKind.FAILURE, {"reason": "insoluble"}))
        result = hit_rate(round_, threshold=7.0)
        assert result["n_measured"] == 2, "the failure must not be in the denominator"
        assert result["hit_rate"] == pytest.approx(0.5)
        assert result["excluded_technical_failures"] == 1

    def test_an_unnamed_failure_reason_is_refused(self):
        from bindra.prospective.ingest import IngestError, Record, RecordKind

        with pytest.raises(IngestError, match="reason"):
            Record("C1", RecordKind.FAILURE, {"reason": "it didn't work"})

    def test_the_funnel_must_nest(self):
        from bindra.prospective.manifest import ProspectiveManifest
        from bindra.prospective.report import ReportError, RoundReport

        manifest = ProspectiveManifest(
            round_id="r1", model_checkpoint="ck", model_provenance_digest="d",
            budget=10, ranking_rule="top-k", selection_date="2026-10-10",
            candidate_codes=("C1", "C2"), candidate_universe_digest="u",
            assignment_key_digest="k",
        )
        with pytest.raises(ReportError, match="not a subset"):
            RoundReport(manifest=manifest, planned={"C1"}, selected={"C1", "C2"})

    def test_a_manifest_cannot_exceed_its_own_budget(self):
        from bindra.prospective.manifest import ManifestError, ProspectiveManifest

        with pytest.raises(ManifestError, match="budget"):
            ProspectiveManifest(
                round_id="r1", model_checkpoint="ck", model_provenance_digest="d",
                budget=1, ranking_rule="top-k", selection_date="2026-10-10",
                candidate_codes=("C1", "C2"), candidate_universe_digest="u",
                assignment_key_digest="k",
            )

    def test_a_refit_cannot_pose_as_the_original_prediction(self):
        from bindra.prospective.manifest import ProspectiveManifest
        from bindra.prospective.report import (
            ReportError,
            RoundReport,
            assert_prediction_is_original,
        )

        manifest = ProspectiveManifest(
            round_id="r1", model_checkpoint="original.pt", model_provenance_digest="d",
            budget=10, ranking_rule="top-k", selection_date="2026-10-10",
            candidate_codes=("C1",), candidate_universe_digest="u", assignment_key_digest="k",
        )
        report = RoundReport(manifest=manifest, planned={"C1"}, selected={"C1"})
        with pytest.raises(ReportError, match="retrospective refit"):
            assert_prediction_is_original(report, "refit.pt")


class TestLegacyQuarantine:
    def test_it_cannot_be_constructed_without_acknowledging_the_limits(self):
        from bindra.generation.legacy_latent import (
            LegacyLatentConfig,
            LegacyLatentDiffusion,
            LegacyLatentError,
        )

        with pytest.raises(LegacyLatentError, match="ablation"):
            LegacyLatentDiffusion(LegacyLatentConfig(feature_dim=8))

    def test_it_has_no_decoder_and_says_so(self):
        from bindra.generation.legacy_latent import LegacyLatentDiffusion, LegacyLatentError

        with pytest.raises(LegacyLatentError, match="no molecular decoder"):
            LegacyLatentDiffusion.decode()
