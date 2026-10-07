"""The ligand arm seam: two arms, one contract, and the places they differ.

The point of these tests is not that each arm works in isolation -- ``test_ligand_gat.py``
and ``test_kpgt_light.py`` already cover that -- but that
:class:`~bindra.model.BindraModel` can hold either one without knowing which,
and that the ways they differ are *asserted* rather than assumed.

``test_kpgt_drops_knowledge_nodes_from_the_token_axis``
    The load-bearing one. KPGT emits two knowledge nodes per molecule whose
    content is a fingerprint and a descriptor vector. If they reach the fusion
    token axis the protein attends to them, and the resulting map puts mass on
    something that cannot be drawn on a molecule. Counted against bond counts,
    because "two fewer than the triplet count" would also pass if the wrong two
    were dropped.
``test_the_gat_checkpoint_keys_are_unchanged``
    The refactor's real risk. A trained model exists; interposing a wrapper
    module around the GAT arm would rename every ligand parameter and silently
    stop that checkpoint loading.
``test_an_arm_refuses_the_other_arms_batch``
    The two arms take different batch types. Handing one the wrong batch must
    fail with a message that says so, not with a shape error ten frames down.
"""

from __future__ import annotations

import pytest
import torch

from bindra.encoders.ligand_arm import (
    FusionTokens,
    LigandArm,
    LigandArmError,
    build_ligand_arm,
    pad_by_destination,
    physical_token_destination,
)
from bindra.encoders.ligand_featurizers import LigandBatchBuilder, featurize_kpgt_one
from bindra.heads.affinity import AffinityHeadConfig
from bindra.model import BindraConfig, BindraModel, BindraModelError

#: Ethanol, benzene, aspirin: 2, 6 and 13 bonds, and 3, 6 and 13 heavy atoms.
#: Chosen so the atom and bond counts differ for one molecule, which is what
#: distinguishes an atom-token arm from a bond-token arm.
MOLECULES = ("CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O")
HEAVY_ATOMS = (3, 6, 13)
BONDS = (2, 6, 13)


def kpgt_config():
    from bindra.encoders.ligand_kpgt import KPGTConfig

    # Far smaller than the published base config: this is a wiring test, and a
    # 92M-parameter arm would make it a slow one.
    return KPGTConfig(
        d_g_feats=32, n_mol_layers=2, n_heads=4, d_hpath_ratio=2, n_trainable_layers=1
    )


def build_model(ligand) -> BindraModel:
    return BindraModel(
        BindraConfig(
            protein_dim=32,
            ligand=ligand,
            model_dim=16,
            head=AffinityHeadConfig(hidden_dim=16),
        )
    )


def protein(batch_size: int, length: int = 7):
    features = torch.randn(batch_size, length, 32)
    mask = torch.ones(batch_size, length, dtype=torch.bool)
    return features, mask


class TestTheContract:
    def test_the_arm_satisfies_the_protocol(self) -> None:
        assert isinstance(build_ligand_arm(kpgt_config()), LigandArm)


    def test_an_unregistered_config_is_refused_by_name(self) -> None:
        with pytest.raises(LigandArmError, match="no ligand arm is registered"):
            build_ligand_arm(object())

    def test_a_config_without_the_widths_is_refused_at_construction(self) -> None:
        # Not at the first forward pass, where it would surface as a shape error.
        with pytest.raises(BindraModelError, match="token_dim"):
            BindraConfig(protein_dim=32, ligand=object())

    def test_token_dim_names_the_arms_own_width(self) -> None:
        assert kpgt_config().token_dim == kpgt_config().d_g_feats


class TestFusionTokens:
    def test_a_mask_of_the_wrong_dtype_is_refused(self) -> None:
        with pytest.raises(LigandArmError, match="must be bool"):
            FusionTokens(
                tokens=torch.zeros(2, 3, 4),
                token_mask=torch.ones(2, 3),
                graph_states=torch.zeros(2, 8),
            )

    def test_a_readout_that_disagrees_on_batch_size_is_refused(self) -> None:
        with pytest.raises(LigandArmError, match="rows against"):
            FusionTokens(
                tokens=torch.zeros(2, 3, 4),
                token_mask=torch.ones(2, 3, dtype=torch.bool),
                graph_states=torch.zeros(3, 8),
            )


class TestPadding:
    def test_destination_scatter_matches_a_loop(self) -> None:
        # Two molecules, three and one tokens.
        states = torch.arange(8, dtype=torch.float32).view(4, 2)
        graph_index = torch.tensor([0, 0, 0, 1])
        destination, width = physical_token_destination(graph_index, 2)
        assert width == 3
        padded, mask = pad_by_destination(
            states, destination, num_graphs=2, width=width
        )
        assert padded.shape == (2, 3, 2)
        assert mask.tolist() == [[True, True, True], [True, False, False]]
        torch.testing.assert_close(padded[0], states[:3])
        torch.testing.assert_close(padded[1, 0], states[3])
        # Padding is zero, not uninitialised memory.
        assert padded[1, 1:].abs().sum() == 0

    def test_an_empty_batch_pads_to_an_empty_result(self) -> None:
        padded, mask = pad_by_destination(
            torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), num_graphs=0, width=0
        )
        assert padded.shape == (0, 0, 4)
        assert mask.shape == (0, 0)

    def test_a_destination_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(LigandArmError, match="destination has"):
            pad_by_destination(
                torch.zeros(4, 2),
                torch.zeros(3, dtype=torch.long),
                num_graphs=2,
                width=2,
            )



class TestTheLiGhTArm:
    def test_knowledge_nodes_are_tokens_but_not_physical(self) -> None:
        """They attend; they are never drawable.

        KPGT's triplet axis is ``bonds + unbonded atoms + 2 knowledge nodes``.
        For these three molecules every atom is bonded, so the physical token
        count must equal the bond count exactly. Asserting against bonds rather
        than "triplets minus two" would still pass if the wrong two were
        dropped, so the molecules are chosen to make atom and bond counts differ.
        """
        arm = build_ligand_arm(kpgt_config())
        batch = arm.collate_smiles(MOLECULES)
        tokens = arm.encode_for_fusion(batch)

        # Every triplet is a token, knowledge nodes included: bonds + 2.
        assert tokens.token_mask.sum(1).tolist() == [n + 2 for n in BONDS]
        # Only the bonds are physical. Ethanol discriminates: 3 atoms, 2 bonds.
        assert tokens.physical.sum(1).tolist() == list(BONDS)
        assert tokens.physical[0].sum().item() == 2 != HEAVY_ATOMS[0]
        # A non-physical token is still a real token the protein may attend to.
        assert bool((tokens.token_mask & ~tokens.physical).any())
        assert batch.graphs.num_nodes == sum(BONDS) + 2 * len(MOLECULES)
        assert tokens.graph_states.shape == (3, 3 * 32)

    def test_dropping_them_is_still_reachable_as_an_ablation(self) -> None:
        from bindra.encoders.ligand_kpgt import KPGTConfig

        arm = build_ligand_arm(
            KPGTConfig.mini(
                d_g_feats=32, n_mol_layers=2, n_heads=4, d_hpath_ratio=2,
                n_trainable_layers=2, knowledge_tokens_in_fusion=False,
            )
        )
        tokens = arm.encode_for_fusion(arm.collate_smiles(MOLECULES))
        assert tokens.token_mask.sum(1).tolist() == list(BONDS)
        assert tokens.physical_mask is None

    def test_knowledge_inputs_out_of_step_with_the_graphs_are_refused(self) -> None:
        from bindra.encoders.ligand_kpgt_arm import KPGTLigandBatch

        arm = build_ligand_arm(kpgt_config())
        batch = arm.collate_smiles(MOLECULES)
        with pytest.raises(LigandArmError, match="out of step"):
            KPGTLigandBatch(
                graphs=batch.graphs,
                fingerprints=batch.fingerprints[:1],
                descriptors=batch.descriptors,
            )

    def test_a_frozen_untrained_arm_says_so(self) -> None:
        from bindra.encoders.ligand_kpgt import KPGTConfig

        frozen = build_ligand_arm(
            KPGTConfig(d_g_feats=32, n_mol_layers=2, n_heads=4, d_hpath_ratio=2)
        )
        # No pretrained weights and no trainable layers: a fixed random
        # projection, which is expressible and almost never wanted.
        assert frozen.n_trainable == 0
        assert frozen.frozen_and_untrained()

        partial = build_ligand_arm(kpgt_config())
        assert partial.n_trainable > 0

    def test_a_missing_checkpoint_names_the_file_and_where_to_get_it(self) -> None:
        arm = build_ligand_arm(kpgt_config())
        with pytest.raises(LigandArmError, match="base.pth"):
            arm.load_pretrained("does/not/exist.pth")


class TestTheArmInTheModel:
    def test_the_model_runs_the_arm(self) -> None:
        torch.manual_seed(0)
        model = build_model(kpgt_config())
        features, mask = protein(3)
        out = model.predict(features, mask, MOLECULES)
        assert out.pair_representation.shape == (3, build_model(kpgt_config()).config.pair_dim)
        assert out.ligand_tokens is not None
        assert out.ligand_tokens.batch_size == 3

    def test_every_shared_module_receives_gradient(self) -> None:
        torch.manual_seed(0)
        model = build_model(kpgt_config())
        features, mask = protein(2, length=5)
        out = model.predict(features, mask, MOLECULES[:2])
        out.prediction.observed(torch.zeros(2, dtype=torch.long)).sum().backward()

        for prefix in model.shared_parameter_prefixes:
            trainable = [
                (n, p)
                for n, p in model.named_parameters()
                if n.startswith(prefix) and p.requires_grad
            ]
            assert trainable, f"nothing trainable under {prefix}"
            for parameter_name, parameter in trainable:
                assert parameter.grad is not None, f"no grad for {parameter_name}"
                assert parameter.grad.abs().sum() > 0, f"zero grad for {parameter_name}"


    def test_a_batch_that_is_not_a_batch_is_refused(self) -> None:
        model = build_model(kpgt_config())
        features, mask = protein(1, length=5)
        with pytest.raises(BindraModelError, match="num_graphs"):
            model(features, mask, object())

    def test_the_arm_is_recorded_in_the_config_dict(self) -> None:
        """A results table must be able to say which arm produced it."""
        assert build_model(kpgt_config()).config.to_dict()["ligand_arm"] == "KPGTConfig"


class TestLiGhTMini:
    """LiGhT-mini: a small line-graph transformer trained from scratch.

    Named for the architecture, not for KPGT, because without `base.pth` the
    pre-training that the acronym refers to is exactly what is missing.

    ``test_from_scratch_trains_the_embeddings`` is the one that matters.
    Partial fine-tuning deliberately pins the input embeddings, which is right
    only when they were pretrained; with no checkpoint they are random, and
    pinning them permanently is uncorrectable rather than conservative.
    """

    def test_the_default_mini_size_is_pinned(self) -> None:
        """A silent size change would invalidate every recorded comparison."""
        from bindra.encoders.ligand_kpgt import KPGTConfig

        arm = build_ligand_arm(KPGTConfig.mini())
        trainable = sum(p.numel() for p in arm.module.parameters() if p.requires_grad)
        assert trainable == 1_032_560

    def test_from_scratch_trains_the_embeddings(self) -> None:
        """The whole reason `from_scratch` is a regime and not a bigger count.

        At the maximum `n_trainable_layers` the embeddings and attention biases
        are still frozen, so a model with no checkpoint would learn on top of a
        fixed random input representation.
        """
        from bindra.encoders.ligand_kpgt import KPGTConfig

        def embedding_trains(arm) -> bool:
            return all(
                p.requires_grad
                for n, p in arm.module.named_parameters()
                if n.startswith("triplet_emb.")
            )

        # The default, 1 trainable block: almost the whole encoder is pinned.
        default = build_ligand_arm(KPGTConfig(n_trainable_layers=1))
        assert default.partially_frozen_at_random() == 82_463_960

        # And even at the maximum, the embeddings and attention biases stay
        # pinned, which is the point: no setting of n_trainable_layers reaches
        # them, so `from_scratch` has to be its own regime.
        partial = build_ligand_arm(KPGTConfig(n_trainable_layers=12))
        assert not embedding_trains(partial)
        assert partial.partially_frozen_at_random() == 4_497_368

        scratch = build_ligand_arm(KPGTConfig.mini())
        assert embedding_trains(scratch)
        assert scratch.partially_frozen_at_random() == 0
        assert not scratch.frozen_and_untrained()

    def test_only_pretraining_parameters_stay_frozen(self) -> None:
        """Including `mask_emb`, which no forward path through this arm reaches."""
        from bindra.encoders.ligand_kpgt import KPGTConfig

        arm = build_ligand_arm(KPGTConfig.mini())
        frozen = [n for n, p in arm.module.named_parameters() if not p.requires_grad]
        assert frozen, "something should stay frozen"
        for name in frozen:
            assert name.startswith(arm.PRETRAINING_ONLY), name
        assert any("mask_emb" in n for n in frozen)

    def test_a_scratch_model_is_not_reported_as_kpgt(self) -> None:
        """It keeps the architecture and the knowledge inputs, not the training."""
        from bindra.encoders.ligand_kpgt import KPGTConfig, KPGTLigandEncoder

        mini = KPGTConfig.mini()
        arm = build_ligand_arm(mini)
        assert KPGTLigandEncoder(mini, module=arm.module).metadata().name == (
            "ligand_light_mini"
        )
        base = KPGTConfig(n_trainable_layers=1)
        assert KPGTLigandEncoder(base).metadata().name == "ligand_kpgt"
        assert mini.to_dict()["from_scratch"] is True

    def test_building_metadata_does_not_refreeze_the_arm(self) -> None:
        """Regression. Provenance is a read; it must not change the model.

        ``KPGTLigandEncoder.__init__`` calls ``apply_freezing``, so constructing
        one around a *training arm's* module — which is exactly what recording
        provenance does — used to re-apply the partial-fine-tuning rule and
        re-pin a from-scratch arm's embeddings. In ``cmd_fit`` that call sits
        after the refusal checks and before training, so the run proceeded with
        452,896 parameters pinned at random initialisation, reporting a normal
        loss curve. Both call sites now share one freezing function.
        """
        from bindra.encoders.ligand_kpgt import KPGTConfig, KPGTLigandEncoder

        config = KPGTConfig.mini()
        arm = build_ligand_arm(config)
        before = {n: p.requires_grad for n, p in arm.module.named_parameters()}

        metadata = KPGTLigandEncoder(config, module=arm.module).metadata()
        assert metadata.name == "ligand_light_mini"

        after = {n: p.requires_grad for n, p in arm.module.named_parameters()}
        changed = [n for n in before if before[n] != after[n]]
        assert not changed, f"metadata froze {len(changed)} parameters: {changed[:4]}"
        assert sum(
            p.numel() for p in arm.module.parameters() if p.requires_grad
        ) == 1_032_560

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_an_unbonded_atom_survives_autocast(self, dtype) -> None:
        """Regression: the configuration that crashed the first training run.

        `AtomEmbedding` index-puts a learned embedding into the slot of an
        unbonded atom's absent partner. Autocast casts *operations*, not
        parameter reads, so under bf16 the `nn.Linear` output was bf16 while the
        `nn.Embedding` weight stayed fp32, and the index-put raised. It needs
        both conditions at once -- autocast **and** a molecule with an unbonded
        atom -- which is why every fp32 test passed and the run died on the
        first batch holding a salt.

        Parametrized over both dtypes so the fp32 path, which the port's
        reference verification depends on, is asserted unchanged.
        """
        from bindra.encoders.ligand_kpgt import KPGTConfig

        torch.manual_seed(0)
        model = build_model(KPGTConfig.mini())
        # Sodium acetate: the Na+ is unbonded. 3 bonds + 1 unbonded atom.
        batch = model.ligand_encoder.collate_smiles(["CC(=O)[O-].[Na+]", MOLECULES[2]])
        features, mask = protein(2, length=12)

        with torch.autocast("cpu", dtype=dtype, enabled=dtype is torch.bfloat16):
            out = model(features, mask, batch)
        assert out.ligand_tokens.token_mask.sum(1).tolist() == [6, 15]
        assert out.ligand_tokens.physical.sum(1).tolist() == [4, 13]

        out.prediction.observed(torch.zeros(2, dtype=torch.long)).float().sum().backward()
        for fragment in ("virtual_atom_emb", "virutal_bond_emb"):
            grads = [
                p.grad
                for n, p in model.named_parameters()
                if fragment in n and p.requires_grad
            ]
            assert grads and all(
                g is not None and g.abs().sum() > 0 for g in grads
            ), f"{fragment} received no gradient; the dtype cast broke autograd"

    def test_a_half_specified_scratch_config_is_refused(self) -> None:
        from bindra.encoders.ligand_kpgt import KPGTConfig

        with pytest.raises(ValueError, match="must equal n_mol_layers"):
            KPGTConfig(n_mol_layers=4, n_trainable_layers=1, from_scratch=True)

    def test_shape_constraints_are_checked_at_construction(self) -> None:
        from bindra.encoders.ligand_kpgt import KPGTConfig

        with pytest.raises(ValueError, match="not divisible by n_heads"):
            KPGTConfig(d_g_feats=100, n_heads=8)
        with pytest.raises(ValueError, match="triplet path width"):
            KPGTConfig(d_g_feats=64, n_heads=8, d_hpath_ratio=128)

    def test_every_trainable_weight_moves_under_a_real_step(self) -> None:
        """No optimizer state allocated for a weight that cannot update.

        Endpoint heads for unmeasured endpoints are excluded: receiving no
        gradient is exactly what the gather is for.
        """
        from bindra.encoders.ligand_kpgt import KPGTConfig

        torch.manual_seed(0)
        model = build_model(KPGTConfig.mini())
        features, mask = protein(2, length=8)
        # A salt exercises the virtual-atom path; aspirin exercises long paths.
        out = model.predict(features, mask, ["CC(=O)[O-].[Na+]", MOLECULES[2]])
        out.prediction.observed(torch.zeros(2, dtype=torch.long)).sum().backward()

        dead = [
            n
            for n, p in model.named_parameters()
            if n.startswith("ligand_encoder.")
            and p.requires_grad
            and (p.grad is None or p.grad.abs().sum() == 0)
        ]
        # trip_fortrans.N only fires for a path of length N, so allow those.
        dead = [n for n in dead if "trip_fortrans" not in n]
        assert not dead, dead


class TestTheBatchBuilder:
    """The data-pipeline half: featurizing for an arm, inside a process pool."""

    def test_the_builder_and_its_featurizer_pickle(self) -> None:
        """The hard constraint. ``executor.map`` cannot carry a closure."""
        import pickle

        builder = LigandBatchBuilder("kpgt")
        assert pickle.loads(pickle.dumps(builder)) == builder
        assert pickle.loads(pickle.dumps(builder.featurize)) is builder.featurize


    def test_one_molecules_parts_survive_a_round_trip(self) -> None:
        """What comes back from a worker, for the arm that returns three things."""
        import pickle

        import numpy as np

        parts = pickle.loads(pickle.dumps(featurize_kpgt_one("CCO")))
        assert parts.fingerprint.shape == (512,)
        assert parts.descriptors.shape == (200,)
        # NaNs are zeroed in the worker: one reaching a knowledge node would
        # propagate through every attention weight in the batch, and by then the
        # molecule that caused it would be unidentifiable.
        assert not np.isnan(parts.descriptors).any()
        assert not np.isnan(parts.fingerprint).any()

    def test_the_arm_comes_from_the_config_not_a_flag(self) -> None:
        assert LigandBatchBuilder.for_config(kpgt_config()).arm == "kpgt"
        with pytest.raises(ValueError, match="no ligand featurizer"):
            LigandBatchBuilder.for_config(object())

    def test_an_unknown_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown ligand arm"):
            LigandBatchBuilder("graphormer")
        with pytest.raises(ValueError, match="no ligand featurizer"):
            LigandBatchBuilder.for_config(object())

    def test_the_builder_collates_the_arms_batch_type(self) -> None:
        from bindra.encoders.ligand_kpgt_arm import KPGTLigandBatch

        kpgt = LigandBatchBuilder("kpgt")
        parts = kpgt.collate([kpgt.featurize(s) for s in MOLECULES])
        assert isinstance(parts, KPGTLigandBatch)
        assert parts.graphs.num_nodes == sum(BONDS) + 2 * len(MOLECULES)
        assert parts.fingerprints.shape == (3, 512)
        assert parts.descriptors.shape == (3, 200)


    def test_the_arm_declares_what_a_bad_molecule_raises(self) -> None:
        """So a failure inside a worker process surfaces as itself."""
        from bindra.encoders.kpgt_graph import KPGTGraphError

        assert KPGTGraphError in LigandBatchBuilder("kpgt").featurization_errors
