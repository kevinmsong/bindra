"""KPGT ported off DGL, verified against the reference implementation.

A port of a published architecture that is not checked against its reference is
a guess. These tests run the real thing: tensors dumped from
``lihan97/KPGT``'s own code executing in the CPU-only DGL environment
(``C:/bindra/envs/kpgt``), compared stage by stage against the pure-PyTorch port.

Staged rather than end-to-end on purpose. A single output mismatch tells you
nothing about where it went wrong, and this port has eleven stages where it
could.

Regenerating the reference (needed only if the port or the reference changes)::

    /c/bindra/envs/kpgt/Scripts/python.exe C:/bindra/oracle/make_reference.py \\
        C:/bindra/oracle/reference.npz small
    /c/bindra/envs/kpgt/Scripts/python.exe C:/bindra/oracle/make_reference.py \\
        C:/bindra/oracle/reference_base.npz base

The base artifact is 327 MB, so neither is committed; tests skip when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bindra.encoders.kpgt_graph import (
    KPGTGraphError,
    atom_features,
    bond_features,
    collate_kpgt,
    featurize_kpgt,
    one_hot_encoding,
    rdk_fingerprint,
)
from bindra.encoders.kpgt_light import BASE_CONFIG, KPGTBatch, KPGTPortError, LiGhTPredictor

ORACLE = Path("C:/bindra/oracle")
SMALL = ORACLE / "reference.npz"
BASE = ORACLE / "reference_base.npz"

SMALL_CONFIG = dict(
    d_node_feats=137, d_edge_feats=14, d_g_feats=48, d_fp_feats=512, d_md_feats=200,
    d_hpath_ratio=4, n_mol_layers=2, path_length=5, n_heads=4, n_ffn_dense_layers=2,
    input_drop=0.0, feat_drop=0.0, attn_drop=0.0, n_node_types=1,
)

#: Float32 reassociation only. The port changes the order of a sum (segment ops
#: against DGL's kernels), so exact equality is not achievable and not expected;
#: anything above this would be a logic difference.
TOLERANCE = 2e-5
BASE_TOLERANCE = 1e-4


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"reference not generated: {path} (see this module's docstring)")
    return np.load(path, allow_pickle=True)


def _state_dict(reference) -> dict[str, torch.Tensor]:
    return {
        key[len("param::") :]: torch.from_numpy(reference[key])
        for key in reference.files
        if key.startswith("param::")
    }


def _batch_from_reference(reference) -> KPGTBatch:
    """Build a batch from the reference's own graph, isolating the model."""
    counts = reference["num_nodes_per_graph"]
    return KPGTBatch(
        begin_end=torch.from_numpy(reference["begin_end"]),
        edge_feats=torch.from_numpy(reference["edge_feats"]),
        vavn=torch.from_numpy(reference["vavn"]).long(),
        edge_index=torch.from_numpy(
            np.stack([reference["edge_src"], reference["edge_dst"]])
        ).long(),
        path=torch.from_numpy(reference["path"]).long(),
        virtual_path=torch.from_numpy(reference["vp"]).bool(),
        self_loop=torch.from_numpy(reference["sl"]).bool(),
        graph_index=torch.from_numpy(
            np.repeat(np.arange(len(counts)), counts)
        ).long(),
        num_graphs=int(len(counts)),
    )


@pytest.fixture(scope="module")
def small():
    return _load(SMALL)


@pytest.fixture(scope="module")
def small_model(small):
    model = LiGhTPredictor(**SMALL_CONFIG)
    model.load_state_dict(_state_dict(small), strict=True)
    model.eval()
    return model


class TestOneHotEncoding:
    """dgllife's convention, which is not the obvious one."""

    def test_unknown_lands_in_a_trailing_slot_not_an_all_zero_block(self):
        assert one_hot_encoding("S", ["C", "O"], True) == [False, False, True]

    def test_without_encode_unknown_the_block_can_be_all_zero(self):
        assert one_hot_encoding("S", ["C", "O"], False) == [False, False]

    def test_the_allowable_set_is_not_mutated_for_the_caller(self):
        allowable = ["C", "O"]
        one_hot_encoding("S", allowable, True)
        assert allowable == ["C", "O"]


class TestFeatureWidths:
    def test_atom_features_are_137_wide(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        assert len(atom_features(mol.GetAtomWithIdx(0))) == 137

    def test_bond_features_are_14_wide(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        assert len(bond_features(mol.GetBondWithIdx(0))) == 14

    def test_the_fingerprint_is_512_bits(self):
        fingerprint = rdk_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        assert fingerprint.shape == (512,)
        assert set(np.unique(fingerprint)) <= {0.0, 1.0}


class TestFeaturizerAgainstReference:
    """The triplet graph, compared element by element with the reference."""

    def test_node_counts_match(self, small):
        graphs = [featurize_kpgt(s) for s in list(small["smiles"])]
        assert [g.num_nodes for g in graphs] == small["num_nodes_per_graph"].tolist()

    def test_edge_counts_match(self, small):
        graphs = [featurize_kpgt(s) for s in list(small["smiles"])]
        assert [g.num_edges for g in graphs] == small["num_edges_per_graph"].tolist()

    def test_atom_and_bond_features_are_bit_identical(self, small):
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        assert np.array_equal(batch.begin_end.numpy(), small["begin_end"])
        assert np.array_equal(batch.edge_feats.numpy(), small["edge_feats"])

    def test_node_role_indicators_are_identical(self, small):
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        assert np.array_equal(batch.vavn.numpy(), small["vavn"])

    def test_the_edge_list_is_identical_in_order_not_just_as_a_set(self, small):
        """Order matters: edge i's path is path[i], so a permutation decouples them."""
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        assert np.array_equal(batch.edge_index[0].numpy(), small["edge_src"])
        assert np.array_equal(batch.edge_index[1].numpy(), small["edge_dst"])

    def test_edge_kind_flags_are_identical(self, small):
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        assert np.array_equal(batch.virtual_path.numpy(), small["vp"])
        assert np.array_equal(batch.self_loop.numpy(), small["sl"])

    def test_paths_agree_on_every_real_index_and_every_absent_slot(self, small):
        """Deliberate deviation, and the only one in the featurizer.

        The reference's ``preprocess_batch_light`` adds each graph's node offset
        to *every* path entry, including the -1e6 "absent step" sentinels, which
        therefore drift to -999996, -999981 and so on. It survives only because
        the later clamp tests ``< -99``; a batch with more than ~1M triplet nodes
        would push a drifted sentinel past that and it would be read as a real
        node index. This port shifts only non-negative entries, so the sentinel
        stays exactly -1e6.

        Equivalence is therefore asserted as: the same slots are absent, and every
        real index is identical.
        """
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        mine, theirs = batch.path.numpy(), small["path"]
        assert np.array_equal(mine < -99, theirs < -99), "absent slots differ"
        assert np.array_equal(
            np.where(mine < -99, -1, mine), np.where(theirs < -99, -1, theirs)
        ), "real path indices differ"

    def test_the_sentinel_does_not_drift_in_this_port(self, small):
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        path = batch.path.numpy()
        assert set(np.unique(path[path < -99]).tolist()) == {-1_000_000}


class TestAtomMapping:
    """The token-to-atom mapping iBAM needs, which KPGT provides natively."""

    def test_a_bond_triplet_maps_to_exactly_two_atoms(self):
        graph = featurize_kpgt("CCO")
        bond_triplets = [
            atoms for atoms, role in zip(graph.triplet_to_atoms, graph.vavn) if role == 0
        ]
        assert bond_triplets and all(len(atoms) == 2 for atoms in bond_triplets)

    def test_a_knowledge_node_maps_to_no_atom(self):
        graph = featurize_kpgt("CCO")
        knowledge = [
            atoms for atoms, role in zip(graph.triplet_to_atoms, graph.vavn) if role >= 1
        ]
        assert len(knowledge) == 2
        assert all(atoms == () for atoms in knowledge)

    def test_an_unbonded_atom_maps_to_one_atom(self):
        graph = featurize_kpgt("C")
        lone = [
            atoms for atoms, role in zip(graph.triplet_to_atoms, graph.vavn) if role == -1
        ]
        assert lone == [(0,)]

    def test_mapped_indices_are_in_the_original_molecule_not_the_renumbered_one(self):
        """The reference renumbers atoms canonically before featurizing.

        Reporting post-renumbering indices would put every iBAM attribution on
        the wrong atom of the depiction the user supplied.
        """
        from rdkit import Chem

        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        graph = featurize_kpgt(smiles)
        mol = Chem.MolFromSmiles(smiles)
        referenced = {a for atoms in graph.triplet_to_atoms for a in atoms}
        assert referenced == set(range(mol.GetNumAtoms()))
        # The renumbering is a real permutation, so this is not vacuous.
        assert list(graph.canonical_order) != sorted(graph.canonical_order) or (
            mol.GetNumAtoms() == 1
        )

    def test_every_bond_of_the_molecule_appears_exactly_once(self):
        from rdkit import Chem

        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        graph = featurize_kpgt(smiles)
        mol = Chem.MolFromSmiles(smiles)
        bonds = {
            frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()
        }
        mapped = [
            frozenset(atoms)
            for atoms, role in zip(graph.triplet_to_atoms, graph.vavn)
            if role == 0
        ]
        assert len(mapped) == len(bonds)
        assert set(mapped) == bonds


class TestModelAgainstReference:
    """Every stage of the forward pass, against the executed reference."""

    def _staged(self, small, small_model):
        batch = _batch_from_reference(small)
        fp = torch.from_numpy(small["fp"])
        md = torch.from_numpy(small["md"])
        with torch.no_grad():
            indicators = batch.vavn
            node_h = small_model.node_emb(batch.begin_end, indicators)
            edge_h = small_model.edge_emb(batch.edge_feats, indicators)
            triplet_h0 = small_model.triplet_emb(node_h, edge_h, fp, md, indicators)
            light = small_model.model
            dist_h = light._featurize_path(batch, batch.path)
            path_h = light._init_path(triplet_h0, batch.path)
            dist_attn = light.dist_attn_layer(dist_h)
            path_attn = light.path_attn_layer(path_h)
            layers = []
            states = triplet_h0
            for layer in light.mol_T_layers:
                states = layer(batch, states, dist_attn, path_attn)
                layers.append(states)
            readout = small_model.readout(batch, states)
        return {
            "node_h": node_h, "edge_h": edge_h, "triplet_h0": triplet_h0,
            "dist_h": dist_h, "path_h": path_h,
            "dist_attn": dist_attn, "path_attn": path_attn,
            "triplet_h_layer0": layers[0], "triplet_h_layer1": layers[1],
            "g_feats": readout,
        }

    def test_the_published_state_dict_loads_strictly(self, small):
        """No missing and no unexpected keys: the checkpoint will load unchanged."""
        model = LiGhTPredictor(**SMALL_CONFIG)
        model.load_state_dict(_state_dict(small), strict=True)

    @pytest.mark.parametrize(
        "stage",
        ["node_h", "edge_h", "triplet_h0", "dist_h", "path_h", "dist_attn",
         "path_attn", "triplet_h_layer0", "triplet_h_layer1", "g_feats"],
    )
    def test_stage_matches_the_reference(self, small, small_model, stage):
        produced = self._staged(small, small_model)[stage].numpy()
        expected = small[stage]
        assert produced.shape == expected.shape
        deviation = np.abs(produced - expected).max()
        assert deviation < TOLERANCE, f"{stage} deviates by {deviation:.3e}"

    def test_the_readout_is_three_blocks_wide(self, small, small_model):
        readout = self._staged(small, small_model)["g_feats"]
        assert readout.shape == (8, 3 * SMALL_CONFIG["d_g_feats"])

    def test_end_to_end_through_my_own_featurizer(self, small, small_model):
        """The whole pipeline: port's featurizer into port's model, vs the reference.

        Distinct from the staged tests above, which feed the reference's own graph
        and so isolate the model. This one can only pass if both halves are right.
        """
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        with torch.no_grad():
            _, g_feats = small_model(
                batch, torch.from_numpy(small["fp"]), torch.from_numpy(small["md"])
            )
        deviation = np.abs(g_feats.numpy() - small["g_feats"]).max()
        assert deviation < TOLERANCE, f"end-to-end deviates by {deviation:.3e}"

    def test_triplet_states_have_one_row_per_node(self, small, small_model):
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        with torch.no_grad():
            states, _ = small_model(
                batch, torch.from_numpy(small["fp"]), torch.from_numpy(small["md"])
            )
        assert states.shape == (batch.num_nodes, SMALL_CONFIG["d_g_feats"])


class TestAtPublishedScale:
    """The 768-dim, 12-layer, 12-head configuration the checkpoint actually uses."""

    @pytest.fixture(scope="class")
    @classmethod
    def base(cls):
        return _load(BASE)

    def _model(self, base):
        config = dict(BASE_CONFIG)
        config.update(
            d_fp_feats=512, d_md_feats=200, n_node_types=1,
            input_drop=0.0, feat_drop=0.0, attn_drop=0.0,
        )
        model = LiGhTPredictor(**config)
        model.load_state_dict(_state_dict(base), strict=True)
        model.eval()
        return model

    def test_the_base_state_dict_loads_strictly(self, base):
        model = self._model(base)
        assert sum(p.numel() for p in model.parameters()) > 90_000_000

    def test_end_to_end_matches_at_base_scale(self, base):
        model = self._model(base)
        batch = collate_kpgt([featurize_kpgt(s) for s in list(base["smiles"])])
        with torch.no_grad():
            _, g_feats = model(
                batch, torch.from_numpy(base["fp"]), torch.from_numpy(base["md"])
            )
        deviation = np.abs(g_feats.numpy() - base["g_feats"]).max()
        assert deviation < BASE_TOLERANCE, f"deviates by {deviation:.3e}"

    def test_the_molecule_representation_is_2304_wide(self, base):
        model = self._model(base)
        batch = collate_kpgt([featurize_kpgt(s) for s in list(base["smiles"])])
        with torch.no_grad():
            _, g_feats = model(
                batch, torch.from_numpy(base["fp"]), torch.from_numpy(base["md"])
            )
        assert g_feats.shape == (8, 2304)


class TestBatching:
    def test_collating_does_not_change_any_molecule(self, small, small_model):
        """A missing offset shift would point every path at the first molecule."""
        molecules = list(small["smiles"])
        together = collate_kpgt([featurize_kpgt(s) for s in molecules])
        fp, md = torch.from_numpy(small["fp"]), torch.from_numpy(small["md"])
        with torch.no_grad():
            _, batched = small_model(together, fp, md)
            for index, smiles in enumerate(molecules):
                alone = collate_kpgt([featurize_kpgt(smiles)])
                _, single = small_model(alone, fp[index : index + 1], md[index : index + 1])
                assert torch.allclose(batched[index], single[0], atol=1e-5), smiles

    def test_an_empty_batch_is_refused(self):
        with pytest.raises(KPGTGraphError, match="empty batch"):
            collate_kpgt([])

    def test_unshifted_edges_are_caught_by_the_contract(self, small):
        graphs = [featurize_kpgt(s) for s in list(small["smiles"])[:2]]
        batch = collate_kpgt(graphs)
        with pytest.raises(KPGTPortError, match="offsets were not applied"):
            KPGTBatch(
                begin_end=batch.begin_end,
                edge_feats=batch.edge_feats,
                vavn=batch.vavn,
                edge_index=batch.edge_index + batch.num_nodes,
                path=batch.path,
                virtual_path=batch.virtual_path,
                self_loop=batch.self_loop,
                graph_index=batch.graph_index,
                num_graphs=batch.num_graphs,
            )


class TestFailureModes:
    def test_unparseable_smiles_raises(self):
        with pytest.raises(KPGTGraphError, match="could not parse"):
            featurize_kpgt("not a molecule")

    def test_a_bondless_molecule_still_builds(self):
        graph = featurize_kpgt("C")
        assert graph.num_nodes == 3  # one virtual-atom triplet + two knowledge nodes
        assert graph.num_edges > 0  # self-loops and knowledge-node edges

    def test_gradients_reach_the_parameters(self, small, small_model):
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        _, g_feats = small_model(
            batch, torch.from_numpy(small["fp"]), torch.from_numpy(small["md"])
        )
        g_feats.sum().backward()
        missing = [
            name
            for name, parameter in small_model.model.named_parameters()
            if parameter.grad is None or parameter.grad.abs().sum() == 0
        ]
        small_model.zero_grad(set_to_none=True)
        # LiGhT.mask_emb is dead weight in the reference as well: it is declared
        # and sits in the checkpoint, but no forward path reads it. Only
        # LiGhTPredictor.mask_emb is used, and only by the masked-node
        # pretraining objective. It must stay for a strict state-dict load, so
        # receiving no gradient during feature extraction is correct.
        assert missing == ["mask_emb.weight"], f"unexpected missing gradients: {missing}"


@pytest.mark.gpu
class TestOnTheBigGpu:
    """KPGT on the RTX 5060 Ti: the thing DGL cannot do at any version."""

    def _device(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from bindra.runtime.device import list_cuda_devices

        for gpu in list_cuda_devices():
            if gpu.capability[0] >= 12:
                return f"cuda:{gpu.index}"
        pytest.skip("no sm_120 device present")

    def test_it_runs_on_sm_120_and_agrees_with_the_cpu(self, small, small_model):
        device = self._device()
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        fp, md = torch.from_numpy(small["fp"]), torch.from_numpy(small["md"])
        with torch.no_grad():
            _, on_cpu = small_model(batch, fp, md)
            moved = small_model.to(device)
            _, on_gpu = moved(batch.to(device), fp.to(device), md.to(device))
            small_model.to("cpu")
        assert torch.allclose(on_cpu, on_gpu.cpu(), atol=1e-4)

    def test_it_still_matches_the_dgl_reference_on_the_gpu(self, small, small_model):
        """The claim that matters: same numbers, on hardware DGL cannot reach."""
        device = self._device()
        batch = collate_kpgt([featurize_kpgt(s) for s in list(small["smiles"])])
        with torch.no_grad():
            moved = small_model.to(device)
            _, g_feats = moved(
                batch.to(device),
                torch.from_numpy(small["fp"]).to(device),
                torch.from_numpy(small["md"]).to(device),
            )
            small_model.to("cpu")
        deviation = np.abs(g_feats.cpu().numpy() - small["g_feats"]).max()
        assert deviation < 1e-4, f"deviates by {deviation:.3e}"
