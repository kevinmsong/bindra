"""The assembled model, including two tests the specification names explicitly.

    Tiny-data overfit and nonzero shared-module gradients for each enabled task.

``test_the_model_overfits_eight_pairs``
    The only test that proves the whole stack is actually connected and learning.
    Every shape can be right while a ``detach()``, a zeroed mask or a mis-ordered
    gather stops learning entirely; the loss would still go down a little, because
    the heads can fit the label mean. Driving eight pairs to near-zero error
    requires the ligand signal to reach the output.
``test_every_task_reaches_every_shared_module``
    The specification's nonzero-shared-gradient requirement, checked per task
    against the prefixes the model itself declares.

Also here, because they are cheap and catch real corruption:

``test_padding_does_not_change_a_prediction``
    End-to-end version of the fusion test. Protein lengths span two orders of
    magnitude, so this is the difference between reproducible and not.
``test_saving_and_reloading_reproduces_inference``
    Required by the specification's own list.
``test_atom_renumbering_does_not_change_a_prediction``
    Also required: predictions must not depend on RDKit's atom ordering.
"""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from bindra.encoders.ligand_kpgt import KPGTConfig
from bindra.encoders.ligand_kpgt_arm import KPGTLigandBatch
from bindra.features.types import EncoderMetadata
from bindra.fusion.adapters import assert_shared, attribute_gradients
from bindra.heads.affinity import AffinityHeadConfig, ENDPOINT_INDEX
from bindra.model import (
    BindraConfig,
    BindraModel,
    BindraModelError,
)
from bindra.training.losses.affinity import (
    AffinityTargets,
    affinity_loss,
)

PROTEIN_DIM = 48
NAN = float("nan")

MOLECULES = (
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CCO",
    "c1ccc2ccccc2c1",
    "C",
    "N[C@@H](C)C(=O)O",
    "CS(=O)(=O)N",
    "FC(F)(F)c1ccccc1",
)


def _config(**kwargs) -> BindraConfig:
    defaults = dict(
        protein_dim=PROTEIN_DIM,
        ligand=KPGTConfig.mini(d_g_feats=32, n_mol_layers=2, n_trainable_layers=2, n_heads=4, d_hpath_ratio=2),
        model_dim=32,
        cross_attention_layers=1,
        cross_attention_heads=4,
        dropout=0.0,
        head=AffinityHeadConfig(hidden_dim=32, dropout=0.0, num_layers=2),
    )
    defaults.update(kwargs)
    return BindraConfig(**defaults)


@pytest.fixture
def model() -> BindraModel:
    torch.manual_seed(20261001)
    built = BindraModel(_config())
    built.eval()
    return built


def _proteins(count: int, length: int = 9, seed: int = 0):
    """Distinct per-pair protein features with a full mask."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, length, PROTEIN_DIM, generator=generator)
    return features, torch.ones(count, length, dtype=torch.bool)



class TestConstruction:
    def test_the_protein_width_comes_from_the_cache_metadata(self):
        """Nothing hardcodes 1152."""
        metadata = EncoderMetadata(
            name="esmc", checkpoint="x", output_dim=1152, precision="float32",
            feature_schema_version="v1",
        )
        built = BindraModel.from_protein_metadata(
            metadata, model_dim=32, cross_attention_heads=4, ligand=KPGTConfig.mini(d_g_feats=32, n_mol_layers=2, n_trainable_layers=2, n_heads=4, d_hpath_ratio=2)
        )
        assert built.config.protein_dim == 1152

    def test_the_pair_representation_counts_its_blocks(self):
        config = _config(model_dim=64)
        assert config.pair_dim == (3 + config.adapter_config().graph_blocks) * 64

    def test_the_adapter_width_cannot_disagree_with_the_ligand_encoder(self):
        config = _config(ligand=KPGTConfig.mini(d_g_feats=64, n_mol_layers=1, n_trainable_layers=1, n_heads=8, d_hpath_ratio=2))
        assert config.adapter_config().ligand_dim == 64
        assert config.adapter_config().ligand_graph_dim == 3 * 64


class TestForward:
    def test_shapes_follow_the_config(self, model):
        features, mask = _proteins(len(MOLECULES))
        out = model.predict(features, mask, list(MOLECULES))
        assert out.prediction.values.shape == (len(MOLECULES), 3)
        assert out.pair_representation.shape == (len(MOLECULES), model.config.pair_dim)
        assert out.fusion.num_layers == 1

    def test_attention_is_retained_for_ibam(self, model):
        features, mask = _proteins(2)
        batch = KPGTLigandBatch.from_smiles(list(MOLECULES[:2]))
        out = model(features, mask, batch, return_ligand_attention=True,
                    need_attention=True)
        # LiGhT does not expose per-layer attention through this port.
        assert out.ligand_attention == ()
        assert len(out.fusion.p2l_attention) == model.config.cross_attention_layers
        assert len(out.fusion.p2l_attention) == 1
        # Unaveraged over heads.
        assert out.fusion.p2l_attention[0].shape[1] == model.config.cross_attention_heads

    def test_a_mismatched_protein_width_is_refused_with_a_useful_message(self, model):
        with pytest.raises(BindraModelError, match="different encoder"):
            model.predict(
                torch.randn(2, 9, PROTEIN_DIM + 1), torch.ones(2, 9, dtype=torch.bool),
                list(MOLECULES[:2]),
            )

    def test_a_pair_count_mismatch_is_refused(self, model):
        """A silent mismatch here would train on wrong protein-ligand pairs."""
        features, mask = _proteins(3)
        with pytest.raises(BindraModelError, match="batch mismatch"):
            model.predict(features, mask, list(MOLECULES[:2]))

    def test_a_float_protein_mask_is_refused(self, model):
        with pytest.raises(BindraModelError, match="True = real residue"):
            model(torch.randn(2, 9, PROTEIN_DIM), torch.ones(2, 9),
                  KPGTLigandBatch.from_smiles(list(MOLECULES[:2])))

    def test_output_is_finite_for_a_single_atom_molecule(self, model):
        features, mask = _proteins(1)
        out = model.predict(features, mask, ["C"])
        assert torch.isfinite(out.prediction.values).all()

    def test_padding_does_not_change_a_prediction(self, model):
        """End to end: a result must not depend on batch composition."""
        features, _ = _proteins(1, length=9, seed=3)
        tight = model.predict(features, torch.ones(1, 9, dtype=torch.bool), ["CCO"])

        padded = torch.cat((features, torch.randn(1, 15, PROTEIN_DIM)), dim=1)
        mask = torch.zeros(1, 24, dtype=torch.bool)
        mask[:, :9] = True
        loose = model.predict(padded, mask, ["CCO"])
        assert torch.allclose(tight.prediction.values, loose.prediction.values, atol=1e-5)

    def test_batching_does_not_change_a_prediction(self, model):
        features, mask = _proteins(len(MOLECULES), seed=11)
        together = model.predict(features, mask, list(MOLECULES))
        for index, smiles in enumerate(MOLECULES):
            alone = model.predict(
                features[index : index + 1], mask[index : index + 1], [smiles]
            )
            assert torch.allclose(
                together.prediction.values[index], alone.prediction.values[0], atol=1e-4
            ), smiles

    def test_atom_renumbering_does_not_change_a_prediction(self, model):
        """Required by the specification; a prediction is not a function of input order."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        order = [3, 0, 7, 1, 11, 2, 9, 4, 12, 5, 10, 6, 8]
        features, mask = _proteins(1, seed=5)
        original = model(
            features, mask, KPGTLigandBatch.from_smiles([Chem.MolToSmiles(mol)])
        ).prediction.values
        renumbered = model(
            features, mask,
            KPGTLigandBatch.from_smiles([Chem.MolToSmiles(Chem.RenumberAtoms(mol, order))]),
        ).prediction.values
        assert torch.allclose(original, renumbered, atol=1e-5)


class TestSharedGradients:
    """The specification's nonzero-shared-module gradients, for each enabled task."""

    def _tasks(self, model):
        features, mask = _proteins(len(MOLECULES), seed=7)
        out = model.predict(features, mask, list(MOLECULES))
        targets = AffinityTargets(
            endpoint_index=torch.tensor([ENDPOINT_INDEX["Ki"]] * len(MOLECULES)),
            p_value=torch.full((len(MOLECULES),), 8.0),
            is_exact=torch.ones(len(MOLECULES), dtype=torch.bool),
            lower_bound=torch.full((len(MOLECULES),), NAN),
            upper_bound=torch.full((len(MOLECULES),), NAN),
            is_primary=torch.ones(len(MOLECULES), dtype=torch.bool),
        )
        return {
            "affinity": affinity_loss(out.prediction, targets).total,
            # Stand-ins for the retrieval and conditioning objectives: what is
            # under test is that they reach the shared modules, not their form.
            "retrieval": out.pair_representation.pow(2).mean(),
            "conditioning": out.fusion.ligand.abs().mean(),
        }

    def test_every_task_reaches_every_shared_module(self, model):
        model.train()
        attribution = attribute_gradients(model, self._tasks(model))
        for prefix in model.shared_parameter_prefixes:
            assert_shared(
                attribution, prefix=prefix, losses=list(attribution)
            )

    def test_the_affinity_task_reaches_the_heads(self, model):
        model.train()
        attribution = attribute_gradients(model, self._tasks(model))
        assert any(name.startswith("heads.") for name in attribution["affinity"])

    def test_the_auxiliary_tasks_do_not_reach_the_heads(self, model):
        """They are defined on the representation, not on the prediction."""
        model.train()
        attribution = attribute_gradients(model, self._tasks(model))
        assert not any(name.startswith("heads.") for name in attribution["retrieval"])

    def test_no_shared_module_is_left_without_gradient_from_the_affinity_loss(self, model):
        """A detach anywhere in the chain shows up here and almost nowhere else."""
        model.train()
        attribution = attribute_gradients(
            model, {"affinity": self._tasks(model)["affinity"]}
        )
        touched = set(attribution["affinity"])
        expected = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.startswith(model.shared_parameter_prefixes)
        }
        missing = sorted(expected - touched)
        assert not missing, f"no gradient from the affinity loss reached: {missing}"


class TestTinyDataOverfit:
    """The specification's tiny-data overfit, and the strongest test here.

    Every shape can be correct while nothing is learned: a stray ``detach()``, a
    mask that is all False, a gather on the wrong axis. Such a model still reduces
    its loss, because the heads alone can fit the label mean -- so "loss went down"
    proves nothing. Driving *distinct* targets for eight pairs to near-zero
    requires the ligand and protein signal to actually reach the output.
    """

    def test_the_model_overfits_eight_pairs(self):
        torch.manual_seed(20261001)
        model = BindraModel(_config(model_dim=64, ligand=KPGTConfig.mini(d_g_feats=64, n_mol_layers=2, n_trainable_layers=2, n_heads=4, d_hpath_ratio=2)))
        model.train()

        features, mask = _proteins(len(MOLECULES), seed=17)
        graphs = KPGTLigandBatch.from_smiles(list(MOLECULES))
        # Distinct labels, spread over the real corpus range, and spread over all
        # three endpoints so no single head can carry the task alone.
        p_values = torch.tensor([4.5, 9.5, 6.0, 11.0, 3.0, 8.0, 7.0, 10.0])
        endpoints = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        targets = AffinityTargets(
            endpoint_index=endpoints,
            p_value=p_values,
            is_exact=torch.ones(8, dtype=torch.bool),
            lower_bound=torch.full((8,), NAN),
            upper_bound=torch.full((8,), NAN),
            is_primary=torch.ones(8, dtype=torch.bool),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = None
        for step in range(400):
            optimizer.zero_grad(set_to_none=True)
            out = model(features, mask, graphs)
            loss = affinity_loss(out.prediction, targets)
            loss.total.backward()
            optimizer.step()
            if step == 0:
                first = float(loss.exact_rmse.detach())
        model.eval()
        with torch.no_grad():
            final = affinity_loss(model(features, mask, graphs).prediction, targets)

        # The label spread is ~2.6 p-units SD, and sigma_rep's smallest value is
        # 0.641, so "below 0.25" is well inside the noise floor and far below what
        # predicting the mean achieves.
        print(
            f"\ntiny-data overfit: exact RMSE {first:.3f} -> "
            f"{float(final.exact_rmse):.4f} over 400 steps on 8 pairs"
        )
        assert float(final.exact_rmse) < 0.25, (
            f"RMSE only reached {float(final.exact_rmse):.3f} from {first:.3f}; "
            "the stack is not learning end to end"
        )

    def test_predicting_the_label_mean_would_not_pass_that_bar(self):
        """Calibrates the threshold above: the bar must exclude the trivial solution."""
        p_values = torch.tensor([4.5, 9.5, 6.0, 11.0, 3.0, 8.0, 7.0, 10.0])
        mean_only_rmse = (p_values - p_values.mean()).pow(2).mean().sqrt()
        assert float(mean_only_rmse) > 2.0

    def test_an_overfit_model_separates_molecules_that_share_a_protein(self):
        """Rules out the protein side carrying everything.

        If only the protein features reached the head, two different molecules
        against the *same* protein would get the same prediction.
        """
        torch.manual_seed(5)
        model = BindraModel(_config(model_dim=64, ligand=KPGTConfig.mini(d_g_feats=64, n_mol_layers=2, n_trainable_layers=2, n_heads=4, d_hpath_ratio=2)))
        model.train()
        one_protein = torch.randn(1, 9, PROTEIN_DIM).repeat(4, 1, 1)
        mask = torch.ones(4, 9, dtype=torch.bool)
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "C"]
        graphs = KPGTLigandBatch.from_smiles(list(smiles))
        targets = AffinityTargets(
            endpoint_index=torch.zeros(4, dtype=torch.long),
            p_value=torch.tensor([4.0, 7.0, 9.0, 11.0]),
            is_exact=torch.ones(4, dtype=torch.bool),
            lower_bound=torch.full((4,), NAN),
            upper_bound=torch.full((4,), NAN),
            is_primary=torch.ones(4, dtype=torch.bool),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(400):
            optimizer.zero_grad(set_to_none=True)
            affinity_loss(model(one_protein, mask, graphs).prediction, targets).total.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = affinity_loss(model(one_protein, mask, graphs).prediction, targets)
        assert float(final.exact_rmse) < 0.25

    def test_censored_rows_can_also_be_satisfied(self):
        """A censored batch must be learnable, not merely non-crashing."""
        torch.manual_seed(11)
        model = BindraModel(_config(model_dim=64, ligand=KPGTConfig.mini(d_g_feats=64, n_mol_layers=2, n_trainable_layers=2, n_heads=4, d_hpath_ratio=2)))
        model.train()
        features, mask = _proteins(4, seed=23)
        graphs = KPGTLigandBatch.from_smiles(list(MOLECULES[:4]))
        targets = AffinityTargets(
            endpoint_index=torch.zeros(4, dtype=torch.long),
            p_value=torch.tensor([5.0, 5.0, 9.0, 9.0]),
            is_exact=torch.zeros(4, dtype=torch.bool),
            lower_bound=torch.tensor([NAN, NAN, 9.0, 9.0]),
            upper_bound=torch.tensor([5.0, 5.0, NAN, NAN]),
            is_primary=torch.ones(4, dtype=torch.bool),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            affinity_loss(model(features, mask, graphs).prediction, targets).total.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = affinity_loss(model(features, mask, graphs).prediction, targets)
        assert float(final.bound_violation_rate) == 0.0
        assert float(final.censored_loss) < 1e-3


class TestSerialisation:
    def test_saving_and_reloading_reproduces_inference(self, model, tmp_path):
        """Required by the specification's own test list."""
        features, mask = _proteins(len(MOLECULES), seed=31)
        before = model.predict(features, mask, list(MOLECULES)).prediction.values

        path = tmp_path / "model.pt"
        torch.save({"config": model.config.to_dict(), "state": model.state_dict()}, path)

        reloaded = BindraModel(model.config)
        reloaded.load_state_dict(torch.load(path, weights_only=False)["state"])
        reloaded.eval()
        after = reloaded.predict(features, mask, list(MOLECULES)).prediction.values
        assert torch.equal(before, after)

    def test_a_config_mismatch_fails_loudly_on_load(self, model, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        wider = BindraModel(_config(model_dim=64))
        with pytest.raises(RuntimeError, match="size mismatch|Missing key|Unexpected key"):
            wider.load_state_dict(torch.load(path, weights_only=False))


@pytest.mark.gpu
class TestOnCuda:
    def test_the_model_trains_a_step_on_the_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        torch.manual_seed(20261001)
        model = BindraModel(_config()).cuda()
        model.train()
        features, mask = _proteins(len(MOLECULES), seed=41)
        graphs = KPGTLigandBatch.from_smiles(list(MOLECULES)).to("cuda")
        targets = AffinityTargets(
            endpoint_index=torch.zeros(len(MOLECULES), dtype=torch.long),
            p_value=torch.full((len(MOLECULES),), 8.0),
            is_exact=torch.ones(len(MOLECULES), dtype=torch.bool),
            lower_bound=torch.full((len(MOLECULES),), NAN),
            upper_bound=torch.full((len(MOLECULES),), NAN),
            is_primary=torch.ones(len(MOLECULES), dtype=torch.bool),
        ).to("cuda")
        out = model(features.cuda(), mask.cuda(), graphs)
        loss = affinity_loss(out.prediction, targets)
        loss.total.backward()
        assert torch.isfinite(loss.total)
        assert all(
            p.grad is not None
            for name, p in model.named_parameters()
            if name.startswith("adapters.")
        )

    def test_cuda_and_cpu_predictions_agree(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        torch.manual_seed(20261001)
        cpu = BindraModel(_config()).eval()
        gpu = BindraModel(_config()).eval().cuda()
        gpu.load_state_dict(cpu.state_dict())
        features, mask = _proteins(len(MOLECULES), seed=43)
        a = cpu.predict(features, mask, list(MOLECULES)).prediction.values
        b = gpu.predict(features.cuda(), mask.cuda(), list(MOLECULES)).prediction.values
        assert torch.allclose(a, b.cpu(), atol=1e-4)
