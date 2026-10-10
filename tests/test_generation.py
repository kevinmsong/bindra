"""Analog generation: the discrete process, the scaffold, and honest validity.

Three properties the module exists to guarantee, each asserted rather than
described:

``test_corruption_is_categorical_not_gaussian``
    Carbon is not between nitrogen and oxygen. A corrupted atom must land on a
    real category, never between two.
``test_the_scaffold_survives_the_whole_chain``
    Re-imposed at every reverse step, not merely initialised. A soft constraint
    lets a long chain drift off the scaffold a fraction at a time.
``test_sanitizer_rescues_are_not_credited_to_the_model``
    Validity before and after RDKit's repair are different numbers, and the
    gap between them belongs to RDKit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bindra.generation.digress.decode import DecodedMolecule, validity_report
from bindra.generation.digress.denoiser import (
    DenoiserConfig,
    GraphDenoiser,
    assert_no_product_leakage,
    timestep_embedding,
)
from bindra.generation.digress.noise import (
    CategoricalMarginals,
    DiscreteGraphNoise,
    cosine_alpha_bar,
)
from bindra.generation.digress.sampler import (
    ScaffoldConstraint,
    sample,
    verify_scaffold,
)

A, E, N, B = 6, 4, 9, 2


def _graph(seed=0):
    torch.manual_seed(seed)
    atoms = torch.nn.functional.one_hot(torch.randint(0, A, (B, N)), A).float()
    raw = torch.randint(0, E, (B, N, N))
    raw = torch.minimum(raw, raw.transpose(1, 2))
    bonds = torch.nn.functional.one_hot(raw, E).float()
    return atoms, bonds


def _denoiser():
    return GraphDenoiser(
        DenoiserConfig(n_atom_types=A, n_bond_types=E, condition_dim=16,
                       model_dim=32, n_layers=2, n_heads=4)
    )


class TestSchedule:
    def test_alpha_bar_runs_from_one_to_almost_zero(self):
        schedule = cosine_alpha_bar(100)
        assert schedule.numel() == 101
        assert float(schedule[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(schedule[-1]) < 0.02

    def test_it_is_monotonically_decreasing(self):
        schedule = cosine_alpha_bar(200)
        assert bool((schedule[1:] <= schedule[:-1]).all())

    def test_zero_timesteps_is_refused(self):
        with pytest.raises(Exception):
            cosine_alpha_bar(0)


class TestMarginals:
    def test_a_marginal_that_does_not_sum_to_one_is_refused(self):
        with pytest.raises(Exception, match="sum to 1"):
            CategoricalMarginals(atom=torch.tensor([0.5, 0.2]), bond=torch.tensor([0.5, 0.5]))

    def test_smoothing_leaves_no_category_impossible(self):
        """A zero-probability category can never be sampled back."""
        marginals = CategoricalMarginals.from_counts(
            torch.tensor([100, 0, 0, 5]), torch.tensor([50, 0, 3])
        )
        assert bool((marginals.atom > 0).all())
        assert bool((marginals.bond > 0).all())


class TestForwardProcess:
    def test_corruption_is_categorical_not_gaussian(self):
        """Every corrupted atom lands on a real category, never between two."""
        atoms, bonds = _graph()
        noise = DiscreteGraphNoise(CategoricalMarginals.uniform(A, E), timesteps=50)
        t = torch.full((B,), 40, dtype=torch.long)
        noisy_atoms, noisy_bonds = noise.q_sample(atoms, bonds, t)
        assert bool((noisy_atoms.sum(-1) == 1).all()), "an atom is not one-hot"
        assert bool((noisy_bonds.sum(-1) == 1).all()), "a bond is not one-hot"
        assert set(noisy_atoms.unique().tolist()) <= {0.0, 1.0}

    def test_bonds_stay_symmetric_under_corruption(self):
        """A bond is one fact about a pair, not two."""
        atoms, bonds = _graph()
        noise = DiscreteGraphNoise(CategoricalMarginals.uniform(A, E), timesteps=50)
        _, noisy = noise.q_sample(atoms, bonds, torch.full((B,), 30, dtype=torch.long))
        assert bool((noisy == noisy.transpose(1, 2)).all())

    def test_t_zero_leaves_the_graph_essentially_clean(self):
        atoms, bonds = _graph()
        noise = DiscreteGraphNoise(CategoricalMarginals.uniform(A, E), timesteps=50)
        probabilities, _ = noise.q_probabilities(atoms, bonds, torch.zeros(B, dtype=torch.long))
        assert torch.allclose(probabilities, atoms, atol=1e-5)

    def test_the_kept_mask_is_never_corrupted(self):
        atoms, bonds = _graph()
        keep = torch.zeros(B, N, dtype=torch.bool)
        keep[:, :4] = True
        noise = DiscreteGraphNoise(CategoricalMarginals.uniform(A, E), timesteps=50)
        noisy_atoms, noisy_bonds = noise.q_sample(
            atoms, bonds, torch.full((B,), 49, dtype=torch.long), keep=keep
        )
        assert torch.equal(noisy_atoms[keep], atoms[keep])
        internal = keep.unsqueeze(1) & keep.unsqueeze(2)
        assert torch.equal(noisy_bonds[internal], bonds[internal])


class TestDenoiser:
    def test_it_predicts_both_streams_at_the_right_width(self):
        atoms, bonds = _graph()
        mask = torch.ones(B, N, dtype=torch.bool)
        atom_logits, bond_logits = _denoiser()(
            atoms, bonds, torch.zeros(B, dtype=torch.long), torch.randn(B, 5, 16), mask
        )
        assert atom_logits.shape == (B, N, A)
        assert bond_logits.shape == (B, N, N, E)

    def test_bond_predictions_are_symmetric(self):
        """An asymmetric prediction makes the sampler's two halves disagree."""
        atoms, bonds = _graph()
        mask = torch.ones(B, N, dtype=torch.bool)
        _, bond_logits = _denoiser()(
            atoms, bonds, torch.zeros(B, dtype=torch.long), torch.randn(B, 5, 16), mask
        )
        assert torch.allclose(bond_logits, bond_logits.transpose(1, 2), atol=1e-5)

    def test_padded_atoms_do_not_produce_nan(self):
        atoms, bonds = _graph()
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[1, 5:] = False
        atom_logits, bond_logits = _denoiser()(
            atoms, bonds, torch.zeros(B, dtype=torch.long), torch.randn(B, 5, 16), mask
        )
        assert not bool(torch.isnan(atom_logits).any())
        assert not bool(torch.isnan(bond_logits).any())

    def test_conditioning_on_the_answer_is_refused(self):
        """A model handed the answer trains beautifully and generates nothing."""
        assert_no_product_leakage({"target": 1, "seed": 2, "scaffold": 3})
        for bad in ("product_graph", "clean_x", "x_0", "answer_key"):
            with pytest.raises(Exception, match="clean product"):
                assert_no_product_leakage({"target": 1, bad: 2})

    def test_the_timestep_embedding_distinguishes_steps(self):
        embedded = timestep_embedding(torch.tensor([0, 1, 500]), 32)
        assert embedded.shape == (3, 32)
        assert not torch.allclose(embedded[0], embedded[2])


class TestSampler:
    def test_the_scaffold_survives_the_whole_chain(self):
        atoms, bonds = _graph()
        keep = torch.zeros(B, N, dtype=torch.bool)
        keep[:, :4] = True
        constraint = ScaffoldConstraint(keep=keep, atoms=atoms, bonds=bonds)
        noise = DiscreteGraphNoise(CategoricalMarginals.uniform(A, E), timesteps=8)
        out_atoms, out_bonds = sample(
            _denoiser(), noise,
            conditions=torch.randn(B, 5, 16),
            atom_mask=torch.ones(B, N, dtype=torch.bool),
            constraint=constraint,
        )
        report = verify_scaffold(out_atoms, out_bonds, constraint)
        assert report["scaffold_atoms_preserved"] is True
        assert report["scaffold_bonds_preserved"] is True

    def test_output_is_one_hot_and_symmetric(self):
        noise = DiscreteGraphNoise(CategoricalMarginals.uniform(A, E), timesteps=5)
        atoms, bonds = sample(
            _denoiser(), noise,
            conditions=torch.randn(B, 5, 16),
            atom_mask=torch.ones(B, N, dtype=torch.bool),
        )
        assert bool((atoms.sum(-1) == 1).all())
        assert bool((bonds == bonds.transpose(1, 2)).all())

    def test_padding_is_zeroed_so_a_decoder_cannot_read_it(self):
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[0, 6:] = False
        noise = DiscreteGraphNoise(CategoricalMarginals.uniform(A, E), timesteps=4)
        atoms, _ = sample(
            _denoiser(), noise, conditions=torch.randn(B, 5, 16), atom_mask=mask
        )
        assert float(atoms[0, 6:].sum()) == 0.0

    def test_a_non_boolean_keep_is_refused(self):
        atoms, bonds = _graph()
        with pytest.raises(Exception, match="bool"):
            ScaffoldConstraint(keep=torch.ones(B, N), atoms=atoms, bonds=bonds)


class TestDecodeHonesty:
    def test_sanitizer_rescues_are_not_credited_to_the_model(self):
        decoded = [
            DecodedMolecule("CCO", True, True),
            DecodedMolecule("CCN", False, True),   # rescued by RDKit
            DecodedMolecule("CCC", False, True),   # rescued by RDKit
            DecodedMolecule(None, False, False),
        ]
        report = validity_report(decoded)
        assert report["valid_before_sanitize"] == pytest.approx(0.25)
        assert report["valid_after_sanitize"] == pytest.approx(0.75)
        assert report["sanitizer_rescue_rate"] == pytest.approx(0.5)
        assert "model's own validity" in report["note"]

    def test_uniqueness_counts_distinct_molecules(self):
        decoded = [DecodedMolecule("CCO", True, True) for _ in range(4)]
        report = validity_report(decoded)
        assert report["unique_smiles"] == 1
        assert report["uniqueness"] == pytest.approx(0.25)

    def test_nothing_decoded_is_a_status_not_a_zero(self):
        assert validity_report([])["n"] == 0

    def test_scaffold_coverage_is_reported_when_unchecked(self):
        report = validity_report([DecodedMolecule("CCO", True, True)])
        assert report["scaffold_preserved"] is None
        assert report["n_scaffold_checked"] == 0

    @pytest.mark.snapshot
    def test_a_real_graph_decodes_to_a_molecule(self):
        from rdkit import Chem

        from bindra.generation.digress.decode import decode_graph

        # Ethanol: C-C-O with two single bonds.
        atoms = [0, 0, 1]
        bonds = np.zeros((3, 3), dtype=int)
        bonds[0, 1] = bonds[1, 0] = 1
        bonds[1, 2] = bonds[2, 1] = 1
        result = decode_graph(
            atoms, bonds, ["C", "O"], [None, Chem.BondType.SINGLE],
            scaffold_indices={0, 1}, scaffold_elements={0: "C", 1: "C"},
        )
        assert result.usable
        assert result.scaffold_preserved is True
        assert Chem.CanonSmiles(result.smiles) == Chem.CanonSmiles("CCO")

    @pytest.mark.snapshot
    def test_the_emitted_smiles_carries_no_internal_bookkeeping(self):
        """Atom map numbers are how the scaffold is checked, not output.

        Left in, two copies of one molecule whose atoms happened to be
        generated in a different order render as different strings, and the
        uniqueness rate becomes meaningless.
        """
        from rdkit import Chem

        from bindra.generation.digress.decode import decode_graph

        bonds = np.zeros((3, 3), dtype=int)
        bonds[0, 1] = bonds[1, 0] = 1
        bonds[1, 2] = bonds[2, 1] = 1
        first = decode_graph([0, 0, 1], bonds, ["C", "O"], [None, Chem.BondType.SINGLE])

        # Same molecule, atoms in a different order.
        other = np.zeros((3, 3), dtype=int)
        other[2, 1] = other[1, 2] = 1
        other[1, 0] = other[0, 1] = 1
        second = decode_graph([1, 0, 0], other, ["C", "O"], [None, Chem.BondType.SINGLE])

        assert ":" not in first.smiles, f"map numbers leaked: {first.smiles}"
        assert first.smiles == second.smiles, "atom order changed the SMILES"
