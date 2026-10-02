"""Standardization: the two keys, the fragment policy, and the atom maps.

The keys carry the load here. ``identity_key`` is what a measurement attaches
to, so it must separate species that were measured separately; ``leakage_key`` is
what a cold-drug split partitions on, so it must fuse everything that is "the
same compound" for leakage purposes. The specification pulls both ways at once,
and most of these tests pin a case where the two answers differ.
"""

from __future__ import annotations

from rdkit import Chem
import pytest

from bindra.config.profiles import StandardizationPolicy
from bindra.data.bindingdb.standardize import (
    SALT_SMILES,
    StandardizationStatus,
    policy_id,
    salt_definitions_sha256,
    standardize_many,
    standardize_smiles,
)


class TestIdentityVersusLeakage:
    def test_enantiomers_are_distinct_molecules_but_one_leakage_group(self):
        left = standardize_smiles("C[C@H](N)C(=O)O")
        right = standardize_smiles("C[C@@H](N)C(=O)O")
        assert left.ok and right.ok
        # Measured separately, so they must not merge into one labelled molecule.
        assert left.identity_key != right.identity_key
        # But a cold-drug split must never put one in train and the other in test.
        assert left.leakage_key == right.leakage_key

    def test_salt_and_free_acid_are_distinct_species_in_one_leakage_group(self):
        """The case where the two requirements genuinely conflict.

        A carboxylate anion and its free acid are different measured species
        (different net charge), but they must never straddle a split.
        """
        salt = standardize_smiles("[Na+].[O-]C(=O)c1ccccc1")
        acid = standardize_smiles("OC(=O)c1ccccc1")
        assert salt.ok and acid.ok
        assert salt.formal_charge == -1
        assert acid.formal_charge == 0
        assert salt.identity_key != acid.identity_key
        assert salt.leakage_key == acid.leakage_key

    def test_common_tautomers_share_a_leakage_group_via_inchi(self):
        """InChI's mobile-hydrogen layer handles these without the enumerator.

        This is why ``canonicalize_tautomers`` is off by default: the expensive
        step costs 96.5 ms/mol and this case already works.
        """
        enol = standardize_smiles("Oc1ccccn1")
        keto = standardize_smiles("O=c1cccc[nH]1")
        assert enol.ok and keto.ok
        assert enol.identity_key != keto.identity_key
        assert enol.leakage_key == keto.leakage_key

    def test_hydrochloride_resolves_to_the_same_species_as_the_free_base(self):
        salt = standardize_smiles("Cl.CCN(CC)CC")
        base = standardize_smiles("CCN(CC)CC")
        assert salt.identity_key == base.identity_key
        assert "Cl" in (salt.removed_fragments or "")

    def test_charges_are_preserved_rather_than_neutralized(self):
        """The specification says preserve formal charges on the identity path."""
        quaternary = standardize_smiles("C[N+](C)(C)C")
        assert quaternary.ok
        assert quaternary.formal_charge == 1


class TestFragmentPolicy:
    def test_two_drug_like_fragments_are_quarantined_not_resolved(self):
        """Largest-fragment selection is disabled by policy.

        Silently keeping the bigger half would discard a co-crystallised active
        with no record, which is the coercion the specification forbids.
        """
        result = standardize_smiles("c1ccc2ccccc2c1CCN.CCc1ccc(F)cc1C(=O)NC")
        assert result.status == StandardizationStatus.MULTIPLE_ORGANIC_FRAGMENTS
        assert result.standardized_smiles is None
        assert result.identity_key is None
        assert "largest-fragment selection is disabled" in result.detail

    def test_recognized_counterions_are_removed_and_recorded(self):
        result = standardize_smiles("[Na+].[Cl-].c1ccc2ccccc2c1CCN")
        assert result.ok
        assert result.standardized_smiles == "NCCc1cccc2ccccc12"
        assert "[Na+]" in result.removed_fragments
        assert "[Cl-]" in result.removed_fragments

    def test_anionic_and_neutral_counterion_spellings_both_match(self):
        """Counterions appear in both forms; matching is on the neutral parent."""
        anionic = standardize_smiles("[O-]C(=O)C(F)(F)F.c1ccc2ccccc2c1CCN")
        neutral = standardize_smiles("OC(=O)C(F)(F)F.c1ccc2ccccc2c1CCN")
        assert anionic.ok and neutral.ok
        assert anionic.identity_key == neutral.identity_key

    def test_a_salt_of_an_allowlisted_acid_still_resolves_to_that_acid(self):
        """Calcium dibenzoate is benzoate, not an ambiguous mixture.

        Identical organic fragments collapse before the allowlist is consulted,
        so the compound survives even though benzoate is itself a counterion.
        """
        result = standardize_smiles("[Ca+2].[O-]C(=O)c1ccccc1.[O-]C(=O)c1ccccc1")
        assert result.ok
        assert result.leakage_key == standardize_smiles("OC(=O)c1ccccc1").leakage_key

    def test_inorganic_only_input_is_quarantined(self):
        result = standardize_smiles("[Na+].[Cl-]")
        assert result.status == StandardizationStatus.ONLY_INORGANIC
        assert result.standardized_smiles is None

    def test_every_allowlist_entry_parses(self):
        """An unparseable entry would silently never match anything."""
        for smiles in SALT_SMILES:
            assert Chem.MolFromSmiles(smiles) is not None, smiles

    def test_allowlist_has_no_duplicates(self):
        assert len(SALT_SMILES) == len(set(SALT_SMILES))


class TestFailuresAreReasons:
    def test_unparseable_smiles_is_a_status_not_a_placeholder_molecule(self):
        """Legacy returned an all-zero graph here, making it a training example."""
        result = standardize_smiles("this-is-not-smiles")
        assert result.status == StandardizationStatus.PARSE_FAILED
        assert result.standardized_smiles is None
        assert result.identity_key is None
        assert result.detail

    def test_empty_input_is_a_status(self):
        assert standardize_smiles("").status == StandardizationStatus.EMPTY
        assert standardize_smiles("   ").status == StandardizationStatus.EMPTY


class TestAtomMaps:
    def test_std_to_orig_is_injective_and_element_preserving(self):
        raw = "[Na+].[O-]C(=O)c1ccccc1"
        result = standardize_smiles(raw)
        assert result.ok
        original = Chem.MolFromSmiles(raw)
        standardized = Chem.MolFromSmiles(result.standardized_smiles)

        assert len(result.std_to_orig) == standardized.GetNumAtoms()
        assert len(set(result.std_to_orig)) == len(result.std_to_orig)
        assert all(index >= 0 for index in result.std_to_orig)

    def test_std_to_output_is_a_permutation(self):
        """Needed so an iBAM contribution lands on the right atom of a depiction."""
        result = standardize_smiles("CC(=O)Nc1ccc(O)cc1")
        assert result.ok
        assert sorted(result.std_to_output) == list(range(len(result.std_to_output)))

    def test_atom_map_tracks_the_surviving_fragment(self):
        """After desalting, the map must point into the ORIGINAL indices.

        The sodium is atom 0 of the input, so a correct map starts at 1.
        """
        result = standardize_smiles("[Na+].[O-]C(=O)c1ccccc1")
        assert 0 not in result.std_to_orig


class TestScaffolds:
    def test_ring_molecule_gets_a_murcko_scaffold(self):
        result = standardize_smiles("CC(=O)Nc1ccc(O)cc1")
        assert result.scaffold_kind == "murcko"
        assert result.scaffold_smiles

    def test_two_different_acyclic_molecules_do_not_share_a_scaffold_group(self):
        """A shared ACYCLIC bucket would fuse unrelated chemistry into one group."""
        acid = standardize_smiles("CC(=O)O")
        amine = standardize_smiles("CCCCN")
        assert acid.scaffold_kind == "acyclic_fallback"
        assert amine.scaffold_kind == "acyclic_fallback"
        assert acid.scaffold_key != amine.scaffold_key

    def test_acyclic_scaffold_key_is_its_own_leakage_key(self):
        acid = standardize_smiles("CC(=O)O")
        assert acid.scaffold_key == acid.leakage_key

    def test_scaffold_ignores_stereochemistry_so_it_does_not_split_by_isomer(self):
        left = standardize_smiles("C[C@H](N)c1ccccc1")
        right = standardize_smiles("C[C@@H](N)c1ccccc1")
        assert left.scaffold_key == right.scaffold_key


class TestDeterminism:
    def test_same_input_gives_byte_identical_keys(self):
        smiles = ["CC(=O)Nc1ccc(O)cc1", "C[C@H](N)C(=O)O", "[Na+].[Cl-].CCO"]
        first = standardize_many(smiles)
        second = standardize_many(smiles)
        assert [r.identity_key for r in first] == [r.identity_key for r in second]
        assert [r.leakage_key for r in first] == [r.leakage_key for r in second]
        assert [r.std_to_orig for r in first] == [r.std_to_orig for r in second]

    def test_policy_id_changes_when_the_policy_changes(self):
        base = StandardizationPolicy()
        changed = StandardizationPolicy(canonicalize_tautomers=True)
        assert policy_id(base) != policy_id(changed)

    def test_salt_definitions_hash_is_stable(self):
        assert salt_definitions_sha256() == salt_definitions_sha256()
        assert len(salt_definitions_sha256()) == 64


class TestTautomerFlag:
    def test_enabling_canonicalization_still_produces_usable_keys(self):
        """The expensive path must remain correct, just unused by default."""
        policy = StandardizationPolicy(canonicalize_tautomers=True)
        enol = standardize_smiles("CC(O)=CC(=O)C", policy)
        keto = standardize_smiles("CC(=O)CC(=O)C", policy)
        assert enol.standardized_smiles and keto.standardized_smiles
        # With canonicalization on, keto-enol collapses too -- the case InChI's
        # mobile-hydrogen layer alone does not cover.
        assert enol.leakage_key == keto.leakage_key

    def test_keto_enol_is_the_documented_gap_when_canonicalization_is_off(self):
        enol = standardize_smiles("CC(O)=CC(=O)C")
        keto = standardize_smiles("CC(=O)CC(=O)C")
        assert enol.ok and keto.ok
        # Documented limitation, asserted so it cannot change silently.
        assert enol.leakage_key != keto.leakage_key
