"""Target constructs: identity, validation flags, and offline mutation calls."""

from __future__ import annotations

import pytest

from bindra.config.profiles import ProteinPolicy
from bindra.data.bindingdb.proteins import (
    ConstructFlag,
    ConstructStatus,
    build_target_constructs,
    corroborate_mutations,
    mutation_tokens_from_name,
    sequence_sha256,
    validate_sequence,
)

BASE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTL"


def _record(ref: int, sequence: str) -> dict:
    return {
        "sequence_ref": ref,
        "sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "sequence_length": len(sequence),
    }


class TestValidation:
    def test_standard_sequence_is_ok_with_no_flags(self):
        status, flags, invalid = validate_sequence(BASE)
        assert status == ConstructStatus.OK
        assert flags == ()
        assert invalid is None

    def test_digits_in_the_sequence_cell_are_rejected(self):
        """Real failure mode: BindingDB sometimes puts range notation here.

        Release 202610 contains constructs whose 'sequence' is text like
        ``[1-18,20-1210,...]``, which must not become a protein.
        """
        status, _, invalid = validate_sequence("MKTAY012345")
        assert status == ConstructStatus.INVALID_RESIDUES
        assert invalid == "012345"

    def test_ambiguous_residues_are_flagged_but_kept(self):
        status, flags, _ = validate_sequence(BASE + "XXX")
        assert status == ConstructStatus.OK
        assert ConstructFlag.CONTAINS_AMBIGUOUS in flags

    def test_nonstandard_residues_are_flagged_but_kept(self):
        status, flags, _ = validate_sequence(BASE + "U")
        assert status == ConstructStatus.OK
        assert ConstructFlag.CONTAINS_NONSTANDARD in flags

    def test_a_very_long_construct_is_flagged_not_rejected(self):
        """Titin is 34,350 residues in this release; rejecting it would lose a target."""
        status, flags, _ = validate_sequence("A" * 34_350)
        assert status == ConstructStatus.OK
        assert ConstructFlag.LONG_CONSTRUCT in flags

    def test_a_very_short_construct_is_flagged_not_rejected(self):
        status, flags, _ = validate_sequence("MKTAY")
        assert status == ConstructStatus.OK
        assert ConstructFlag.SHORT_CONSTRUCT in flags

    def test_empty_sequence_is_empty_status(self):
        assert validate_sequence("")[0] == ConstructStatus.EMPTY


class TestConstructIdentity:
    def test_identity_is_the_construct_hash_not_the_uniprot_accession(self):
        """Two constructs of one protein are two targets, as they were measured."""
        mutant = BASE[:10] + "G" + BASE[11:]
        constructs, _ = build_target_constructs(
            [_record(0, BASE), _record(1, mutant)],
            {
                0: {"uniprot_primary": "P00001", "uniprot_source": "swissprot"},
                1: {"uniprot_primary": "P00001", "uniprot_source": "swissprot"},
            },
        )
        assert constructs[0].construct_id != constructs[1].construct_id
        assert constructs[0].uniprot_primary == constructs[1].uniprot_primary
        assert all(c.construct_id.startswith("P00001:") for c in constructs)

    def test_construct_without_uniprot_is_still_identified(self):
        constructs, _ = build_target_constructs([_record(0, BASE)], {})
        assert constructs[0].construct_id.startswith("NOUP:")
        assert ConstructFlag.NO_UNIPROT in constructs[0].flags

    def test_multiple_uniprot_ids_are_flagged_and_retained(self):
        constructs, _ = build_target_constructs(
            [_record(0, BASE)],
            {0: {"uniprot_primary": "P00001", "uniprot_all": "P00001,P00002"}},
        )
        assert ConstructFlag.MULTIPLE_UNIPROT_IDS in constructs[0].flags
        assert constructs[0].uniprot_all == "P00001,P00002"


class TestMutationDetection:
    def test_single_point_difference_is_called(self):
        mutant = BASE[:10] + "G" + BASE[11:]
        constructs, report = build_target_constructs(
            [_record(0, BASE), _record(1, mutant)],
            {
                0: {"uniprot_primary": "P1", "uniprot_source": "swissprot"},
                1: {"uniprot_primary": "P1", "uniprot_source": "swissprot"},
            },
        )
        mutated = [c for c in constructs if c.mutations]
        assert len(mutated) == 1
        assert str(mutated[0].mutations[0]) == f"{BASE[10]}11G"
        assert mutated[0].mutation_source == "alignment_vs_longest_construct"
        assert report["n_constructs_with_mutations"] == 1

    def test_different_lengths_are_not_called_as_point_mutations(self):
        """A truncation is an indel relationship, not a substitution."""
        constructs, report = build_target_constructs(
            [_record(0, BASE), _record(1, BASE[:40])],
            {
                0: {"uniprot_primary": "P1"},
                1: {"uniprot_primary": "P1"},
            },
        )
        assert report["n_constructs_with_mutations"] == 0
        # ...but the fragment relationship is recorded.
        short = [c for c in constructs if c.sequence_length == 40][0]
        assert ConstructFlag.SUBSTRING_OF_ANOTHER in short.flags

    def test_too_many_differences_is_not_a_mutation_call(self):
        """Below the identity floor it is a different protein, not a mutant."""
        other = "".join("G" if i % 2 else c for i, c in enumerate(BASE))
        _, report = build_target_constructs(
            [_record(0, BASE), _record(1, other)],
            {0: {"uniprot_primary": "P1"}, 1: {"uniprot_primary": "P1"}},
        )
        assert report["n_constructs_with_mutations"] == 0

    def test_mutations_are_not_called_across_different_uniprot_groups(self):
        mutant = BASE[:10] + "G" + BASE[11:]
        _, report = build_target_constructs(
            [_record(0, BASE), _record(1, mutant)],
            {0: {"uniprot_primary": "P1"}, 1: {"uniprot_primary": "P2"}},
        )
        assert report["n_constructs_with_mutations"] == 0

    def test_name_tokens_are_parsed_but_are_only_corroboration(self):
        assert mutation_tokens_from_name("EGFR [T790M]") == ("T790M",)
        assert mutation_tokens_from_name("Kinase [L49I,T183A]") == ("L49I", "T183A")
        assert mutation_tokens_from_name("Endochitinase B1") == ()
        assert mutation_tokens_from_name(None) == ()

    def test_corroboration_reports_agreement_and_offsets(self):
        """Alignment and free-text tokens are independent signals.

        On the real release they agree on the substitution for 91.1% of
        comparable constructs, with the residual explained by construct-relative
        versus canonical numbering.
        """
        mutant = BASE[:10] + "G" + BASE[11:]
        constructs, _ = build_target_constructs(
            [_record(0, BASE), _record(1, mutant)],
            {
                0: {"uniprot_primary": "P1", "target_name": "Thing"},
                1: {
                    "uniprot_primary": "P1",
                    "target_name": f"Thing [{BASE[10]}11G]",
                },
            },
        )
        summary = corroborate_mutations(constructs)
        assert summary["n_with_mutations"] == 1
        assert summary["n_exact_agreement"] == 1
        assert summary["substitution_agreement_rate"] == 1.0

    def test_corroboration_detects_a_constant_numbering_offset(self):
        mutant = BASE[:10] + "G" + BASE[11:]
        constructs, _ = build_target_constructs(
            [_record(0, BASE), _record(1, mutant)],
            {
                0: {"uniprot_primary": "P1"},
                # Named with canonical numbering, one lower than construct numbering.
                1: {"uniprot_primary": "P1", "target_name": f"Thing [{BASE[10]}10G]"},
            },
        )
        summary = corroborate_mutations(constructs)
        assert summary["n_exact_agreement"] == 0
        assert summary["n_agreement_at_constant_offset"] == 1
        assert summary["offset_histogram"]["-1"] == 1


class TestReport:
    def test_length_statistics_are_reported(self):
        _, report = build_target_constructs(
            [_record(i, "A" * length) for i, length in enumerate([10, 500, 3000])], {}
        )
        assert report["length"]["min"] == 10
        assert report["length"]["max"] == 3000
        assert report["length"]["total_residues"] == 3510
        assert report["length"]["over_2046"] == 1

    def test_report_carries_the_wild_type_caveat(self):
        _, report = build_target_constructs([_record(0, BASE)], {})
        assert "not established to be wild type" in report["caveat"]
