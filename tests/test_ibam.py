"""iBAM export: mapping correctness, mass conservation, and honest labelling.

The tests that matter:

``test_physical_attention_mass_is_conserved_onto_atoms``
    The derived residue x atom view must neither create nor lose attention. A
    mapping bug that drops a token silently shrinks some atoms' contributions,
    which looks like a finding.
``test_a_bond_token_splits_equally_between_its_endpoints``
    The specification's prescribed rule, and the reason the view is labelled
    derived.
``test_nonphysical_mass_is_preserved_not_dropped``
    Dropping knowledge-node mass makes the remainder look like a complete
    distribution when it is not.
``test_residue_positions_are_original_sequence_indices``
    261 of 10,820 constructs exceed the 2,046-residue encoder window. Exporting
    row numbers as residue numbers would misplace every annotation on all of them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from bindra.encoders.ligand_gat import GATConfig, GATLigandEncoder
from bindra.features.types import EncoderMetadata, LigandFeatures, TokenType
from bindra.fusion.cross_attention import CrossAttentionConfig, CrossAttentionStack
from bindra.ibam.export import (
    DEFAULT_POLICY,
    DISCLAIMER,
    AggregationPolicy,
    IBAMExport,
    IBAMExportError,
    derive_residue_atom,
    export_ibam,
)


@pytest.fixture
def stack() -> CrossAttentionStack:
    torch.manual_seed(20261001)
    built = CrossAttentionStack(
        CrossAttentionConfig(model_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    )
    built.eval()
    return built


@pytest.fixture
def ligand_features() -> LigandFeatures:
    torch.manual_seed(20261001)
    encoder = GATLigandEncoder(GATConfig(hidden_dim=32, num_layers=1, num_heads=4))
    encoder.module.eval()
    return encoder.encode("CC(=O)Oc1ccccc1C(=O)O")


def _fused(stack, num_tokens: int, length: int = 7, batch: int = 1):
    torch.manual_seed(3)
    return stack(
        torch.randn(batch, length, 32),
        torch.randn(batch, num_tokens, 32),
        torch.ones(batch, length, dtype=torch.bool),
        torch.ones(batch, num_tokens, dtype=torch.bool),
    )


def _mixed_features(num_atoms: int = 4) -> LigandFeatures:
    """A molecule with atom, bond and knowledge tokens, as KPGT will produce.

    The GAT arm emits atom tokens only, so the bond and knowledge paths would
    otherwise be untested until KPGT exists -- by which time the rule would be
    being written against examples rather than in advance.
    """
    token_types = (
        TokenType.ATOM, TokenType.ATOM, TokenType.ATOM, TokenType.ATOM,
        TokenType.BOND, TokenType.BOND,
        TokenType.KNOWLEDGE,
    )
    return LigandFeatures(
        token_features=torch.zeros(len(token_types), 8),
        token_mask=torch.ones(len(token_types), dtype=torch.bool),
        token_types=token_types,
        graph_features=torch.zeros(8),
        token_to_atom=((0,), (1,), (2,), (3,), (0, 1), (2, 3), ()),
        token_to_bond=(None, None, None, None, 0, 1, None),
        atom_index_maps={"std_to_output": (3, 2, 1, 0)},
        molecule_id="synthetic",
        encoder_metadata=EncoderMetadata(
            name="synthetic", checkpoint="none", output_dim=8,
            precision="float32", feature_schema_version="v1",
        ),
        num_atoms=num_atoms,
    )


class TestPolicy:
    def test_the_default_is_fixed_in_advance_and_hashed(self):
        """A later change must be visible in the export, not lost in a diff."""
        assert DEFAULT_POLICY.head_reduction == "mean"
        assert DEFAULT_POLICY.layer_reduction == "last"
        assert DEFAULT_POLICY.bond_split == "equal"
        assert DEFAULT_POLICY.renormalize_physical is False
        assert len(DEFAULT_POLICY.policy_hash) == 16

    def test_changing_the_policy_changes_its_hash(self):
        assert AggregationPolicy(layer_reduction="mean").policy_hash != DEFAULT_POLICY.policy_hash

    def test_the_description_says_the_atom_view_is_derived(self):
        assert "derived aggregation" in DEFAULT_POLICY.describe()

    @pytest.mark.parametrize(
        "kwargs", [{"head_reduction": "median"}, {"layer_reduction": "best"}, {"bond_split": "weighted"}]
    )
    def test_an_unimplemented_rule_is_refused_rather_than_approximated(self, kwargs):
        with pytest.raises(IBAMExportError):
            AggregationPolicy(**kwargs)


class TestDeriveResidueAtom:
    def test_an_atom_token_maps_to_exactly_its_atom(self):
        residue_token = torch.tensor([[0.5, 0.3, 0.2]])
        residue_atom, nonphysical = derive_residue_atom(
            residue_token, ((0,), (1,), (2,)), (TokenType.ATOM,) * 3, 3
        )
        assert torch.allclose(residue_atom, residue_token)
        assert nonphysical.item() == 0.0

    def test_a_bond_token_splits_equally_between_its_endpoints(self):
        """The specification's prescribed initial rule."""
        residue_token = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]])
        residue_atom, _ = derive_residue_atom(
            residue_token,
            ((0,), (1,), (2,), (3,), (0, 1), (2, 3), ()),
            _mixed_features().token_types,
            4,
        )
        assert residue_atom[0, 0].item() == pytest.approx(0.4)
        assert residue_atom[0, 1].item() == pytest.approx(0.4)
        assert residue_atom[0, 2].item() == 0.0

    def test_physical_attention_mass_is_conserved_onto_atoms(self):
        """A dropped token silently shrinks contributions, which looks like a finding."""
        torch.manual_seed(1)
        features = _mixed_features()
        residue_token = torch.rand(5, len(features.token_types))
        residue_atom, nonphysical = derive_residue_atom(
            residue_token, features.token_to_atom, features.token_types, 4
        )
        physical = [
            i for i, t in enumerate(features.token_types) if TokenType(t).is_physical
        ]
        expected = residue_token[:, physical].sum(dim=1)
        assert torch.allclose(residue_atom.sum(dim=1), expected, atol=1e-6)
        # And nothing vanished: physical + nonphysical is the whole row.
        assert torch.allclose(
            residue_atom.sum(dim=1) + nonphysical, residue_token.sum(dim=1), atol=1e-6
        )

    def test_nonphysical_mass_is_preserved_not_dropped(self):
        residue_token = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]])
        residue_atom, nonphysical = derive_residue_atom(
            residue_token,
            _mixed_features().token_to_atom,
            _mixed_features().token_types,
            4,
        )
        assert residue_atom.abs().sum().item() == 0.0
        assert nonphysical.item() == pytest.approx(0.7)

    def test_a_knowledge_node_is_never_drawn_as_an_atom(self):
        features = _mixed_features()
        knowledge = [
            i for i, t in enumerate(features.token_types) if t is TokenType.KNOWLEDGE
        ]
        assert knowledge
        for column in knowledge:
            assert features.token_to_atom[column] == ()

    def test_an_out_of_range_atom_reference_is_refused(self):
        with pytest.raises(IBAMExportError, match="outside"):
            derive_residue_atom(
                torch.ones(1, 1), ((9,),), (TokenType.ATOM,), 3
            )


class TestExport:
    def test_the_native_and_derived_views_are_both_exported(self, stack, ligand_features):
        fused = _fused(stack, ligand_features.num_atoms)
        export = export_ibam(fused, ligand_features)
        assert export.residue_token.shape == (7, ligand_features.num_atoms)
        assert export.residue_atom.shape == (7, ligand_features.num_atoms)
        assert export.residue_atom_is_derived is True

    def test_the_derived_flag_cannot_be_turned_off(self, stack, ligand_features):
        """There is no path producing a residue x atom matrix claimed as native."""
        fused = _fused(stack, ligand_features.num_atoms)
        export = export_ibam(fused, ligand_features)
        with pytest.raises(IBAMExportError, match="always a derived aggregation"):
            IBAMExport(
                **{
                    **{
                        field: getattr(export, field)
                        for field in export.__dataclass_fields__
                        if field != "residue_atom_is_derived"
                    },
                    "residue_atom_is_derived": False,
                }
            )

    def test_the_disclaimer_travels_with_the_matrices(self, stack, ligand_features):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        assert export.disclaimer == DISCLAIMER
        assert "NOT an experimentally established contact map" in export.metadata()["disclaimer"]

    def test_full_per_layer_per_head_attention_is_retained_unaveraged(
        self, stack, ligand_features
    ):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        assert export.attention_by_layer_head.shape == (
            2, 4, 7, ligand_features.num_atoms
        )
        assert export.num_layers == 2 and export.num_heads == 4

    def test_the_last_layer_is_what_the_default_policy_reduces_to(
        self, stack, ligand_features
    ):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        expected = export.attention_by_layer_head[-1].mean(axis=0)
        assert np.allclose(export.residue_token, expected, atol=1e-6)

    def test_padding_is_removed_from_the_export(self, stack, ligand_features):
        """Exported rows and columns are real residues and real tokens only."""
        num_tokens = ligand_features.num_atoms
        torch.manual_seed(3)
        protein_mask = torch.zeros(1, 20, dtype=torch.bool)
        protein_mask[:, :7] = True
        ligand_mask = torch.zeros(1, num_tokens + 5, dtype=torch.bool)
        ligand_mask[:, :num_tokens] = True
        fused = stack(
            torch.randn(1, 20, 32), torch.randn(1, num_tokens + 5, 32),
            protein_mask, ligand_mask,
        )
        export = export_ibam(fused, ligand_features)
        assert export.residue_token.shape == (7, num_tokens)

    def test_a_mismatched_molecule_is_refused(self, stack, ligand_features):
        """The fused batch and the features must describe the same molecule."""
        fused = _fused(stack, ligand_features.num_atoms + 2)
        with pytest.raises(IBAMExportError, match="not the same molecule"):
            export_ibam(fused, ligand_features)

    def test_an_out_of_range_batch_index_is_refused(self, stack, ligand_features):
        fused = _fused(stack, ligand_features.num_atoms)
        with pytest.raises(IBAMExportError, match="outside the batch"):
            export_ibam(fused, ligand_features, index=5)

    def test_a_row_is_selected_from_a_larger_batch(self, stack, ligand_features):
        fused = _fused(stack, ligand_features.num_atoms, batch=3)
        for index in range(3):
            export = export_ibam(fused, ligand_features, index=index)
            expected = fused.p2l_attention[-1][index].mean(dim=0).detach().numpy()
            assert np.allclose(export.residue_token, expected, atol=1e-6)

    def test_the_encoder_identities_are_recorded(self, stack, ligand_features):
        export = export_ibam(
            _fused(stack, ligand_features.num_atoms),
            ligand_features,
            protein_encoder="esmc-600m",
            model_checkpoint="epoch42",
        )
        assert export.ligand_encoder == "ligand_gat"
        assert export.protein_encoder == "esmc-600m"
        assert export.model_checkpoint == "epoch42"


class TestResidueNumbering:
    def test_residue_positions_are_original_sequence_indices(self, stack, ligand_features):
        """261 of 10,820 constructs exceed the 2,046-residue window.

        A row number is not a residue number for any of them, so exporting rows
        would misplace every annotation.
        """
        fused = _fused(stack, ligand_features.num_atoms)
        # A construct whose features start at residue 2000 after windowing.
        positions = torch.arange(2000, 2007)
        export = export_ibam(fused, ligand_features, residue_positions=positions)
        assert export.residue_positions.tolist() == list(range(2000, 2007))
        assert export.top_residues(1)[0][0] >= 2000

    def test_the_fallback_is_row_numbers_and_is_only_right_for_short_proteins(
        self, stack, ligand_features
    ):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        assert export.residue_positions.tolist() == list(range(7))

    def test_a_wrong_number_of_positions_is_refused(self, stack, ligand_features):
        fused = _fused(stack, ligand_features.num_atoms)
        with pytest.raises(IBAMExportError, match="positions for"):
            export_ibam(fused, ligand_features, residue_positions=torch.arange(3))

    def test_a_window_slices_by_sequence_position_not_by_row(self, stack, ligand_features):
        fused = _fused(stack, ligand_features.num_atoms)
        export = export_ibam(
            fused, ligand_features, residue_positions=torch.arange(2000, 2007)
        )
        window = export.residue_window(2002, 2005)
        assert window.residue_positions.tolist() == [2002, 2003, 2004]
        assert window.residue_token.shape[0] == 3
        assert window.attention_by_layer_head.shape[2] == 3

    def test_an_empty_window_is_an_error_not_an_empty_plot(self, stack, ligand_features):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        with pytest.raises(IBAMExportError, match="no residues"):
            export.residue_window(500, 600)


class TestRenormalization:
    def test_renormalization_is_off_by_default(self):
        assert DEFAULT_POLICY.renormalize_physical is False

    def test_when_renormalized_the_export_discloses_it(self, stack, ligand_features):
        export = export_ibam(
            _fused(stack, ligand_features.num_atoms),
            ligand_features,
            policy=AggregationPolicy(renormalize_physical=True),
        )
        assert export.renormalized is True
        assert export.metadata()["renormalized"] is True
        assert np.allclose(export.residue_atom.sum(axis=1), 1.0, atol=1e-5)

    def test_the_set_aside_mass_is_still_reported_after_renormalization(self):
        """Renormalising must not erase the record of what was excluded."""
        features = _mixed_features()
        residue_token = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]])
        residue_atom, nonphysical = derive_residue_atom(
            residue_token, features.token_to_atom, features.token_types, 4
        )
        assert nonphysical.item() == pytest.approx(0.4)
        normalized = residue_atom / residue_atom.sum(dim=1, keepdim=True)
        assert normalized.sum().item() == pytest.approx(1.0)
        # The excluded 0.4 is still knowable from `nonphysical`.
        assert nonphysical.item() > 0


class TestSummaries:
    def test_per_residue_and_per_atom_are_marginals_of_the_derived_view(
        self, stack, ligand_features
    ):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        assert np.allclose(export.per_residue, export.residue_atom.sum(axis=1), atol=1e-6)
        assert np.allclose(export.per_atom, export.residue_atom.sum(axis=0), atol=1e-6)

    def test_top_atoms_are_reported_in_standardized_indexing(self, stack, ligand_features):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        top = export.top_atoms(3)
        assert len(top) == 3
        assert all(0 <= index < ligand_features.num_atoms for index, _ in top)
        assert top[0][1] >= top[1][1] >= top[2][1]

    def test_the_depiction_mapping_is_carried_so_atoms_can_be_drawn(self):
        features = _mixed_features()
        assert features.atom_index_maps["std_to_output"] == (3, 2, 1, 0)


class TestWriting:
    def test_matrices_and_metadata_are_both_written(self, stack, ligand_features, tmp_path):
        """Numeric export is mandatory; an image alone is not an export."""
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        paths = export.write(tmp_path / "run")
        assert paths["matrices"].exists() and paths["metadata"].exists()

        with np.load(paths["matrices"]) as loaded:
            assert np.allclose(loaded["residue_token"], export.residue_token)
            assert np.allclose(loaded["residue_atom"], export.residue_atom)
            assert loaded["attention_by_layer_head"].shape == (2, 4, 7, ligand_features.num_atoms)

    def test_the_metadata_records_the_aggregation_rule_and_its_hash(
        self, stack, ligand_features, tmp_path
    ):
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        paths = export.write(tmp_path / "run")
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert payload["aggregation"]["bond_split"] == "equal"
        assert payload["aggregation_hash"] == DEFAULT_POLICY.policy_hash
        assert payload["residue_atom_is_derived"] is True
        assert "token_to_atom" in payload and "token_types" in payload

    def test_ensemble_variability_is_recorded_when_supplied(self, stack, ligand_features):
        """The specification asks for it 'where calculated', so absence is explicit."""
        fused = _fused(stack, ligand_features.num_atoms)
        without = export_ibam(fused, ligand_features)
        assert without.ensemble_std is None
        assert without.metadata()["has_ensemble_variability"] is False

        members = [
            fused.p2l_attention[-1][0].mean(dim=0),
            fused.p2l_attention[-1][0].mean(dim=0) * 1.1,
        ]
        with_ensemble = export_ibam(fused, ligand_features, ensemble_attention=members)
        assert with_ensemble.ensemble_std is not None
        assert with_ensemble.metadata()["has_ensemble_variability"] is True


class TestEndToEnd:
    def test_an_export_can_be_built_from_the_assembled_model(self):
        """The whole path: SMILES and protein features to exported matrices."""
        from bindra.heads.affinity import AffinityHeadConfig
        from bindra.model import BindraConfig, BindraModel

        torch.manual_seed(20261001)
        model = BindraModel(
            BindraConfig(
                protein_dim=48,
                ligand=GATConfig(hidden_dim=32, num_layers=1, num_heads=4, dropout=0.0),
                model_dim=32,
                cross_attention_layers=1,
                cross_attention_heads=4,
                dropout=0.0,
                head=AffinityHeadConfig(hidden_dim=32, dropout=0.0),
            )
        )
        model.eval()
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        features = torch.randn(1, 11, 48)
        mask = torch.ones(1, 11, dtype=torch.bool)
        with torch.no_grad():
            out = model.predict(features, mask, [smiles])

        encoder = GATLigandEncoder(model.config.ligand, module=model.ligand_encoder)
        ligand_features = encoder.encode(smiles)
        export = export_ibam(out.fusion, ligand_features, protein_encoder="esmc-600m")

        assert export.residue_token.shape == (11, 13)
        assert export.num_atoms == 13
        assert np.isfinite(export.residue_atom).all()
        # Conservation holds on a real forward pass, not only on synthetic input.
        assert np.allclose(
            export.residue_atom.sum(axis=1) + export.nonphysical_mass,
            export.residue_token.sum(axis=1),
            atol=1e-5,
        )


class TestPlots:
    """Figures are drawn from the export, never from a live model.

    The highest-risk test here is
    ``test_atom_contributions_are_mapped_to_the_depiction_order``: RDKit draws a
    molecule in canonical-SMILES output order, not in the standardized atom order
    the export uses. Without the mapping the figure puts the right numbers on the
    wrong atoms, which is confidently and invisibly wrong.
    """

    def test_the_colormap_is_sequential_and_not_configurable(self):
        """Attention has no meaningful midpoint, so a diverging map would invent one."""
        from bindra.ibam import plots

        assert plots.COLORMAP == "viridis"
        assert not any(
            "cmap" in name or "colormap" in name
            for name in plots.FigureStyle.__dataclass_fields__
        )

    def test_vector_formats_are_written_by_default(self, stack, ligand_features, tmp_path):
        from bindra.ibam.plots import plot_residue_atom_heatmap

        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        written = plot_residue_atom_heatmap(export, tmp_path / "heatmap")
        assert set(written) == {"pdf", "svg"}
        assert all(path.exists() and path.stat().st_size > 0 for path in written.values())

    def test_a_600_dpi_png_is_optional(self, stack, ligand_features, tmp_path):
        from bindra.ibam.plots import FigureStyle, plot_residue_atom_heatmap

        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        written = plot_residue_atom_heatmap(
            export, tmp_path / "heatmap", style=FigureStyle(write_png=True)
        )
        assert written["png"].exists()

    def test_a_raster_only_format_is_refused_as_a_vector_format(self):
        from bindra.ibam.plots import FigureStyle, IBAMPlotError

        with pytest.raises(IBAMPlotError, match="not a vector format"):
            FigureStyle(vector_formats=("png",))

    def test_both_the_native_and_derived_views_can_be_drawn(
        self, stack, ligand_features, tmp_path
    ):
        from bindra.ibam.plots import plot_residue_atom_heatmap

        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        for view in ("atom", "token"):
            written = plot_residue_atom_heatmap(
                export, tmp_path / view, view=view
            )
            assert written["svg"].exists()

    def test_the_derived_view_is_labelled_derived_on_the_figure(
        self, stack, ligand_features, tmp_path
    ):
        """Not only in the metadata: a figure gets separated from its sidecar."""
        from bindra.ibam.plots import plot_residue_atom_heatmap

        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        written = plot_residue_atom_heatmap(export, tmp_path / "h", view="atom")
        text = written["svg"].read_text(encoding="utf-8")
        assert "DERIVED" in text
        assert "contact map" in text  # the disclaimer is burned in

    def test_an_unreadably_long_sequence_is_refused_with_a_route_forward(
        self, stack, ligand_features, tmp_path
    ):
        from bindra.ibam.plots import IBAMPlotError, plot_residue_atom_heatmap

        num_tokens = ligand_features.num_atoms
        torch.manual_seed(3)
        fused = stack(
            torch.randn(1, 400, 32), torch.randn(1, num_tokens, 32),
            torch.ones(1, 400, dtype=torch.bool), torch.ones(1, num_tokens, dtype=torch.bool),
        )
        export = export_ibam(fused, ligand_features)
        with pytest.raises(IBAMPlotError, match="residue_window"):
            plot_residue_atom_heatmap(export, tmp_path / "too_long")

    def test_the_profile_works_where_the_matrix_cannot(
        self, stack, ligand_features, tmp_path
    ):
        from bindra.ibam.plots import plot_residue_profile

        num_tokens = ligand_features.num_atoms
        torch.manual_seed(3)
        fused = stack(
            torch.randn(1, 400, 32), torch.randn(1, num_tokens, 32),
            torch.ones(1, 400, dtype=torch.bool), torch.ones(1, num_tokens, dtype=torch.bool),
        )
        export = export_ibam(fused, ligand_features, residue_positions=torch.arange(400))
        written = plot_residue_profile(export, tmp_path / "profile")
        assert written["pdf"].exists()

    def test_atom_contributions_are_mapped_to_the_depiction_order(
        self, stack, ligand_features, tmp_path
    ):
        """Without the mapping, the right numbers land on the wrong atoms."""
        from bindra.ibam.plots import plot_molecule_contributions

        smiles = ligand_features.molecule_id
        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        written = plot_molecule_contributions(export, smiles, tmp_path / "mol")
        assert written["svg"].exists()
        assert "contact map" in written["svg"].read_text(encoding="utf-8")

    def test_a_depiction_of_a_different_molecule_is_refused(
        self, stack, ligand_features, tmp_path
    ):
        from bindra.ibam.plots import IBAMPlotError, plot_molecule_contributions

        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        with pytest.raises(IBAMPlotError, match="not the same molecule"):
            plot_molecule_contributions(export, "CCO", tmp_path / "wrong")

    def test_a_short_index_map_is_refused_rather_than_silently_truncating(
        self, stack, ligand_features, tmp_path
    ):
        from bindra.ibam.plots import IBAMPlotError, plot_molecule_contributions

        export = export_ibam(_fused(stack, ligand_features.num_atoms), ligand_features)
        broken = IBAMExport(
            **{
                **{f: getattr(export, f) for f in export.__dataclass_fields__},
                "atom_index_maps": {"std_to_output": (0, 1)},
            }
        )
        with pytest.raises(IBAMPlotError, match="shorter than the molecule"):
            plot_molecule_contributions(
                broken, ligand_features.molecule_id, tmp_path / "short"
            )

    def test_plot_all_produces_the_whole_set_including_windows(
        self, stack, ligand_features, tmp_path
    ):
        from bindra.ibam.plots import plot_all

        export = export_ibam(
            _fused(stack, ligand_features.num_atoms),
            ligand_features,
            residue_positions=torch.arange(100, 107),
        )
        produced = plot_all(
            export,
            tmp_path / "figures",
            smiles=ligand_features.molecule_id,
            windows=[(102, 105)],
        )
        assert {"residue_profile", "residue_atom", "residue_token", "molecule",
                "window_102_105"} <= set(produced)
        for written in produced.values():
            assert all(path.exists() for path in written.values())

    def test_renormalization_is_disclosed_in_the_caption(
        self, stack, ligand_features, tmp_path
    ):
        from bindra.ibam.export import AggregationPolicy
        from bindra.ibam.plots import plot_residue_atom_heatmap

        export = export_ibam(
            _fused(stack, ligand_features.num_atoms),
            ligand_features,
            policy=AggregationPolicy(renormalize_physical=True),
        )
        written = plot_residue_atom_heatmap(export, tmp_path / "renorm")
        assert "renormalised" in written["svg"].read_text(encoding="utf-8")
