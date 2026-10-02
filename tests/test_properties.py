"""Molecular property store: cached, parameterised, and honest about failures."""

from __future__ import annotations

import numpy as np
import pytest

from bindra.data.properties import (
    PROPERTY_COLUMNS,
    PropertyStore,
    build_property_store,
    compute_properties,
)

MOLECULES = ["CC(=O)Nc1ccc(O)cc1", "CCO", "c1ccc2ccccc2c1CCN", "C[N+](C)(C)C"]


class TestCompute:
    def test_every_declared_column_is_produced(self):
        values, valid = compute_properties(MOLECULES)
        assert values.shape == (len(MOLECULES), len(PROPERTY_COLUMNS))
        assert valid.all()

    def test_values_are_chemically_sensible(self):
        values, _ = compute_properties(["CCO"])
        store = PropertyStore(("ethanol",), values, np.array([True]))
        assert 44 < store.column("molecular_weight")[0] < 48
        assert store.column("heavy_atoms")[0] == 3
        assert store.column("rings")[0] == 0

    def test_formal_charge_is_captured(self):
        values, _ = compute_properties(["C[N+](C)(C)C"])
        store = PropertyStore(("quat",), values, np.array([True]))
        assert store.column("formal_charge")[0] == 1

    def test_an_unparseable_molecule_is_nan_and_invalid_not_an_exception(self):
        values, valid = compute_properties(["CCO", "not-a-molecule", ""])
        assert list(valid) == [True, False, False]
        assert np.isnan(values[1]).all()
        assert np.isnan(values[2]).all()


class TestStore:
    def test_column_lookup_and_index(self):
        values, valid = compute_properties(MOLECULES)
        store = PropertyStore(tuple(f"L{i}" for i in range(len(MOLECULES))), values, valid)
        assert store.index["L2"] == 2
        assert store.column("tpsa").shape == (len(MOLECULES),)

    def test_frame_round_trip_keeps_keys_and_validity(self):
        values, valid = compute_properties(MOLECULES)
        store = PropertyStore(tuple(f"L{i}" for i in range(len(MOLECULES))), values, valid)
        frame = store.to_frame()
        assert list(frame["identity_key"]) == [f"L{i}" for i in range(len(MOLECULES))]
        assert frame["valid"].all()
        for column in PROPERTY_COLUMNS:
            assert column in frame.columns

    def test_meta_reports_counts(self):
        values, valid = compute_properties(MOLECULES + ["bad"])
        store = PropertyStore(tuple(f"L{i}" for i in range(len(MOLECULES) + 1)), values, valid)
        meta = store.to_meta()
        assert meta["n_molecules"] == len(MOLECULES) + 1
        assert meta["n_valid"] == len(MOLECULES)


class TestCaching:
    def _processed(self, tmp_path):
        import pandas as pd

        from bindra.data.tables import write_table

        frame = pd.DataFrame(
            {
                "identity_key": [f"L{i}" for i in range(len(MOLECULES))],
                "standardized_smiles": MOLECULES,
            }
        )
        write_table(frame, tmp_path / "ligands" / "part-00000")
        return tmp_path

    def test_build_then_reload_is_consistent(self, tmp_path):
        processed = self._processed(tmp_path)
        first = build_property_store(processed, workers=1)
        second = build_property_store(processed, workers=1)
        assert first.identity_keys == second.identity_keys
        assert np.allclose(np.asarray(first.values), np.asarray(second.values), equal_nan=True)

    def test_reload_does_not_recompute(self, tmp_path):
        processed = self._processed(tmp_path)
        build_property_store(processed, workers=1)
        mtime = (processed / "properties" / "values.npy").stat().st_mtime_ns
        build_property_store(processed, workers=1)
        assert (processed / "properties" / "values.npy").stat().st_mtime_ns == mtime

    def test_rebuild_forces_recomputation(self, tmp_path):
        processed = self._processed(tmp_path)
        build_property_store(processed, workers=1)
        store = build_property_store(processed, workers=1, rebuild=True)
        assert len(store.identity_keys) == len(MOLECULES)
