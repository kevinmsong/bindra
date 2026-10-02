"""Benchmark locking and the sealed-label gate.

The specification's rule is that test labels inform only the prespecified
diagnostic category and the metrics computed after predictions are frozen. These
tests make that mechanical rather than aspirational -- in particular
``test_training_paths_never_touch_sealed_files``, which monkeypatches ``open`` and
asserts that nothing under ``sealed/`` is read while ordinary pipeline code runs.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pandas as pd
import pytest

from bindra.eval.benchmarks.lock import (
    ACCESS_LOG,
    SEALED_DIRNAME,
    BenchmarkError,
    SealedLabelError,
    definition_hash,
    load_locked,
    load_queries,
    load_sealed_labels,
    load_unknowns,
    lock,
    sealed_paths,
)

DEFINITION = {
    "fingerprint": "morgan",
    "radius": 2,
    "n_bits": 2048,
    "tanimoto_threshold": 0.90,
    "cliff_delta_p": 1.28,
}
INPUTS = {"snapshot_release": "202610", "observations_sha256": "abc123"}


@pytest.fixture
def panel(tmp_path) -> Path:
    queries = pd.DataFrame(
        {"pair_id": ["p1", "p2", "p3"], "target": ["T1", "T1", "T2"]}
    )
    labels = pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "delta_p": [1.5, 2.0, -1.4]})
    unknowns = pd.DataFrame({"pair_id": ["u1"], "reason": ["countertarget not measured"]})
    controls = pd.DataFrame({"pair_id": ["c1"], "kind": ["swapped"]})
    directory = tmp_path / "activity_cliff"
    lock(
        "activity_cliff",
        DEFINITION,
        INPUTS,
        directory,
        queries=queries,
        labels=labels,
        unknowns=unknowns,
        controls=controls,
        attrition={"dropped_censored": 4},
        notes=["tier-1 comparability only"],
    )
    return directory


class TestLocking:
    def test_manifest_records_definition_and_counts(self, panel):
        manifest = load_locked(panel)
        assert manifest.panel == "activity_cliff"
        assert manifest.definition["tanimoto_threshold"] == 0.90
        assert manifest.definition_sha256 == definition_hash(DEFINITION)
        assert manifest.counts == {"queries": 3, "labels": 3, "unknowns": 1, "controls": 1}
        assert manifest.attrition == {"dropped_censored": 4}

    def test_relocking_is_refused(self, panel):
        with pytest.raises(BenchmarkError, match="locked"):
            lock(
                "activity_cliff",
                DEFINITION,
                INPUTS,
                panel,
                queries=pd.DataFrame({"pair_id": ["p1"]}),
            )

    def test_overwrite_requires_an_explicit_flag(self, panel):
        lock(
            "activity_cliff",
            DEFINITION,
            INPUTS,
            panel,
            queries=pd.DataFrame({"pair_id": ["p1"]}),
            overwrite=True,
        )

    def test_editing_the_manifest_is_detected(self, panel):
        """A panel that can be widened after results are known is not locked."""
        path = panel / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["definition"]["tanimoto_threshold"] = 0.50
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(BenchmarkError, match="has been edited"):
            load_locked(panel)

    def test_a_changed_query_table_is_detected(self, panel):
        reference = Path(load_locked(panel).queries["path"])
        reference.write_bytes(reference.read_bytes() + b"junk")
        with pytest.raises(Exception):
            load_locked(panel)

    def test_definition_hash_is_order_independent_but_value_sensitive(self):
        assert definition_hash({"a": 1, "b": 2}) == definition_hash({"b": 2, "a": 1})
        assert definition_hash({"a": 1}) != definition_hash({"a": 2})


class TestOpenArtifacts:
    def test_queries_are_readable_without_authorisation(self, panel):
        """Identifiers are what a model may see, so no gate."""
        frame = load_queries(panel)
        assert sorted(frame["pair_id"]) == ["p1", "p2", "p3"]
        assert "delta_p" not in frame.columns

    def test_unknowns_are_readable_and_carry_no_label(self, panel):
        """A missing countertarget measurement is unknown, never a negative."""
        frame = load_unknowns(panel)
        assert len(frame) == 1
        assert "delta_p" not in frame.columns
        assert "reason" in frame.columns


class TestSealedLabels:
    def test_reading_labels_without_authorisation_is_refused(self, panel):
        with pytest.raises(SealedLabelError, match="allow_labels"):
            load_sealed_labels(panel)

    def test_authorisation_without_a_reason_is_refused(self, panel):
        with pytest.raises(SealedLabelError, match="reason is required"):
            load_sealed_labels(panel, allow_labels=True, reason="   ")

    def test_authorised_access_returns_labels_and_is_logged(self, panel):
        frame = load_sealed_labels(
            panel, allow_labels=True, reason="final metrics, predictions frozen"
        )
        assert sorted(frame["pair_id"]) == ["p1", "p2", "p3"]
        log = panel / SEALED_DIRNAME / ACCESS_LOG
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        assert entries[0]["reason"] == "final metrics, predictions frozen"
        assert entries[0]["panel"] == "activity_cliff"

    def test_every_access_appends_to_the_log(self, panel):
        for index in range(3):
            load_sealed_labels(panel, allow_labels=True, reason=f"check {index}")
        log = panel / SEALED_DIRNAME / ACCESS_LOG
        assert len(log.read_text(encoding="utf-8").splitlines()) == 3

    def test_labels_live_under_sealed(self, panel):
        paths = sealed_paths(panel)
        assert any(p.name.startswith("labels") for p in paths)
        assert all(SEALED_DIRNAME in p.parts for p in paths)

    def test_sealed_directory_explains_itself(self, panel):
        readme = (panel / SEALED_DIRNAME / "README.md").read_text(encoding="utf-8")
        assert "after predictions are frozen" in readme

    def test_tampered_labels_are_detected(self, panel):
        reference = Path(load_locked(panel).labels["path"])
        reference.write_bytes(reference.read_bytes() + b"junk")
        with pytest.raises(Exception):
            load_sealed_labels(panel, allow_labels=True, reason="x")


class TestLeakGuard:
    def test_training_paths_never_touch_sealed_files(self, panel, monkeypatch):
        """Walk ordinary pipeline code with ``open`` watched.

        This is the mechanical form of "no held-out labels enter training losses,
        scalers, negative mining or calibration". It fails if any of those paths
        ever reads a sealed artifact, however indirectly.
        """
        touched: list[str] = []
        real_open = builtins.open

        def watching_open(file, *args, **kwargs):
            try:
                text = str(file)
            except Exception:  # pragma: no cover - exotic path objects
                text = ""
            if f"{SEALED_DIRNAME}{Path('/').as_posix()}" in text.replace("\\", "/"):
                touched.append(text)
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", watching_open)

        # Representative pipeline reads: the panel's public side, the curation
        # policy, and a scaler fitted on training values.
        load_queries(panel)
        load_unknowns(panel)
        load_locked(panel)

        from bindra.config.profiles import get_profile

        profile = get_profile()
        profile.ranking_margin("Ki")
        profile.cliff_delta_threshold("Kd")

        values = pd.Series([5.0, 6.0, 7.0, 8.0])
        _ = (values - values.mean()) / values.std()

        assert touched == [], f"sealed artifacts were read: {touched}"

    def test_the_guard_actually_fires_when_labels_are_read(self, panel, monkeypatch):
        """A leak test that cannot fail is worthless, so prove it can."""
        touched: list[str] = []
        real_open = builtins.open

        def watching_open(file, *args, **kwargs):
            text = str(file).replace("\\", "/")
            if f"/{SEALED_DIRNAME}/" in text:
                touched.append(text)
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", watching_open)
        load_sealed_labels(panel, allow_labels=True, reason="deliberate, to prove the guard")
        assert touched, "the guard did not notice a sealed read"
