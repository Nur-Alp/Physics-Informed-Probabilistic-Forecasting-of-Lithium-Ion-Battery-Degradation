"""Sanity tests for data_pipeline.metrics. Run: python -m pytest tests -q"""
import numpy as np

from data_pipeline import metrics as M


def test_crps_samples_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    mu, sigma = 0.3, 0.7
    y = np.array([-1.0, 0.3, 0.9, 2.5])
    s = rng.normal(mu, sigma, size=(200_000, 1)).repeat(len(y), axis=1)
    np.testing.assert_allclose(M.crps_samples(y, s), M.crps_gaussian(y, mu, sigma), atol=3e-3)


def test_crps_point_mass_is_absolute_error():
    s = np.full((10, 3), 2.0)
    np.testing.assert_allclose(M.crps_samples(np.array([0.0, 2.0, 5.0]), s), [2.0, 0.0, 3.0])


def test_coverage_gaussian_95():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(4000, 1)).repeat(20_000, axis=1)
    y = rng.normal(size=20_000)
    assert abs(M.coverage(y, s) - 0.95) < 0.01


def test_eol_crossing_interpolates_and_censors():
    x = np.array([0.0, 10.0, 20.0, 30.0])
    s = np.array([[1.0, 0.9, 0.8, 0.7],     # crosses 0.85 at x=15
                  [1.0, 0.99, 0.98, 0.97]])  # never crosses
    e = M.eol_samples(x, s, 0.85)
    assert np.isclose(e[0], 15.0) and np.isinf(e[1])


def test_split_lock_detects_tampering(tmp_path, monkeypatch):
    import json
    from data_pipeline import splits
    rec = {"train": ["a"], "val": ["b"], "test": ["c", "d"],
           "test_sha256": splits._hash(["c", "d"])}
    (tmp_path / "x.json").write_text(json.dumps(rec))
    monkeypatch.setattr(splits, "SPLIT_DIR", tmp_path)
    assert "test" not in splits.get("x")
    assert splits.get("x", final_evaluation=True)["test"] == ["c", "d"]
    rec["test"] = ["c"]
    (tmp_path / "x.json").write_text(json.dumps(rec))
    import pytest
    with pytest.raises(RuntimeError):
        splits.get("x")
