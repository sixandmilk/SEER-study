"""Read-only compatibility tests for the shared WF151 spectral cache."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.validate_graph_spectral import _build_structural_config
from ts_benchmark.baselines.seer.graph_spectral.graph_builder import load_training_values
from ts_benchmark.baselines.seer.graph_spectral.spectral_basis import validate_spectral_arrays
from ts_benchmark.baselines.seer.graph_spectral.spectral_cache import (
    CacheInvalidError,
    load_spectral_cache,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset/forecasting/WF151.csv"
CACHE_DIR = ROOT / "checkpoints/graph_spectral/WF151"


class SpectralCacheCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = SimpleNamespace(
            seq_len=96,
            patch_len=16,
            stride=8,
            k=10,
        )
        # pred_len is deliberately varied in the tests; this data-derived
        # structural expectation contains no horizon or window-count fields.
        training = load_training_values(
            DATA_PATH,
            seq_len=cls.args.seq_len,
            patch_len=cls.args.patch_len,
            stride=cls.args.stride,
            pred_len=1,
            target_channel=[-1],
            num_test=1440,
            normalize=True,
        )
        cls.structural_config = _build_structural_config(cls.args, training)

    def _load(self, pred_len: int):
        expected = copy.deepcopy(self.structural_config)
        # These are provenance values only and must not affect compatibility.
        expected["graph_config"]["training"]["pred_len"] = pred_len
        expected["graph_config"]["training"]["num_training_windows"] = 1
        expected["graph_config"]["descriptor"]["training_window_count"] = 1
        return load_spectral_cache(
            CACHE_DIR,
            structural_config=expected,
            data_path=DATA_PATH,
        )

    def test_all_short_horizons_share_cache(self):
        for pred_len in (1, 2, 4, 8, 16, 20):
            with self.subTest(pred_len=pred_len):
                cache = self._load(pred_len)
                self.assertEqual(cache["metadata"]["graph_config"]["training"]["pred_len"], 8)
                self.assertEqual(
                    cache["metadata"]["graph_config"]["training"]["num_training_windows"],
                    175384,
                )
                self.assertEqual(
                    cache["metadata"]["graph_config"]["descriptor"]["training_window_count"],
                    175384,
                )

    def test_structural_mismatch_is_rejected(self):
        cases = {
            "seq_len": ("training", "seq_len", 95),
            "patch_len": ("training", "patch_len", 8),
            "stride": ("training", "stride", 4),
            "channel_order": ("training", "model_channel_order", ["wrong"]),
            "split_boundary": ("training", "train_end", 1),
            "k": ("graph", "k", 9),
            "graph_method": ("graph", "knn", "different_graph_method"),
        }
        for name, (section, field, value) in cases.items():
            with self.subTest(case=name):
                expected = copy.deepcopy(self.structural_config)
                expected["graph_config"][section][field] = value
                with self.assertRaises(CacheInvalidError):
                    load_spectral_cache(
                        CACHE_DIR,
                        structural_config=expected,
                        data_path=DATA_PATH,
                    )

    def test_legacy_expected_config_remains_strict(self):
        with self.assertRaises(CacheInvalidError):
            load_spectral_cache(
                CACHE_DIR,
                expected_config={"training": {"pred_len": 1}},
                data_path=DATA_PATH,
            )

    def test_data_fingerprint_mismatch_is_rejected(self):
        fd, temp_name = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(b"different dataset")
            with self.assertRaises(CacheInvalidError):
                load_spectral_cache(
                    CACHE_DIR,
                    structural_config=self.structural_config,
                    data_path=temp_name,
                )
        finally:
            os.unlink(temp_name)

    def test_cached_arrays_shape_and_spectral_validation(self):
        cache = self._load(8)
        arrays = cache["arrays"]
        self.assertEqual(arrays["adjacency"].shape, (1812, 1812))
        self.assertEqual(arrays["laplacian"].shape, (1812, 1812))
        self.assertEqual(arrays["eigenvectors"].shape, (1812, 1812))
        self.assertEqual(arrays["eigenvalues"].shape, (1812,))
        self.assertEqual(arrays["band_masks"].shape, (3, 1812))
        validation = validate_spectral_arrays(
            arrays["adjacency"],
            arrays["laplacian"],
            arrays["eigenvalues"],
            arrays["eigenvectors"],
            arrays["band_masks"],
            tolerance=1e-5,
            error_threshold=1e-4,
        )
        self.assertTrue(validation["validation_pass"])
        self.assertTrue(np.isfinite(arrays["eigenvectors"]).all())

    def test_invalid_cached_array_shape_or_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cache = Path(temp_dir)
            shutil.copy2(CACHE_DIR / "metadata.json", temp_cache / "metadata.json")
            with np.load(CACHE_DIR / "arrays.npz", allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            arrays["eigenvalues"] = arrays["eigenvalues"][:-1]
            np.savez_compressed(temp_cache / "arrays.npz", **arrays)
            with self.assertRaises(CacheInvalidError):
                load_spectral_cache(
                    temp_cache,
                    structural_config=self.structural_config,
                    data_path=DATA_PATH,
                )

            arrays["eigenvalues"] = np.pad(arrays["eigenvalues"], (0, 1), constant_values=0)
            arrays["eigenvalues"][0] = np.nan
            np.savez_compressed(temp_cache / "arrays.npz", **arrays)
            with self.assertRaises(CacheInvalidError):
                load_spectral_cache(
                    temp_cache,
                    structural_config=self.structural_config,
                    data_path=DATA_PATH,
                )


if __name__ == "__main__":
    unittest.main()
