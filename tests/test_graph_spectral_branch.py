"""Integration tests for SEER's fixed-basis Graph Spectral Branch."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ts_benchmark.baselines.seer.graph_spectral.spectral_cache import CacheInvalidError
from ts_benchmark.baselines.seer.layers.Embed import AugmentedEmbedding
from ts_benchmark.baselines.seer.layers.graph_spectral_branch import (
    GatedResidualFusion,
    GraphSpectralBranch,
)
from ts_benchmark.baselines.seer.models.seer_model import SEERModel


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "checkpoints/graph_spectral/WF151"
DATA_PATH = ROOT / "dataset/forecasting/WF151.csv"
METADATA = json.loads((CACHE_DIR / "metadata.json").read_text(encoding="utf-8"))
CHANNEL_ORDER = METADATA["graph_config"]["training"]["model_channel_order"]


def _config(**overrides):
    values = {
        "enc_in": 151,
        "seq_len": 96,
        "pred_len": 2,
        "patch_len": 16,
        "stride": 8,
        "d_model": 16,
        "d_score": 4,
        "d_ff": 32,
        "n_heads": 4,
        "factor": 3,
        "dropout": 0.0,
        "head_dropout": 0.0,
        "activation": "gelu",
        "output_attention": False,
        "e_layers": 1,
        "num_experts": 2,
        "noisy_gating": False,
        "k": 1,
        "CI": True,
        "scaling": 1,
        "score_threshold": 0,
        "graph_spectral_enabled": True,
        "graph_hidden_dim": 4,
        "graph_spectral_cache_dir": str(CACHE_DIR),
        "graph_spectral_data_path": str(DATA_PATH),
        "graph_spectral_target_channel": [-1],
        "model_channel_order": CHANNEL_ORDER,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class GraphSpectralBranchTest(unittest.TestCase):
    def test_branch_shape_gradient_and_fixed_basis(self):
        branch = GraphSpectralBranch(
            16,
            4,
            channel_count=151,
            patch_count=12,
            seq_len=96,
            patch_len=16,
            stride=8,
            cache_dir=CACHE_DIR,
            data_path=DATA_PATH,
            channel_order=CHANNEL_ORDER,
            target_channel=[-1],
        )
        h0 = torch.randn(1, 1812, 16, requires_grad=True)
        output = branch(h0)
        self.assertEqual(output.shape, h0.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(h0.grad).all())
        self.assertFalse(branch.eigenvectors.requires_grad)
        self.assertIsNone(branch.eigenvectors.grad)
        self.assertNotIn("eigenvectors", dict(branch.named_parameters()))
        self.assertTrue(
            torch.equal(branch.band_masks.sum(dim=0), torch.ones(1812))
        )

    def test_channel_order_mismatch_is_rejected(self):
        wrong_order = list(CHANNEL_ORDER)
        wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
        with self.assertRaises(CacheInvalidError):
            GraphSpectralBranch(
                16,
                4,
                channel_count=151,
                patch_count=12,
                seq_len=96,
                patch_len=16,
                stride=8,
                cache_dir=CACHE_DIR,
                data_path=DATA_PATH,
                channel_order=wrong_order,
                target_channel=[-1],
            )

    def test_gated_residual_fusion_shape_and_gradients(self):
        fusion = GatedResidualFusion(16)
        h_seer = torch.randn(2, 10, 16, requires_grad=True)
        h_graph = torch.randn(2, 10, 16, requires_grad=True)
        output = fusion(h_seer, h_graph)
        self.assertEqual(output.shape, h_seer.shape)
        output.mean().backward()
        self.assertIsNotNone(h_seer.grad)
        self.assertIsNotNone(h_graph.grad)

    def test_augmented_embedding_preserves_public_shape(self):
        config = _config()
        embedding = AugmentedEmbedding(
            config,
            config.d_model,
            config.patch_len,
            config.stride,
            config.stride,
            config.dropout,
        ).eval()
        x_enc = torch.randn(1, 151, 96)
        x_mark = torch.randn(1, 96, 4)
        with torch.no_grad():
            output, importance = embedding(x_enc, x_mark)
        self.assertEqual(output.shape, (151, 13, 16))
        self.assertEqual(importance.ndim, 0)

    def test_graph_branch_is_opt_in_for_existing_seer_configs(self):
        config = _config(
            graph_spectral_enabled=False,
            graph_spectral_cache_dir=None,
            graph_spectral_data_path=None,
            model_channel_order=None,
        )
        embedding = AugmentedEmbedding(
            config,
            config.d_model,
            config.patch_len,
            config.stride,
            config.stride,
            config.dropout,
        ).eval()
        self.assertFalse(embedding.graph_spectral_enabled)
        self.assertFalse(hasattr(embedding, "graph_spectral_branch"))
        with torch.no_grad():
            output, importance = embedding(
                torch.randn(1, 151, 96), torch.randn(1, 96, 4)
            )
        self.assertEqual(output.shape, (151, 13, 16))
        self.assertEqual(importance.ndim, 0)

    def test_seer_model_forward_interface_is_unchanged(self):
        model = SEERModel(_config()).eval()
        input_tensor = torch.randn(1, 96, 151)
        input_mark = torch.randn(1, 96, 4)
        with torch.no_grad():
            output, importance = model(input_tensor, input_mark)
        self.assertEqual(output.shape, (1, 2, 151))
        self.assertEqual(importance.ndim, 0)


if __name__ == "__main__":
    unittest.main()
