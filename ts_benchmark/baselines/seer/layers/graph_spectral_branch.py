"""Trainable graph-spectral branch for SEER patch embeddings.

The graph basis is an offline artifact.  It is loaded once during model
construction, kept as non-parameter buffers, and moved with the module when
``.to(device)`` is called.  The branch itself only learns the input/output
projections, frequency router, and the fusion gate.

This module borrows xCPD's Channel-Patch graph/spectral perspective, but it is
not an implementation of xCPD's full method.  In particular, the first SEER
version uses a fixed cached basis and fixed Low/Mid/High masks, a dense soft
frequency router, and no ego-graph construction, DyMoE edge filtering, or GNN
message passing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ts_benchmark.baselines.seer.graph_spectral.spectral_cache import (
    cache_arrays_to_torch,
    load_spectral_cache,
)


class GraphSpectralBranch(nn.Module):
    """Apply fixed Channel-Patch spectral filters to embedded patch tokens.

    Input and output use node-major layout ``[B, V, D]`` where
    ``V = channel_count * patch_count`` and node index is
    ``channel_index * patch_count + patch_position``.
    """

    def __init__(
        self,
        d_model: int,
        graph_hidden_dim: int,
        *,
        channel_count: int,
        patch_count: int,
        seq_len: int,
        patch_len: int,
        stride: int,
        graph_k: int,
        cache_dir: str | Path,
        data_path: Optional[str | Path] = None,
        channel_order: Optional[list[str]] = None,
        target_channel: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        if graph_hidden_dim <= 0:
            raise ValueError("graph_hidden_dim must be positive")
        self.d_model = int(d_model)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.channel_count = int(channel_count)
        self.patch_count = int(patch_count)
        self.node_count = self.channel_count * self.patch_count

        expected_config = {
            "training": {
                "seq_len": int(seq_len),
                "patch_len": int(patch_len),
                "stride": int(stride),
            },
            "graph": {
                "k": int(graph_k),
                "C": self.channel_count,
                "N": self.patch_count,
                "V": self.node_count,
                "similarity": "cosine_similarity",
                "edge_weight": "max(cosine_similarity, 0)",
                "knn": "union_kNN_symmetric_max",
                "self_loops": False,
                "node_definition": "(channel_index, patch_position)",
                "node_index": "channel_index * N + patch_position",
            },
            "laplacian": "I - D^(-1/2) A D^(-1/2)",
            "frequency_bands": {"mode": "sorted_eigenvalue_count"},
            "descriptor": {
                "descriptor_names": [
                    "mean",
                    "std",
                    "rms",
                    "linear_slope",
                    "quantile_25",
                    "quantile_50",
                    "quantile_75",
                ],
                "patch_count": self.patch_count,
            },
            "validation": {"validation_pass": True},
        }
        if channel_order is not None:
            expected_config["training"]["model_channel_order"] = list(channel_order)
            expected_config["training"]["channel_names"] = list(channel_order)
            expected_config["channel_names"] = list(channel_order)
        if target_channel is not None:
            expected_config["training"]["target_channel"] = list(target_channel)
        cache = load_spectral_cache(
            cache_dir,
            expected_config=expected_config,
            data_path=data_path,
        )
        cache_tensors = cache_arrays_to_torch(cache, device="cpu")
        if channel_order is not None:
            self._validate_node_mapping(
                cache["metadata"].get("node_mapping", []), list(channel_order)
            )
        eigenvectors = (
            cache_tensors["eigenvectors"].detach().to(dtype=torch.float32).clone()
        )
        eigenvalues = (
            cache_tensors["eigenvalues"].detach().to(dtype=torch.float32).clone()
        )
        band_masks = (
            cache_tensors["band_masks"].detach().to(dtype=torch.float32).clone()
        )
        if eigenvectors.shape != (self.node_count, self.node_count):
            raise ValueError(
                "Spectral eigenvectors shape mismatch: "
                f"expected {(self.node_count, self.node_count)}, got {tuple(eigenvectors.shape)}"
            )
        if eigenvalues.shape != (self.node_count,):
            raise ValueError(
                f"Spectral eigenvalues shape mismatch: expected {(self.node_count,)}, "
                f"got {tuple(eigenvalues.shape)}"
            )
        if band_masks.shape != (3, self.node_count):
            raise ValueError(
                f"Frequency band mask shape mismatch: expected {(3, self.node_count)}, "
                f"got {tuple(band_masks.shape)}"
            )
        if not torch.isfinite(eigenvectors).all() or not torch.isfinite(eigenvalues).all():
            raise ValueError("Spectral basis contains NaN or Inf")
        if not torch.allclose(
            band_masks.sum(dim=0),
            torch.ones(self.node_count, dtype=band_masks.dtype),
        ):
            raise ValueError("Frequency bands must cover every spectral mode exactly once")

        # These are deliberately buffers rather than Parameters.  persistent=False
        # avoids copying the large fixed basis into every model checkpoint while
        # retaining automatic device movement with the module.
        self.register_buffer("eigenvectors", eigenvectors, persistent=False)
        self.register_buffer("eigenvalues", eigenvalues, persistent=False)
        self.register_buffer("band_masks", band_masks, persistent=False)

        self.input_projection = nn.Linear(self.d_model, self.graph_hidden_dim)
        self.router = nn.Linear(self.d_model, 3)
        self.output_projection = nn.Linear(self.graph_hidden_dim, self.d_model)

    def _validate_node_mapping(self, mapping, channel_order: list[str]) -> None:
        if len(channel_order) != self.channel_count:
            raise ValueError(
                f"channel_order has {len(channel_order)} entries, expected {self.channel_count}"
            )
        if len(mapping) != self.node_count:
            raise ValueError(
                f"Cache node_mapping has {len(mapping)} entries, expected {self.node_count}"
            )
        for expected_node, item in enumerate(mapping):
            channel_index, patch_position = divmod(expected_node, self.patch_count)
            expected_name = channel_order[channel_index]
            actual = (
                int(item.get("node_index", -1)),
                int(item.get("channel_index", -1)),
                int(item.get("patch_position", -1)),
                str(item.get("channel_name", "")),
            )
            expected = (expected_node, channel_index, patch_position, expected_name)
            if actual != expected:
                raise ValueError(
                    "Cache node_mapping does not match model input order at "
                    f"node {expected_node}: expected {expected}, got {actual}"
                )

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        if h0.ndim != 3:
            raise ValueError(f"GraphSpectralBranch expects [B,V,D], got {tuple(h0.shape)}")
        _, node_count, dim = h0.shape
        if node_count != self.node_count or dim != self.d_model:
            raise ValueError(
                "GraphSpectralBranch input shape mismatch: "
                f"expected [B,{self.node_count},{self.d_model}], got {tuple(h0.shape)}"
            )

        z = self.input_projection(h0)  # [B,V,Hg]
        # U is stored node-by-mode, so U^T Z projects nodes to graph modes.
        z_spec = torch.einsum("vm,bvh->bmh", self.eigenvectors, z)
        band_spec = z_spec.unsqueeze(1) * self.band_masks.unsqueeze(0).unsqueeze(-1)
        # Return each filtered band to node space: [B,V,3,Hg].
        band_nodes = torch.einsum("vm,bkmh->bvkh", self.eigenvectors, band_spec)
        router_weights = torch.softmax(self.router(h0), dim=-1)  # [B,V,3]
        z_graph = (band_nodes * router_weights.unsqueeze(-1)).sum(dim=2)
        return self.output_projection(z_graph)


class GatedResidualFusion(nn.Module):
    """Fuse the original SEER representation with the graph representation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.graph_projection = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h_seer: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        if h_seer.shape != h_graph.shape:
            raise ValueError(
                f"Fusion inputs must have identical shapes, got {tuple(h_seer.shape)} "
                f"and {tuple(h_graph.shape)}"
            )
        gate = torch.sigmoid(self.gate(torch.cat([h_seer, h_graph], dim=-1)))
        return h_seer + gate * self.graph_projection(h_graph)

