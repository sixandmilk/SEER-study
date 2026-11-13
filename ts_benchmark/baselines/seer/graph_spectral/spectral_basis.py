"""Normalized graph Laplacian, full CPU float64 spectrum and validation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch


@dataclass
class SpectralBasisResult:
    adjacency: np.ndarray
    laplacian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    band_masks: np.ndarray
    band_indices: np.ndarray
    band_offsets: np.ndarray
    validation: Dict[str, Any]


def _connected_components(adjacency: np.ndarray) -> int:
    v = adjacency.shape[0]
    neighbors = adjacency > 0
    visited = np.zeros(v, dtype=bool)
    count = 0
    for start in range(v):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            unseen = np.flatnonzero(neighbors[node] & ~visited)
            if unseen.size:
                visited[unseen] = True
                stack.extend(unseen.tolist())
    return count


def compute_normalized_laplacian(adjacency: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    a = np.asarray(adjacency, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.isfinite(a).all():
        raise ValueError("adjacency contains NaN or Inf")
    symmetry_error = float(np.max(np.abs(a - a.T)))
    diagonal_max = float(np.max(np.abs(np.diag(a))))
    if symmetry_error > 1e-10:
        raise ValueError(f"adjacency is not symmetric; max error={symmetry_error}")
    if diagonal_max > 1e-10:
        raise ValueError(f"adjacency diagonal is not zero; max={diagonal_max}")
    degrees = a.sum(axis=1)
    isolated = np.flatnonzero(degrees <= 0)
    if isolated.size:
        raise ValueError(f"adjacency contains {isolated.size} isolated nodes")
    inv_sqrt = 1.0 / np.sqrt(degrees)
    laplacian = np.eye(a.shape[0], dtype=np.float64) - (
        inv_sqrt[:, None] * a * inv_sqrt[None, :]
    )
    laplacian = (laplacian + laplacian.T) * 0.5
    if not np.isfinite(laplacian).all():
        raise ValueError("laplacian contains NaN or Inf")
    diagnostics = {
        "adjacency_symmetry_error": symmetry_error,
        "adjacency_diagonal_max": diagonal_max,
        "isolated_node_count": int(isolated.size),
        "connected_component_count": _connected_components(a),
        "degree_min": float(degrees.min()),
        "degree_max": float(degrees.max()),
        "laplacian_symmetry_error": float(np.max(np.abs(laplacian - laplacian.T))),
    }
    return laplacian, diagnostics


def make_band_masks(num_modes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the mandated equal-count masks for V=1812.

    The task specification fixes [0,603), [603,1208), [1208,1812) for V=1812.
    For other sizes, the remainder is distributed as evenly as possible.
    """

    if num_modes <= 0:
        raise ValueError("num_modes must be positive")
    if num_modes == 1812:
        ends = (603, 1208, 1812)
    else:
        base, remainder = divmod(num_modes, 3)
        counts = [base, base, base]
        for i in range(remainder):
            counts[i] += 1
        ends = (counts[0], counts[0] + counts[1], num_modes)
    starts = (0, ends[0], ends[1])
    masks = np.zeros((3, num_modes), dtype=np.float32)
    indices = []
    for band, (start, end) in enumerate(zip(starts, ends)):
        if end <= start:
            raise ValueError(f"empty frequency band {band}: [{start},{end})")
        masks[band, start:end] = 1.0
        indices.append(np.arange(start, end, dtype=np.int64))
    concatenated = np.concatenate(indices)
    offsets = np.array([0, len(indices[0]), len(indices[0]) + len(indices[1]), len(concatenated)], dtype=np.int64)
    if not np.array_equal(masks.sum(axis=0), np.ones(num_modes, dtype=np.float32)):
        raise ValueError("frequency masks are not disjoint and exhaustive")
    return masks, concatenated, offsets


def validate_spectral_arrays(
    adjacency: np.ndarray,
    laplacian: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    band_masks: np.ndarray,
    *,
    tolerance: float = 1e-8,
    error_threshold: float = 1e-6,
) -> Dict[str, Any]:
    a = np.asarray(adjacency, dtype=np.float64)
    l = np.asarray(laplacian, dtype=np.float64)
    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    masks = np.asarray(band_masks)
    v = a.shape[0]
    identity = np.eye(v, dtype=np.float64)
    orthogonality_error = float(np.linalg.norm(vectors.T @ vectors - identity, ord="fro"))
    l_norm = float(np.linalg.norm(l, ord="fro"))
    reconstruction_error = float(
        np.linalg.norm(l - vectors @ np.diag(values) @ vectors.T, ord="fro")
        / max(l_norm, np.finfo(np.float64).eps)
    )
    residual_error = float(
        np.linalg.norm(l @ vectors - vectors * values[None, :], ord="fro")
        / max(l_norm, np.finfo(np.float64).eps)
    )
    sorted_error = float(np.max(np.maximum(-np.diff(values), 0.0))) if values.size > 1 else 0.0
    finite_eigenvalues = bool(np.isfinite(values).all())
    finite_eigenvectors = bool(np.isfinite(vectors).all())
    below_zero_count = int(np.count_nonzero(values < -tolerance))
    above_two_count = int(np.count_nonzero(values > 2.0 + tolerance))
    zero_eigenvalue_count = int(np.count_nonzero(np.abs(values) <= tolerance))
    mask_counts = masks.sum(axis=1).astype(int).tolist()
    mask_overlap = int(np.count_nonzero(masks.sum(axis=0) > 1))
    mask_gap = int(np.count_nonzero(masks.sum(axis=0) < 1))
    band_ranges = []
    for mask in masks:
        selected = values[mask.astype(bool)]
        band_ranges.append([float(selected.min()), float(selected.max())])
    return {
        "orthogonality_error_fro": orthogonality_error,
        "reconstruction_relative_error": reconstruction_error,
        "eigen_equation_relative_error": residual_error,
        "eigenvalue_sort_error": sorted_error,
        "eigenvalue_min": float(values.min()),
        "eigenvalue_max": float(values.max()),
        "eigenvalue_below_zero_count": below_zero_count,
        "eigenvalue_above_two_count": above_two_count,
        "eigenvalue_zero_count": zero_eigenvalue_count,
        "band_eigenvalue_ranges": band_ranges,
        "eigenvalues_finite": finite_eigenvalues,
        "eigenvectors_finite": finite_eigenvectors,
        "band_counts": mask_counts,
        "band_overlap_count": mask_overlap,
        "band_gap_count": mask_gap,
        "validation_pass": bool(
            orthogonality_error < error_threshold
            and reconstruction_error < error_threshold
            and residual_error < error_threshold
            and sorted_error == 0.0
            and finite_eigenvalues
            and finite_eigenvectors
            and below_zero_count == 0
            and above_two_count == 0
            and mask_overlap == 0
            and mask_gap == 0
            and all(count > 0 for count in mask_counts)
        ),
    }


def build_spectral_basis(adjacency: np.ndarray) -> SpectralBasisResult:
    """Compute float64 Laplacian/eigh, validate, then expose float32 storage arrays."""

    laplacian_start = time.perf_counter()
    laplacian64, graph_validation = compute_normalized_laplacian(adjacency)
    laplacian_seconds = time.perf_counter() - laplacian_start
    laplacian_tensor = torch.from_numpy(laplacian64).to(dtype=torch.float64, device="cpu")
    eigh_start = time.perf_counter()
    eigenvalues_tensor, eigenvectors_tensor = torch.linalg.eigh(laplacian_tensor)
    eigh_seconds = time.perf_counter() - eigh_start
    eigenvalues64 = eigenvalues_tensor.detach().cpu().numpy()
    eigenvectors64 = eigenvectors_tensor.detach().cpu().numpy()
    band_masks, band_indices, band_offsets = make_band_masks(laplacian64.shape[0])
    spectral_validation = validate_spectral_arrays(
        np.asarray(adjacency, dtype=np.float64),
        laplacian64,
        eigenvalues64,
        eigenvectors64,
        band_masks,
    )
    if not spectral_validation["validation_pass"]:
        raise ValueError(f"Spectral validation failed: {spectral_validation}")
    validation = {**graph_validation, **spectral_validation}
    validation["laplacian_seconds"] = laplacian_seconds
    validation["cpu_float64_eigh_seconds"] = eigh_seconds
    return SpectralBasisResult(
        adjacency=np.asarray(adjacency, dtype=np.float32),
        laplacian=laplacian64.astype(np.float32),
        eigenvalues=eigenvalues64.astype(np.float32),
        eigenvectors=eigenvectors64.astype(np.float32),
        band_masks=band_masks.astype(np.float32),
        band_indices=band_indices,
        band_offsets=band_offsets,
        validation=validation,
    )
