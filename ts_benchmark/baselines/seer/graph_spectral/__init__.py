"""Offline shared Channel-Patch graph spectral utilities for SEER.

This package is intentionally independent from the SEER model.  It builds and
validates an offline graph-spectral basis that can be consumed by a later
Graph Spectral Branch integration.
"""

from .graph_builder import (
    GraphConstructionError,
    TrainingDataInfo,
    build_union_knn_graph,
    extract_node_descriptors,
    load_training_values,
)
from .spectral_basis import (
    SpectralBasisResult,
    build_spectral_basis,
    validate_spectral_arrays,
)
from .spectral_cache import (
    CacheInvalidError,
    cache_arrays_to_torch,
    load_spectral_cache,
    save_spectral_cache,
)

__all__ = [
    "CacheInvalidError",
    "GraphConstructionError",
    "SpectralBasisResult",
    "TrainingDataInfo",
    "build_spectral_basis",
    "build_union_knn_graph",
    "cache_arrays_to_torch",
    "extract_node_descriptors",
    "load_spectral_cache",
    "load_training_values",
    "save_spectral_cache",
    "validate_spectral_arrays",
]
