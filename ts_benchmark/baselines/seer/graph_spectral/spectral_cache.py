"""Atomic cache persistence and configuration validation."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


CACHE_VERSION = 1


class CacheInvalidError(RuntimeError):
    """Raised when a cache is missing or does not match the requested config."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_npz_atomic(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        actual_name = temp_name if temp_name.endswith(".npz") else f"{temp_name}.npz"
        os.replace(actual_name, path)
    finally:
        for candidate in (temp_name, f"{temp_name}.npz"):
            if os.path.exists(candidate):
                os.unlink(candidate)


def _file_fingerprint(path: str) -> Dict[str, Any]:
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def save_spectral_cache(
    cache_dir: str | Path,
    *,
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    """Save arrays and metadata with per-file atomic replacement.

    ``metadata.json`` is written last and acts as the committed cache marker.
    """

    target = Path(cache_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.setdefault("cache_version", CACHE_VERSION)
    metadata.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    metadata.setdefault("python_version", platform.python_version())
    metadata.setdefault("numpy_version", np.__version__)
    metadata.setdefault("torch_version", torch.__version__)
    metadata.setdefault("compute_dtype", "float64")
    metadata.setdefault("storage_dtype", "float32")
    if "data_path" in metadata:
        metadata["data_fingerprint"] = _file_fingerprint(metadata["data_path"])
    _write_npz_atomic(target / "arrays.npz", arrays)
    _write_json_atomic(target / "metadata.json", metadata)


def _compare(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict):
        mismatches = []
        if not isinstance(actual, dict):
            return [f"{prefix}: expected object, got {type(actual).__name__}"]
        for key, value in expected.items():
            if key not in actual:
                mismatches.append(f"{prefix}{key}: missing")
            else:
                mismatches.extend(_compare(value, actual[key], f"{prefix}{key}."))
        return mismatches
    if isinstance(expected, list):
        return [] if expected == actual else [f"{prefix[:-1]}: expected {expected!r}, got {actual!r}"]
    return [] if expected == actual else [f"{prefix[:-1]}: expected {expected!r}, got {actual!r}"]


_HORIZON_ONLY_PATHS = {
    "graph_config.training.pred_len",
    "graph_config.training.num_training_windows",
    "graph_config.descriptor.training_window_count",
}


def _compare_structural(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    """Compare structural metadata while ignoring horizon-dependent statistics.

    The expected object normally contains ``graph_config`` plus optional
    top-level metadata fields such as ``node_mapping``.  The three ignored
    paths are deliberately retained in ``metadata.json`` as provenance: the
    graph descriptors were built from the cache's original horizon (WF151 is
    currently ``pred_len=8``), but the resulting fixed graph/spectral basis is
    intentionally shared by other prediction horizons.
    """

    if isinstance(expected, dict):
        mismatches = []
        if not isinstance(actual, dict):
            return [f"{prefix}: expected object, got {type(actual).__name__}"]
        for key, value in expected.items():
            path = f"{prefix}{key}".rstrip(".")
            if path in _HORIZON_ONLY_PATHS:
                continue
            if key not in actual:
                mismatches.append(f"{prefix}{key}: missing")
            else:
                mismatches.extend(_compare_structural(value, actual[key], f"{prefix}{key}."))
        return mismatches
    if isinstance(expected, list):
        return [] if expected == actual else [f"{prefix[:-1]}: expected {expected!r}, got {actual!r}"]
    return [] if expected == actual else [f"{prefix[:-1]}: expected {expected!r}, got {actual!r}"]


def _compare_structural_config(expected: Dict[str, Any], metadata: Dict[str, Any]) -> list[str]:
    """Compare a horizon-compatible structural configuration.

    ``expected`` may be either a graph-config-shaped mapping (compared with
    ``metadata['graph_config']``) or a wrapper containing ``graph_config`` and
    top-level metadata expectations such as ``node_mapping``.  This keeps the
    existing metadata schema unchanged while allowing node-index mappings to
    be checked as part of structural compatibility.
    """

    if "graph_config" in expected:
        mismatches = _compare_structural(
            expected["graph_config"],
            metadata.get("graph_config", {}),
            "graph_config.",
        )
        top_level = {key: value for key, value in expected.items() if key != "graph_config"}
        if top_level:
            mismatches.extend(_compare_structural(top_level, metadata, ""))
        return mismatches
    return _compare_structural(expected, metadata.get("graph_config", {}), "graph_config.")


def _validate_loaded_array_shapes(arrays: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    """Reject caches whose stored tensors do not match the graph dimensions."""

    graph = metadata.get("graph_config", {}).get("graph", {})
    try:
        c = int(graph["C"])
        n = int(graph["N"])
        v = int(graph["V"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheInvalidError("Cache graph dimensions are missing or invalid") from exc
    if c <= 0 or n <= 0 or v != c * n:
        raise CacheInvalidError(f"Cache graph dimensions are inconsistent: C={c}, N={n}, V={v}")
    descriptor_width = len(metadata.get("graph_config", {}).get("descriptor", {}).get("descriptor_names", []))
    expected_shapes = {
        "adjacency": (v, v),
        "laplacian": (v, v),
        "eigenvalues": (v,),
        "eigenvectors": (v, v),
        "band_masks": (3, v),
        "band_indices": (v,),
        "band_offsets": (4,),
    }
    if descriptor_width:
        expected_shapes.update(
            {
                "node_descriptors_raw": (v, descriptor_width),
                "node_descriptors": (v, descriptor_width),
                "descriptor_mean": (descriptor_width,),
                "descriptor_std": (descriptor_width,),
            }
        )
    missing = [name for name in expected_shapes if name not in arrays]
    if missing:
        raise CacheInvalidError(f"Cache arrays are missing required entries: {missing}")
    mismatches = [
        f"{name}: expected shape {shape}, got {arrays[name].shape}"
        for name, shape in expected_shapes.items()
        if tuple(arrays[name].shape) != shape
    ]
    if mismatches:
        raise CacheInvalidError("Cache array shape mismatch:\n- " + "\n- ".join(mismatches))
    non_finite = [name for name, array in arrays.items() if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()]
    if non_finite:
        raise CacheInvalidError(f"Cache arrays contain NaN or Inf: {non_finite}")


def load_spectral_cache(
    cache_dir: str | Path,
    *,
    expected_config: Optional[Dict[str, Any]] = None,
    structural_config: Optional[Dict[str, Any]] = None,
    data_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load a cache after validating its provenance and compatibility.

    ``expected_config`` preserves the original strict behavior and compares
    every supplied graph-config field, including ``pred_len``.  New callers
    that reuse a fixed graph basis across forecasting horizons should pass
    ``structural_config`` instead.  Structural validation still checks the
    dataset (via ``data_path`` and its fingerprint), split boundaries, patch
    geometry, channel/node mapping, graph/Laplacian/descriptor definitions,
    and matrix metadata.  It intentionally ignores only
    ``training.pred_len``, ``training.num_training_windows`` and
    ``descriptor.training_window_count``; those values remain in the original
    metadata as cache-build provenance.  In particular, a basis built from
    the ``pred_len=8`` training-window set can be shared by other horizons by
    design, without rebuilding descriptors, the graph, or the eigendecomposition.
    """
    target = Path(cache_dir)
    metadata_path = target / "metadata.json"
    arrays_path = target / "arrays.npz"
    if not metadata_path.exists() or not arrays_path.exists():
        raise CacheInvalidError(
            f"Spectral cache is missing or incomplete at {target}. "
            "Run with --rebuild to create it."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheInvalidError(f"Unable to read cache metadata: {exc}") from exc
    if metadata.get("cache_version") != CACHE_VERSION:
        raise CacheInvalidError(
            f"Cache version mismatch: {metadata.get('cache_version')} != {CACHE_VERSION}. "
            "Run with --rebuild."
        )
    if data_path is not None:
        stored_fingerprint = metadata.get("data_fingerprint")
        if stored_fingerprint is None:
            raise CacheInvalidError(
                "Cache has no data fingerprint; refusing to load it. "
                "Run with --rebuild."
            )
        current_fingerprint = _file_fingerprint(str(data_path))
        if stored_fingerprint != current_fingerprint:
            raise CacheInvalidError(
                "Dataset fingerprint mismatch; stale cache was not loaded:\n"
                f"stored={stored_fingerprint}\ncurrent={current_fingerprint}\n"
                "Run with --rebuild to explicitly rebuild it."
            )
    if expected_config is not None:
        mismatches = _compare(expected_config, metadata.get("graph_config", {}))
        if mismatches:
            raise CacheInvalidError(
                "Cache configuration mismatch; stale cache was not loaded:\n- "
                + "\n- ".join(mismatches)
                + "\nRun with --rebuild to explicitly rebuild it."
            )
    if structural_config is not None:
        mismatches = _compare_structural_config(structural_config, metadata)
        if mismatches:
            raise CacheInvalidError(
                "Cache structural configuration mismatch; stale cache was not loaded:\n- "
                + "\n- ".join(mismatches)
                + "\nRun with --rebuild to explicitly rebuild it."
            )
    try:
        with np.load(arrays_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
    except Exception as exc:
        raise CacheInvalidError(f"Unable to read cache arrays: {exc}") from exc
    _validate_loaded_array_shapes(arrays, metadata)
    return {"arrays": arrays, "metadata": metadata}


def cache_arrays_to_torch(cache: Dict[str, Any], device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Move cache arrays to a device without registering model parameters."""

    arrays = cache["arrays"]
    result: Dict[str, torch.Tensor] = {}
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.floating):
            tensor = torch.from_numpy(array.astype(np.float32, copy=False))
        else:
            tensor = torch.from_numpy(array)
        result[name] = tensor.to(device=device)
    return result
