"""Build, cache and validate the offline SEER shared graph spectral basis.

This script never imports or runs the SEER model.  Example:

    python scripts/validate_graph_spectral.py \
      --data-path dataset/forecasting/WF151.csv \
      --cache-dir checkpoints/graph_spectral/WF151 \
      --pred-len 8 --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.baselines.seer.graph_spectral.graph_builder import (
    GraphConstructionError,
    build_union_knn_graph,
    extract_node_descriptors,
    load_training_values,
)
from ts_benchmark.baselines.seer.graph_spectral.spectral_basis import (
    build_spectral_basis,
    validate_spectral_arrays,
)
from ts_benchmark.baselines.seer.graph_spectral.spectral_cache import (
    CacheInvalidError,
    cache_arrays_to_torch,
    load_spectral_cache,
    save_spectral_cache,
)


class PeakMemoryMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except Exception:
            self._process = None
        self._windows_process = None
        if self._process is None and os.name == "nt":
            try:
                import ctypes

                class _Counters(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                psapi = ctypes.WinDLL("psapi", use_last_error=True)
                get_current = kernel32.GetCurrentProcess
                get_current.restype = ctypes.c_void_p
                get_info = psapi.GetProcessMemoryInfo
                get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
                get_info.restype = ctypes.c_int
                self._windows_process = (ctypes, _Counters, get_current, get_info)
            except Exception:
                self._windows_process = None

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(self.interval)

    def _rss(self):
        if self._process is not None:
            return int(self._process.memory_info().rss)
        if self._windows_process is not None:
            ctypes, counters_type, get_current, get_info = self._windows_process
            counters = counters_type()
            counters.cb = ctypes.sizeof(counters)
            if get_info(get_current(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        return 0

    def __enter__(self):
        self.peak = self._rss()
        if self._process is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._rss())


def _default_project_path(relative: str) -> str:
    return str((Path(__file__).resolve().parents[1] / relative).resolve())


def _build_config(args, training, descriptor_meta, graph_diag, basis) -> Dict[str, Any]:
    path = Path(training.data_path)
    return {
        "dataset_name": path.name,
        "data_path": str(path),
        "training": {
            **training.config,
            "data_length": training.data_length,
            "train_valid_end": training.train_valid_end,
            "train_end": training.train_end,
            "validation_start": training.validation_start,
            "validation_end": training.validation_end,
            "test_start": training.test_start,
            "test_end": training.test_end,
            "num_training_windows": training.num_training_windows,
            "channel_names": training.channel_names,
            "original_channel_indices": training.original_channel_indices,
        },
        "graph": {
            "k": int(args.k),
            "similarity": "cosine_similarity",
            "edge_weight": "max(cosine_similarity, 0)",
            "knn": "union_kNN_symmetric_max",
            "self_loops": False,
            "node_definition": "(channel_index, patch_position)",
            "node_index": "channel_index * N + patch_position",
            "C": len(training.channel_names),
            "N": int(descriptor_meta["patch_count"]),
            "V": len(training.channel_names) * int(descriptor_meta["patch_count"]),
            "diagnostics": graph_diag,
        },
        "laplacian": "I - D^(-1/2) A D^(-1/2)",
        "frequency_bands": {
            "mode": "sorted_eigenvalue_count",
            "ranges": [[0, 603], [603, 1208], [1208, 1812]]
            if basis.eigenvalues.shape[0] == 1812
            else None,
        },
        "descriptor": descriptor_meta,
        "validation": basis.validation,
        "channel_names": training.channel_names,
    }


def _arrays_for_cache(training, raw, standardized, node_mapping, descriptor_meta, adjacency, basis):
    return {
        "adjacency": basis.adjacency,
        "laplacian": basis.laplacian,
        "node_descriptors_raw": raw.astype(np.float32),
        "node_descriptors": standardized.astype(np.float32),
        "descriptor_mean": np.asarray(descriptor_meta["descriptor_mean"], dtype=np.float32),
        "descriptor_std": np.asarray(descriptor_meta["descriptor_std"], dtype=np.float32),
        "eigenvalues": basis.eigenvalues,
        "eigenvectors": basis.eigenvectors,
        "band_masks": basis.band_masks,
        "band_indices": basis.band_indices,
        "band_offsets": basis.band_offsets,
    }, node_mapping


def _validate_cached(cache: Dict[str, Any]) -> Dict[str, Any]:
    a = cache["arrays"]
    validation = validate_spectral_arrays(
        a["adjacency"],
        a["laplacian"],
        a["eigenvalues"],
        a["eigenvectors"],
        a["band_masks"],
        tolerance=1e-5,
        error_threshold=1e-4,
    )
    return validation


def _cache_shape_dtype_report(cache: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in cache["arrays"].items()
    }


def _frequency_ranges(num_modes: int) -> list[list[int]]:
    if num_modes == 1812:
        return [[0, 603], [603, 1208], [1208, 1812]]
    base, remainder = divmod(num_modes, 3)
    counts = [base + (1 if i < remainder else 0) for i in range(3)]
    return [[0, counts[0]], [counts[0], counts[0] + counts[1]], [counts[0] + counts[1], num_modes]]


def _expected_node_mapping(training, patch_count: int) -> list[Dict[str, Any]]:
    mapping = []
    for channel_index, (channel_name, original_index) in enumerate(
        zip(training.channel_names, training.original_channel_indices)
    ):
        for patch_position in range(patch_count):
            mapping.append(
                {
                    "node_index": channel_index * patch_count + patch_position,
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "original_channel_index": original_index,
                    "patch_position": patch_position,
                    "patch_start": patch_position * int(training.config["stride"]),
                }
            )
    return mapping


def _build_structural_config(args, training) -> Dict[str, Any]:
    """Build cache-load expectations independent of prediction horizon.

    ``pred_len`` and the resulting training-window counts are intentionally
    omitted.  The cache metadata retains those original values (currently
    pred_len=8) as provenance, while this fixed basis is shared by all short
    forecast horizons.
    """

    seq_len = int(args.seq_len)
    patch_len = int(args.patch_len)
    stride = int(args.stride)
    patch_count = int(np.floor((seq_len - patch_len) / stride + 2))
    channel_count = len(training.channel_names)
    node_count = channel_count * patch_count
    training_structural = {
        key: value
        for key, value in training.config.items()
        if key != "pred_len"
    }
    training_structural.update(
        {
            "data_length": training.data_length,
            "train_valid_end": training.train_valid_end,
            "train_end": training.train_end,
            "validation_start": training.validation_start,
            "validation_end": training.validation_end,
            "test_start": training.test_start,
            "test_end": training.test_end,
            "channel_names": training.channel_names,
            "original_channel_indices": training.original_channel_indices,
        }
    )
    graph_config = {
        "dataset_name": Path(training.data_path).name,
        "data_path": training.data_path,
        "training": training_structural,
        "graph": {
            "k": int(args.k),
            "similarity": "cosine_similarity",
            "edge_weight": "max(cosine_similarity, 0)",
            "knn": "union_kNN_symmetric_max",
            "self_loops": False,
            "node_definition": "(channel_index, patch_position)",
            "node_index": "channel_index * N + patch_position",
            "C": channel_count,
            "N": patch_count,
            "V": node_count,
        },
        "laplacian": "I - D^(-1/2) A D^(-1/2)",
        "frequency_bands": {
            "mode": "sorted_eigenvalue_count",
            "ranges": _frequency_ranges(node_count),
        },
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
            "patch_count": patch_count,
        },
        "channel_names": training.channel_names,
    }
    return {
        "graph_config": graph_config,
        "node_mapping": _expected_node_mapping(training, patch_count),
    }


def _validate_node_mapping(cache: Dict[str, Any]) -> Dict[str, Any]:
    metadata = cache["metadata"]
    mapping = metadata.get("node_mapping", [])
    graph_config = metadata.get("graph_config", {}).get("graph", {})
    patch_count = int(graph_config.get("N", 0))
    errors = []
    for item in mapping:
        expected = int(item["channel_index"]) * patch_count + int(item["patch_position"])
        if int(item["node_index"]) != expected:
            errors.append({"item": item, "expected": expected})
    return {
        "count": len(mapping),
        "expected_count": int(graph_config.get("V", 0)),
        "valid": len(errors) == 0 and len(mapping) == int(graph_config.get("V", 0)),
        "errors": errors[:5],
    }


def _measure_gpu_load(cache: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is not available"}
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    # Some torch/CUDA combinations reject a torch.device object here even
    # though tensor placement accepts it; the current device is unambiguous.
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tensors = cache_arrays_to_torch(cache, device=device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "available": True,
        "device": torch.cuda.get_device_name(device),
        "load_seconds": elapsed,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "tensor_names": sorted(tensors),
        "tensor_shapes_dtypes": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=_default_project_path("dataset/forecasting/WF151.csv"))
    parser.add_argument("--cache-dir", default=_default_project_path("checkpoints/graph_spectral/WF151"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-test", type=int, default=1440)
    parser.add_argument("--target-channel", type=int, default=-1)
    parser.add_argument("--all-channels", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_channel = None if args.all_channels else [args.target_channel]
    cache_dir = Path(args.cache_dir).resolve()
    print(f"data_path={Path(args.data_path).resolve()}")
    print(f"cache_dir={cache_dir}")
    print(f"torch={torch.__version__}; cuda={torch.cuda.is_available()}")

    if cache_dir.exists() and not args.rebuild:
        cache_load_start = time.perf_counter()
        with PeakMemoryMonitor() as cache_monitor:
            try:
                # Recompute only the data/split/channel provenance needed for
                # structural checks.  Descriptor statistics and the graph
                # basis are loaded from cache; they are not rebuilt here.
                expected_training = load_training_values(
                    args.data_path,
                    seq_len=args.seq_len,
                    patch_len=args.patch_len,
                    stride=args.stride,
                    pred_len=args.pred_len,
                    target_channel=target_channel,
                    num_test=args.num_test,
                    normalize=True,
                )
                cache = load_spectral_cache(
                    cache_dir,
                    structural_config=_build_structural_config(args, expected_training),
                    data_path=args.data_path,
                )
            except CacheInvalidError as exc:
                print(f"CACHE_INVALID: {exc}", file=sys.stderr)
                return 2
            validation = _validate_cached(cache)
        print(
            json.dumps(
                {
                    "mode": "cache_load",
                    "cache_load_seconds": time.perf_counter() - cache_load_start,
                    "cpu_peak_rss_bytes": cache_monitor.peak,
                    "array_shapes_dtypes": _cache_shape_dtype_report(cache),
                    "node_mapping": _validate_node_mapping(cache),
                    "validation": validation,
                },
                indent=2,
            )
        )
        print(json.dumps({"gpu_load": _measure_gpu_load(cache)}, indent=2))
        return 0 if validation["validation_pass"] else 3

    overall_start = time.perf_counter()
    with PeakMemoryMonitor() as monitor:
        stage_start = time.perf_counter()
        training = load_training_values(
            args.data_path,
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            stride=args.stride,
            pred_len=args.pred_len,
            target_channel=target_channel,
            num_test=args.num_test,
            normalize=True,
        )
        load_seconds = time.perf_counter() - stage_start
        print(
            json.dumps(
                {
                    "data": {
                        "data_length": training.data_length,
                        "train_valid_end": training.train_valid_end,
                        "train_end": training.train_end,
                        "validation_range": [training.validation_start, training.validation_end],
                        "test_range": [training.test_start, training.test_end],
                        "channel_count": len(training.channel_names),
                        "first_channels": training.channel_names[:5],
                        "last_channels": training.channel_names[-5:],
                        "num_training_windows": training.num_training_windows,
                        "load_and_normalize_seconds": load_seconds,
                    }
                },
                indent=2,
            )
        )

        stage_start = time.perf_counter()
        raw, standardized, node_mapping, descriptor_meta = extract_node_descriptors(training)
        descriptor_seconds = time.perf_counter() - stage_start
        print(json.dumps({"descriptor_seconds": descriptor_seconds, "descriptor_meta": descriptor_meta}, indent=2))

        stage_start = time.perf_counter()
        try:
            adjacency, graph_diag = build_union_knn_graph(standardized, k=args.k)
        except GraphConstructionError as exc:
            print(f"GRAPH_INVALID: {exc}", file=sys.stderr)
            return 4
        graph_seconds = time.perf_counter() - stage_start
        print(json.dumps({"graph_seconds": graph_seconds, "graph": graph_diag}, indent=2))

        stage_start = time.perf_counter()
        basis = build_spectral_basis(adjacency)
        spectral_stage_seconds = time.perf_counter() - stage_start
        laplacian_seconds = basis.validation["laplacian_seconds"]
        eigh_seconds = basis.validation["cpu_float64_eigh_seconds"]
        print(
            json.dumps(
                {
                    "spectral_seconds_including_validation": spectral_stage_seconds,
                    "laplacian_seconds": laplacian_seconds,
                    "cpu_float64_eigh_seconds": eigh_seconds,
                    "spectral_validation": basis.validation,
                },
                indent=2,
            )
        )

        graph_config = _build_config(args, training, descriptor_meta, graph_diag, basis)
        arrays, _ = _arrays_for_cache(training, raw, standardized, node_mapping, descriptor_meta, adjacency, basis)
        metadata = {
            "cache_version": 1,
            "data_path": training.data_path,
            "graph_config": graph_config,
            "channel_names": training.channel_names,
            "node_mapping": node_mapping,
            "scaler_mean": training.scaler_mean.tolist(),
            "scaler_scale": training.scaler_scale.tolist(),
        }
        stage_start = time.perf_counter()
        save_spectral_cache(cache_dir, arrays=arrays, metadata=metadata)
        save_seconds = time.perf_counter() - stage_start

    cache_bytes = sum(path.stat().st_size for path in cache_dir.iterdir() if path.is_file())
    print(
        json.dumps(
            {
                "timing": {
                    "data_load_and_normalize_seconds": load_seconds,
                    "descriptor_seconds": descriptor_seconds,
                    "graph_seconds": graph_seconds,
                    "laplacian_seconds": laplacian_seconds,
                    "cpu_float64_eigh_seconds": eigh_seconds,
                    "spectral_stage_seconds_including_validation": spectral_stage_seconds,
                    "cache_save_seconds": save_seconds,
                    "total_seconds": time.perf_counter() - overall_start,
                },
                "memory": {"cpu_peak_rss_bytes": monitor.peak},
                "cache": {"path": str(cache_dir), "size_bytes": cache_bytes},
                "gpu_load": _measure_gpu_load({"arrays": arrays, "metadata": metadata}),
                "success": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
