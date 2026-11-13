"""Training-only Channel-Patch descriptor and graph construction helpers.

The functions in this module deliberately operate outside the SEER model.  The
input follows the benchmark's actual model-facing channel order and its
training-only normalization path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.utils.data_processing import split_channel
from ts_benchmark.utils.data_processing import split_time
from ts_benchmark.data.utils import read_data


class GraphConstructionError(RuntimeError):
    """Raised when the requested graph rules produce an invalid graph."""


@dataclass
class TrainingDataInfo:
    """The exact training data slice used for descriptor extraction."""

    values: np.ndarray
    channel_names: List[str]
    original_channel_indices: List[int]
    data_path: str
    data_length: int
    train_valid_end: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    num_training_windows: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    config: Dict[str, Any]


def _resolve_target_channel(
    target_channel: Optional[Sequence[int]], num_columns: int
) -> Optional[List[int]]:
    if target_channel is None:
        return None
    resolved: List[int] = []
    for item in target_channel:
        if not isinstance(item, (int, np.integer)):
            raise ValueError(
                "This offline builder accepts integer target_channel entries "
                "only; use None for all channels."
            )
        index = int(item)
        if index < 0:
            index += num_columns
        if index < 0 or index >= num_columns:
            raise IndexError(
                f"target_channel index {item} is out of range for {num_columns} columns"
            )
        resolved.append(index)
    return list(dict.fromkeys(resolved))


def load_training_values(
    data_path: str | Path,
    *,
    seq_len: int,
    patch_len: int,
    stride: int,
    pred_len: int,
    target_channel: Optional[Sequence[int]] = (-1,),
    num_test: Optional[int] = 1440,
    tv_ratio: Optional[float] = None,
    train_ratio_in_tv: Optional[float] = None,
    normalize: bool = True,
) -> TrainingDataInfo:
    """Load exactly the model-facing training slice used by the benchmark.

    For the WF151 rolling setup, ``target_channel=[-1]`` selects OT and the
    remaining 150 columns are appended as exogenous variables.  The resulting
    column order is therefore the actual order seen by ``SEERModel``.  The
    StandardScaler is fit only on the training slice, matching
    ``DeepForecastingModelBase.forecast_fit``.
    """

    if seq_len <= 0 or patch_len <= 0 or stride <= 0 or pred_len <= 0:
        raise ValueError("seq_len, patch_len, stride and pred_len must be positive")
    if patch_len > seq_len + stride:
        raise ValueError("patch_len is incompatible with the padded sequence")

    path = Path(data_path).resolve()
    data = read_data(str(path))
    if data.shape[1] == 0:
        raise ValueError(f"No value columns found in {path}")

    resolved_target = _resolve_target_channel(target_channel, data.shape[1])
    target_df, exog_df = split_channel(data, resolved_target)
    model_data = target_df if exog_df is None else pd.concat([target_df, exog_df], axis=1)
    channel_names = [str(name) for name in model_data.columns]
    original_indices = []
    original_name_to_index = {str(name): i for i, name in enumerate(data.columns)}
    for name in channel_names:
        original_indices.append(original_name_to_index[name])

    data_length = int(model_data.shape[0])
    if num_test is not None:
        if isinstance(num_test, bool) or int(num_test) <= 0 or int(num_test) >= data_length:
            raise ValueError("num_test must be positive and smaller than the data length")
        train_valid_end = data_length - int(num_test)
        actual_tv_ratio = (train_valid_end - int(num_test)) / train_valid_end
        train_valid_data, _ = split_time(model_data, train_valid_end)
        # This is the exact policy in RollingForecastStrategy when num_test is set.
        train_ratio = actual_tv_ratio
    else:
        if tv_ratio is None:
            tv_ratio = 1.0
        if not 0 < float(tv_ratio) <= 1:
            raise ValueError("tv_ratio must be in (0, 1]")
        train_valid_end = int(float(tv_ratio) * data_length)
        train_valid_data, _ = split_time(model_data, train_valid_end)
        train_ratio = 1.0 if train_ratio_in_tv is None else float(train_ratio_in_tv)

    if train_ratio_in_tv is not None and num_test is None:
        train_ratio = float(train_ratio_in_tv)
    train_data, valid_data = train_val_split(train_valid_data, train_ratio, seq_len)
    train_end = int(train_data.shape[0])
    validation_start = train_end
    validation_end = int(train_valid_data.shape[0])
    test_start = int(train_valid_end)
    test_end = data_length

    if train_end < seq_len + pred_len:
        raise ValueError(
            "The training slice is too short for the requested seq_len and pred_len"
        )

    scaler = StandardScaler()
    if normalize:
        values = scaler.fit_transform(train_data.values)
    else:
        values = train_data.values.astype(np.float64, copy=True)
        scaler.fit(train_data.values)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Training values contain NaN or Inf after preprocessing")

    num_training_windows = train_end - seq_len - pred_len + 1
    config = {
        "seq_len": int(seq_len),
        "patch_len": int(patch_len),
        "stride": int(stride),
        "pred_len": int(pred_len),
        "target_channel": None if target_channel is None else list(target_channel),
        "target_channel_resolved": resolved_target,
        "normalize": bool(normalize),
        "num_test": None if num_test is None else int(num_test),
        "tv_ratio": None if tv_ratio is None else float(tv_ratio),
        "train_ratio_in_tv": float(train_ratio),
        "model_channel_order": channel_names,
    }
    return TrainingDataInfo(
        values=values,
        channel_names=channel_names,
        original_channel_indices=original_indices,
        data_path=str(path),
        data_length=data_length,
        train_valid_end=train_valid_end,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        num_training_windows=num_training_windows,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        config=config,
    )


def _padded_patch_values(
    values: np.ndarray,
    *,
    starts: np.ndarray,
    seq_len: int,
    patch_position: int,
    patch_len: int,
    stride: int,
) -> np.ndarray:
    """Return model-equivalent replicated-right-padded patches.

    ``values`` is one channel of the normalized training series.  ``starts``
    indexes the model input windows, not arbitrary rows.
    """

    patch_start = patch_position * stride
    window_count = starts.size
    # All windows have equal length; materialize only this channel/patch.
    window = np.empty((window_count, patch_len), dtype=np.float64)
    for offset in range(patch_len):
        position = patch_start + offset
        if position < seq_len:
            window[:, offset] = values[starts + position]
        else:
            # ReplicationPad1d((0, stride)) repeats the last sample of each input window.
            window[:, offset] = values[starts + seq_len - 1]
    return window


def extract_node_descriptors(
    training: TrainingDataInfo,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Extract raw and column-standardized ``Z:[V,7]`` node descriptors.

    Statistics are computed from model input windows in the training slice only.
    Mean/std/RMS/quantiles aggregate all values in each node's ``[window, P]``
    collection.  Linear slope is computed per window over the patch positions
    and then averaged over training windows.
    """

    values = training.values
    seq_len = int(training.config["seq_len"])
    patch_len = int(training.config["patch_len"])
    stride = int(training.config["stride"])
    pred_len = int(training.config["pred_len"])
    channels = values.shape[1]
    patch_count = int(math.floor((seq_len - patch_len) / stride + 2))
    if patch_count <= 0:
        raise ValueError("Computed patch count must be positive")
    if training.num_training_windows <= 0:
        raise ValueError("No training windows are available")

    starts = np.arange(training.num_training_windows, dtype=np.int64)
    raw = np.empty((channels * patch_count, 7), dtype=np.float64)
    node_mapping: List[Dict[str, Any]] = []
    time = np.arange(patch_len, dtype=np.float64)
    time_centered = time - time.mean()
    slope_denominator = float(np.square(time_centered).sum())
    node_index = 0

    for channel_index in range(channels):
        channel_values = values[:, channel_index]
        # One sliding-window view per channel lets NumPy calculate all N patch
        # positions together.  The final stride positions exactly reproduce
        # ReplicationPad1d((0, stride)) for every training window.
        windows = np.lib.stride_tricks.sliding_window_view(channel_values, seq_len)[
            : training.num_training_windows
        ]
        padded = np.concatenate(
            [windows, np.repeat(windows[:, -1:], stride, axis=1)], axis=1
        )
        patches = np.lib.stride_tricks.sliding_window_view(
            padded, patch_len, axis=1
        )[:, ::stride, :]
        if patches.shape != (training.num_training_windows, patch_count, patch_len):
            raise RuntimeError(f"Unexpected patch view shape: {patches.shape}")
        if not np.isfinite(patches).all():
            raise ValueError(f"Non-finite patch values at channel={channel_index}")
        node_patches = patches.transpose(1, 0, 2)
        patch_means = node_patches.mean(axis=(1, 2))
        patch_stds = node_patches.std(axis=(1, 2), ddof=0)
        patch_rms = np.sqrt(np.mean(np.square(node_patches), axis=(1, 2)))
        patch_quantiles = np.quantile(
            node_patches, [0.25, 0.50, 0.75], axis=(1, 2)
        )
        per_window_mean = patches.mean(axis=2, keepdims=True)
        per_window_slope = (
            (patches - per_window_mean) * time_centered[None, None, :]
        ).sum(axis=2) / slope_denominator
        patch_slopes = per_window_slope.mean(axis=0)
        for patch_position in range(patch_count):
            raw[node_index] = np.array(
                [
                    patch_means[patch_position],
                    patch_stds[patch_position],
                    patch_rms[patch_position],
                    patch_slopes[patch_position],
                    patch_quantiles[0, patch_position],
                    patch_quantiles[1, patch_position],
                    patch_quantiles[2, patch_position],
                ],
                dtype=np.float64,
            )
            node_mapping.append(
                {
                    "node_index": node_index,
                    "channel_index": channel_index,
                    "channel_name": training.channel_names[channel_index],
                    "original_channel_index": training.original_channel_indices[channel_index],
                    "patch_position": patch_position,
                    "patch_start": patch_position * stride,
                }
            )
            node_index += 1

    descriptor_mean = raw.mean(axis=0)
    descriptor_std = raw.std(axis=0, ddof=0)
    zero_variance = descriptor_std == 0
    safe_std = descriptor_std.copy()
    safe_std[zero_variance] = 1.0
    standardized = (raw - descriptor_mean[None, :]) / safe_std[None, :]
    if not np.isfinite(standardized).all():
        raise ValueError("Non-finite standardized node descriptors")
    row_norms = np.linalg.norm(standardized, axis=1)
    zero_norm_count = int(np.count_nonzero(row_norms == 0))
    if zero_norm_count:
        raise GraphConstructionError(
            f"{zero_norm_count} standardized node descriptors have zero norm; "
            "cosine similarity is undefined."
        )

    descriptor_metadata = {
        "descriptor_names": [
            "mean",
            "std",
            "rms",
            "linear_slope",
            "quantile_25",
            "quantile_50",
            "quantile_75",
        ],
        "descriptor_mean": descriptor_mean.tolist(),
        "descriptor_std": descriptor_std.tolist(),
        "zero_variance_descriptor_count": int(np.count_nonzero(zero_variance)),
        "zero_norm_node_count": zero_norm_count,
        "training_window_count": int(training.num_training_windows),
        "patch_count": patch_count,
    }
    return raw, standardized, node_mapping, descriptor_metadata


def build_union_knn_graph(
    descriptors: np.ndarray,
    *,
    k: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the required positive-weight union-kNN cosine graph."""

    z = np.asarray(descriptors, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("descriptors must have shape [V,D] with V >= 2")
    if not np.isfinite(z).all():
        raise GraphConstructionError("Descriptors contain NaN or Inf")
    v = z.shape[0]
    if k <= 0 or k >= v:
        raise ValueError(f"k must be in [1, {v - 1}]")
    norms = np.linalg.norm(z, axis=1)
    if np.any(norms <= 0):
        raise GraphConstructionError("At least one node has zero descriptor norm")
    similarity = (z @ z.T) / (norms[:, None] * norms[None, :])
    np.fill_diagonal(similarity, -np.inf)

    neighbor_indices = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    adjacency = np.zeros((v, v), dtype=np.float64)
    selected_weights = []
    for i in range(v):
        neighbors = neighbor_indices[i]
        weights = np.maximum(similarity[i, neighbors], 0.0)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            count = int(np.count_nonzero(weights <= 0))
            raise GraphConstructionError(
                f"Node {i} has {count} non-positive selected kNN edge weights; "
                "the requested max(cosine_similarity, 0) rule would create invalid edges."
            )
        adjacency[i, neighbors] = weights
        selected_weights.append(weights)

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    if not np.isfinite(adjacency).all():
        raise GraphConstructionError("Adjacency contains NaN or Inf")
    degrees = adjacency.sum(axis=1)
    isolated = np.flatnonzero(degrees <= 0)
    if isolated.size:
        raise GraphConstructionError(
            f"Union-kNN graph contains {isolated.size} isolated nodes: "
            f"{isolated[:10].tolist()}"
        )
    diagnostics = {
        "k": int(k),
        "directed_edge_count_before_union": int(v * k),
        "undirected_edge_count": int(np.count_nonzero(np.triu(adjacency > 0, 1))),
        "positive_edge_count": int(np.count_nonzero(adjacency > 0)),
        "minimum_edge_weight": float(adjacency[adjacency > 0].min()),
        "maximum_edge_weight": float(adjacency.max()),
        "isolated_node_count": int(isolated.size),
        "degree_min": float(degrees.min()),
        "degree_max": float(degrees.max()),
        "degree_mean": float(degrees.mean()),
        "degree_quantiles": [float(x) for x in np.quantile(degrees, [0.0, 0.25, 0.5, 0.75, 1.0])],
    }
    return adjacency, diagnostics
