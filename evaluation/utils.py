"""Evaluation utilities: model loading, metric orchestration, and result serialization."""

import os
import random
from copy import deepcopy

import cv2
import numpy as np
import torch

from argus.heads.utils import reorder_by_reference
from argus.models.argus import Argus
from argus.utils.normalization import normalize_camera_extrinsics_and_points_batch
from argus.utils.pose_enc import pose_encoding_to_extri360
from metric_covis import compute_covis_metrics
from metric_depth import compute_campoint_depth_metrics, compute_depth_metrics
from metric_points import compute_points_metrics
from metric_pose import compute_pose_metrics


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(device: str, model_path: str, reorder_by_learning_ref: bool) -> Argus:
    """Load and prepare the Argus model for evaluation.

    Args:
        device: Target device ('cuda' or 'cpu').
        model_path: Path to the model checkpoint file.
        reorder_by_learning_ref: Whether to enable learned reference reordering.

    Returns:
        The model in eval mode on the specified device.
    """
    print(f"Initializing and loading model from {model_path}...")
    model = Argus(
        enable_point=True,
        reorder_by_learning_ref=reorder_by_learning_ref,
        restore_metric_scale=True,
    )
    model.load_state_dict(torch.load(model_path)["model"], strict=False)
    model.eval()
    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_random_seeds(seed: int) -> None:
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics result management
# ---------------------------------------------------------------------------


def init_metrics_results() -> dict:
    """Initialize empty containers for all evaluation metrics."""
    pose_keys = (
        [f"RRA_{t}" for t in (5, 10, 15, 20, 30)]
        + [f"RTA_{t}" for t in (5, 10, 15, 20, 30)]
        + [f"AUC_{t}" for t in (3, 5, 10, 15, 20, 30)]
        + ["ATE_RMSE", "accept_num", "total_num"]
    )
    depth_keys = ["Abs Rel", "δ < 1.03", "δ < 1.25", "RMSE", "MAE"]
    points_keys = ["accuracy", "accuracy_median", "completion", "completion_median", "nc", "nc_median"]
    covis_keys = ["top3_acc"]

    return {
        "pose": {k: [] for k in pose_keys},
        "opt_aligned_depth": {k: [] for k in depth_keys},
        "median_aligned_depth": {k: [] for k in depth_keys},
        "none_aligned_depth": {k: [] for k in depth_keys},
        "points": {k: [] for k in points_keys},
        "metric_points": {k: [] for k in points_keys},
        "covis": {k: [] for k in covis_keys},
    }


def integrate_metrics_results(metrics_results: dict) -> dict:
    """Aggregate per-sample metrics into final averages.

    Modifies the input dictionary in-place and returns it.
    """
    # Pose: sum for counts, mean for rates.
    pose = metrics_results["pose"]
    for key in pose:
        if key in ("accept_num", "total_num"):
            pose[key] = np.sum(pose[key])
        else:
            pose[key] = np.mean(pose[key]).round(3)
    pose["accept_rate"] = 100.0 * (pose["accept_num"] / pose["total_num"]).round(3)

    # Depth variants and point metrics: simple mean.
    for group in ("opt_aligned_depth", "median_aligned_depth", "none_aligned_depth",
                  "points", "metric_points", "covis"):
        for key in metrics_results[group]:
            metrics_results[group][key] = np.mean(metrics_results[group][key]).round(3)

    return metrics_results


# ---------------------------------------------------------------------------
# Batch normalization and metric computation
# ---------------------------------------------------------------------------


def _normalize_batch(predictions: dict, batch: dict) -> None:
    """Apply reference reordering and coordinate normalization to the GT batch.

    This aligns the GT data with the model's predicted ordering and normalizes
    camera extrinsics for consistent evaluation.
    """
    if "ref_idx" in predictions:
        ref_idx = predictions["ref_idx"].detach().cpu()
        for key in ("extrinsics", "depths", "cam_points", "rotated_points", "world_points", "point_masks"):
            batch[key] = reorder_by_reference(batch[key], ref_idx)

    (
        normalized_extrinsics,
        normalized_cam_points,
        normalized_rotated_points,
        normalized_world_points,
        normalized_depths,
    ) = normalize_camera_extrinsics_and_points_batch(
        extrinsics=batch["extrinsics"],
        cam_points=batch["cam_points"],
        depths=batch["depths"],
        point_masks=batch["point_masks"],
        scale_mode="none",
    )

    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["rotated_points"] = normalized_rotated_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths


def compute_all_metrics(predictions: dict, batch: dict, metrics_results: dict) -> None:
    """Compute all applicable metrics and append results to the accumulator.

    Metrics are conditionally computed based on which keys are present in predictions.
    """
    _normalize_batch(predictions, batch)

    if "pose_enc" in predictions:
        pose_results = compute_pose_metrics(predictions, batch)
        for key, value in pose_results.items():
            metrics_results["pose"][key].append(value)

    if "depth" in predictions:
        for align_mode in ("opt_aligned", "median_aligned", "none_aligned"):
            depth_results = compute_depth_metrics(predictions, batch, align_with_scale=align_mode)
            group_key = f"{align_mode}_depth"
            for key, value in depth_results.items():
                metrics_results[group_key][key].append(value)

    if "world_points" in predictions:
        # Aligned (Umeyama + ICP) point cloud metrics.
        points_results = compute_points_metrics(predictions, batch)
        for key, value in points_results.items():
            metrics_results["points"][key].append(value)

        # Metric-scale point cloud metrics (no alignment).
        metric_points_results = compute_points_metrics(predictions, batch, align=False)
        for key, value in metric_points_results.items():
            metrics_results["metric_points"][key].append(value)

    if "ref_idx" in predictions:
        covis_results = compute_covis_metrics(predictions, batch)
        for key, value in covis_results.items():
            metrics_results["covis"][key].append(value)


# ---------------------------------------------------------------------------
# Prediction output saving
# ---------------------------------------------------------------------------


def save_output(predictions: dict, batch: dict, output_dir: str) -> None:
    """Save model predictions (pose, depth, points) to disk.

    Saves per-viewpoint extrinsics as txt, depth as uint16 PNG, and world points as npy.
    Only saves outputs that are present in predictions. Assumes batch_size = 1.
    """
    scene_id = batch["seq_name"][0]
    scene_dir = os.path.join(output_dir, scene_id, "viewpoints")
    os.makedirs(scene_dir, exist_ok=True)

    ids = batch["ids"][0]
    viewpoints = batch["viewpoints"]

    # Decode available predictions.
    pred_extrinsics = None
    if "pose_enc" in predictions:
        extrinsics, _ = pose_encoding_to_extri360(predictions["pose_enc"])
        pred_extrinsics = extrinsics[0].cpu().numpy()  # [S, 4, 4]

    pred_depths = None
    if "depth" in predictions:
        pred_depths = predictions["depth"][0].squeeze(-1).cpu().numpy()  # [S, H, W]

    pred_pts = None
    if "world_points" in predictions:
        pred_pts = predictions["world_points"][0].cpu().numpy()  # [S, H, W, 3]

    depth_scale = 1000.0
    for i, frame_id in enumerate(ids):
        vp = viewpoints[frame_id.item()][0]
        vp_dir = os.path.join(scene_dir, vp)
        os.makedirs(vp_dir, exist_ok=True)

        # Save extrinsics.
        if pred_extrinsics is not None:
            np.savetxt(os.path.join(vp_dir, "extrinsics.txt"), pred_extrinsics[i])

        # Save depth as uint16 PNG.
        if pred_depths is not None:
            depth_mm = pred_depths[i] * depth_scale
            depth_mm[(depth_mm < 0) | (depth_mm > 65535)] = 0
            cv2.imwrite(os.path.join(vp_dir, "depth_image.png"), depth_mm.astype(np.uint16))

            with open(os.path.join(vp_dir, "depth_scale.txt"), "w") as f:
                f.write(f"{depth_scale:.1f}\n")

        # Save world points.
        if pred_pts is not None:
            np.save(os.path.join(vp_dir, "world_points.npy"), pred_pts[i])


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


def serialize_numpy_and_round(obj):
    """Custom JSON serializer handling numpy types with 3-decimal rounding."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return round(float(obj), 3)
    if isinstance(obj, (np.ndarray, list)):
        data = obj.tolist() if isinstance(obj, np.ndarray) else obj
        return [round(float(x), 3) if isinstance(x, (float, np.floating)) else x for x in data]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
