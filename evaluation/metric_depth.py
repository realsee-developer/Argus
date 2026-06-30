"""Depth evaluation metrics: Abs Rel, RMSE, MAE, and threshold accuracy."""

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core depth metric computation
# ---------------------------------------------------------------------------


def _align_scale(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Align predicted depth to ground truth via scale estimation.

    Args:
        predicted: Predicted depth values (masked, 1D).
        ground_truth: Ground truth depth values (masked, 1D).
        method: One of 'opt_aligned', 'median_aligned', 'none_aligned'.

    Returns:
        Scale-aligned predicted depth.
    """
    if method == "none_aligned":
        return predicted, 1.0

    if method == "median_aligned":
        s = torch.median(ground_truth) / torch.median(predicted)
        return predicted * s, s

    # opt_aligned: L2 closed-form initial estimate + IRLS refinement.
    s = torch.sum(predicted * ground_truth) / torch.sum(predicted ** 2)

    for _ in range(10):
        residuals = s * predicted - ground_truth
        weights = 1.0 / (residuals.abs() + 1e-8)
        s = torch.sum(weights * predicted * ground_truth) / torch.sum(weights * predicted ** 2)

    s = s.clamp(min=1e-3).detach()
    return predicted * s, s


def get_depth_metrics(
    predicted_depth: np.ndarray,
    ground_truth_depth: np.ndarray,
    max_depth: float = 100.0,
    align_with_scale: str = "opt_aligned",
    use_gpu: bool = True,
) -> dict:
    """Compute standard depth evaluation metrics for a single frame.

    Args:
        predicted_depth: Predicted depth map, shape [H, W] or [S, H, W].
        ground_truth_depth: Ground truth depth map, same shape.
        max_depth: Maximum valid depth threshold in meters.
        align_with_scale: Scale alignment method.
        use_gpu: Whether to run computation on GPU.

    Returns:
        Tuple of (metrics_dict, error_map, scaled_prediction, gt_map).
    """
    predicted = torch.from_numpy(predicted_depth.astype(np.float32))
    gt = torch.from_numpy(ground_truth_depth.astype(np.float32))

    # Flatten multi-frame input to 2D.
    if predicted.dim() == 3:
        _, h, w = predicted.shape
        predicted = predicted.view(-1, w)
        gt = gt.view(-1, w)

    if use_gpu:
        predicted = predicted.cuda()
        gt = gt.cuda()

    # Valid depth mask.
    mask = (gt > 0) & (gt < max_depth)
    pred_masked = predicted[mask]
    gt_masked = gt[mask]

    num_valid = mask.sum().item()
    if num_valid == 0:
        return (
            {"Abs Rel": 0, "RMSE": 0, "MAE": 0, "δ < 1.03": 0, "δ < 1.25": 0, "valid_pixels": 0},
            torch.zeros_like(gt),
            predicted,
            gt,
        )

    # Scale alignment.
    pred_aligned, s = _align_scale(pred_masked, gt_masked, align_with_scale)

    # Core metrics.
    abs_diff = torch.abs(pred_aligned - gt_masked)
    abs_rel = (abs_diff / gt_masked).mean().item()
    mae = abs_diff.mean().item()
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()

    # Threshold accuracy.
    pred_safe = pred_aligned.clamp(min=1e-5)
    max_ratio = torch.maximum(pred_safe / gt_masked, gt_masked / pred_safe)
    delta_103 = (max_ratio < 1.03).float().mean().item()
    delta_125 = (max_ratio < 1.25).float().mean().item()

    # Error parity map (full spatial map for visualization).
    predicted_scaled = predicted * (s if isinstance(s, float) else s.item() if hasattr(s, 'item') else s)
    error_map = torch.abs(predicted_scaled - gt) / gt.clamp(min=1e-8)
    error_map = torch.where(mask, error_map, torch.zeros_like(error_map))

    results = {
        "Abs Rel": abs_rel,
        "RMSE": rmse,
        "MAE": mae,
        "δ < 1.03": delta_103,
        "δ < 1.25": delta_125,
        "valid_pixels": num_valid,
    }

    return results, error_map, predicted_scaled, gt


# ---------------------------------------------------------------------------
# Per-scene depth metrics (multi-frame)
# ---------------------------------------------------------------------------


def _aggregate_frame_metrics(gathered: list) -> dict:
    """Aggregate per-frame depth metrics using valid-pixel-weighted average.

    Args:
        gathered: List of per-frame metric dicts (each containing 'valid_pixels').

    Returns:
        Weighted-average metrics dict (excluding 'valid_pixels').
    """
    weights = [m["valid_pixels"] for m in gathered]
    return {
        key: np.average([m[key] for m in gathered], weights=weights).item()
        for key in gathered[0]
        if key != "valid_pixels"
    }


def compute_depth_metrics(predictions: dict, gt: dict, align_with_scale: str) -> dict:
    """Compute depth metrics across all frames in a scene.

    Args:
        predictions: Model output dict containing 'depth' of shape [B, S, H, W] or [B, S, H, W, 1].
        gt: Ground truth dict containing 'depths' of shape [B, S, H, W].
        align_with_scale: Scale alignment method.

    Returns:
        Aggregated metrics dict with percentage-scaled threshold values.
    """
    pred_depths = predictions["depth"][0].squeeze(-1).clone().cpu().numpy()  # [S, H, W]
    gt_depths = gt["depths"][0].clone().cpu().numpy()  # [S, H, W]

    S, H, W = gt_depths.shape
    gathered = []
    for idx in range(S):
        pred_frame = pred_depths[idx]
        # Resize if prediction resolution differs from GT.
        if pred_frame.shape[1] != W:
            pred_frame = cv2.resize(pred_frame, (W, H), interpolation=cv2.INTER_CUBIC)

        metrics, _, _, _ = get_depth_metrics(pred_frame, gt_depths[idx], align_with_scale=align_with_scale)
        gathered.append(metrics)

    avg = _aggregate_frame_metrics(gathered)
    return {
        "Abs Rel": avg["Abs Rel"],
        "RMSE": avg["RMSE"],
        "MAE": avg["MAE"],
        "δ < 1.03": avg["δ < 1.03"] * 100.0,
        "δ < 1.25": avg["δ < 1.25"] * 100.0,
    }


def compute_campoint_depth_metrics(predictions: dict, gt: dict, align_with_scale: str) -> dict:
    """Compute depth metrics using camera-space point norms as depth proxy.

    Args:
        predictions: Model output dict containing 'cam_points'.
        gt: Ground truth dict containing 'cam_points'.
        align_with_scale: Scale alignment method.

    Returns:
        Aggregated metrics dict.
    """
    pred_depths = torch.norm(predictions["cam_points"][0], dim=-1).clone().cpu().numpy()
    gt_depths = torch.norm(gt["cam_points"][0], dim=-1).clone().cpu().numpy()

    S, H, W = gt_depths.shape
    gathered = []
    for idx in range(S):
        pred_frame = pred_depths[idx]
        if pred_frame.shape[1] != W:
            pred_frame = cv2.resize(pred_frame, (W, H), interpolation=cv2.INTER_CUBIC)

        metrics, _, _, _ = get_depth_metrics(pred_frame, gt_depths[idx], align_with_scale=align_with_scale)
        gathered.append(metrics)

    avg = _aggregate_frame_metrics(gathered)
    return {
        "Abs Rel": avg["Abs Rel"],
        "RMSE": avg["RMSE"],
        "MAE": avg["MAE"],
        "δ < 1.03": avg["δ < 1.03"] * 100.0,
        "δ < 1.25": avg["δ < 1.25"] * 100.0,
    }
