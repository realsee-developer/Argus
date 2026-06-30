"""Camera pose evaluation metrics: relative rotation/translation accuracy, AUC, and ATE."""

import numpy as np
import torch

from argus.utils.pose_enc import pose_encoding_to_extri360
from argus.utils.rotation import mat_to_quat


# ---------------------------------------------------------------------------
# Pair-wise relative pose error
# ---------------------------------------------------------------------------


def build_pair_index(num_frames: int, batch_size: int = 1):
    """Build indices for all unique frame pairs.

    Args:
        num_frames: Number of frames (N).
        batch_size: Batch size (B).

    Returns:
        Tuple of (i1, i2) index tensors for all C(N,2) pairs per batch.
    """
    i1_, i2_ = torch.combinations(torch.arange(num_frames), 2, with_replacement=False).unbind(-1)
    i1, i2 = [
        (idx[None] + torch.arange(batch_size)[:, None] * num_frames).reshape(-1)
        for idx in [i1_, i2_]
    ]
    return i1, i2


def rotation_angle(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Compute rotation angle error between GT and predicted rotation matrices.

    Uses quaternion-based geodesic distance for numerical stability.

    Args:
        rot_gt: Ground truth rotations, shape [N, 3, 3].
        rot_pred: Predicted rotations, shape [N, 3, 3].
        eps: Clamping epsilon to avoid sqrt(0) in arccos.

    Returns:
        Rotation angle error in degrees, shape [N].
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    # Geodesic distance via quaternion dot product.
    dot_sq = (q_pred * q_gt).sum(dim=1) ** 2
    loss_q = (1.0 - dot_sq).clamp(min=eps)
    err_q = torch.arccos(1.0 - 2.0 * loss_q)

    return err_q * (180.0 / np.pi)


def translation_angle(tvec_gt: torch.Tensor, tvec_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Compute translation direction angle error between GT and predicted translations.

    Handles direction ambiguity by taking the minimum of angle and (180 - angle).

    Args:
        tvec_gt: Ground truth translations, shape [N, 3].
        tvec_pred: Predicted translations, shape [N, 3].
        eps: Small value to avoid division by zero.

    Returns:
        Translation angle error in degrees, shape [N].
    """
    # Normalize translation vectors.
    t_gt = tvec_gt / (torch.norm(tvec_gt, dim=1, keepdim=True) + eps)
    t_pred = tvec_pred / (torch.norm(tvec_pred, dim=1, keepdim=True) + eps)

    # Angle via: arccos(sqrt(1 - (1 - dot^2))) = arccos(|dot|)
    dot_sq = torch.sum(t_gt * t_pred, dim=1) ** 2
    loss_t = (1.0 - dot_sq).clamp(min=eps)
    err_t = torch.acos(torch.sqrt(1.0 - loss_t))

    # Replace NaN/Inf with large error.
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = 1e6

    rel_tangle_deg = err_t * (180.0 / np.pi)

    # Handle direction ambiguity: min(angle, 180 - angle).
    rel_tangle_deg = torch.min(rel_tangle_deg, (180.0 - rel_tangle_deg).abs())

    return rel_tangle_deg


def se3_to_relative_pose_error(pred_se3: torch.Tensor, gt_se3: torch.Tensor, num_frames: int):
    """Compute pairwise relative pose errors between predicted and GT trajectories.

    Both inputs are assumed to be camera-to-world (c2w) transformations.
    Relative pose from frame i to frame j is computed as T_i^{-1} @ T_j.

    Args:
        pred_se3: Predicted c2w transforms, shape [N, 4, 4].
        gt_se3: Ground truth c2w transforms, shape [N, 4, 4].
        num_frames: Number of frames.

    Returns:
        Tuple of (rotation_error_deg, translation_error_deg), each shape [num_pairs].
    """
    i1, i2 = build_pair_index(num_frames)

    relative_pose_gt = torch.inverse(gt_se3[i1]).bmm(gt_se3[i2])
    relative_pose_pred = torch.inverse(pred_se3[i1]).bmm(pred_se3[i2])

    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------


def calculate_auc(r_error: torch.Tensor, t_error: torch.Tensor, max_threshold: int = 30) -> torch.Tensor:
    """Calculate Area Under the Curve for pose error.

    Uses a histogram with 1-degree bins from [0, max_threshold].

    Args:
        r_error: Rotation errors in degrees, shape [N].
        t_error: Translation errors in degrees, shape [N].
        max_threshold: Maximum error threshold for binning.

    Returns:
        AUC value (scalar tensor).
    """
    max_errors, _ = torch.max(torch.stack((r_error, t_error), dim=1), dim=1)

    histogram = torch.histc(max_errors, bins=max_threshold, min=0, max=max_threshold)
    normalized_histogram = histogram / float(max_errors.size(0))

    return torch.cumsum(normalized_histogram, dim=0).mean()


# ---------------------------------------------------------------------------
# Absolute Trajectory Error (ATE)
# ---------------------------------------------------------------------------


def _horn_alignment(model: np.ndarray, data: np.ndarray):
    """Align two 3D trajectories using Horn's closed-form method (rigid body).

    Args:
        model: First trajectory, shape [3, N].
        data: Second trajectory, shape [3, N].

    Returns:
        Tuple of (rotation [3,3], translation [3,1], per-point error [N]).
    """
    model_centered = model - model.mean(axis=1, keepdims=True)
    data_centered = data - data.mean(axis=1, keepdims=True)

    W = (data_centered @ model_centered.T) / model.shape[1]
    U, _, Vt = np.linalg.svd(W)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    t = data.mean(axis=1, keepdims=True) - R @ model.mean(axis=1, keepdims=True)

    aligned = R @ model + t
    error = np.linalg.norm(aligned - data, axis=0)

    return R, t, error


def evaluate_ate(pred_extrinsics: torch.Tensor, gt_extrinsics: torch.Tensor) -> float:
    """Compute Absolute Trajectory Error (RMSE) after rigid alignment.

    Args:
        pred_extrinsics: Predicted c2w transforms, shape [N, 4, 4].
        gt_extrinsics: Ground truth c2w transforms, shape [N, 4, 4].

    Returns:
        ATE RMSE value (float).
    """
    pred_positions = pred_extrinsics[:, :3, 3].numpy().T  # [3, N]
    gt_positions = gt_extrinsics[:, :3, 3].numpy().T  # [3, N]

    _, _, trans_error = _horn_alignment(gt_positions, pred_positions)

    return np.sqrt((trans_error ** 2).mean()).item()


# ---------------------------------------------------------------------------
# Per-frame absolute pose error (for accept rate)
# ---------------------------------------------------------------------------


def _compute_absolute_pose_errors(T1: np.ndarray, T2: np.ndarray):
    """Compute per-frame rotation and translation errors between two pose sets.

    Args:
        T1: Pose matrices, shape [N, 4, 4].
        T2: Pose matrices, shape [N, 4, 4].

    Returns:
        Tuple of (rotation_error_degrees [N], translation_error_meters [N]).
    """
    R1, t1 = T1[:, :3, :3], T1[:, :3, 3]
    R2, t2 = T2[:, :3, :3], T2[:, :3, 3]

    # Rotation angle error via trace of relative rotation.
    R_rel = R2 @ np.transpose(R1, (0, 2, 1))
    trace = np.trace(R_rel, axis1=1, axis2=2)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_error_deg = np.degrees(np.arccos(cos_theta))

    # Translation absolute error (L2 distance).
    trans_error = np.linalg.norm(t1 - t2, axis=1)

    return rot_error_deg, trans_error


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_pose_metrics(predictions: dict, gt: dict) -> dict:
    """Compute all pose evaluation metrics.

    Metrics include:
        - RRA@τ / RTA@τ: Relative rotation/translation accuracy at various thresholds.
        - AUC@τ: Area under the pose error curve.
        - ATE_RMSE: Absolute trajectory error (RMSE after Horn alignment).
        - accept_num / total_num: Frames passing R<10° and t<0.5m thresholds.

    Args:
        predictions: Model output dict containing 'pose_enc'.
        gt: Ground truth dict containing 'extrinsics'.

    Returns:
        Dictionary of metric name -> value.
    """
    pred_extrinsics, _ = pose_encoding_to_extri360(predictions["pose_enc"])
    pred_extrinsics = pred_extrinsics[0].clone().cpu()  # [S, 4, 4]
    gt_extrinsics = gt["extrinsics"][0].clone().cpu()  # [S, 4, 4]

    # Pairwise relative pose errors.
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
        pred_extrinsics, gt_extrinsics, num_frames=pred_extrinsics.shape[0]
    )

    results = {}

    # Relative accuracy at various thresholds.
    for tau in (5, 10, 15, 20, 30):
        results[f"RRA_{tau}"] = (rel_rangle_deg < tau).float().mean().item() * 100.0
        results[f"RTA_{tau}"] = (rel_tangle_deg < tau).float().mean().item() * 100.0

    # AUC at various thresholds.
    for threshold in (3, 5, 10, 15, 20, 30):
        results[f"AUC_{threshold}"] = (
            calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=threshold) * 100.0
        )

    # Absolute Trajectory Error.
    results["ATE_RMSE"] = evaluate_ate(pred_extrinsics, gt_extrinsics)

    # Accept rate (per-frame absolute error thresholds).
    rot_err_deg, trans_err = _compute_absolute_pose_errors(
        pred_extrinsics.numpy(), gt_extrinsics.numpy()
    )
    results["accept_num"] = ((rot_err_deg < 10.0) & (trans_err < 0.5)).sum().item()
    results["total_num"] = rot_err_deg.shape[0]

    if results["total_num"] != results["accept_num"]:
        print(
            f"failed seq_name: {gt['seq_name']}, "
            f"total_num: {results['total_num']}, accept_num: {results['accept_num']}"
        )

    return results
