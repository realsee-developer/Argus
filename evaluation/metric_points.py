"""3D point cloud evaluation metrics: accuracy, completion, and normal consistency."""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree


# ---------------------------------------------------------------------------
# Sim(3) alignment (Umeyama)
# ---------------------------------------------------------------------------


def umeyama(X: np.ndarray, Y: np.ndarray):
    """Estimate Sim(3) transformation (c*R@X + t ≈ Y) using Umeyama's method.

    Args:
        X: Source points, shape [3, N].
        Y: Target points, shape [3, N].

    Returns:
        Tuple of (scale, rotation [3,3], translation [3,1]).
    """
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)

    X_centered = X - mu_x
    Y_centered = Y - mu_y

    var_x = np.square(X_centered).sum(axis=0).mean()
    cov_xy = (Y_centered @ X_centered.T) / X.shape[1]

    U, D, Vt = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ Vt
    t = mu_y - c * R @ mu_x

    return c, R, t


# ---------------------------------------------------------------------------
# Point cloud distance metrics
# ---------------------------------------------------------------------------


def _accuracy(gt_points: np.ndarray, rec_points: np.ndarray,
              gt_normals: np.ndarray = None, rec_normals: np.ndarray = None):
    """Compute accuracy: mean/median distance from reconstructed points to GT.

    Args:
        gt_points: Ground truth point cloud, shape [M, 3].
        rec_points: Reconstructed point cloud, shape [N, 3].
        gt_normals: GT normals, shape [M, 3] (optional).
        rec_normals: Reconstructed normals, shape [N, 3] (optional).

    Returns:
        Tuple of (mean_acc, median_acc, mean_nc, median_nc) if normals provided,
        otherwise (mean_acc, median_acc).
    """
    tree = KDTree(gt_points)
    distances, indices = tree.query(rec_points, workers=24)

    acc_mean = np.mean(distances)
    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.abs(np.sum(gt_normals[indices] * rec_normals, axis=-1))
        return acc_mean, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc_mean, acc_median


def _completion(gt_points: np.ndarray, rec_points: np.ndarray,
                gt_normals: np.ndarray = None, rec_normals: np.ndarray = None):
    """Compute completion: mean/median distance from GT points to reconstruction.

    Args:
        gt_points: Ground truth point cloud, shape [M, 3].
        rec_points: Reconstructed point cloud, shape [N, 3].
        gt_normals: GT normals, shape [M, 3] (optional).
        rec_normals: Reconstructed normals, shape [N, 3] (optional).

    Returns:
        Tuple of (mean_comp, median_comp, mean_nc, median_nc) if normals provided,
        otherwise (mean_comp, median_comp).
    """
    tree = KDTree(rec_points)
    distances, indices = tree.query(gt_points, workers=24)

    comp_mean = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.abs(np.sum(gt_normals * rec_normals[indices], axis=-1))
        return comp_mean, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp_mean, comp_median


# ---------------------------------------------------------------------------
# ICP refinement
# ---------------------------------------------------------------------------


def _icp_refine(pred_pcd: o3d.geometry.PointCloud, gt_pcd: o3d.geometry.PointCloud,
                threshold: float = 0.1) -> o3d.geometry.PointCloud:
    """Refine alignment with point-to-point ICP.

    Args:
        pred_pcd: Coarsely aligned predicted point cloud.
        gt_pcd: Ground truth point cloud.
        threshold: Maximum correspondence distance.

    Returns:
        ICP-refined predicted point cloud.
    """
    reg = o3d.pipelines.registration.registration_icp(
        pred_pcd,
        gt_pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return pred_pcd.transform(reg.transformation)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_points_metrics(predictions: dict, gt: dict, align: bool = True) -> dict:
    """Compute 3D point cloud evaluation metrics.

    When align=True, performs Umeyama + ICP alignment before evaluation.
    When align=False, assumes predictions are already in metric world coordinates.

    Args:
        predictions: Model output dict containing 'world_points' of shape [B, S, H, W, 3].
        gt: Ground truth dict containing 'world_points' and 'point_masks'.
        align: Whether to perform Sim(3) + ICP alignment.

    Returns:
        Dictionary with accuracy, completion, and normal consistency metrics.
    """
    gt_pts = gt["world_points"][0].clone().cpu().numpy()  # [S, H, W, 3]
    valid_mask = gt["point_masks"][0].clone().cpu().numpy()  # [S, H, W]
    pred_pts = predictions["world_points"][0].clone().cpu().numpy()  # [S, H, W, 3]

    assert pred_pts.shape == gt_pts.shape, (
        f"Shape mismatch: pred {pred_pts.shape} vs gt {gt_pts.shape}"
    )

    if align:
        # Coarse alignment with Umeyama (Sim3).
        c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
        pred_pts = c * np.einsum("nhwj, ij -> nhwi", pred_pts, R) + t.T

    # Extract valid points.
    pred_valid = pred_pts[valid_mask].reshape(-1, 3)
    gt_valid = gt_pts[valid_mask].reshape(-1, 3)

    # Build Open3D point clouds.
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_valid)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_valid)

    if align:
        # Fine alignment with ICP.
        pred_pcd = _icp_refine(pred_pcd, gt_pcd)

    # Estimate normals for normal consistency evaluation.
    pred_pcd.estimate_normals()
    gt_pcd.estimate_normals()

    pred_points_np = np.asarray(pred_pcd.points)
    gt_points_np = np.asarray(gt_pcd.points)
    pred_normals = np.asarray(pred_pcd.normals)
    gt_normals = np.asarray(gt_pcd.normals)

    # Compute bidirectional metrics.
    acc, acc_med, nc_acc, nc_acc_med = _accuracy(gt_points_np, pred_points_np, gt_normals, pred_normals)
    comp, comp_med, nc_comp, nc_comp_med = _completion(gt_points_np, pred_points_np, gt_normals, pred_normals)

    return {
        "accuracy": acc,
        "accuracy_median": acc_med,
        "completion": comp,
        "completion_median": comp_med,
        "nc": (nc_acc + nc_comp) / 2.0,
        "nc_median": (nc_acc_med + nc_comp_med) / 2.0,
    }
