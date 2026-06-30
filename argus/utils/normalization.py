import torch
from typing import Optional, Tuple
from argus.utils.geometry import closed_form_inverse_se3

def cal_scale_by_points(points: torch.Tensor, point_masks: torch.Tensor) -> torch.Tensor:
    # Calculate average distance of valid 3D points (batch-wise)
    dist = points.norm(dim=-1)
    dist_sum = (dist * point_masks).sum(dim=[1, 2, 3])  # Shape: [B,]
    valid_count = point_masks.sum(dim=[1, 2, 3])
    avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
    return avg_scale

def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: torch.Tensor,
    depths: torch.Tensor,
    point_masks: torch.Tensor,
    scale_mode: str = "none",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Basic input validation
    assert extrinsics.ndim == 4 and extrinsics.shape[2:] == (4, 4), \
        f"Extrinsics must be (B, S, 4, 4), got {extrinsics.shape}"
    B, S = extrinsics.shape[:2]
    device = extrinsics.device
    
    # Step 1: Transform all extrinsics to reference frame (1st frame of each batch)
    ref_extrinsics = extrinsics[:,0,:,:]  # (B, 4, 4)
    ref_extr_inv = closed_form_inverse_se3(ref_extrinsics)
    new_extrinsics = torch.matmul(ref_extr_inv.unsqueeze(1), extrinsics)  # (B, S, 4, 4) world coordinate
    
    # Step 2: Clone tensors to avoid in-place modification
    new_depths = depths.clone()
    new_cam_points = cam_points.clone()

    # Step 3: Compute rotated/world points from new extrinsics
    R_new = new_extrinsics[:, :, :3, :3]  # (B, S, 3, 3)
    t_new = new_extrinsics[:, :, :3, 3]  # (B, S, 3)
    new_rotated_points = torch.matmul(R_new.unsqueeze(2).unsqueeze(3), new_cam_points.unsqueeze(-1)).squeeze(-1) # (B,S,1,1,3,3) × (B,S,H,W,3,1) -> (B,S,H,W,3)
    new_world_points = new_rotated_points + t_new.unsqueeze(2).unsqueeze(3)

    # Step 4: Apply scene scaling
    if scale_mode == "avg_dist":
        avg_scale = cal_scale_by_points(new_world_points, point_masks)  # (B,)
        # Reshape scale for broadcasting with different tensor shapes
        scale_3d = avg_scale.view(-1, 1, 1)      # For extrinsics (B, S, 4, 4)
        scale_4d = avg_scale.view(-1, 1, 1, 1)   # For depths (B, S, H, W)
        scale_5d = avg_scale.view(-1, 1, 1, 1, 1) # For 3D points (B, S, H, W, 3)
        new_extrinsics[:, :, :3, 3] /= scale_3d
        new_depths /= scale_4d
        new_cam_points /= scale_5d
        new_rotated_points /= scale_5d
        new_world_points /= scale_5d
    elif scale_mode == "abs":
        metric_scale = 10.0
        new_extrinsics[:, :, :3, 3] /= metric_scale
        new_depths /= metric_scale
        new_cam_points /= metric_scale
        new_rotated_points /= metric_scale
        new_world_points /= metric_scale
    elif scale_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    
    return new_extrinsics, new_cam_points, new_rotated_points, new_world_points, new_depths