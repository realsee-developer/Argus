import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Callable
from math import ceil, floor

# Import external utilities
from argus.utils.pose_enc import extri_to_pose_encoding360
from train_utils.general import check_and_fix_inf_nan
from argus.utils.normalization import normalize_camera_extrinsics_and_points_batch
from argus.heads.utils import reorder_by_reference
from argus.utils.rotation import quat_to_mat
from argus.utils.geometry import pano_depth_to_points, points_to_pano_depth, camera_points_to_rotated_points, rotated_points_to_world_points

# -------------------------- Type Aliases --------------------------
Tensor = torch.Tensor
LossDict = Dict[str, Tensor]
ActivationType = Union[str, "l1", "l2", "normal", "grad"]


# -------------------------- Multi-task Loss Module --------------------------
@dataclass(eq=False)
class MultitaskLoss(nn.Module):
    """
    Multi-task loss module for Argus model combining multiple vision tasks:
    - Camera pose estimation loss
    - Depth prediction loss
    - 3D point reconstruction loss (world/camera/rotated)
    - Covisibility prediction loss
    
    Each task loss can be independently configured and weighted through initialization parameters.
    """
    def __init__(
        self,
        camera: Optional[Dict] = None,
        depth: Optional[Dict] = None,
        point: Optional[Dict] = None,
        cam_point: Optional[Dict] = None,
        rotated_point: Optional[Dict] = None,
        covisibility: Optional[Dict] = None,
        joint: Optional[Dict] = None,
        **kwargs
    ) -> None:
        super().__init__()
        
        # Task-specific loss configurations (None = task disabled)
        self.camera = camera
        self.depth = depth
        self.point = point
        self.cam_point = cam_point
        self.rotated_point = rotated_point
        self.covisibility = covisibility
        self.joint = joint

    def forward(self, predictions: Dict[str, Tensor], batch: Dict[str, Tensor]) -> LossDict:
        """
        Compute total multi-task loss by aggregating losses from enabled tasks.
        
        Args:
            predictions: Model output dictionary containing task-specific predictions
            batch: Ground truth data dictionary with labels, masks, and metadata
            
        Returns:
            Dictionary containing individual task losses and total aggregated objective
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        loss_dict = {}
        
        # Reorder ground truth by predicted reference indices for correct supervision
        if "ref_idx" in predictions:
            ref_idx = predictions["ref_idx"].detach()
            # Reorder all spatial/temporal data (exclude adjacency matrix and IDs)
            batch["extrinsics"] = reorder_by_reference(batch["extrinsics"], ref_idx)
            batch["depths"] = reorder_by_reference(batch["depths"], ref_idx)
            batch["cam_points"] = reorder_by_reference(batch["cam_points"], ref_idx)
            batch["rotated_points"] = reorder_by_reference(batch["rotated_points"], ref_idx)
            batch["world_points"] = reorder_by_reference(batch["world_points"], ref_idx)
            batch["point_masks"] = reorder_by_reference(batch["point_masks"], ref_idx)
        
        # Normalize camera extrinsics and 3D points for consistent loss calculation
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
            scale_mode="abs"
        )
        
        # Update batch with normalized values
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["rotated_points"] = normalized_rotated_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths
        
        # Calculate loss for each enabled task
        loss_dict = self._compute_task_losses(predictions, batch, total_loss)
        loss_dict["objective"] = total_loss

        return loss_dict

    def _compute_task_losses(
        self,
        predictions: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        total_loss: Tensor
    ) -> LossDict:
        """Helper method to compute losses for all enabled tasks"""
        loss_dict = {}
        
        # Camera pose loss
        if "pose_enc_list" in predictions and self.camera is not None:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)
            camera_loss = camera_loss_dict["loss_camera"] + camera_loss_dict["loss_camera_conf"] 
            total_loss += camera_loss * self.camera["weight"]
            loss_dict.update(camera_loss_dict)

        # Depth prediction loss
        if "depth" in predictions and self.depth is not None:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = (depth_loss_dict["loss_conf_depth"] + 
                          depth_loss_dict["loss_reg_depth"] + 
                          depth_loss_dict["loss_grad_depth"])
            total_loss += depth_loss * self.depth["weight"]
            loss_dict.update(depth_loss_dict)

        # World 3D point loss
        if "world_points" in predictions and self.point is not None:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = (point_loss_dict["loss_conf_world_points"] + 
                          point_loss_dict["loss_reg_world_points"] + 
                          point_loss_dict["loss_grad_world_points"])
            total_loss += point_loss * self.point["weight"]
            loss_dict.update(point_loss_dict)

        # Camera coordinate 3D point loss
        if "cam_points" in predictions and self.cam_point is not None:
            cam_point_loss_dict = compute_cam_point_loss(predictions, batch, **self.cam_point)
            cam_point_loss = (cam_point_loss_dict["loss_conf_cam_points"] + 
                              cam_point_loss_dict["loss_reg_cam_points"] + 
                              cam_point_loss_dict["loss_grad_cam_points"])
            total_loss += cam_point_loss * self.cam_point["weight"]
            loss_dict.update(cam_point_loss_dict)

        # Rotated 3D point loss
        if "rotated_points" in predictions and self.rotated_point is not None:
            rotated_point_loss_dict = compute_rotated_point_loss(predictions, batch, **self.rotated_point)
            rotated_point_loss = (rotated_point_loss_dict["loss_conf_rotated_points"] + 
                                  rotated_point_loss_dict["loss_reg_rotated_points"] + 
                                  rotated_point_loss_dict["loss_grad_rotated_points"])
            total_loss += rotated_point_loss * self.rotated_point["weight"]
            loss_dict.update(rotated_point_loss_dict)

        # Covisibility prediction loss
        if "covisibility_scores" in predictions and self.covisibility is not None:
            covisibility_loss_dict = compute_covisibility_loss(predictions, batch, **self.covisibility)
            covisibility_loss = covisibility_loss_dict["loss_covisibility"]
            total_loss += covisibility_loss * self.covisibility["weight"]
            loss_dict.update(covisibility_loss_dict)
            
        if self.joint is not None:
            if "pose_enc_list" in predictions and self.camera is not None and "depth" in predictions and self.depth is not None:
                joint_loss_dict = compute_joint_loss(predictions, batch, **self.joint)
                
                D2CP_loss = joint_loss_dict["loss_reg_D2CP"] + joint_loss_dict["loss_conf_D2CP"]+joint_loss_dict["loss_grad_D2CP"]
                total_loss += D2CP_loss * self.joint["weight_D2CP"]
                
                CP2RP_loss = joint_loss_dict["loss_reg_CP2RP"]+joint_loss_dict["loss_conf_CP2RP"]+joint_loss_dict["loss_grad_CP2RP"]
                total_loss += CP2RP_loss * self.joint["weight_CP2RP"]
                
                RP2WP_loss = joint_loss_dict["loss_reg_RP2WP"]+joint_loss_dict["loss_conf_RP2WP"]+joint_loss_dict["loss_grad_RP2WP"]
                total_loss += RP2WP_loss * self.joint["weight_RP2WP"]
                
                WP2RP_loss = joint_loss_dict["loss_reg_WP2RP"]+joint_loss_dict["loss_conf_WP2RP"]+joint_loss_dict["loss_grad_WP2RP"]
                total_loss += WP2RP_loss * self.joint["weight_WP2RP"]
                
                RP2CP_loss = joint_loss_dict["loss_reg_RP2CP"]+joint_loss_dict["loss_conf_RP2CP"]
                total_loss += RP2CP_loss * self.joint["weight_RP2CP"]
                
                CP2D_loss = joint_loss_dict["loss_reg_CP2D"]+joint_loss_dict["loss_conf_CP2D"]+joint_loss_dict["loss_grad_CP2D"]
                total_loss += CP2D_loss * self.joint["weight_CP2D"]
                
                loss_dict.update(joint_loss_dict)
            
        return loss_dict


# -------------------------- Task-Specific Loss Functions --------------------------
def compute_joint_loss(predictions: Dict[str, Tensor], batch: Dict[str, Tensor], **kwargs) -> LossDict:
    pred_pose_enc_list = predictions["pose_enc_list"]
    pred_last_pose_enc = pred_pose_enc_list[-1]
    t = pred_last_pose_enc[...,:3]
    q = pred_last_pose_enc[...,3:7]
    R = quat_to_mat(q)
    
    # unproject Depth to Camera Points
    D = predictions["depth"]
    D_conf = predictions["depth_conf"]
    D2CP = pano_depth_to_points(D)
    D2CP_loss_dict = _compute_joint_3d_point_loss_base(D2CP, D_conf, batch["cam_points"], batch["point_masks"], point_type="D2CP", gradient_loss_fn="normal")
   
    # Camera Points to Rotated Points
    CP = predictions["cam_points"]
    CP_conf = predictions["cam_points_conf"]
    CP2RP = camera_points_to_rotated_points(CP, R)
    CP2RP_loss_dict = _compute_joint_3d_point_loss_base(CP2RP, CP_conf, batch["rotated_points"], batch["point_masks"], point_type="CP2RP", gradient_loss_fn="normal")
    
    # Rotated Points to World Points
    RP = predictions["rotated_points"]
    RP_conf = predictions["rotated_points_conf"]
    RP2WP = rotated_points_to_world_points(RP, t)
    RP2WP_loss_dict = _compute_joint_3d_point_loss_base(RP2WP, RP_conf, batch["world_points"], batch["point_masks"], point_type="RP2WP", gradient_loss_fn="normal")
    
    # World Points to Rotated Points
    WP = predictions["world_points"]
    WP_conf = predictions["world_points_conf"]
    WP2RP = rotated_points_to_world_points(WP, -t)
    WP2RP_loss_dict = _compute_joint_3d_point_loss_base(WP2RP, WP_conf, batch["rotated_points"], batch["point_masks"], point_type="WP2RP", gradient_loss_fn="normal")
    
    # Rotated Points to Camera Points
    RP2CP = camera_points_to_rotated_points(RP, R.transpose(-1, -2))
    RP2CP_loss_dict = _compute_joint_3d_point_loss_base(RP2CP, RP_conf, batch["cam_points"], batch["point_masks"], point_type="RP2CP", gradient_loss_fn="normal")
    
    # project Camera Points to Depth
    CP2D = points_to_pano_depth(CP)
    
    CP2D_loss_dict = _compute_joint_3d_point_loss_base(CP2D, CP_conf, batch["depths"].unsqueeze(-1), batch["point_masks"], point_type="CP2D", gradient_loss_fn="grad")
    
    return {"loss_reg_D2CP": D2CP_loss_dict["loss_reg_D2CP"],
            "loss_conf_D2CP": D2CP_loss_dict["loss_conf_D2CP"],
            "loss_grad_D2CP":D2CP_loss_dict["loss_grad_D2CP"],
            "loss_reg_CP2RP":CP2RP_loss_dict["loss_reg_CP2RP"],
            "loss_conf_CP2RP":CP2RP_loss_dict["loss_conf_CP2RP"],
            "loss_grad_CP2RP":CP2RP_loss_dict["loss_grad_CP2RP"],
            "loss_reg_RP2WP":RP2WP_loss_dict["loss_reg_RP2WP"],
            "loss_conf_RP2WP":RP2WP_loss_dict["loss_conf_RP2WP"],
            "loss_grad_RP2WP":RP2WP_loss_dict["loss_grad_RP2WP"],
            "loss_reg_WP2RP":WP2RP_loss_dict["loss_reg_WP2RP"],
            "loss_conf_WP2RP":WP2RP_loss_dict["loss_conf_WP2RP"],
            "loss_grad_WP2RP":WP2RP_loss_dict["loss_grad_WP2RP"],
            "loss_reg_RP2CP":RP2CP_loss_dict["loss_reg_RP2CP"],
            "loss_conf_RP2CP":RP2CP_loss_dict["loss_conf_RP2CP"],
            "loss_grad_RP2CP":RP2CP_loss_dict["loss_grad_RP2CP"],
            "loss_reg_CP2D":CP2D_loss_dict["loss_reg_CP2D"],
            "loss_conf_CP2D":CP2D_loss_dict["loss_conf_CP2D"],
            "loss_grad_CP2D":CP2D_loss_dict["loss_grad_CP2D"],
            }



def overlap_to_distance(overlap_matrix: torch.Tensor, method: str = "reciprocal", epsilon: float = 1e-5) -> torch.Tensor:
    """Convert overlap matrix to distance matrix (higher overlap = smaller distance)
    Args:
        overlap_matrix: Input overlap adjacency matrix, shape [*, S, S]
        method: Conversion method, "reciprocal" (default) or "linear"
        epsilon: Small value to avoid division by zero
    Returns:
        Distance matrix with zero diagonal, shape [*, S, S]
    """
    if method == "reciprocal":
        dist_mat = 1.0 / (overlap_matrix + epsilon)
    elif method == "linear":
        dist_mat = 1.0 - overlap_matrix
    else:
        raise ValueError(f"Invalid method: {method}, choose 'reciprocal' or 'linear'")
    
    # Set diagonal (self-distance) to 0 with broadcast
    *dims, S, _ = dist_mat.shape
    eye = torch.eye(S, device=dist_mat.device, dtype=dist_mat.dtype)
    return dist_mat * (1 - eye)

def batch_dijkstra(dist_matrix: torch.Tensor) -> torch.Tensor:
    """Batch Dijkstra's algorithm for all sources (single batch)
    Args:
        dist_matrix: Distance adjacency matrix for one batch, shape [S, S]
    Returns:
        Shortest path matrix (source x target), shape [S, S]
    """
    S = dist_matrix.shape[0]
    device, dtype = dist_matrix.device, dist_matrix.dtype
    INF = torch.tensor(float('inf'), device=device, dtype=dtype)

    # Initialize distance matrix: [S, S] (each row is a source)
    dist = dist_matrix.clone()
    # Self-distance = 0
    dist.fill_diagonal_(0.0)
    # Visited mask: [S, S] (per source)
    visited = torch.zeros((S, S), dtype=torch.bool, device=device)

    for _ in range(S):
        # Find minimum distance node for each source (avoid visited nodes)
        min_dists, u_indices = torch.min(dist.masked_fill(visited, INF), dim=1)
        # Terminate if all sources have no reachable nodes left
        if (min_dists == INF).all():
            break
        
        # Mark visited nodes for each source
        source_indices = torch.arange(S, device=device)
        visited[source_indices, u_indices] = True
        
        # Relaxation step: update distances for all sources in parallel
        # New distance: dist[source, u] + dist_matrix[u, target]
        u_dist = dist[source_indices, u_indices].unsqueeze(1)  # [S, 1]
        new_dists = u_dist + dist_matrix[u_indices, :]  # [S, S] (broadcast)
        # Update dist with minimum of current and new distance
        dist = torch.min(dist, new_dists)

    return dist

def find_best_reference_mask(gt_adj_matrix: torch.Tensor, 
                             overlap2dist_method: str = "reciprocal",
                             top_k: int = 1) -> torch.Tensor:
    """Efficiently find the top-k best reference frames and generate corresponding mask matrix
    Args:
        gt_adj_matrix: Input overlap adjacency matrix with shape [B, S, S] 
                       (B = batch size, S = number of cameras/frames)
        overlap2dist_method: Conversion method from overlap degree to distance metric, default is 'reciprocal'
        top_k: Number of frames with the smallest total distance to select for each batch, must be positive integer
               Default is 1, which is fully compatible with the original logic
    Returns:
        Best reference frame mask with shape [B, S], where 1 indicates the top-k optimal frames 
        and 0 indicates the non-selected frames (float dtype)
    """
    B, S, _ = gt_adj_matrix.shape
    device, dtype = gt_adj_matrix.device, gt_adj_matrix.dtype

    # Boundary validation for top_k parameter to avoid runtime errors
    assert isinstance(top_k, int) and top_k >= 1, f"top_k must be a positive integer, but got {top_k}"
    top_k = S if top_k > S else top_k

    # Step 1: Convert overlap adjacency matrix to distance matrix (batch-wise vectorized operation)
    dist_mat = overlap_to_distance(gt_adj_matrix, overlap2dist_method)
    
    # Step 2: Calculate batch-wise shortest path and sum total distance for each frame
    # Eliminate Python loop over frame sources, keep only batch loop
    total_dist = torch.zeros((B, S), device=device, dtype=dtype)
    for b in range(B):
        # Get shortest path matrix for current batch sample, shape [S, S]
        shortest_paths = batch_dijkstra(dist_mat[b])
        # Sum along target dimension to get total distance from each source frame to all others, shape [S]
        total_dist[b] = shortest_paths.sum(dim=1)

    # Step 3: Core logic - Generate top-k mask matrix (vectorized implementation without Python loops)
    # Get indices of the top-k smallest total distance values along frame dimension (dim=1)
    # Set largest=False to select minimum values instead of maximum values
    topk_vals, topk_indices = torch.topk(total_dist, k=top_k, dim=1, largest=False)

    # Initialize mask matrix with all zeros, same shape/dtype/device as total_dist
    topk_mask = torch.zeros_like(total_dist, device=device, dtype=dtype)
    # Batch assignment: set the positions of top-k indices to 1.0 for each batch
    topk_mask.scatter_(dim=1, index=topk_indices, value=1.0)

    return topk_mask


def compute_covisibility_loss(predictions: Dict[str, Tensor], batch: Dict[str, Tensor], **kwargs) -> LossDict:
    """
    Compute covisibility loss for reference frame selection.
    
    Loss measures how well predicted covisibility scores match ground truth adjacency matrix
    to identify the most covisible reference frame in each sequence.
    
    Args:
        predictions: Contains "covisibility_scores" [B, S]
        batch: Contains "sample_adj_matrix" [B, S, S] and "ids" [B, S]
        
    Returns:
        Loss dictionary with binary cross-entropy loss for covisibility prediction
    """
    # Extract predictions and ground truth
    pred_covis = predictions["covisibility_scores"]  # [B, S]
    gt_adj_matrix = batch["sample_adj_matrix"]       # [B, S, S]
      
    gt_ref_mask = find_best_reference_mask(gt_adj_matrix)

    # Binary cross-entropy loss for reference frame classification
    bce_loss = F.binary_cross_entropy_with_logits(pred_covis, gt_ref_mask)

    return {"loss_covisibility": bce_loss}


def compute_camera_loss(
    predictions: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    loss_type: str = "l1",
    gamma: float = 0.6,
    pose_encoding_type: str = "absT_quaR",
    **kwargs
) -> LossDict:
    """
    Compute camera pose loss with temporal weighting for multi-stage predictions.
    
    Args:
        predictions: Contains "pose_enc_list" (list of pose encodings per stage)
        batch: Contains "extrinsics" (ground truth pose) and "point_masks" (valid frame mask)
        loss_type: Loss type ("l1" or "l2")
        gamma: Temporal decay weight for multi-stage predictions
        pose_encoding_type: Format of pose encoding (e.g., "absT_quaR")
        weight_trans: Weight for translation component loss
        weight_rot: Weight for rotation component loss
        
    Returns:
        Loss dictionary with translation, rotation, and confidence losses
    """
    # Extract predictions and ground truth
    pred_pose_list = predictions["pose_enc_list"]
    point_masks = batch["point_masks"]
    gt_extrinsics = batch["extrinsics"]
    n_stages = len(pred_pose_list)

    # Filter valid frames (minimum 100 valid points)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100

    # Encode ground truth pose to match prediction format
    gt_pose_enc = extri_to_pose_encoding360(gt_extrinsics, pose_encoding_type=pose_encoding_type)

    # Initialize loss accumulators
    total_loss_T = total_loss_R = total_loss_conf_T = total_loss_conf_R = 0.0
    # total_rel_loss_T = total_rel_loss_R = 0.0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        stage_weight = gamma ** (n_stages - stage_idx - 1)  # Later stages get higher weight
        pred_pose = pred_pose_list[stage_idx]

        if valid_frame_mask.sum() == 0:
            # No valid frames - return zero loss to avoid gradient issues
            loss_T = loss_R = loss_conf_T = loss_conf_R = (pred_pose * 0).mean()
            # rel_loss_T = rel_loss_R = (pred_pose * 0).mean()
            
        else:
            # Compute abs loss only for valid frames
            abs_loss_T, abs_loss_R, abs_loss_conf_T, abs_loss_conf_R = abs_camera_loss(
                pred_pose[valid_frame_mask].clone(),
                gt_pose_enc[valid_frame_mask].clone(),
                loss_type=loss_type
            )
   
            loss_T = abs_loss_T
            loss_R = abs_loss_R
            loss_conf_T = abs_loss_conf_T
            loss_conf_R = abs_loss_conf_R
            
            # unhelpful
            # rel_loss_T, rel_loss_R = rel_camera_loss(
            #     pred_pose[valid_frame_mask].clone(),
            #     gt_extrinsics[valid_frame_mask].clone()
            # )
            
        # Accumulate weighted losses
        total_loss_T += loss_T * stage_weight
        total_loss_R += loss_R * stage_weight
        total_loss_conf_T += loss_conf_T * stage_weight
        total_loss_conf_R += loss_conf_R * stage_weight
        # total_rel_loss_T += rel_loss_T * stage_weight
        # total_rel_loss_R += rel_loss_R * stage_weight
     

    # Average losses across stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_conf_T = total_loss_conf_T / n_stages
    avg_loss_conf_R = total_loss_conf_R / n_stages
    # avg_rel_loss_T = total_rel_loss_T / n_stages
    # avg_rel_loss_R = total_rel_loss_R / n_stages
    
   
    # Total weighted camera loss
    total_camera_loss = avg_loss_T + avg_loss_R
    total_camera_conf_loss = avg_loss_conf_T + avg_loss_conf_R
    # total_camera_rel_loss = avg_rel_loss_T + avg_rel_loss_R

    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_camera_conf": total_camera_conf_loss,
        "loss_conf_T": avg_loss_conf_T,
        "loss_conf_R": avg_loss_conf_R,
        # "loss_camera_rel": total_camera_rel_loss,
        # "loss_rel_T": avg_rel_loss_T,
        # "loss_rel_R": avg_rel_loss_R,
    }


def compute_point_loss(
    predictions: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: Optional[str] = None,
    valid_range: float = -1,
    **kwargs
) -> LossDict:
    """
    Compute 3D world point reconstruction loss with confidence weighting and gradient loss.
    
    Args:
        predictions: Contains "world_points" [B, S, H, W, 3] and "world_points_conf" [B, S, H, W]
        batch: Contains "world_points" (ground truth) and "point_masks" (valid pixel mask)
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        gradient_loss_fn: Type of gradient loss ("normal" or "grad")
        valid_range: Quantile range for outlier filtering
        
    Returns:
        Loss dictionary with confidence, regression, and gradient losses
    """
    return _compute_3d_point_loss_base(
        predictions=predictions,
        batch=batch,
        point_type="world_points",
        gamma=gamma,
        alpha=alpha,
        gradient_loss_fn=gradient_loss_fn,
        valid_range=valid_range
    )


def compute_cam_point_loss(
    predictions: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: Optional[str] = None,
    valid_range: float = -1,
    **kwargs
) -> LossDict:
    """
    Compute 3D camera point reconstruction loss (see compute_point_loss for details).
    
    Args:
        predictions: Contains "cam_points" and "cam_points_conf"
        batch: Contains "cam_points" (ground truth) and "point_masks"
        
    Returns:
        Loss dictionary with confidence, regression, and gradient losses
    """
    return _compute_3d_point_loss_base(
        predictions=predictions,
        batch=batch,
        point_type="cam_points",
        gamma=gamma,
        alpha=alpha,
        gradient_loss_fn=gradient_loss_fn,
        valid_range=valid_range
    )


def compute_rotated_point_loss(
    predictions: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: Optional[str] = None,
    valid_range: float = -1,
    **kwargs
) -> LossDict:
    """
    Compute rotated 3D point reconstruction loss (see compute_point_loss for details).
    
    Args:
        predictions: Contains "rotated_points" and "rotated_points_conf"
        batch: Contains "rotated_points" (ground truth) and "point_masks"
        
    Returns:
        Loss dictionary with confidence, regression, and gradient losses
    """
    return _compute_3d_point_loss_base(
        predictions=predictions,
        batch=batch,
        point_type="rotated_points",
        gamma=gamma,
        alpha=alpha,
        gradient_loss_fn=gradient_loss_fn,
        valid_range=valid_range
    )


def compute_depth_loss(
    predictions: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: Optional[str] = None,
    valid_range: float = -1,
    **kwargs
) -> LossDict:
    """
    Compute depth prediction loss with confidence weighting and gradient loss for spatial smoothness.
    
    Args:
        predictions: Contains "depth" [B, S, H, W, 1] and "depth_conf" [B, S, H, W]
        batch: Contains "depths" (ground truth) and "point_masks" (valid pixel mask)
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        gradient_loss_fn: Type of gradient loss ("normal" or "grad")
        valid_range: Quantile range for outlier filtering
        
    Returns:
        Loss dictionary with confidence, regression, and gradient losses
    """
    # Extract predictions and ground truth
    pred_depth = predictions["depth"]
    pred_depth_conf = predictions["depth_conf"]
    gt_depth = batch["depths"]
    gt_depth_mask = batch["point_masks"].clone()

    # Sanitize ground truth depth
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]  # Add channel dimension [B, S, H, W, 1]

    # Return zero loss if insufficient valid points
    if gt_depth_mask.sum() < 100:
        dummy_loss = (0.0 * pred_depth).mean()
        return {
            "loss_conf_depth": dummy_loss,
            "loss_reg_depth": dummy_loss,
            "loss_grad_depth": dummy_loss,
        }

    # Compute core regression loss with confidence and gradient terms
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred=pred_depth,
        gt=gt_depth,
        mask=gt_depth_mask,
        conf=pred_depth_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range
    )

    return {
        "loss_conf_depth": loss_conf,
        "loss_reg_depth": loss_reg,
        "loss_grad_depth": loss_grad,
    }


# -------------------------- Core Loss Helpers --------------------------
def camera_loss_single(
    pred_pose_enc: Tensor,
    gt_pose_enc: Tensor,
    loss_type: str = "l1"
) -> Tuple[Tensor, Tensor]:
    """
    Compute basic camera pose loss (translation + rotation) without confidence weighting.
    
    Args:
        pred_pose_enc: Predicted pose encoding [B, S, D]
        gt_pose_enc: Ground truth pose encoding [B, S, D]
        loss_type: Loss type ("l1" or "l2")
        
    Returns:
        Tuple of translation loss and rotation loss
    """
    # Split pose encoding into translation (first 3 dims) and rotation (next 4 dims)
    if loss_type == "l1":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
    elif loss_type == "l2":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type} (use 'l1' or 'l2')")

    # Fix numerical issues (NaN/Inf)
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")

    # Clamp extreme values and average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()

    return loss_T, loss_R


def abs_camera_loss(
    pred_pose_enc: Tensor,
    gt_pose_enc: Tensor,
    loss_type: str = "l1",
    gamma: float = 1.0,
    alpha: float = 0.2
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute abs camera pose loss with confidence weighting for translation and rotation.
    
    Confidence loss formula: gamma * loss * conf - alpha * log(conf)
    
    Args:
        pred_pose_enc: Predicted pose encoding [B, S, D] (includes confidence channels)
        gt_pose_enc: Ground truth pose encoding [B, S, D]
        loss_type: Loss type ("l1" or "l2")
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        
    Returns:
        Tuple of (translation loss, rotation loss, translation confidence loss, rotation confidence loss)
    """
    # Compute basic pose loss
    loss_T, loss_R = camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type)

    # Extract confidence values (last two channels)
    conf_T = pred_pose_enc[..., -2:-1]
    conf_R = pred_pose_enc[..., -1:]

    # Compute confidence-weighted loss
    loss_conf_T = gamma * loss_T * conf_T - alpha * torch.log(conf_T)
    loss_conf_R = gamma * loss_R * conf_R - alpha * torch.log(conf_R)

    # Fix numerical issues
    loss_conf_T = check_and_fix_inf_nan(loss_conf_T, "loss_conf_T")
    loss_conf_R = check_and_fix_inf_nan(loss_conf_R, "loss_conf_R")

    # Average confidence losses
    loss_conf_T = loss_conf_T.mean()
    loss_conf_R = loss_conf_R.mean()

    return loss_T, loss_R, loss_conf_T, loss_conf_R


# ---------------------------------------------------------------------------
# Pi3's CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------
from argus.utils.geometry import closed_form_inverse_se3

def rot_ang_loss(R, Rgt, eps=1e-6):
    """
    Args:
        R: estimated rotation matrix [B, 3, 3]
        Rgt: ground-truth rotation matrix [B, 3, 3]
    Returns:  
        R_err: rotation angular error 
    """
    residual = torch.matmul(R.transpose(1, 2), Rgt)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
    return R_err.mean()         # [0, 3.14]

def rel_camera_loss(pred_pose_enc, gt_pose):
    B, S, _ = pred_pose_enc.shape
    t = pred_pose_enc[..., :3]  # [B, S, 3]
    q = F.normalize(pred_pose_enc[..., 3:7], dim=-1)  # [B, S, 4], unit quaternion
    R = quat_to_mat(q)  # [B, S, 3, 3] # Rotation matrix from quaternion

    pred_pose = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # [B, S, 3, 4]
    pred_pose = torch.cat([pred_pose, torch.tensor([[[0, 0, 0, 1]]], device=pred_pose.device).repeat(B, S, 1, 1)], dim=-2)  # [B, S, 4, 4]
    
    pred_w2c = closed_form_inverse_se3(pred_pose.reshape(-1, 4, 4)).reshape(B, S, 4, 4)
    gt_w2c = closed_form_inverse_se3(gt_pose.reshape(-1, 4, 4)).reshape(B, S, 4, 4)
    
    pred_w2c_exp = pred_w2c.unsqueeze(2)
    pred_pose_exp = pred_pose.unsqueeze(1)
    
    gt_w2c_exp = gt_w2c.unsqueeze(2)
    gt_pose_exp = gt_pose.unsqueeze(1)
    
    pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
    gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

    mask = ~torch.eye(S, dtype=torch.bool, device=pred_pose.device)

    t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
    R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
    
    t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
    R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

    rel_loss_T = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1) * 100.0
    
    rel_loss_R = rot_ang_loss(
        R_pred.reshape(-1, 3, 3), 
        R_gt.reshape(-1, 3, 3)
    )
    
    # Fix numerical issues and average
    rel_loss_T = check_and_fix_inf_nan(rel_loss_T, "rel_loss_T")
    rel_loss_R = check_and_fix_inf_nan(rel_loss_R, "rel_loss_R")
    
    return rel_loss_T, rel_loss_R


def _compute_3d_point_loss_base(
    predictions: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    point_type: str,
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: Optional[str] = None,
    valid_range: float = -1
) -> LossDict:
    """
    Base function for 3D point loss calculation (world/camera/rotated points).
    
    Args:
        predictions: Model predictions containing point_type and point_type + "_conf"
        batch: Ground truth containing point_type and "point_masks"
        point_type: Type of points ("world_points", "cam_points", "rotated_points")
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        gradient_loss_fn: Type of gradient loss ("normal" or "grad")
        valid_range: Quantile range for outlier filtering
        
    Returns:
        Loss dictionary with confidence, regression, and gradient losses
    """
    # Extract predictions and ground truth
    pred_pts = predictions[point_type]
    pred_pts_conf = predictions[f"{point_type}_conf"]
    gt_pts = batch[point_type]
    gt_pts_mask = batch["point_masks"]

    # Sanitize ground truth points
    gt_pts = check_and_fix_inf_nan(gt_pts, f"gt_{point_type}")

    # Return zero loss if insufficient valid points
    if gt_pts_mask.sum() < 100:
        dummy_loss = (0.0 * pred_pts).mean()
        return {
            f"loss_conf_{point_type}": dummy_loss,
            f"loss_reg_{point_type}": dummy_loss,
            f"loss_grad_{point_type}": dummy_loss,
        }

    # Compute core regression loss with confidence and gradient terms
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred=pred_pts,
        gt=gt_pts,
        mask=gt_pts_mask,
        conf=pred_pts_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range
    )

    return {
        f"loss_conf_{point_type}": loss_conf,
        f"loss_reg_{point_type}": loss_reg,
        f"loss_grad_{point_type}": loss_grad,
    }


def _compute_joint_3d_point_loss_base(
    pred_pts: Tensor,
    pred_pts_conf: Tensor,
    gt_pts: Tensor,
    gt_pts_mask: Tensor,
    point_type: str,
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: Optional[str] = None,
    valid_range: float = -1
) -> LossDict:
    """
    Base function for 3D point loss calculation (world/camera/rotated points).
    
    Args:
        
        point_type: Type of points ("world_points", "cam_points", "rotated_points")
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        gradient_loss_fn: Type of gradient loss ("normal" or "grad")
        valid_range: Quantile range for outlier filtering
        
    Returns:
        Loss dictionary with confidence, regression, and gradient losses
    """
    # Sanitize ground truth points
    gt_pts = check_and_fix_inf_nan(gt_pts, f"gt_{point_type}")

    # Return zero loss if insufficient valid points
    if gt_pts_mask.sum() < 100:
        dummy_loss = (0.0 * pred_pts).mean()
        return {
            f"loss_conf_{point_type}": dummy_loss,
            f"loss_reg_{point_type}": dummy_loss,
            f"loss_grad_{point_type}": dummy_loss,
        }

    # Compute core regression loss with confidence and gradient terms
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred=pred_pts,
        gt=gt_pts,
        mask=gt_pts_mask,
        conf=pred_pts_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range
    )

    return {
        f"loss_conf_{point_type}": loss_conf,
        f"loss_reg_{point_type}": loss_reg,
        f"loss_grad_{point_type}": loss_grad,
    }
    

def regression_loss(
    pred: Tensor,
    gt: Tensor,
    mask: Tensor,
    conf: Optional[Tensor] = None,
    gradient_loss_fn: Optional[str] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
    valid_range: float = -1
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Core regression loss with confidence weighting and optional gradient loss for spatial smoothness.
    
    Confidence loss formula: gamma * ||pred - gt|| * conf - alpha * log(conf)
    
    Args:
        pred: Predicted values [B, S, H, W, C]
        gt: Ground truth values [B, S, H, W, C]
        mask: Valid pixel mask [B, S, H, W]
        conf: Confidence scores [B, S, H, W] (optional)
        gradient_loss_fn: Type of gradient loss ("normal" or "grad")
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        valid_range: Quantile range for outlier filtering
        
    Returns:
        Tuple of (confidence loss, gradient loss, regression loss)
    """
    # Get tensor dimensions
    b, s, h, w, c = pred.shape

    # Compute L2 regression loss for valid pixels only
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Compute confidence-weighted loss if confidence is provided
    if conf is not None:
        loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
        loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
    else:
        loss_conf = loss_reg.clone()

    # Compute gradient loss for spatial smoothness (multi-scale)
    loss_grad = torch.tensor(0.0, device=pred.device)
    if gradient_loss_fn is not None:
        # Prepare confidence for gradient loss
        conf_grad = conf.reshape(b*s, h, w) if gradient_loss_fn == "conf" else None
        
        # Compute multi-scale gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            prediction=pred.reshape(b*s, h, w, c),
            target=gt.reshape(b*s, h, w, c),
            mask=mask.reshape(b*s, h, w),
            scales=3,
            gradient_loss_fn=normal_loss if gradient_loss_fn == "normal" else gradient_loss,
            conf=conf_grad
        )

    # Process confidence loss (filter outliers and average)
    loss_conf = _process_loss_component(loss_conf, mask, valid_range, "loss_conf")
    
    # Process regression loss (filter outliers and average)
    loss_reg = _process_loss_component(loss_reg, mask, valid_range, "loss_reg")

    return loss_conf, loss_grad, loss_reg


# -------------------------- Spatial Loss Helpers --------------------------
def gradient_loss_multi_scale_wrapper(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    scales: int = 4,
    gradient_loss_fn: Callable = None,
    conf: Optional[Tensor] = None
) -> Tensor:
    """
    Multi-scale gradient loss wrapper for capturing spatial structure at different resolutions.
    
    Applies gradient loss at multiple scales by subsampling input with step size 2^scale.
    
    Args:
        prediction: Predicted values [B, H, W, C]
        target: Ground truth values [B, H, W, C]
        mask: Valid pixel mask [B, H, W]
        scales: Number of scales to apply
        gradient_loss_fn: Gradient loss function to apply
        conf: Confidence scores [B, H, W] (optional)
        
    Returns:
        Average gradient loss across all scales
    """
    total_loss = torch.tensor(0.0, device=prediction.device)
    
    for scale in range(scales):
        step = 2 ** scale  # Subsampling step size
        total_loss += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )
    
    return total_loss / scales


def normal_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    cos_eps: float = 1e-8,
    conf: Optional[Tensor] = None,
    gamma: float = 1.0,
    alpha: float = 0.2
) -> Tensor:
    """
    Surface normal loss for geometric consistency of 3D point maps.
    
    Computes surface normals from 3D points using cross products of neighboring points,
    then measures angular difference between predicted and ground truth normals.
    
    Args:
        prediction: Predicted 3D points [B, H, W, 3]
        target: Ground truth 3D points [B, H, W, 3]
        mask: Valid pixel mask [B, H, W]
        cos_eps: Epsilon for numerical stability in cosine calculation
        conf: Confidence scores [B, H, W] (optional)
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        
    Returns:
        Average surface normal loss
    """
    # Convert point maps to surface normals
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    # Only consider pixels with valid normals in both prediction and ground truth
    valid_mask = pred_valids & gt_valids
    valid_count = torch.sum(valid_mask)

    # Return zero loss if insufficient valid normals
    if valid_count < 10:
        return torch.tensor(0.0, device=prediction.device)

    # Extract valid normals and compute cosine similarity
    pred_normals_valid = pred_normals[valid_mask].clone()
    gt_normals_valid = gt_normals[valid_mask].clone()
    dot_product = torch.sum(pred_normals_valid * gt_normals_valid, dim=-1)
    dot_product = torch.clamp(dot_product, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta) (numerically stable alternative to arccos)
    loss = 1 - dot_product
    loss = check_and_fix_inf_nan(loss, "normal_loss")

    # Apply confidence weighting if provided
    if conf is not None:
        conf_expanded = conf[None, ...].expand(4, -1, -1, -1)
        conf_valid = conf_expanded[valid_mask].clone()
        loss = gamma * loss * conf_valid - alpha * torch.log(conf_valid)

    return loss.mean() if loss.numel() >= 10 else torch.tensor(0.0, device=prediction.device)


def gradient_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    conf: Optional[Tensor] = None,
    gamma: float = 1.0,
    alpha: float = 0.2
) -> Tensor:
    """
    Gradient loss for spatial smoothness (L1 difference of adjacent pixels).
    
    Computes horizontal and vertical gradient differences between prediction and ground truth.
    
    Args:
        prediction: Predicted values [B, H, W, C]
        target: Ground truth values [B, H, W, C]
        mask: Valid pixel mask [B, H, W]
        conf: Confidence scores [B, H, W] (optional)
        gamma: Weight for confidence loss term
        alpha: Weight for confidence regularization term
        
    Returns:
        Average gradient loss
    """
    # Expand mask to match channel dimension
    mask_expanded = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    valid_pixel_count = torch.sum(mask_expanded)

    # Return zero loss if no valid pixels
    if valid_pixel_count == 0:
        return torch.tensor(0.0, device=prediction.device)

    # Compute prediction-target difference
    diff = prediction - target
    diff = diff * mask_expanded

    # Compute horizontal gradients (x-direction)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask_expanded[:, :, 1:] * mask_expanded[:, :, :-1]
    grad_x = grad_x * mask_x

    # Compute vertical gradients (y-direction)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask_expanded[:, 1:, :] * mask_expanded[:, :-1, :]
    grad_y = grad_y * mask_y

    # Apply confidence weighting if provided
    if conf is not None:
        conf_expanded = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf_expanded[:, :, 1:]
        conf_y = conf_expanded[:, 1:, :]
        
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Clamp extreme values and compute total gradient loss
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)
    total_grad = torch.sum(grad_x) + torch.sum(grad_y)

    return total_grad / valid_pixel_count


def point_map_to_normal(
    point_map: Tensor,
    mask: Tensor,
    eps: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    """
    Convert 3D point map to surface normals using cross products of neighboring points.
    
    Computes normals in four different directions for robustness:
    1. Up × Left
    2. Left × Down
    3. Down × Right
    4. Right × Up
    
    Args:
        point_map: 3D point map [B, H, W, 3]
        mask: Valid pixel mask [B, H, W]
        eps: Epsilon for numerical stability in normalization
        
    Returns:
        Tuple of (surface normals [4, B, H, W, 3], validity mask [4, B, H, W])
    """
    with torch.amp.autocast("cuda", enabled=False):
        # Pad inputs to handle boundary pixels
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
        padded_pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0)
        padded_pts = padded_pts.permute(0, 2, 3, 1)

        # Extract neighboring points
        center = padded_pts[:, 1:-1, 1:-1, :]  # Center pixel
        up = padded_pts[:, :-2, 1:-1, :]       # Upper neighbor
        left = padded_pts[:, 1:-1, :-2, :]     # Left neighbor
        down = padded_pts[:, 2:, 1:-1, :]      # Lower neighbor
        right = padded_pts[:, 1:-1, 2:, :]     # Right neighbor

        # Compute direction vectors from center to neighbors
        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        # Compute surface normals via cross products
        n1 = torch.cross(up_dir, left_dir, dim=-1)    # Up × Left
        n2 = torch.cross(left_dir, down_dir, dim=-1)  # Left × Down
        n3 = torch.cross(down_dir, right_dir, dim=-1) # Down × Right
        n4 = torch.cross(right_dir, up_dir, dim=-1)   # Right × Up

        # Compute validity masks (all required neighbors must be valid)
        v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
        v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)   # [4, B, H, W]

        # Normalize normals to unit length
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


# -------------------------- Utility Functions --------------------------
def _process_loss_component(
    loss_tensor: Tensor,
    mask: Tensor,
    valid_range: float,
    loss_name: str
) -> Tensor:
    """
    Helper to process loss components (filter outliers, fix NaN/Inf, average).
    
    Args:
        loss_tensor: Raw loss values
        mask: Valid pixel mask
        valid_range: Quantile range for outlier filtering
        loss_name: Name for error reporting
        
    Returns:
        Processed and averaged loss value
    """
    if loss_tensor.numel() == 0:
        return (0.0 * mask).mean()

    # Filter outliers using quantile thresholding
    if valid_range > 0:
        loss_tensor = filter_by_quantile(loss_tensor, valid_range)

    # Fix numerical issues and average
    loss_tensor = check_and_fix_inf_nan(loss_tensor, loss_name)
    return loss_tensor.mean() if loss_tensor.numel() > 0 else (0.0 * mask).mean()


def filter_by_quantile(
    loss_tensor: Tensor,
    valid_range: float,
    min_elements: int = 1000,
    hard_max: float = 100
) -> Tensor:
    """
    Filter loss tensor by keeping only values below a quantile threshold to remove outliers.
    
    Args:
        loss_tensor: Raw loss values
        valid_range: Quantile threshold (0-1)
        min_elements: Minimum elements required for filtering
        hard_max: Hard upper limit for loss values
        
    Returns:
        Filtered loss tensor with outliers removed
    """
    # Skip filtering if too few elements
    if loss_tensor.numel() <= min_elements:
        return loss_tensor

    # Random subsample for large tensors to avoid memory issues
    if loss_tensor.numel() > 100_000_000:
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # Clamp extreme values
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold and filter
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh.item(), hard_max)
    quantile_mask = loss_tensor < quantile_thresh

    # Return filtered tensor if enough elements remain
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    
    return loss_tensor


def torch_quantile(
    input: Tensor,
    q: float,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: Optional[Tensor] = None
) -> Tensor:
    """
    Efficient quantile calculation using k-th value (avoids PyTorch's 2^24 element limit).
    
    Args:
        input: Input tensor
        q: Quantile value (0-1, scalar only)
        dim: Dimension to compute quantile over (None = flatten)
        keepdim: Keep dimension after computation (only False supported)
        interpolation: Interpolation method ("nearest", "lower", "higher")
        out: Output tensor (only None supported)
        
    Returns:
        Quantile value tensor
    """
    # Validate input quantile
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Quantile must be scalar between 0 and 1 (got {q})")

    # Handle flattening for dim=None
    dim_was_none = dim is None
    if dim_was_none:
        dim = 0
        input = input.reshape(-1, *[1] * (input.ndim - 1))

    # Validate parameters
    if out is not None:
        raise ValueError("Output tensor not supported (use None)")
    if keepdim:
        raise ValueError("keepdim=True not supported (use False)")

    # Select interpolation method
    if interpolation == "nearest":
        interp_fn = round
    elif interpolation == "lower":
        interp_fn = floor
    elif interpolation == "higher":
        interp_fn = ceil
    else:
        raise ValueError(f"Interpolation '{interpolation}' not supported (use nearest/lower/higher)")

    # Compute k-th value index
    k = interp_fn(q * (input.shape[dim] - 1)) + 1
    quantile_val = torch.kthvalue(input, k, dim=dim, keepdim=True)[0]

    # Reshape output
    if dim_was_none:
        quantile_val = quantile_val.squeeze()
    else:
        quantile_val = quantile_val.squeeze(dim)

    return quantile_val