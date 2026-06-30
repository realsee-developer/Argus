import torch
import torch.nn as nn
from typing import Optional, Dict
from huggingface_hub import PyTorchModelHubMixin

# Import model components
from argus.models.aggregator import Aggregator
from argus.heads.camera_head import CameraHead
from argus.heads.dpt_head import DPTHead
from argus.heads.utils import reorder_by_reference


class Argus(nn.Module, PyTorchModelHubMixin):
    """
    Argus multi-task vision model for camera pose estimation, depth prediction, and 3D points.
    
    Integrates an aggregator backbone with task-specific heads for:
    - Camera pose encoding
    - Depth map prediction
    - 3D camera/rotated/world point prediction
    
    Args:
        img_size: Input image size (height/width, assumes square) (default: 518)
        patch_size: Patch size for vision transformer backbone (default: 14)
        embed_dim: Embedding dimension for transformer features (default: 1024)
        enable_camera: Enable camera pose estimation head (default: True)
        enable_depth: Enable depth prediction head (default: True)
        enable_cam_point: Enable camera coordinate 3D point prediction head (default: False)
        enable_rotated_point: Enable rotated 3D point prediction head (default: False)
        enable_point: Enable world coordinate 3D point prediction head (default: False, Please do not set it to True during training)
    
    Note:
        All heads share the same aggregated transformer features from the Aggregator backbone.
        Each DPT-based head outputs both predictions and confidence scores.
    """
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_cam_point: bool = False,
        enable_rotated_point: bool = False,
        enable_point: bool = False,
        reorder_by_learning_ref: bool = True,
        restore_metric_scale: bool = False 
    ) -> None:
        super().__init__()
        # For inference
        self.restore_metric_scale = restore_metric_scale
        self.reorder_by_learning_ref = reorder_by_learning_ref
        
        # Backbone and geometry transformer
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            reorder_by_learning_ref=reorder_by_learning_ref,
        )
        
        # Task-specific prediction heads (lazy initialization based on flags)
        self.camera_head: Optional[CameraHead] = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.depth_head: Optional[DPTHead] = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1"
        ) if enable_depth else None
        
        # 3D point prediction heads (shared architecture, different output semantics)
        self.cam_point_head: Optional[DPTHead] = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1"
        ) if enable_cam_point else None
        
        self.rotated_point_head: Optional[DPTHead] = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1"
        ) if enable_rotated_point else None
        
        self.point_head: Optional[DPTHead] = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1"
        ) if enable_point else None

    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Argus model.
        
        Automatically adds batch dimension if missing and processes multi-task predictions.
        
        Args:
            images: Input RGB images with shape:
                - [S, 3, H, W] (sequence without batch) or 
                - [B, S, 3, H, W] (batch of sequences)
                Values in range [0, 1], where:
                - B: batch size
                - S: sequence length (number of frames)
                - 3: RGB channels
                - H/W: image height/width (matches img_size)
        
        Returns:
            Dictionary of model predictions with task-specific outputs:
                Common outputs:
                    - covisibility_scores: Covisibility scores from aggregator (shape varies)
                    - ref_idx: Reference frame indices (shape varies)
                
                Camera head outputs (if enabled):
                    - pose_enc: Final camera pose encoding [B, S, 9]
                    - pose_enc_list: List of pose encodings from all iterations [List[torch.Tensor]]
                
                Depth head outputs (if enabled):
                    - depth: Predicted depth maps [B, S, H, W, 1]
                    - depth_conf: Depth prediction confidence [B, S, H, W]
                
                Camera point head outputs (if enabled):
                    - cam_points: 3D camera coordinates per pixel [B, S, H, W, 3]
                    - cam_points_conf: Camera point confidence [B, S, H, W]
                
                Rotated point head outputs (if enabled):
                    - rotated_points: Rotated 3D coordinates per pixel [B, S, H, W, 3]
                    - rotated_points_conf: Rotated point confidence [B, S, H, W]
                
                World point head outputs (if enabled):
                    - world_points: 3D world coordinates per pixel [B, S, H, W, 3]
                    - world_points_conf: World point confidence [B, S, H, W]
                    
                Inference-only outputs (not training):
                    - images: Original input images (for visualization) [B, S, 3, H, W]
        """
        # Add batch dimension if missing (handle [S,3,H,W] -> [1,S,3,H,W])
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        # Extract aggregated features from backbone
        (
            aggregated_tokens_list,  # List of aggregated transformer tokens across iterations
            patch_start_idx,         # Patch start indices for feature reconstruction
            covisibility_scores,     # Covisibility scores between frames
            ref_idx                  # Reference frame indices
        ) = self.aggregator(images)

        # Initialize prediction dictionary
        predictions: Dict[str, torch.Tensor] = {}

        # Disable mixed precision for precise prediction calculations
        with torch.amp.autocast("cuda", enabled=False):
            # Add aggregator outputs to predictions
            if covisibility_scores is not None:
                predictions["covisibility_scores"] = covisibility_scores
            if ref_idx is not None:
                predictions["ref_idx"] = ref_idx
            
            # Camera pose prediction (if enabled)
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # Use final iteration encoding
                predictions["pose_enc_list"] = pose_enc_list # Mutil-layer supervision
            
            # Depth prediction (if enabled)
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # Camera 3D point prediction (if enabled)
            if self.cam_point_head is not None:
                cam_pts3d, cam_pts3d_conf = self.cam_point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx
                )
                predictions["cam_points"] = cam_pts3d
                predictions["cam_points_conf"] = cam_pts3d_conf

            # Rotated 3D point prediction (if enabled)
            if self.rotated_point_head is not None:
                rotated_pts3d, rotated_pts3d_conf = self.rotated_point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx
                )
                predictions["rotated_points"] = rotated_pts3d
                predictions["rotated_points_conf"] = rotated_pts3d_conf

            # World 3D point prediction (if enabled)
            if self.point_head is not None:
                world_pts3d, world_pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = world_pts3d
                predictions["world_points_conf"] = world_pts3d_conf
            
            
        # Store input images for visualization during inference (skip in training)
        if not self.training:
            predictions["images"] = images
            if "ref_idx" in predictions:
                ref_idx = predictions["ref_idx"].detach()
                # Reorder all spatial/temporal data (exclude adjacency matrix and IDs)
                predictions["images"] = reorder_by_reference(predictions["images"], ref_idx)
        
            if self.restore_metric_scale:
                # Restore metric scale
                abs_scale = 10.0
                if self.camera_head is not None:
                    predictions["pose_enc"][...,:3] *= abs_scale
                if self.depth_head is not None:
                    predictions["depth"] *= abs_scale
                if self.cam_point_head is not None:
                    predictions["cam_points"] *= abs_scale
                if self.rotated_point_head is not None:
                    predictions["rotated_points"] *= abs_scale
                if self.point_head is not None:
                    predictions["world_points"] *= abs_scale
            

        return predictions