import torch
from typing import Tuple, Union
from .rotation import quat_to_mat, mat_to_quat


def extri_to_pose_encoding360(
    extrinsics: torch.Tensor,
    pose_encoding_type: Union[str, "absT_quaR"] = "absT_quaR"
) -> torch.Tensor:
    """
    Convert camera extrinsic parameters to a compact pose encoding (absolute translation + quaternion rotation).
    
    Transforms OpenCV-style camera extrinsics (3x4 [R|t] matrix) into a flattened encoding format
    suitable for machine learning tasks like pose prediction or representation learning.
    
    Args:
        extrinsics: Camera extrinsic matrices with shape [B, S, 3, 4] or [B, S, 4, 4]
            - B: Batch size
            - S: Sequence length (number of frames)
            - 3x4/4x4: Extrinsic matrix in OpenCV coordinate system (x-right, y-down, z-forward)
              representing the transformation from world to camera space ([R|t] where R=3x3 rotation, t=3x1 translation)
        pose_encoding_type: Type of pose encoding format (only "absT_quaR" supported):
            - "absT_quaR": Absolute translation (3D) + quaternion rotation (4D)
        
    Returns:
        Encoded pose tensor with shape [B, S, 7]
            - [:3]: Absolute translation vector (T) in world coordinates
            - [3:7]: Rotation represented as unit quaternion (quat)
    """
    # Extract rotation matrix (R) and translation vector (T) from extrinsics
    # Handle both 3x4 and 4x4 extrinsic matrix inputs
    R = extrinsics[:, :, :3, :3]  # [B, S, 3, 3] - rotation matrix
    T = extrinsics[:, :, :3, 3]   # [B, S, 3]    - translation vector

    if pose_encoding_type == "absT_quaR":
        # Convert rotation matrix to quaternion (4D)
        quat = mat_to_quat(R)
        
        # Concatenate translation and quaternion to form compact pose encoding
        pose_encoding = torch.cat([T, quat], dim=-1).float()
    else:
        raise NotImplementedError(f"Pose encoding type '{pose_encoding_type}' not supported. Only 'absT_quaR' is implemented.")

    return pose_encoding


def pose_encoding_to_extri360(
    pose_encoding: torch.Tensor,
    pose_encoding_type: Union[str, "absT_quaR"] = "absT_quaR"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert compact pose encoding back to full camera extrinsic parameters (inverse of extri_to_pose_encoding360).
    
    Reconstructs the 4x4 homogeneous extrinsic matrix from the flattened pose encoding,
    including extraction of confidence scores from the encoding's extra dimensions.
    
    Args:
        pose_encoding: Encoded pose tensor with shape [B, S, 9]
            - B: Batch size
            - S: Sequence length (number of frames)
            - [:3]: Absolute translation vector (T)
            - [3:7]: Rotation quaternion (quat)
            - [-2:]: Confidence scores for translation and rotation
        pose_encoding_type: Type of pose encoding format (only "absT_quaR" supported):
            - "absT_quaR": Absolute translation (3D) + quaternion rotation (4D)
        
    Returns:
        Tuple containing:
            1. extrinsics: Reconstructed camera extrinsic matrices with shape [B, S, 4, 4]
               (homogeneous matrix in OpenCV coordinate system: [R|t; 0 0 0 1])
            2. conf: Confidence scores with shape [B, S, 2]
               - [:, :, 0]: Translation confidence
               - [:, :, 1]: Rotation confidence
    
    Raises:
        NotImplementedError: If unsupported pose encoding type is provided
    """
    if pose_encoding_type == "absT_quaR":
        # Extract translation (T) and rotation quaternion (quat) from pose encoding
        T = pose_encoding[..., :3]       # [B, S, 3] - translation vector
        quat = pose_encoding[..., 3:7]   # [B, S, 4] - rotation quaternion

        # Convert quaternion back to rotation matrix (3x3)
        R = quat_to_mat(quat)  # [B, S, 3, 3]

        # Reconstruct 3x4 [R|t] matrix (rotation + translation)
        extri_3x4 = torch.cat([R, T[..., None]], dim=-1)  # [B, S, 3, 4]

        # Add homogeneous row [0, 0, 0, 1] to form 4x4 extrinsic matrix
        batch_size, seq_len = extri_3x4.shape[:2]
        homogenous_row = torch.tensor(
            [0, 0, 0, 1], 
            device=extri_3x4.device,
            dtype=extri_3x4.dtype
        ).expand(batch_size, seq_len, 1, 4)  # [B, S, 1, 4]
        
        # Combine to form 4x4 homogeneous extrinsic matrix
        extrinsics = torch.cat((extri_3x4, homogenous_row), dim=2)  # [B, S, 4, 4]

        # Extract confidence scores (last two dimensions of pose encoding)
        conf = pose_encoding[..., -2:]  # [B, S, 2]

        return extrinsics, conf
    
    raise NotImplementedError(f"Pose encoding type '{pose_encoding_type}' not supported. Only 'absT_quaR' is implemented.")