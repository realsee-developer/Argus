import torch
import numpy as np


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def pano_depth_to_points(depth_map, original_pano_shape=(560, 280), crop_ratio=0.15):
    """
    Convert batched cropped panoramic depth maps to 3D point clouds (PyTorch implementation).
    Assumption: Input depth maps are already cropped by crop_ratio on top and bottom.
    
    Args:
        depth_map (torch.Tensor): Input cropped depth map, shape [B, S, H_crop, W, 1]
        original_pano_shape (tuple): Original uncropped panorama size (W_ori, H_ori), default (560, 280)
        crop_ratio (float): Crop ratio of original panorama (top and bottom respectively), default 0.15
    
    Returns:
        torch.Tensor: 3D point cloud with shape [B, S, H_crop, W, 3]
    """
    # Validate input shape
    assert depth_map.dim() == 5 and depth_map.shape[-1] == 1, \
        f"Input must be [B, S, H_crop, W, 1], got {depth_map.shape}"
    
    B, S, H_crop, W, _ = depth_map.shape
    W_ori, H_ori = original_pano_shape
    device = depth_map.device  # Align tensor device automatically
    
    # Generate pixel grid coordinates (H_crop, W)
    px_grid, py_grid = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H_crop, device=device),
        indexing='xy'  # Consistent with numpy's meshgrid
    )
    
    # Restore to original panorama y-coordinates (compensate for cropping)
    crop_top = int(crop_ratio * H_ori)
    py_ori = py_grid + crop_top
    
    # Compute spherical coordinates (lat: latitude, long: longitude)
    lat = (py_ori / H_ori - 0.5) * torch.pi
    long = (px_grid / W_ori - 0.5) * 2 * torch.pi
    
    # Remove channel dim and compute 3D Cartesian coordinates
    dist = depth_map.squeeze(-1)  # [B, S, H_crop, W]
    y = dist * torch.sin(lat)
    tmp = dist * torch.cos(lat)
    x = tmp * torch.sin(long)
    z = tmp * torch.cos(long)
    
    # Concatenate to form 3D point cloud
    point_cloud = torch.stack([x, y, z], dim=-1)
    
    return point_cloud


def points_to_pano_depth(points):
    """
    Convert 3D point cloud back to ray panoramic depth map.
    Ignore the error in direction.
    
    Args:
        points (torch.Tensor): Input 3D point cloud, shape [B, S, H, W, 3]
    
    Returns:
        torch.Tensor: panoramic depth map, shape [B, S, H, W, 1]
    """
    # Validate input shape and fill mode
    assert points.dim() == 5 and points.shape[-1] == 3, \
        f"Input point cloud must be [B, S, H, W, 3], got {points.shape}"
   
    # Compute radial depth (dist = sqrt(x² + y² + z²))
    dist = torch.norm(points, dim=-1, keepdim=True)  # [B, S, H, W, 1]
    
    return dist


def camera_points_to_rotated_points(cam_points, R):
    """
    Rotate batched panoramic camera point clouds with corresponding rotation matrices.
    
    Args:
        cam_points (torch.Tensor): Input camera 3D point cloud, shape [B, S, H, W, 3]
        R (torch.Tensor): Corresponding rotation matrices, shape [B, S, 3, 3]
        
    Returns:
        torch.Tensor: Rotated 3D point cloud, shape [B, S, H, W, 3] (same as input cam_points)
    """
    # Validate input shapes and dimensions matching
    assert cam_points.dim() == 5 and cam_points.shape[-1] == 3, \
        f"Camera points must be [B, S, H, W, 3], got {cam_points.shape}"
    assert R.dim() == 4 and R.shape[2:] == (3, 3), \
        f"Rotation matrices R must be [B, S, 3, 3], got {R.shape}"
    assert cam_points.shape[:2] == R.shape[:2], \
        f"Batch/Sequence dim mismatch: cam_points {cam_points.shape[:2]} vs R {R.shape[:2]}"
    
    # Expand dimensions for broadcasting (align spatial dimensions H, W)
    cam_points_expanded = cam_points.unsqueeze(-1)  # [B, S, H, W, 3, 1]
    R_expanded = R.unsqueeze(2).unsqueeze(2)        # [B, S, 1, 1, 3, 3]
    
    # Batch matrix multiplication: R @ p (rotation operation)
    rotated_points_expanded = torch.matmul(R_expanded, cam_points_expanded)
    
    # Squeeze redundant dimension to recover original shape
    rotated_points = rotated_points_expanded.squeeze(-1)
    
    return rotated_points


def rotated_points_to_world_points(rotated_points, t):
    """
    Transform rotated camera points to world coordinates by adding translation vector.
    
    Args:
        rotated_points (torch.Tensor): Rotated 3D point cloud, shape [B, S, H, W, 3]
        t (torch.Tensor): Translation vector, shape [B, S, 3] (per batch-sequence translation)
        
    Returns:
        torch.Tensor: World-coordinate 3D point cloud, shape [B, S, H, W, 3] (same as input)
    """
    # Validate input shapes and dimension matching
    assert rotated_points.dim() == 5 and rotated_points.shape[-1] == 3, \
        f"Rotated points must be [B, S, H, W, 3], got {rotated_points.shape}"
    assert t.dim() == 3 and t.shape[-1] == 3, \
        f"Translation t must be [B, S, 3], got {t.shape}"
    assert rotated_points.shape[:2] == t.shape[:2], \
        f"Batch/Sequence dim mismatch: rotated_points {rotated_points.shape[:2]} vs t {t.shape[:2]}"
    
    # Expand translation dimensions for broadcasting with spatial dimensions (H, W)
    # t: [B, S, 3] -> [B, S, 1, 1, 3] (broadcast to H and W)
    t_expanded = t.unsqueeze(2).unsqueeze(2)
    
    # Add translation (broadcasting automatically applies t to all H×W points per B-S pair)
    world_points = rotated_points + t_expanded
    
    return world_points



def unproject_depth_to_world_points(depth, extrinsic, size=560):
    '''
    Args:
        depth: [S, H, W, 1]
        extrinsic: [S, 4, 4]
    Returns:
        world_points: [S, H, W, 3]
    '''
    camera_points = pano_depth_to_points(depth, original_pano_shape=(size, size//2))
    rotated_points = camera_points_to_rotated_points(camera_points, extrinsic[:, :, :3, :3])
    world_points = rotated_points_to_world_points(rotated_points, extrinsic[:, :, :3, 3])
    
    return world_points
