import torch


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid



def reorder_by_reference(x: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """Reorder tensor views to place the selected reference view at the first position (index 0),
    while keeping the remaining views in their original order (excluding the reference view).

    Args:
        x: Input tensor with shape (B, S, ...) where B = batch size, S = number of views,
           and trailing dimensions can be arbitrary (e.g., N, C for patch tokens).
        b_idx: 1D tensor of shape (B,) containing the index of the reference view for each batch element,
               each value must be in the range [0, S-1].

    Returns:
        Reordered tensor with the same shape as input, where the reference view is at position 0
        and other views retain their original order (skipping the reference view).

    Example:
        If B=1, S=5, b_idx=[2], input view order is [0,1,2,3,4],
        output order becomes [2,0,1,3,4].
    """
    # Extract batch size (B) and number of views (S) from input shape
    B, S = x.shape[0], x.shape[1]
    
    # No reordering needed if only one view exists
    if S <= 1:
        return x
    
    # Generate base index matrix (B, S): each row is [0, 1, ..., S-1] (same across batches)
    idx = torch.arange(S, device=x.device).expand(B, -1)
    
    # Create mask to exclude reference view indices (True for non-reference positions)
    mask = idx != b_idx.unsqueeze(1)
    
    # Build reorder indices: [reference_idx] + [all non-reference indices in original order]
    # Reshape non-reference indices to (B, S-1) to match batch dimension, then concatenate
    reorder_idx = torch.cat([b_idx.unsqueeze(1), idx[mask].reshape(B, S-1)], dim=1)
    
    # Advanced indexing to reorder: batch indices (B,1) paired with reorder indices (B,S)
    return x[torch.arange(B).unsqueeze(1), reorder_idx]