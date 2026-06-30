"""Co-visibility evaluation metrics: reference frame selection accuracy."""

import torch


# ---------------------------------------------------------------------------
# Graph-based reference selection
# ---------------------------------------------------------------------------


def _overlap_to_distance(overlap_matrix: torch.Tensor, method: str = "reciprocal",
                         epsilon: float = 1e-5) -> torch.Tensor:
    """Convert overlap adjacency matrix to distance matrix.

    Higher overlap corresponds to smaller distance.

    Args:
        overlap_matrix: Overlap scores, shape [*, S, S].
        method: 'reciprocal' (1/overlap) or 'linear' (1 - overlap).
        epsilon: Small constant to avoid division by zero.

    Returns:
        Distance matrix with zero diagonal, same shape as input.
    """
    if method == "reciprocal":
        dist_mat = 1.0 / (overlap_matrix + epsilon)
    elif method == "linear":
        dist_mat = 1.0 - overlap_matrix
    else:
        raise ValueError(f"Invalid method: {method}, choose 'reciprocal' or 'linear'")

    # Zero out diagonal (self-distance).
    S = dist_mat.shape[-1]
    eye = torch.eye(S, device=dist_mat.device, dtype=dist_mat.dtype)
    return dist_mat * (1.0 - eye)


def _batch_dijkstra(dist_matrix: torch.Tensor) -> torch.Tensor:
    """Parallel Dijkstra from all sources on a single graph.

    Args:
        dist_matrix: Distance adjacency matrix, shape [S, S].

    Returns:
        Shortest-path distance matrix, shape [S, S].
    """
    S = dist_matrix.shape[0]
    device, dtype = dist_matrix.device, dist_matrix.dtype
    INF = torch.tensor(float("inf"), device=device, dtype=dtype)

    # Each row i holds shortest distances from source i to all targets.
    dist = dist_matrix.clone()
    dist.fill_diagonal_(0.0)

    visited = torch.zeros((S, S), dtype=torch.bool, device=device)

    for _ in range(S):
        # Select the nearest unvisited node for each source.
        min_dists, u_indices = torch.min(dist.masked_fill(visited, INF), dim=1)
        if (min_dists == INF).all():
            break

        # Mark selected nodes as visited.
        sources = torch.arange(S, device=device)
        visited[sources, u_indices] = True

        # Relaxation: try to improve distances through the selected nodes.
        u_dist = dist[sources, u_indices].unsqueeze(1)  # [S, 1]
        new_dists = u_dist + dist_matrix[u_indices, :]  # [S, S]
        dist = torch.min(dist, new_dists)

    return dist


def _find_best_reference_mask(adj_matrix: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    """Find top-k best reference frames based on minimum total graph distance.

    A reference frame is "best" if the sum of shortest-path distances to all other
    frames is minimal (i.e., it is most central in the co-visibility graph).

    Args:
        adj_matrix: Overlap adjacency matrix, shape [B, S, S].
        top_k: Number of best references to select per batch.

    Returns:
        Binary mask of shape [B, S] (1.0 for selected frames, 0.0 otherwise).
    """
    B, S, _ = adj_matrix.shape
    device, dtype = adj_matrix.device, adj_matrix.dtype

    assert isinstance(top_k, int) and top_k >= 1
    top_k = min(top_k, S)

    dist_mat = _overlap_to_distance(adj_matrix)

    # Compute total shortest-path distance from each frame to all others.
    total_dist = torch.zeros((B, S), device=device, dtype=dtype)
    for b in range(B):
        shortest_paths = _batch_dijkstra(dist_mat[b])
        total_dist[b] = shortest_paths.sum(dim=1)

    # Select top-k frames with smallest total distance.
    _, topk_indices = torch.topk(total_dist, k=top_k, dim=1, largest=False)

    mask = torch.zeros_like(total_dist)
    mask.scatter_(dim=1, index=topk_indices, value=1.0)

    return mask


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_covis_metrics(predictions: dict, batch: dict) -> dict:
    """Evaluate reference frame selection accuracy.

    Checks whether the model's predicted reference frame is among the top-3
    most central frames in the ground-truth co-visibility graph.

    Args:
        predictions: Model output dict containing 'ref_idx'.
        batch: Ground truth dict containing 'adj_matrix'.

    Returns:
        Dictionary with 'top3_acc' (1 if correct, 0 otherwise).
    """
    pred_ref_idx = predictions["ref_idx"][0].cpu()
    gt_adj_matrix = batch["adj_matrix"]  # [B, S, S]
    gt_ref_mask = _find_best_reference_mask(gt_adj_matrix, top_k=3)  # [B, S]

    return {
        "top3_acc": 1 if gt_ref_mask[0, pred_ref_idx] == 1.0 else 0,
    }
