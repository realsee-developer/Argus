import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from collections import deque

# Re-export shared I/O utilities from argus.utils.data_io
# (canonical implementations live there; kept here for backward compatibility)
from argus.utils.data_io import (  # noqa: F401
    read_image_cv2_360,
    read_depth_360,
    random_rotate_theta,
    rotate_y,
    pano_depth_to_points,
    crop_panorama,
    rotate_panorama,
)


#####################################################################################################################
# Training-only utilities below
#####################################################################################################################


def generate_connected_sample(adj_matrix, N, allow_repeat=True, max_attempts=1000):
    """
    Randomly sample a connected sequence of length N from adjacency matrix, optimized to minimize repeated IDs

    Args:
    adj_matrix: numpy array, adjacency matrix representing connectivity between nodes
    N: int, length of sampled sequence
    allow_repeat: bool, whether to allow sampling the same node repeatedly
    max_attempts: int, maximum number of attempts

    Returns:
    list, connected sequence of length N; empty list if failed
    """
    components = get_connected_components(adj_matrix)
    if not components:
        return []

    for _ in range(max_attempts):
        # Weighted selection of connected component by size (prioritize larger components for longer sequences)
        sizes = [len(c) for c in components]
        component = components[
            np.random.choice(len(components), p=sizes / np.sum(sizes))
        ]
        component_set = set(component)  # For fast lookup

        # Check feasibility without repetition
        if not allow_repeat and len(component) < N:
            continue

        # Handle single-node component
        if len(component) == 1:
            return component * N if allow_repeat else component

        # Randomly select start node
        start_node = np.random.choice(component)
        sample = [start_node]
        sampled = {start_node}  # Set of sampled nodes

        # Calculate node degree within connected component (prioritize nodes with higher freedom)
        def get_degree(node):
            return sum(adj_matrix[node][np.array(component)])

        # Global available neighbors: unsampled neighbors of all sampled nodes
        all_available = (
            set(np.where(adj_matrix[start_node])[0]) & component_set - sampled
        )

        # Extend sampled sequence
        while len(sample) < N:
            if all_available:
                # Prioritize high-degree nodes when new nodes are available (easier to extend)
                candidates = list(all_available)
                degrees = [get_degree(node) for node in candidates]
                probs = np.array(degrees) / sum(degrees) if sum(degrees) > 0 else None
                next_node = np.random.choice(candidates, p=probs)

                sample.append(next_node)
                sampled.add(next_node)
                all_available.remove(next_node)  # Remove sampled node from available

                # Add unsampled neighbors of new node to available set
                new_neighbors = (
                    set(np.where(adj_matrix[next_node])[0]) & component_set - sampled
                )
                all_available.update(new_neighbors)

            else:
                # No new nodes available, check if repetition is allowed
                if allow_repeat:
                    # Select from sampled nodes, prioritize high-degree nodes (more new paths)
                    nodes = list(sampled)
                    degrees = [get_degree(node) for node in nodes]
                    probs = (
                        np.array(degrees) / sum(degrees) if sum(degrees) > 0 else None
                    )
                    next_node = np.random.choice(nodes, p=probs)
                    sample.append(next_node)
                else:
                    break  # Cannot extend without repetition, attempt failed

        if len(sample) == N:
            return sample

    return []  # Failed after multiple attempts


def get_connected_components(adj_matrix):
    """Calculate connected components of the graph (same as before)"""
    n = len(adj_matrix)
    visited = np.zeros(n, dtype=bool)
    components = []

    for i in range(n):
        if not visited[i]:
            queue = deque([i])
            visited[i] = True
            component = []

            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in np.where(adj_matrix[node])[0]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            components.append(component)

    return components


def generate_less_repeats(obs_num, img_per_seq):
    # Minimize repetitions
    if img_per_seq <= obs_num:
        # No repetition if required number ≤ total number
        return np.random.choice(obs_num, img_per_seq, replace=False)
    else:
        # First take all unique, then supplement remaining randomly
        unique_part = np.random.choice(obs_num, obs_num, replace=False)
        remaining = img_per_seq - obs_num
        repeat_part = np.random.choice(obs_num, remaining, replace=True)
        return np.concatenate([unique_part, repeat_part])


def sample_ids(covisibility_path, img_per_seq, allow_repeat=True, th=0.001):
    adj_matrix = np.loadtxt(covisibility_path, delimiter=" ")
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix_mask = adj_matrix > th
    sample = generate_connected_sample(adj_matrix_mask, img_per_seq, allow_repeat)
    ids = np.array(sample)
    sample_adj_matrix = adj_matrix[ids][:, ids]
    return ids, sample_adj_matrix
