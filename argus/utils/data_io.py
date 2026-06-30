# Copyright 2026 Realsee. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Shared I/O and preprocessing utilities for panoramic image data.

These functions are used by both evaluation and training pipelines.
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np


def read_image_cv2_360(path: str, rgb: bool = True, shape=(560, 280)) -> np.ndarray:
    """Read and resize a 360 panorama image.

    Args:
        path: Path to the image file.
        rgb: If True, convert BGR to RGB (default: True).
        shape: Target (width, height) tuple.

    Returns:
        Image as numpy array with shape (H, W, 3).
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[1] != shape[0]:
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    return img


def read_depth_360(path: str, depth_scale=5000.0, shape=(560, 280)) -> np.ndarray:
    """Read and normalize a 360 depth map.

    Args:
        path: Path to the depth image file.
        depth_scale: Scale factor to convert raw depth to meters.
        shape: Target (width, height) tuple.

    Returns:
        Depth map as float32 numpy array with shape (H, W).
    """
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d.shape[1] != shape[0]:
        d = cv2.resize(d, shape, interpolation=cv2.INTER_NEAREST)
    d = d.astype(np.float32) / depth_scale
    return d


def random_rotate_theta(W=560, max_shift_percent=0.5):
    """Generate a random rotation angle for panorama augmentation.

    Args:
        W: Panorama width in pixels.
        max_shift_percent: Maximum horizontal shift as fraction of width.

    Returns:
        Rotation angle in radians.
    """
    max_shift = int(W * max_shift_percent)
    shift_pixels = np.random.randint(-max_shift, max_shift + 1)
    theta = (shift_pixels * 2 * np.pi) / W
    return theta


def rotate_y(theta):
    """Create a 3x3 rotation matrix around the Y-axis.

    Args:
        theta: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as float64 numpy array.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array(
        [[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]],
        dtype=np.float64,
    )


def pano_depth_to_points(depth_map, pano_shape=(560, 280), crop=True, crop_ratio=0.15):
    """Convert a panorama depth map to 3D point cloud.

    Args:
        depth_map: 2D depth map (H, W) or flattened array.
        pano_shape: Original panorama (width, height) tuple.
        crop: Whether the depth map has been vertically cropped.
        crop_ratio: Crop ratio applied to top and bottom.

    Returns:
        Point cloud as numpy array with shape (N, 3).
    """
    w, h = pano_shape

    if not crop:
        px = np.tile(np.arange(w), int(h))
        py = np.arange(0, int(h)).repeat(w)
    else:
        px = np.tile(np.arange(w), int(h * (1 - 2 * crop_ratio)))
        py = np.arange(int(crop_ratio * h), int((1 - crop_ratio) * h)).repeat(w)

    dist = depth_map.reshape(-1)

    lat = (py / h - 0.5) * np.pi
    long = (px / w - 0.5) * np.pi * 2.0

    y = dist * np.sin(lat)
    tmp = dist * np.cos(lat)
    x = tmp * np.sin(long)
    z = tmp * np.cos(long)

    point_map = np.concatenate([i.reshape(-1, 1) for i in (x, y, z)], axis=-1)

    return point_map  # (h*w, 3)


def crop_panorama(pano, crop_ratio=0.15):
    """Crop the top and bottom of a panorama by a given ratio.

    Args:
        pano: Input panorama array with shape (H, W, ...).
        crop_ratio: Fraction to crop from top and bottom.

    Returns:
        Cropped panorama.
    """
    H, W = pano.shape[:2]
    crop_H_top = int(crop_ratio * H)
    crop_H_bottom = H - int(crop_ratio * H)
    crop_pano = pano[crop_H_top:crop_H_bottom, ...]
    return crop_pano


def rotate_panorama(panorama, theta):
    """Horizontally rotate a panorama by shifting pixels.

    Args:
        panorama: Input panorama array with shape (H, W, ...).
        theta: Rotation angle in radians.

    Returns:
        Shifted panorama.
    """
    H, W = panorama.shape[:2]
    shift_pixels = int((theta * W) / (2 * np.pi))
    shifted = np.roll(panorama, shift_pixels, axis=1)
    return shifted
