import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

from argus.utils.data_io import (
    read_image_cv2_360,
    read_depth_360,
    random_rotate_theta,
    rotate_y,
    pano_depth_to_points,
    crop_panorama,
    rotate_panorama,
)


class Realsee3DDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_list,
        pano_width: int = 1600,
        pano_crop: bool = True,
        pano_crop_ratio: float = 0.15,
        pano_rotate: bool = True,
    ):
        self.pano_w = pano_width
        self.pano_h = pano_width // 2

        self.pano_crop = pano_crop
        self.pano_crop_ratio = pano_crop_ratio
        self.pano_rotate = pano_rotate

        self.data_store = {}

        with open(data_list, "r") as f:
            scene_list = [p.strip() for p in f.readlines()]

        for i, scene_id in enumerate(scene_list):
            scene_path = os.path.join(data_dir, scene_id)
            self.data_store[scene_id] = {
                "scene_path": scene_path,
            }

        self.sequence_list = list(self.data_store.keys())

    def __len__(self):

        return len(self.sequence_list)

    def __getitem__(self, idx):
        seq_name = self.sequence_list[idx]
        scene_data = self.data_store[seq_name]
        scene_path = scene_data["scene_path"]

        viewpoints_path = osp.join(scene_path, "viewpoints.txt")
        covisibility_path = osp.join(scene_path, "covisibility.txt")
        viewpoint_dir_path = osp.join(scene_path, "viewpoints")

        with open(viewpoints_path, "r") as f:
            viewpoints = [l.strip() for l in f.readlines()]

        # defualt maximize covis score of first view 
        adj_matrix = np.zeros((len(viewpoints), len(viewpoints)))
        adj_matrix[0][0] = 1.
        if os.path.exists(covisibility_path):
            adj_matrix = np.loadtxt(covisibility_path, delimiter=" ")
        
        images = []
        depths = []
        cam_points = []
        rotated_points = []
        world_points = []
        point_masks = []
        extrinsics = []

        ids = np.arange(len(viewpoints))
        for choiced_id in ids:
            vp = viewpoints[choiced_id]
            vp_dir = osp.join(viewpoint_dir_path, vp)

            image = read_image_cv2_360(
                osp.join(vp_dir, "panoImage_1600.jpg"),
                shape=(self.pano_w, self.pano_h),
            )
            extri_opencv = np.loadtxt(
                osp.join(vp_dir, "extrinsics.txt"), dtype=np.float32
            )
            depth_scale = np.loadtxt(
                osp.join(vp_dir, "depth_scale.txt"), dtype=np.float32
            )
            depth = read_depth_360(
                osp.join(vp_dir, "depth_image.png"),
                depth_scale,
                shape=(self.pano_w, self.pano_h),
            )
                
            if self.pano_rotate:
                theta = random_rotate_theta(self.pano_w)
                extri_opencv[:3, :3] = extri_opencv[:3, :3] @ rotate_y(theta)
                image = rotate_panorama(image, theta)
                depth = rotate_panorama(depth, theta)

            if self.pano_crop:
                image = crop_panorama(image, crop_ratio=self.pano_crop_ratio)
                depth = crop_panorama(depth, crop_ratio=self.pano_crop_ratio)

            point_mask = depth > 1e-8
            points = pano_depth_to_points(
                depth,
                pano_shape=(self.pano_w, self.pano_h),
                crop=self.pano_crop,
                crop_ratio=self.pano_crop_ratio,
            )
            R_points = extri_opencv[:3, :3] @ points.T
            world_coords_points = (
                extri_opencv[:3, :3] @ points.T + extri_opencv[:3, [3]]
            ).T

            images.append(image)
            depths.append(depth)
            extrinsics.append(extri_opencv)
            H, W = depth.shape[:2]
            cam_points.append(points.reshape(H, W, 3))
            rotated_points.append(R_points.reshape(H, W, 3))
            world_points.append(world_coords_points.reshape(H, W, 3))
            point_masks.append(point_mask)

        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(images).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types
        depths = torch.from_numpy(np.stack(depths).astype(np.float32))
        extrinsics = torch.from_numpy(np.stack(extrinsics).astype(np.float32))
        cam_points = torch.from_numpy(np.stack(cam_points).astype(np.float32))
        rotated_points = torch.from_numpy(np.stack(rotated_points).astype(np.float32))
        world_points = torch.from_numpy(np.stack(world_points).astype(np.float32))
        point_masks = torch.from_numpy(
            np.stack(point_masks)
        )  # Mask indicating valid depths / world points / cam points per frame
        ids = torch.from_numpy(ids)  # Frame indices sampled from the original sequence
        adj_matrix = torch.from_numpy(adj_matrix)

        batch = {
            "seq_name": seq_name,
            "ids": ids,
            "adj_matrix": adj_matrix,
            "images": images,
            "depths": depths,
            "cam_points": cam_points,
            "rotated_points": rotated_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "extrinsics": extrinsics,
            "viewpoints": viewpoints,
        }

        return batch
