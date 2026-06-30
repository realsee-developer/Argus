import os.path as osp
import os
import logging

import cv2
import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset


class Realsee3DDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = None,
        DATA_LIST: str = None,
        len_train: int = 9000,
        len_test: int = 1000,
    ):
        """
        Initialize the Realsee3D Dataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DATA_DIR (str): Directory path to data.
            DATA_LIST (str): Directory path to data list.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If DATA_DIR or DATA_LIST is not specified.
        """
        super().__init__(common_conf=common_conf)
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.img_size = common_conf.img_size

        self.pano_w = int(self.img_size)
        self.pano_h = int(self.pano_w / 2)

        self.pano_crop = common_conf.pano_crop
        self.pano_crop_ratio = common_conf.pano_crop_ratio
        self.pano_rotate = common_conf.pano_rotate
        self.load_covis = common_conf.load_covis

        if DATA_DIR is None or DATA_LIST is None:
            raise ValueError("Both DATA_DIR and DATA_LIST must be specified.")

        self.data_store = {}

        logging.info(f"DATA_DIR is {DATA_DIR}")

        self.DATA_DIR = DATA_DIR
        self.DATA_LIST = DATA_LIST

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        with open(DATA_LIST, "r") as f:
            scene_list = [p.strip() for p in f.readlines()]

        for i, scene_id in enumerate(scene_list):
            scene_path = os.path.join(DATA_DIR, scene_id)
            self.data_store[scene_id] = {
                "scene_path": scene_path,
            }

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]  # scene_id

        scene_data = self.data_store[seq_name]
        scene_path = scene_data["scene_path"]

        viewpoints_path = osp.join(scene_path, "viewpoints.txt")
        covisibility_path = osp.join(scene_path, "covisibility.txt")
        viewpoint_dir_path = osp.join(scene_path, "viewpoints")

        with open(viewpoints_path, "r") as f:
            viewpoints = [l.strip() for l in f.readlines()]

        ids = None
        sample_adj_matrix = None
   
        if self.load_covis:
            ids, sample_adj_matrix = sample_ids(
                covisibility_path, img_per_seq, self.allow_duplicate_img
            )
        else:
            ids = np.random.choice(
                len(viewpoints), img_per_seq, replace=self.allow_duplicate_img
            )
            sample_adj_matrix = np.ones((img_per_seq, img_per_seq))
            
        images = []
        depths = []
        cam_points = []
        rotated_points = []
        world_points = []
        point_masks = []
        extrinsics = []

        for choiced_id in ids:
            vp = viewpoints[choiced_id]

            image = read_image_cv2_360(
                osp.join(viewpoint_dir_path, vp, "panoImage_1600.jpg"),
                shape=(self.pano_w, self.pano_h),
            )
            extri_opencv = np.loadtxt(
                osp.join(viewpoint_dir_path, vp, "extrinsics.txt"), dtype=np.float32
            )
            depth_scale = np.loadtxt(
                osp.join(viewpoint_dir_path, vp, "depth_scale.txt"), dtype=np.float32
            )
            depth = read_depth_360(
                osp.join(viewpoint_dir_path, vp, "depth_image.png"),
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
            world_coords_points = (
                extri_opencv[:3, :3] @ points.T + extri_opencv[:3, [3]]
            ).T
            
            R_points = (extri_opencv[:3, :3] @ points.T).T

            images.append(image)
            depths.append(depth)
            extrinsics.append(extri_opencv)
            H, W = depth.shape[:2]
            cam_points.append(points.reshape(H, W, 3))
            rotated_points.append(R_points.reshape(H, W, 3))
            world_points.append(world_coords_points.reshape(H, W, 3))
            point_masks.append(point_mask)
            
        
        batch = {
            "seq_name": seq_name,
            "ids": ids,
            "sample_adj_matrix": sample_adj_matrix,
            "frame_num": len(ids),
            "images": images,
            "depths": depths,
            "cam_points": cam_points,
            "rotated_points": rotated_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "extrinsics": extrinsics,
        }
        return batch
