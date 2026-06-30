# Standard library imports
import os
import sys
import shutil
import glob
import gc
import time
import base64
import argparse
import tempfile
from datetime import datetime
from pathlib import Path

# Third-party library imports
import cv2
import torch
import trimesh
import numpy as np
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Custom module imports
from argus.models.argus import Argus
from argus.utils.pose_enc import pose_encoding_to_extri360
from argus.utils.geometry import unproject_depth_to_world_points


# -------------------------- Argument Parsing --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Argus Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pre-trained model weights (.pt file). "
             "If not specified, auto-downloads from HuggingFace.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=560,
        help="Input panoramic image target width (height = width // 2)",
    )
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=0.15,
        help="Vertical crop ratio for panoramic image preprocessing (0-0.5)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number for Gradio server",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable Gradio public sharing link",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host address (0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Default: auto-detect",
    )
    parser.add_argument(
        "--examples_dir",
        type=str,
        default="examples",
        help="Directory containing example scenes",
    )
    parser.add_argument(
        "--save_tmp",
        type=str,
        default=None,
        help="Directory to persist intermediate files (images, predictions, GLB). "
             "If not set, uses system temp dir and cleans up automatically.",
    )
    return parser.parse_args()


args = parse_args()

# -------------------------- Global Configuration --------------------------
# Device configuration: use specified device or auto-detect
DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
# Input panoramic image target size (ERP: W=img_size, H=img_size//2)
IMG_SIZE = args.img_size
# Vertical crop ratio for panoramic image preprocessing
CROP_RATIO = args.crop_ratio


def resolve_model_path(model_path: str) -> str:
    """
    Resolve model path: if a local file is specified and exists, use it directly;
    otherwise download from HuggingFace Hub.
    Requires `huggingface-cli login` for gated repos.
    """
    if model_path is not None and os.path.isfile(model_path):
        return model_path

    if model_path is not None:
        print(f"Specified model path '{model_path}' not found.")

    print("Downloading model from HuggingFace (RealseeTechnology/argus-realsee3d)...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id="RealseeTechnology/argus-realsee3d",
            filename="argus_realsee3d.pt",
        )
        print(f"Model downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        error_msg = str(e)
        if "GatedRepoError" in type(e).__name__ or "401" in error_msg:
            raise RuntimeError(
                "Cannot access gated model repo. Please authenticate first:\n"
                "  1. Run: hf auth login\n"
                "  2. Accept the model license at: https://huggingface.co/RealseeTechnology/argus-realsee3d\n"
                "  3. Re-run this script.\n"
                "Or download manually and specify --model_path."
            ) from e
        raise


# Pre-trained model path (auto-download if not found locally)
MODEL_PATH = resolve_model_path(args.model_path)

# -------------------------- Model Initialization --------------------------
print("Initializing and loading Argus model...")
# Initialize Argus model with metric scale and learning ref reorder
model = Argus(reorder_by_learning_ref=True, restore_metric_scale=True)
# Load model weights (non-strict to ignore unused parameters)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model"], strict=False)
# Set model to evaluation mode and move to target device
model.eval()
model = model.to(DEVICE)


# -------------------------- Image Preprocessing --------------------------
def load_and_preprocess_images(image_path_list, target_size=IMG_SIZE):
    """
    Load and preprocess panoramic images for model inference
    Args:
        image_path_list (list): List of input image file paths
        target_size (int): Target width of panoramic image (height = target_size//2)
    Returns:
        torch.Tensor: Preprocessed tensor with shape (S, C, H, W)
                      S: sequence length, C: 3(RGB), H/W: image size
    """
    images = []
    pano_W, pano_H = target_size, target_size // 2

    # Load and resize each image
    for image_path in image_path_list:
        img = cv2.imread(image_path)  # Load as BGR (H, W, C)
        h, w = img.shape[:2]
        if w != pano_W or h != pano_H:
            img = cv2.resize(img, (pano_W, pano_H), interpolation=cv2.INTER_AREA)
        images.append(img)

    # Stack and preprocess: crop vertical → BGR2RGB → normalize → reshape
    images = np.stack(images)  # (S, H, W, C)
    # Crop top/bottom 15% of height and convert BGR to RGB
    images = np.ascontiguousarray(
        images[:, int(pano_H * CROP_RATIO) : int(pano_H * (1 - CROP_RATIO)), :, ::-1]
    )
    # Convert to tensor and normalize to [0,1]
    images = torch.from_numpy(images).float() / 255.0
    # Reshape to (S, C, H, W) for PyTorch model input
    images = images.permute(0, 3, 1, 2)

    return images


# -------------------------- Point Cloud Utils --------------------------
def save_point_cloud_to_ply(points: np.ndarray, save_path: str):
    """
    Save 3D point cloud (N,3) to PLY format (ASCII) for universal compatibility
    Args:
        points (np.ndarray): 3D point cloud with shape [N, 3] (x, y, z for each point)
        save_path (str): Output PLY file path
    Raises:
        ValueError: If input points shape is not [N, 3]
    """
    # Validate input point cloud shape
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must be [N,3], got {points.shape}")

    num_points = points.shape[0]
    # PLY format header (follow official specification)
    ply_header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
    # Write header and point data to file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(ply_header)
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")


# -------------------------- Core Model Inference --------------------------
def run_model(target_dir, model) -> dict:
    """
    Run Argus model inference on images in target_dir/images
    Args:
        target_dir (str): Root directory containing 'images' subfolder
        model (Argus): Pre-initialized Argus model
    Returns:
        dict: Model predictions with tensor converted to numpy array
    Raises:
        ValueError: If CUDA unavailable or no images found in target_dir
    """
    print(f"Processing images from {target_dir}")

    # Enforce CUDA for inference
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Inference requires GPU acceleration.")

    model = model.to(DEVICE)
    model.eval()

    # Load and sort input images
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    print(f"Found {len(image_names)} input images")
    if len(image_names) == 0:
        raise ValueError("No images found in target_dir/images. Check your upload.")

    # Preprocess images and move to device
    images = load_and_preprocess_images(image_names, target_size=IMG_SIZE).to(DEVICE)
    print(f"Preprocessed images shape: {images.shape}")

    # Mixed precision inference for speed and memory efficiency
    print("Running model inference...")
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    inference_time = t1 - t0
    print(f"Inference time: {inference_time:.3f} s")

    # Convert pose encoding to extrinsic/intrinsic matrices
    print("Converting pose encoding to extrinsic matrices...")
    extrinsic, conf = pose_encoding_to_extri360(pose_encoding=predictions["pose_enc"])
    predictions["extrinsic"] = extrinsic[:, :, :3, :]

    # Unproject depth map to 3D world coordinates
    print("Computing 3D world points from depth map...")
    world_points = unproject_depth_to_world_points(
        predictions["depth"], predictions["extrinsic"], size=IMG_SIZE
    )
    predictions["world_points_from_depth"] = world_points

    # Convert all torch tensors to numpy arrays and remove batch dimension
    print("Converting model outputs to numpy arrays...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().float().numpy().squeeze(0)
        elif isinstance(predictions[key], list):
            for i in range(len(predictions[key])):
                if isinstance(predictions[key][i], torch.Tensor):
                    predictions[key][i] = (
                        predictions[key][i].cpu().float().numpy().squeeze(0)
                    )

    print(f"Model prediction keys: {predictions.keys()}")
    # Clear CUDA cache to save memory
    torch.cuda.empty_cache()
    return predictions, inference_time


# -------------------------- Upload File Handling --------------------------
def handle_uploads(input_images):
    """
    Create directory for uploaded images and copy files to target path.
    Uses system temp dir by default; uses --save_tmp dir if specified.
    Args:
        input_images: Gradio uploaded file data
    Returns:
        tuple: (target_dir, sorted_image_paths)
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create target directory: persistent if --save_tmp is set, otherwise temp
    if args.save_tmp:
        os.makedirs(args.save_tmp, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = os.path.join(args.save_tmp, f"input_images_{timestamp}")
    else:
        target_dir = tempfile.mkdtemp(prefix="argus_")
    target_img_dir = os.path.join(target_dir, "images")

    # Clean up if directory exists (edge case)
    if os.path.exists(target_dir) and args.save_tmp:
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_img_dir, exist_ok=True)

    # Copy uploaded images to target directory
    image_paths = []
    if input_images is not None:
        for file_data in input_images:
            # Get file path from Gradio file data
            file_path = file_data["name"] if isinstance(file_data, dict) else file_data
            dst_path = os.path.join(target_img_dir, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # Sort images for consistent processing
    image_paths = sorted(image_paths)
    print(
        f"Files copied to {target_img_dir} | Time cost: {time.time() - start_time:.3f}s"
    )
    return target_dir, image_paths


def update_gallery_on_upload(input_images):
    """
    Update image gallery immediately after file upload
    Args:
        input_images: Gradio uploaded file data
    Returns:
        tuple: Gradio component update values
    """
    if not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_images)
    return (
        None,
        target_dir,
        image_paths,
        "Upload complete. Click 'Reconstruct' to begin 3D processing.",
    )


# -------------------------- 3D Reconstruction Pipeline --------------------------
def gradio_demo(
    target_dir,
    conf_thres=5.0,
    frame_filter="All",
    show_cam=True,
    show_index=True,
    ceiling_remove=25,
):
    """
    Main 3D reconstruction pipeline for Gradio interface
    Args:
        target_dir (str): Directory with input images
        conf_thres (float): Confidence threshold for point cloud filtering
        frame_filter (str): Filter frames to show in 3D model
        show_cam (bool): Whether to show camera poses in 3D model
        show_index (bool): Whether to show frame indices in 3D model
        ceiling_remove (float): Percentage of top Y-coordinate points to remove as ceiling (0-100, 0=disabled)
    Returns:
        tuple: Gradio component update values (3D model, logs, dropdown, etc.)
    """
    # Validate target directory
    if not os.path.isdir(target_dir) or target_dir == "None":
        return (
            None,
            "No valid target directory. Please upload images first.",
            None,
            None,
            None,
            "",
            None,
        )

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame filter dropdown options
    target_img_dir = os.path.join(target_dir, "images")
    all_files = (
        sorted(os.listdir(target_img_dir)) if os.path.isdir(target_img_dir) else []
    )
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    # Run model inference
    with torch.no_grad():
        predictions, inference_time = run_model(target_dir, model)

    # Save predictions to NPZ for later visualization update
    pred_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(pred_save_path, **predictions)

    # Default frame filter to All if None
    frame_filter = frame_filter if frame_filter is not None else "All"

    # Generate unique GLB filename with parameters
    glb_filename = f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}_index{show_index}_ceiling{ceiling_remove}.glb"
    glbfile = os.path.join(target_dir, glb_filename)

    # Convert model predictions to GLB 3D model
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
        show_index=show_index,
        ceiling_remove=ceiling_remove,
        target_dir=target_dir,
    )
    glbscene.export(file_obj=glbfile)

    # Prepare measure view
    measure_img, _ = update_measure_view(predictions, 0)
    # Create view selector based on number of input images
    num_views = (
        predictions["images"].shape[0] if predictions["images"].shape[0] > 0 else 1
    )
    view_choices = [f"View {i + 1}" for i in range(num_views)]
    measure_selector = gr.Dropdown(choices=view_choices, value=view_choices[0])

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Inference: {inference_time:.2f}s | Total: {total_time:.2f}s"
    print(f"Reconstruction complete | Inference: {inference_time:.2f}s | Total: {total_time:.2f}s")

    return (
        glbfile,
        log_msg,
        gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True),
        predictions,
        measure_img,
        "",
        measure_selector,
    )


# -------------------------- UI Utility Functions --------------------------
def clear_fields():
    """Clear 3D model viewer for Gradio interface"""
    return None


def update_log():
    """Update log message during model processing"""
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir,
    conf_thres,
    frame_filter,
    show_cam,
    show_index,
    ceiling_remove,
    is_example,
):
    """
    Update 3D visualization when parameters change (without re-running model)
    Args:
        is_example (str): Whether it's example data (skip if "True")
    Returns:
        tuple: (GLB file path, log message)
    """
    # Skip if loading example data
    if is_example == "True":
        return (
            None,
            "No reconstruction available. Please click the Reconstruct button first.",
        )
    # Validate target directory and prediction file
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No valid reconstruction. Please upload and reconstruct first."

    pred_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(pred_path):
        return None, f"No prediction file found at {pred_path}. Run Reconstruct first."

    # Load saved predictions
    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "images",
        "extrinsic",
        "world_points_from_depth",
    ]
    loaded = np.load(pred_path)
    predictions = {key: np.array(loaded[key]) for key in key_list if key in loaded}

    # Generate GLB file (create if not exists)
    glb_filename = f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}_index{show_index}_ceiling{ceiling_remove}.glb"
    glbfile = os.path.join(target_dir, glb_filename)

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            show_cam=show_cam,
            show_index=show_index,
            ceiling_remove=ceiling_remove,
            target_dir=target_dir,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Visualization updated successfully"


# -------------------------- Metric Measurement --------------------------
def update_measure_view(predictions, view_index):
    """
    Update measure view with depth confidence mask overlay
    Args:
        predictions (dict): Model predictions with images and depth confidence
        view_index (int): Index of the view to show
    Returns:
        tuple: (processed_image, empty_list)
    """
    # Get image and depth confidence
    image = predictions["images"][view_index].transpose(1, 2, 0).copy()
    depth_conf = predictions["depth_conf"][view_index].copy()

    # Convert image to uint8 format
    if image.dtype != np.uint8:
        image = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1.0
            else image.astype(np.uint8)
        )

    # Create depth confidence mask (filter low confidence areas)
    depth_conf_norm = (depth_conf - depth_conf.min()) / (
        depth_conf.max() - depth_conf.min()
    )
    mask = depth_conf_norm > 0.05
    invalid_mask = ~mask

    # Apply red overlay to invalid areas (low confidence)
    if invalid_mask.any():
        overlay_color = np.array([255, 220, 220], dtype=np.uint8)
        alpha = 0.5  # Transparency
        for c in range(3):
            image[:, :, c] = np.where(
                invalid_mask,
                (1 - alpha) * image[:, :, c] + alpha * overlay_color[c],
                image[:, :, c],
            ).astype(np.uint8)

    return image, []


def navigate_measure_view(processed_data, current_selector_value, direction):
    """
    Navigate between different measure views (previous/next)
    Args:
        direction (int): -1 for previous, +1 for next
    Returns:
        tuple: (new_selector_value, measure_image, empty_points)
    """
    if processed_data["images"].shape[0] == 0:
        return "View 1", None, []

    # Parse current view index from selector
    try:
        current_view = int(current_selector_value.split()[1]) - 1
    except:
        current_view = 0

    # Calculate new view index (circular navigation)
    num_views = processed_data["images"].shape[0]
    new_view = (current_view + direction) % num_views

    # Update selector and image
    new_selector = f"View {new_view + 1}"
    measure_image, _ = update_measure_view(processed_data, new_view)
    return new_selector, measure_image, []


def measure(
    processed_data, measure_points, current_view_selector, event: gr.SelectData
):
    """
    Core metric measurement function: click to select points and calculate 3D distance
    Args:
        event (gr.SelectData): Gradio click event data (image coordinates)
    Returns:
        tuple: (annotated_image, measure_points, measurement_text)
    """
    try:
        # Get current view index
        try:
            current_view = int(current_view_selector.split()[1]) - 1
        except:
            current_view = 0
        # Validate view index
        current_view = (
            0
            if current_view < 0 or current_view >= processed_data["images"].shape[0]
            else current_view
        )

        # Get clicked 2D point
        point2d = event.index[0], event.index[1]
        measure_points.append(point2d)
        print(f"Measuring: clicked point {point2d} (view {current_view + 1})")

        # Get base image and 3D points
        image, _ = update_measure_view(processed_data, current_view)
        image = image.copy()
        points3d = processed_data["world_points_from_depth"][current_view]

        # Draw blue circles for clicked points
        for p in measure_points:
            if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)

        # Calculate depth for single point
        depth_text = ""
        depth = processed_data["depth"][current_view].squeeze(axis=-1)
        for i, p in enumerate(measure_points):
            try:
                if 0 <= p[1] < depth.shape[0] and 0 <= p[0] < depth.shape[1]:
                    d = depth[p[1], p[0]]
                    depth_text += f"- **P{i + 1} depth: {d:.2f}m.**\n"
                else:
                    d = np.linalg.norm(points3d[p[1], p[0]], ord=2)
                    depth_text += f"- **P{i + 1} dist: {d:.2f}m.**\n"
            except:
                depth_text += f"- **P{i + 1}: Depth unavailable**\n"

        # Calculate 3D distance for two points
        if len(measure_points) == 2:
            p1, p2 = measure_points
            # Draw blue line between two points
            if all(
                0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]
                for p in [p1, p2]
            ):
                image = cv2.line(image, p1, p2, color=(255, 0, 0), thickness=2)
            # Calculate 3D Euclidean distance
            try:
                p1_3d = points3d[p1[1], p1[0]]
                p2_3d = points3d[p2[1], p2[0]]
                distance = np.linalg.norm(p1_3d - p2_3d)
                distance_text = f"- **Distance: {distance:.2f}m**"
            except:
                distance_text = "- **Distance: Unable to compute**"
            # Reset points after measurement
            measure_points = []
            return [image, measure_points, depth_text + distance_text]

        return [image, measure_points, depth_text]
    except Exception as e:
        print(f"Measurement error: {str(e)}")
        return None, [], f"Measure error: {str(e)}"


# -------------------------- Example Data Loader --------------------------
def get_scene_info(examples_dir):
    """
    Load example scene information from examples directory
    Args:
        examples_dir (str): Directory containing example scenes
    Returns:
        list: List of scene dicts with name, path, thumbnail, image files
    """
    scenes = []
    if not os.path.exists(examples_dir):
        return scenes

    # Iterate over example scene folders
    for scene_folder in sorted(os.listdir(examples_dir)):
        scene_path = os.path.join(examples_dir, scene_folder)
        if not os.path.isdir(scene_path):
            continue
        # Load all image files
        img_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        image_files = []
        for ext in img_exts:
            image_files.extend(glob.glob(os.path.join(scene_path, ext)))
            image_files.extend(glob.glob(os.path.join(scene_path, ext.upper())))
        # Skip empty folders
        if not image_files:
            continue
        # Sort images and get thumbnail
        image_files = sorted(image_files)
        scenes.append(
            {
                "name": scene_folder,
                "path": scene_path,
                "thumbnail": image_files[0],
                "num_images": len(image_files),
                "image_files": image_files,
            }
        )
    return scenes


def example_pipeline(
    scene,
    conf_thres=5.0,
    show_cam=True,
    show_index=True,
    ceiling_remove=25,
):
    """
    Pipeline for loading example scenes and running reconstruction
    Args:
        scene (dict): Example scene info from get_scene_info
    Returns:
        tuple: Gradio component update values
    """
    input_image_paths = scene["image_files"]
    target_dir, image_paths = handle_uploads(input_image_paths)
    frame_filter = "All"  # Default to all frames for examples
    # Run reconstruction
    (
        glbfile,
        log_msg,
        dropdown,
        predictions,
        measure_img,
        measure_text,
        measure_selector,
    ) = gradio_demo(
        target_dir, conf_thres, frame_filter, show_cam, show_index, ceiling_remove
    )
    return (
        glbfile,
        log_msg,
        target_dir,
        dropdown,
        image_paths,
        predictions,
        measure_img,
        measure_text,
        measure_selector,
    )


# -------------------------- 3D Visualization Utilities --------------------------
class SevenSegmentDigit:
    """7-segment display definition for digital watch style 3D point cloud generation"""
    # 7 segments definition: A(top), B(upper right), C(lower right), D(bottom), E(lower left), F(upper left), G(middle)
    SEGMENTS = {
        'A': np.array([(x, 0.5, 0) for x in np.linspace(-0.4, 0.4, 80) for y in np.linspace(0.45, 0.55, 10)]),
        'B': np.array([(x, y, 0) for x in np.linspace(0.4, 0.5, 10) for y in np.linspace(0, 0.5, 80)]),
        'C': np.array([(x, y, 0) for x in np.linspace(0.4, 0.5, 10) for y in np.linspace(-0.5, 0, 80)]),
        'D': np.array([(x, y, 0) for x in np.linspace(-0.4, 0.4, 80) for y in np.linspace(-0.55, -0.45, 10)]),
        'E': np.array([(x, y, 0) for x in np.linspace(-0.5, -0.4, 10) for y in np.linspace(-0.5, 0, 80)]),
        'F': np.array([(x, y, 0) for x in np.linspace(-0.5, -0.4, 10) for y in np.linspace(0, 0.5, 80)]),
        'G': np.array([(x, y, 0) for x in np.linspace(-0.4, 0.4, 80) for y in np.linspace(-0.05, 0.05, 10)])
    }

    # Segment mapping for standard 0-9 digits (specify lit segments for each digit)
    DIGIT_SEGMENTS = {
        0: ['A', 'B', 'C', 'D', 'E', 'F'],
        1: ['B', 'C'],
        2: ['A', 'B', 'G', 'E', 'D'],
        3: ['A', 'B', 'G', 'C', 'D'],
        4: ['F', 'G', 'B', 'C'],
        5: ['A', 'F', 'G', 'C', 'D'],
        6: ['A', 'F', 'G', 'C', 'D', 'E'],
        7: ['A', 'B', 'C'],
        8: ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        9: ['A', 'B', 'C', 'D', 'F', 'G']
    }

    @classmethod
    def get_digit_points(cls, digit, scale=0.05):
        """
        Generate 3D point cloud for a single digital watch style digit (0-9)
        Args:
            digit (int): Target digit (0-9 only)
            scale (float): Scale factor for point cloud size
        Returns:
            np.ndarray: N×3 array of 3D points for the digit
        Raises:
            ValueError: If digit is not in 0-9 range
        """
        if not 0 <= digit <= 9:
            raise ValueError(f"Digit must be 0-9, got {digit}")

        # Combine lit segments for the target digit
        segments = cls.DIGIT_SEGMENTS[digit]
        points = np.vstack([cls.SEGMENTS[seg] for seg in segments])

        # Scale point cloud and center to origin
        points = points * scale
        points -= points.mean(axis=0)

        # Remove duplicate points and supplement sparse points (ensure dense distribution)
        points = np.unique(points.round(6), axis=0)
        if len(points) < 200:
            points = trimesh.sample.sample_surface(trimesh.Trimesh(points), 500)[0]

        return points


def create_number_point_cloud(number, scale=0.05):
    """
    Generate 3D point cloud for multi-digit number (digital watch style), facing +Y axis
    Args:
        number (int): Non-negative target integer (any digit length)
        scale (float): Scale factor for single digit point cloud size
    Returns:
        trimesh.PointCloud: Colored (red) 3D point cloud of the number
    Raises:
        ValueError: If number is negative or non-integer
    """
    if not isinstance(number, int) or number < 0:
        raise ValueError(f"Number must be non-negative integer, got {number}")

    # Split number into individual digits and handle 0 specially
    digits = [int(d) for d in str(number)] if number != 0 else [0]
    all_points, spacing = [], scale * 1.2
    total_width = (len(digits)-1) * spacing

    # Arrange digits horizontally and center the whole number
    for idx, d in enumerate(digits):
        digit_points = SevenSegmentDigit.get_digit_points(d, scale)
        digit_points[:, 0] += -total_width/2 + idx * spacing
        all_points.append(digit_points)

    # Merge all digit points and apply rotation to face +Y axis
    all_points = np.vstack(all_points)
    rotation = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
    all_points = np.dot(all_points, rotation.T)

    # Create red point cloud (classic digital watch color)
    colors = np.full((len(all_points), 3), [255, 0, 0], dtype=np.uint8)

    return trimesh.PointCloud(all_points, colors)


def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    show_cam=True,
    show_index=True,
    ceiling_remove=25,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
) -> trimesh.Scene:
    """
    Convert VGGT model predictions to a 3D trimesh Scene (exportable to GLB)
    Integrates colored point cloud, camera meshes and digital camera indexes
    Args:
        predictions (dict): Model prediction dict with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Low-confidence point filter (percentile, 0-100)
        filter_by_frames (str): Frame filter ("all" or specific frame index like "0:")
        show_cam (bool): Whether to add camera mesh visualization to scene
        show_index (bool): Whether to add digital index point cloud above cameras
        ceiling_remove (float): Percentage of top Y-coordinate points to remove as ceiling (0-100, 0=disabled)
        target_dir (str): Directory for intermediate files (images)
        prediction_mode (str): Prediction branch ("Predicted Pointmap" / others for depth-based)
    Returns:
        trimesh.Scene: 3D scene with point cloud, cameras and indexes (if enabled)
    Raises:
        ValueError: If predictions is not a dictionary
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    conf_thres = 10.0 if conf_thres is None else conf_thres
    print("Building GLB scene")
    selected_frame_idx = None

    # Parse selected frame index from filter string (e.g., "0:" -> 0)
    if filter_by_frames not in ["all", "All"]:
        try:
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    # Select prediction branch (Pointmap direct / Depthmap derived)
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found, falling back to depth-based world points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Extract core prediction data: images and camera extrinsic matrices
    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    # Filter prediction data to selected single frame if specified
    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    # Reshape 3D points and convert image colors to 8-bit RGB (match point cloud)
    vertices_3d = pred_world_points.reshape(-1, 3)
    if images.ndim == 4 and images.shape[1] == 3:  # Convert NCHW to NHWC format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Direct use if already NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Filter points by confidence threshold (remove low-confidence points)
    conf = pred_world_points_conf.reshape(-1)
    conf_threshold = 0.0 if conf_thres == 0.0 else np.percentile(conf, conf_thres)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    # Create dummy point if no valid points left (avoid scene empty error)
    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate scene scale by 5th/95th percentile bounding box diagonal
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    # Initialize 3D scene and colormap for camera unique colors
    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene_3d = trimesh.Scene()

    # Filter out ceiling points (remove top N% of Y-coordinates by percentile)
    if ceiling_remove > 0 and vertices_3d.size > 1:
        y_coords = vertices_3d[:, 1]
        y_percentile = np.percentile(y_coords, ceiling_remove)
        mask = y_coords > y_percentile
        vertices_3d = vertices_3d[mask]
        colors_rgb = colors_rgb[mask]

    # Add colored 3D point cloud to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    # Convert 3x4 camera extrinsics to 4x4 homogeneous matrices
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    # Add camera meshes and digital index point clouds to the scene
    for i in range(num_cameras):
        camera_to_world = extrinsics_matrices[i]
        rgba_color = colormap(i / num_cameras)  # Unique color for each camera
        current_color = tuple(int(255 * x) for x in rgba_color[:3])

        # Add camera mesh to scene
        if show_cam:
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

        # Add digital index point cloud above each camera (red, digital watch style)
        if show_index:
            camera_center = camera_to_world[:3, 3]
            y_offset = 0.5  # Y-axis offset for index position (above camera)
            number_position = camera_center + np.array([0, y_offset, 0])

            # Generate index point cloud and translate to target position
            number_scale = 0.3
            number_pc = create_number_point_cloud(number=i, scale=number_scale)
            number_pc.apply_translation(number_position)
            scene_3d.add_geometry(number_pc)

    # Align the whole scene to the first camera's viewing perspective
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built successfully")
    return scene_3d


def integrate_camera_into_scene(
    scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float
):
    """
    Add a 3D cone-shaped camera mesh to the 3D scene with specified transform and color
    Args:
        scene (trimesh.Scene): Target 3D scene to add camera mesh
        transform (np.ndarray): 4x4 camera-to-world transformation matrix
        face_colors (tuple): RGB color tuple (0-255) for camera mesh faces
        scene_scale (float): Overall scale of the 3D scene (for camera size adaptation)
    """
    # Set camera mesh size based on scene scale
    cam_width = scene_scale * 0.02
    cam_height = scene_scale * 0.02

    # 45° Z-axis rotation for camera cone shape and backward translation
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    # Combine OpenGL conversion, rotation and camera transform matrices
    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Slight Z-axis rotation for camera mesh detail enhancement
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    # Combine original, scaled and rotated cone vertices for dense camera mesh
    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    # Compute camera mesh faces from cone shape
    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Create camera mesh with specified color and add to scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(
    scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray
) -> trimesh.Scene:
    """
    Align the 3D scene to the first camera's viewing perspective with OpenGL conversion
    Args:
        scene_3d (trimesh.Scene): Unaligned 3D scene
        extrinsics_matrices (np.ndarray): N×4×4 camera extrinsic matrices
    Returns:
        trimesh.Scene: Aligned 3D scene
    """
    # Get OpenGL coordinate conversion matrix and 180° Y-axis rotation for alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Combine transformation matrices and apply to the whole scene
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Create 4x4 OpenGL coordinate system conversion matrix (flip Y and Z axes)
    Returns:
        np.ndarray: 4x4 identity-based conversion matrix
    """
    matrix = np.identity(4)
    matrix[1, 1] = -1  # Flip Y axis
    matrix[2, 2] = -1  # Flip Z axis
    return matrix


def transform_points(
    transformation: np.ndarray, points: np.ndarray, dim: int = None
) -> np.ndarray:
    """
    Apply 4x4 homogeneous transformation matrix to a set of 3D points
    Args:
        transformation (np.ndarray): 4x4 transformation matrix
        points (np.ndarray): N×3 array of 3D points to transform
        dim (int, optional): Target dimension of output points (default: 3)
    Returns:
        np.ndarray: N×dim array of transformed points (same shape as input except last dim)
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Transpose matrix and apply affine transformation to points
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape transformed points to original shape (excluding last dimension)
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Compute face indices for camera mesh from original cone shape faces (enhance detail)
    Args:
        cone_shape (trimesh.Trimesh): Original cone mesh for camera base shape
    Returns:
        np.ndarray: M×3 array of face indices for the camera mesh
    """
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    # Generate enhanced faces from cone faces (skip origin vertex 0)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        # Add multiple face variations for dense camera mesh
        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    # Add reversed faces for double-sided rendering
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


# -------------------------- Gradio UI Construction --------------------------
if __name__ == "__main__":
    # Gradio theme configuration
    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        title="Argus - 3D Reconstruction",
        css="""
        .custom-log * {
            font-style: italic;
            font-size: 20px !important;
            background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: 600 !important;
            color: transparent !important;
            text-align: center !important;
        }
        .example-log * {
            font-size: 15px !important;
            background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
            font-weight: 500 !important;
        }
        .header-banner {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 16px;
            padding: 32px 24px 24px;
            margin-bottom: 16px;
            border: 1px solid #e2e8f0;
            text-align: center;
        }
        .header-banner h1 {
            font-size: 28px;
            font-weight: 700;
            color: #1e293b;
            margin: 12px 0 8px;
        }
        .header-banner .links {
            margin-top: 12px;
            font-size: 15px;
        }
        .header-banner .links a {
            margin: 0 10px;
            color: #4f46e5;
            text-decoration: none;
            font-weight: 500;
        }
        .header-banner .links a:hover {
            text-decoration: underline;
        }
        .instructions {
            font-size: 14px;
            color: #475569;
            line-height: 1.7;
            padding: 12px 20px;
            background: #f8fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        .instructions ol {
            padding-left: 20px;
            margin: 8px 0;
        }
        .instructions li {
            margin-bottom: 4px;
        }
        .param-group {
            padding: 8px 0;
        }
        footer {visibility: hidden;}
        """,
    ) as demo:
        # Hidden state components for data passing
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        processed_data_state = gr.State(value=None)
        measure_points_state = gr.State(value=[])
        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        # Load and display logo (base64 encoded)
        root_dir = Path(__file__).parent
        logo_path = root_dir / "assets" / "argus_logo.png"
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_base64 = base64.b64encode(f.read()).decode()
                logo_src = f"data:image/png;base64,{logo_base64}"
        else:
            logo_src = ""  # Fallback if logo not found

        # UI Header and Instructions
        gr.HTML(
            f"""
        <div class="header-banner">
            <div style="display: flex; justify-content: center;">
                <img src="{logo_src}" alt="Argus Logo" style="height: 72px; border-radius: 8px;">
            </div>
            <h1>Argus: Metric Panoramic 3D Reconstruction for Indoor Scenes</h1>
            <div class="links">
                <a href="https://github.com/realsee-developer/Argus" target="_blank">🌟 GitHub</a>
                <a href="https://argus-paper.realsee.ai" target="_blank">🚀 Project Page</a>
                <a href="https://arxiv.org/abs/2606.30047" target="_blank">📄 Paper</a>
            </div>
        </div>
        <div class="instructions">
            <ol>
                <li><strong>Upload</strong> a set of ERP panoramic images on the left.</li>
                <li><strong>Click "Reconstruct"</strong> to run the 3D reconstruction pipeline.</li>
                <li><strong>Explore</strong> the 3D model — rotate, pan, zoom, and download the GLB.</li>
                <li><strong>Measure</strong> — switch to the Metric tab and click two points to measure real-world distance.</li>
            </ol>
        </div>
        """
        )

        # Main UI Layout (2 columns: upload/gallery | 3D model/measurement)
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=280):
                input_images = gr.File(
                    file_count="multiple", label="📁 Upload Panoramic Images", interactive=True
                )
                image_gallery = gr.Gallery(
                    label="Preview",
                    columns=3,
                    height="280px",
                    object_fit="contain",
                    preview=True,
                )

            with gr.Column(scale=5):
                # Log output
                log_output = gr.Markdown(
                    "Upload panoramic images (ERP), then click Reconstruct.",
                    elem_classes=["custom-log"],
                )
                # Tabbed interface: 3D Model + Metric Measure
                with gr.Tabs():
                    with gr.Tab("🏠 3D Model"):
                        reconstruction_output = gr.Model3D(
                            height=540, zoom_speed=0.5, pan_speed=0.5
                        )
                    with gr.Tab("📏 Metric Measure"):
                        gr.Markdown(
                            "Click two points on the panorama to measure the real-world distance between them."
                        )
                        with gr.Row():
                            prev_measure_btn = gr.Button(
                                "◀ Prev", size="sm", scale=1
                            )
                            measure_view_selector = gr.Dropdown(
                                choices=["View 1"],
                                value="View 1",
                                label="Select View",
                                scale=3,
                                interactive=True,
                                allow_custom_value=True,
                            )
                            next_measure_btn = gr.Button("Next ▶", size="sm", scale=1)
                        measure_image = gr.Image(
                            type="numpy",
                            show_label=False,
                            format="webp",
                            interactive=False,
                            sources=[],
                        )
                        measure_text = gr.Markdown("")

                # Action buttons
                with gr.Row():
                    submit_btn = gr.Button("🔨 Reconstruct", scale=2, variant="primary")
                    clear_btn = gr.ClearButton(
                        [
                            input_images,
                            reconstruction_output,
                            log_output,
                            target_dir_output,
                            image_gallery,
                        ],
                        value="🗑️ Clear",
                        scale=1,
                    )

                # Reconstruction parameters
                gr.Markdown("**Visualization Settings**")
                with gr.Row():
                    conf_thres = gr.Slider(
                        0, 100, 5, 1, label="Confidence Threshold (%)"
                    )
                    ceiling_remove = gr.Slider(
                        0, 100, 25, 1, label="Ceiling Remove (%)"
                    )
                with gr.Row():
                    frame_filter = gr.Dropdown(
                        ["All"], "All", label="Show Points from Frame", scale=2
                    )
                    show_cam = gr.Checkbox(True, label="Show Camera")
                    show_index = gr.Checkbox(True, label="Show Index")

        # Example Scenes Section
        gr.Markdown("---")
        gr.Markdown("### 🖼️ Example Scenes")
        gr.Markdown("Click any thumbnail to load and reconstruct.", elem_classes=["example-log"])
        example_scenes = get_scene_info(args.examples_dir)
        # Create 4-column example thumbnail grid
        if example_scenes:
            for i in range(0, len(example_scenes), 4):
                with gr.Row():
                    for j in range(4):
                        idx = i + j
                        if idx < len(example_scenes):
                            scene = example_scenes[idx]
                            with gr.Column(scale=1):
                                scene_state = gr.State(value=scene)
                                scene_img = gr.Image(
                                    value=scene["thumbnail"],
                                    height=150,
                                    interactive=False,
                                    show_label=False,
                                    sources=[],
                                )
                                gr.Markdown(
                                    f"**{scene['name']}** \n {scene['num_images']} images"
                                )
                                # Bind thumbnail click to example pipeline
                                scene_img.select(
                                    example_pipeline,
                                    [scene_state],
                                    [
                                        reconstruction_output,
                                        log_output,
                                        target_dir_output,
                                        frame_filter,
                                        image_gallery,
                                        processed_data_state,
                                        measure_image,
                                        measure_text,
                                        measure_view_selector,
                                    ],
                                )
                        else:
                            with gr.Column(scale=1):
                                pass  # Empty column for grid alignment

        # -------------------------- Gradio Event Bindings --------------------------
        # Reconstruct button logic
        submit_btn.click(clear_fields, [], [reconstruction_output]).then(
            update_log, [], [log_output]
        ).then(
            gradio_demo,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                show_index,
                ceiling_remove,
            ],
            [
                reconstruction_output,
                log_output,
                frame_filter,
                processed_data_state,
                measure_image,
                measure_text,
                measure_view_selector,
            ],
        ).then(
            lambda: "False", [], [is_example]
        )

        # Real-time parameter update for 3D visualization
        for param in [conf_thres, frame_filter, show_cam, show_index, ceiling_remove]:
            param.change(
                update_visualization,
                [
                    target_dir_output,
                    conf_thres,
                    frame_filter,
                    show_cam,
                    show_index,
                    ceiling_remove,
                    is_example,
                ],
                [reconstruction_output, log_output],
            )

        # Auto-update gallery on file upload
        input_images.change(
            update_gallery_on_upload,
            [input_images],
            [reconstruction_output, target_dir_output, image_gallery, log_output],
        )

        # Metric measure event bindings
        measure_image.select(
            measure,
            [processed_data_state, measure_points_state, measure_view_selector],
            [measure_image, measure_points_state, measure_text],
        )
        # Measure view navigation
        prev_measure_btn.click(
            lambda d, s: navigate_measure_view(d, s, -1),
            [processed_data_state, measure_view_selector],
            [measure_view_selector, measure_image, measure_points_state],
        )
        next_measure_btn.click(
            lambda d, s: navigate_measure_view(d, s, 1),
            [processed_data_state, measure_view_selector],
            [measure_view_selector, measure_image, measure_points_state],
        )
        # Update measure view when selector changes
        measure_view_selector.change(
            lambda d, s: (
                update_measure_view(d, int(s.split()[1]) - 1) if s else (None, [])
            ),
            [processed_data_state, measure_view_selector],
            [measure_image, measure_points_state],
        )

        # Footer acknowledgement
        gr.HTML(
            """
        <hr style="margin-top: 40px; margin-bottom: 20px; border-color: #e2e8f0;">
        <div style="text-align: center; font-size: 13px; color: #94a3b8; margin-bottom: 20px;">
            <p style="margin-bottom: 8px; font-weight: 500; color: #64748b;">Acknowledgements</p>
            <p>Built upon
                <a href="https://github.com/facebookresearch/vggt" style="color: #6366f1;">VGGT</a> &
                <a href="https://github.com/facebookresearch/map-anything" style="color: #6366f1;">Map-Anything</a>
            </p>
        </div>
        """
        )

    # Launch Gradio demo
    demo.queue(max_size=20).launch(
        show_error=True,
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
    )
