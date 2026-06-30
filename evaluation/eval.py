"""Evaluation entry point for the Argus model on Realsee3D dataset."""

import argparse
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Realsee3DDataset
from utils import (
    compute_all_metrics,
    init_metrics_results,
    integrate_metrics_results,
    load_model,
    save_output,
    serialize_numpy_and_round,
    set_random_seeds,
)

# Enforce highest precision for reproducibility.
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False


def setup_args() -> argparse.Namespace:
    """Parse command-line arguments for Realsee3D evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Argus on Realsee3D dataset")
    parser.add_argument("--exam", type=str, default="argus0", help="Exam name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="./data/Realsee3D", help="Dataset root")
    parser.add_argument("--split", type=str, default="both", choices=["both", "real_world", "synthetic"], help="Evaluation split")
    parser.add_argument("--save_path", type=str, default="argus_exam_eval", help="Output directory")
    parser.add_argument("--dump", action="store_true", help="Dump raw predictions to disk")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pano_width", type=int, default=560, help="Panorama input width")
    parser.add_argument("--ref", action="store_true", help="Enable learned reference reordering")
    parser.add_argument("--mono", action="store_true", help="Use mono (single-view) inference mode")
    return parser.parse_args()


def select_dtype() -> torch.dtype:
    """Select the best available AMP dtype for the current GPU."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def run_inference(model, images: torch.Tensor, device: str, dtype: torch.dtype, mono: bool) -> dict:
    """Run model inference in the appropriate mode (multi-view or mono).

    Args:
        model: The Argus model instance.
        images: Input images tensor of shape [B, S, C, H, W].
        device: Target device string.
        dtype: AMP dtype for mixed-precision inference.
        mono: If True, run single-view inference per frame and aggregate depths.

    Returns:
        Dictionary of model predictions.
    """
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if not mono:
            return model(images.to(device))

        # Mono mode: predict depth independently for each frame.
        depth_predictions = []
        for s in range(images.shape[1]):
            mono_pred = model(images[:, [s], :, :].to(device))
            depth_predictions.append(mono_pred["depth"])  # [B, 1, H, W]
        return {"depth": torch.cat(depth_predictions, dim=1)}  # [B, S, H, W]


def evaluate_split(
    model,
    dataset: Realsee3DDataset,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
    output_dir: str,
) -> dict:
    """Evaluate on a single dataset split and return aggregated metrics.

    Args:
        model: The Argus model instance.
        dataset: Dataset for evaluation.
        device: Target device string.
        dtype: AMP dtype.
        args: Parsed arguments.
        output_dir: Directory for saving predictions (if args.dump is True).

    Returns:
        Aggregated metrics dictionary.
    """
    dataloader = DataLoader(dataset, batch_size=1)
    
    metrics_results = init_metrics_results()
    for _, batch in enumerate(tqdm(dataloader)):
        predictions = run_inference(model, batch["images"], device, dtype, args.mono)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            compute_all_metrics(predictions, batch, metrics_results)

        if args.dump:
            os.makedirs(output_dir, exist_ok=True)
            save_output(predictions, batch, output_dir)

        torch.cuda.empty_cache()

    return integrate_metrics_results(metrics_results)


def main():
    """Main evaluation loop."""
    args = setup_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype()

    model = load_model(device, model_path=args.model_path, reorder_by_learning_ref=args.ref)
    set_random_seeds(args.seed)

    save_dir = osp.join(args.save_path, args.exam)
    if args.mono:
        save_dir += "_mono"
    os.makedirs(save_dir, exist_ok=True)

    # Evaluate real-world split.
    if args.split in ("real_world", "both"):
        dataset = Realsee3DDataset(
            data_dir=osp.join(args.dataset_path, "dataset", "real_world_data"),
            data_list=osp.join(args.dataset_path, "dataset", "real_world_scene_test.txt"),
            pano_width=args.pano_width,
        )
        output_dir = osp.join(save_dir, "real_world_data")
        metrics = evaluate_split(model, dataset, device, dtype, args, output_dir)

        print("evaluate real world dataset")
        print(json.dumps(metrics, indent=2, default=serialize_numpy_and_round))
        with open(osp.join(save_dir, "real_world.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=serialize_numpy_and_round)

    # Evaluate synthetic split.
    if args.split in ("synthetic", "both"):
        dataset = Realsee3DDataset(
            data_dir=osp.join(args.dataset_path, "dataset", "synthetic_data"),
            data_list=osp.join(args.dataset_path, "dataset", "synthetic_scene_test.txt"),
            pano_width=args.pano_width,
        )
        output_dir = osp.join(save_dir, "synthetic_data")
        metrics = evaluate_split(model, dataset, device, dtype, args, output_dir)

        print("evaluate synthetic dataset")
        print(json.dumps(metrics, indent=2, default=serialize_numpy_and_round))
        with open(osp.join(save_dir, "synthetic.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=serialize_numpy_and_round)


if __name__ == "__main__":
    main()
