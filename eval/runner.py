import os
from glob import glob
from typing import Sequence

import numpy as np

from PIL import Image
from tqdm import tqdm

from .metrics import compute_all_metrics


def load_masks_from_folder(
    pred_dir: str,
    gt_dir: str,
    suffix_pred: str = "_saliency.png",
    suffix_gt: str = ".png",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pred_paths = sorted(glob(os.path.join(pred_dir, f"*{suffix_pred}")))
    if not pred_paths:
        raise FileNotFoundError(
            f"No prediction files found in {pred_dir} with suffix {suffix_pred}"
        )

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []

    for pred_path in tqdm(pred_paths, desc="Loading masks", ncols=100):
        base = os.path.basename(pred_path).replace(suffix_pred, "")
        gt_path = os.path.join(gt_dir, base + suffix_gt)
        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found for {pred_path}, expected {gt_path}, skipping.")
            continue

        pred = np.array(Image.open(pred_path).convert("L"), dtype=np.float32) / 255.0
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0

        if pred.shape != gt.shape:
            gt = np.array(
                Image.fromarray((gt * 255).astype(np.uint8)).resize(
                    pred.shape[::-1], Image.BILINEAR
                )
            ) / 255.0

        preds.append(pred)
        gts.append(gt)

    if not preds:
        raise RuntimeError("No valid prediction/GT pairs found.")

    return preds, gts


def evaluate_folder(
    pred_dir: str,
    gt_dir: str,
    suffix_pred: str = "_saliency.png",
    suffix_gt: str = ".png",
    metrics: Sequence[str] | None = None,
) -> dict:
    preds, gts = load_masks_from_folder(pred_dir, gt_dir, suffix_pred, suffix_gt)
    results = compute_all_metrics(preds, gts)
    if metrics is not None:
        results = {k: v for k, v in results.items() if k in metrics}
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Unified evaluation for Selfment.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Folder with prediction masks")
    parser.add_argument("--gt_dir", type=str, required=True, help="Folder with ground-truth masks")
    parser.add_argument("--suffix_pred", type=str, default="_saliency.png")
    parser.add_argument("--suffix_gt", type=str, default=".png")
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma separated list, e.g. IoU,accuracy,F_max,wFmeasure,MAE,Smeasure,meanEm",
    )
    args = parser.parse_args()

    if args.metrics.strip().lower() == "all":
        keep = None
    else:
        keep = [m.strip() for m in args.metrics.split(",") if m.strip()]

    results = evaluate_folder(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        suffix_pred=args.suffix_pred,
        suffix_gt=args.suffix_gt,
        metrics=keep,
    )

    print("\n=== Evaluation Results ===")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"{k:10s}: {v:.4f}")
        else:
            print(f"{k:10s}: {v}")
    print("==========================\n")


if __name__ == "__main__":
    main()
