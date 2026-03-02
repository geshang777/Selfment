import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference import infer_single_image, save_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--head_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./demo_outputs")
    parser.add_argument(
        "--dino_type",
        type=str,
        default="dinov3",
        choices=["dinov1", "dinov3"],
    )
    parser.add_argument("--dino_repo", type=str, required=True)
    parser.add_argument("--dino_model_name", type=str, required=True)
    parser.add_argument("--dino_weights", type=str, default=None)
    parser.add_argument("--dino_depth", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=1536)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--postprocess",
        type=str,
        default="none",
        choices=["none", "bs", "crf", "bs+crf"],
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    pil, sal_prob = infer_single_image(
        image_path=args.image,
        head_ckpt=args.head_ckpt,
        dino_type=args.dino_type,
        dino_repo=args.dino_repo,
        dino_model_name=args.dino_model_name,
        dino_weights=args.dino_weights,
        dino_depth=args.dino_depth,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        device=device,
    )

    out = save_outputs(pil, args.image, sal_prob, args.output_dir, postprocess=args.postprocess)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
