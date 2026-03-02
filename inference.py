import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.multiprocessing import spawn
from torchvision import transforms
from tqdm import tqdm


def make_transform(resize_size: int = 768):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_dino_model(
    dino_type: str,
    repo_path: str,
    model_name: str,
    weights_path: str | None,
    device: torch.device,
):
    repo_path = os.path.abspath(repo_path)
    hubconf_path = os.path.join(repo_path, "hubconf.py")

    if os.path.isfile(hubconf_path):
        model = torch.hub.load(repo_path, model_name, source="local", weights=weights_path)
    else:
        if "dinov3" not in dino_type.lower():
            raise FileNotFoundError(
                f"hubconf.py not found in {repo_path}. "
                f"If this is not a DINOv3 repo, please pass a repo path that contains hubconf.py."
            )

        import importlib
        import sys

        repo_parent = os.path.dirname(repo_path)
        if repo_parent not in sys.path:
            sys.path.insert(0, repo_parent)

        pkg_name = os.path.basename(repo_path)
        backbones = importlib.import_module(f"{pkg_name}.hub.backbones")
        if not hasattr(backbones, model_name):
            raise ValueError(
                f"Model '{model_name}' not found in {pkg_name}.hub.backbones."
            )
        fn = getattr(backbones, model_name)
        if weights_path is None:
            model = fn(pretrained=False)
        else:
            model = fn(pretrained=True, weights=weights_path)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _normalize_dino_type(dino_type: str, dino_repo: str, dino_model_name: str) -> str:
    t = (dino_type or "").lower()
    repo = (dino_repo or "").lower()
    name = (dino_model_name or "").lower()

    if "dinov3" in name or "dinov3" in repo:
        if t != "dinov3":
            print(f"[WARN] dino_type={dino_type} but model/repo looks like DINOv3; using dinov3")
        return "dinov3"
    if "dinov1" in name or "dino" in repo:
        if t != "dinov1":
            print(f"[WARN] dino_type={dino_type} but model/repo looks like DINOv1; using dinov1")
        return "dinov1"
    return dino_type


def _expected_head_shapes(state: dict) -> tuple[int | None, int | None]:
    w = state.get("proj.0.weight")
    if w is None:
        return None, None
    if torch.is_tensor(w) and w.ndim == 2:
        embed_dim = int(w.shape[0])
        in_dim = int(w.shape[1])
        return in_dim, embed_dim
    return None, None


@torch.no_grad()
def extract_features(
    model,
    pil_image: Image.Image,
    transform,
    device: torch.device,
    dino_type: str,
    depth: int,
):
    inputs = transform(pil_image)[None].to(device)

    dino_type = dino_type.lower()
    if "dinov3" in dino_type:
        feats, _ = model.get_intermediate_layers(
            inputs, n=depth, reshape=True, return_class_token=True
        )[-1]
        _, dim, h, w = feats.shape
        feats = feats.squeeze(0).permute(1, 2, 0).reshape(h * w, dim)
        return feats, (h, w)

    if "dinov1" in dino_type:
        if hasattr(model, "get_intermediate_layers"):
            outputs = model.get_intermediate_layers(inputs, n=1)
            feats = outputs[-1][0].squeeze(0)
        elif hasattr(model, "forward_features"):
            feats = model.forward_features(inputs).squeeze(0)
        else:
            raise AttributeError("DINOv1 model does not support feature extraction")
        h = w = int(math.sqrt(feats.shape[0]))
        return feats[1:, :], (h, w)

    raise ValueError(f"Unknown dino_type: {dino_type}")


class PatchHead(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 128):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, embed_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.classifier = torch.nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor):
        emb = self.proj(x)
        logits = self.classifier(emb)
        return logits, emb


def _cache_tag(args) -> str:
    payload = "|".join(
        [
            str(args.dino_type),
            str(args.dino_model_name),
            str(args.dino_depth),
            str(args.img_size),
            str(args.dino_weights or ""),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def visualize_mask_clean(
    image: Image.Image,
    mask_np: np.ndarray,
    alpha_dark: float = 0.6,
    highlight_gain: float = 1.1,
    edge_thickness: int = 0,
) -> Image.Image:
    import cv2

    image_np = np.array(image).astype(np.float32)

    if mask_np.dtype == np.bool_:
        mask_gray = (mask_np.astype(np.uint8) * 255)
    else:
        mask_f = mask_np.astype(np.float32)
        if mask_f.max(initial=0.0) <= 1.0:
            mask_gray = np.clip(mask_f * 255.0, 0, 255).astype(np.uint8)
        else:
            mask_gray = np.clip(mask_f, 0, 255).astype(np.uint8)

    _, mask_bin = cv2.threshold(
        mask_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask_bin = (mask_bin > 128).astype(np.uint8)

    if mask_bin.shape[:2] != image_np.shape[:2]:
        mask_bin = cv2.resize(
            mask_bin,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_bin_3 = np.repeat(mask_bin[:, :, None], 3, axis=2).astype(np.float32)

    dark_image = image_np * (1.0 - alpha_dark)

    highlight_image = np.clip(image_np * highlight_gain, 0, 255)

    final = highlight_image * mask_bin_3 + dark_image * (1.0 - mask_bin_3)

    edges = cv2.Canny((mask_bin * 255).astype(np.uint8), 30, 120)
    if edge_thickness is not None and edge_thickness > 1:
        edges = cv2.dilate(
            edges,
            np.ones((edge_thickness, edge_thickness), np.uint8),
        )

    final[edges > 0] = 255.0

    return Image.fromarray(final.astype(np.uint8))


@torch.no_grad()
def infer_single_image(
    image_path: str,
    head_ckpt: str,
    dino_type: str,
    dino_repo: str,
    dino_model_name: str,
    dino_weights: str | None,
    dino_depth: int,
    img_size: int,
    embed_dim: int,
    device: torch.device,
):
    pil = Image.open(image_path).convert("RGB")
    transform = make_transform(img_size)

    model = load_dino_model(dino_type, dino_repo, dino_model_name, dino_weights, device)

    feats, (h, w) = extract_features(model, pil, transform, device, dino_type, dino_depth)
    feats = F.normalize(feats, dim=1)

    ckpt = torch.load(head_ckpt, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    head = PatchHead(in_dim=feats.shape[1], embed_dim=embed_dim).to(device)
    head.load_state_dict(state)
    head.eval()

    logits, _ = head(feats)
    sal = logits[:, 1].reshape(h, w)
    sal_t = sal[None, None]
    sal_up = F.interpolate(
        sal_t,
        size=pil.size[::-1],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    sal_prob = torch.sigmoid(sal_up).cpu().numpy().astype(np.float32)
    return pil, sal_prob


def save_outputs(
    pil: Image.Image,
    image_path: str,
    sal_prob: np.ndarray,
    output_dir: str,
    postprocess: str = "none",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    out_prob = os.path.join(output_dir, stem + "_saliency.png")
    Image.fromarray((sal_prob * 255).astype(np.uint8)).save(out_prob)

    postprocess = postprocess.strip().lower()
    enable_bs = postprocess in {"bs", "bs+crf"}
    enable_crf = postprocess in {"crf", "bs+crf"}

    out: dict[str, str] = {"saliency": out_prob}

    mask_for_next = sal_prob
    # import pdb;pdb.set_trace()
    if enable_bs:
        from utils.bilateral_solver import bilateral_solver_output

        output_solver, _ = bilateral_solver_output(image_path, sal_prob)
        mask_for_next = (output_solver > 0.5).astype(np.float32)
        out_bs = os.path.join(output_dir, stem + "_bs.png")
        Image.fromarray((mask_for_next * 255).astype(np.uint8)).save(out_bs)
        out["bs"] = out_bs

    if enable_crf:
        import cv2

        from utils.crf import dense_crf

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        mask_refine = dense_crf(image_bgr, mask_for_next)
        out_crf = os.path.join(output_dir, stem + "_crf.png")
        Image.fromarray((mask_refine * 255).astype(np.uint8)).save(out_crf)
        mask_for_next = mask_refine.astype(np.float32)
        out["crf"] = out_crf

    vis_out = visualize_mask_clean(pil, mask_for_next)
    out_vis = os.path.join(output_dir, stem + "_vis.png")
    vis_out.save(out_vis)
    out["vis"] = out_vis

    return out


@torch.no_grad()
def inference_worker(rank: int, world_size: int, args, img_chunks):
    img_paths = img_chunks[rank]
    if not img_paths:
        return
    device = torch.device(f"cuda:{rank}")

    args.dino_type = _normalize_dino_type(args.dino_type, args.dino_repo, args.dino_model_name)

    transform = make_transform(args.img_size)
    dino = load_dino_model(
        args.dino_type,
        args.dino_repo,
        args.dino_model_name,
        args.dino_weights,
        device,
    )

    first_pil = Image.open(str(img_paths[0])).convert("RGB")
    tmp_feats, _ = extract_features(dino, first_pil, transform, device, args.dino_type, args.dino_depth)
    in_dim = tmp_feats.shape[1]
    del tmp_feats

    ckpt = torch.load(args.head_ckpt, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    expected_in_dim, expected_embed_dim = _expected_head_shapes(state)
    if expected_embed_dim is not None and args.embed_dim != expected_embed_dim:
        print(
            f"[WARN] embed_dim={args.embed_dim} but checkpoint expects {expected_embed_dim}; using {expected_embed_dim}"
        )
        args.embed_dim = expected_embed_dim
    head = PatchHead(in_dim=in_dim, embed_dim=args.embed_dim).to(device)
    if expected_in_dim is not None and expected_in_dim != in_dim:
        train_args = ckpt.get("args") if isinstance(ckpt, dict) else None
        msg = (
            f"Checkpoint head expects in_dim={expected_in_dim}, but current DINO features are in_dim={in_dim}. "
            f"This usually means you are using a different DINO backbone than the one used for training."
        )
        if isinstance(train_args, dict):
            msg += (
                "\nCheckpoint training config: "
                + ", ".join(
                    [
                        f"dino_type={train_args.get('dino_type')}",
                        f"dino_repo={train_args.get('dino_repo')}",
                        f"dino_model_name={train_args.get('dino_model_name')}",
                        f"dino_depth={train_args.get('dino_depth')}",
                        f"img_size={train_args.get('img_size')}",
                    ]
                )
            )
        raise RuntimeError(msg)
    head.load_state_dict(state)
    head.eval()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    tag = _cache_tag(args)
    expected_dim = head.proj[0].in_features

    for p in tqdm(img_paths, desc=f"[GPU {rank}]", ncols=100):
        p = Path(p)
        cache_path = Path(args.cache_dir) / f"{p.stem}_feats_{tag}.pt"
        pil = Image.open(str(p)).convert("RGB")

        if cache_path.exists():
            cache_data = torch.load(cache_path, map_location=device)
            feats = cache_data["feats"].to(device)
            h, w = cache_data["shape"]
            if feats.shape[1] != expected_dim:
                cache_path.unlink(missing_ok=True)
                feats, (h, w) = extract_features(
                    dino, pil, transform, device, args.dino_type, args.dino_depth
                )
                feats = F.normalize(feats, dim=1)
                torch.save({"feats": feats.cpu(), "shape": (h, w)}, cache_path)
        else:
            feats, (h, w) = extract_features(
                dino, pil, transform, device, args.dino_type, args.dino_depth
            )
            feats = F.normalize(feats, dim=1)
            torch.save({"feats": feats.cpu(), "shape": (h, w)}, cache_path)

        logits, _ = head(feats)
        prob = F.softmax(logits, dim=1)[:, 1].reshape(h, w)
        prob_up = F.interpolate(
            prob[None, None],
            size=pil.size[::-1],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        sal_prob = prob_up.clamp(0, 1).cpu().numpy().astype(np.float32)

        save_outputs(pil, str(p), sal_prob, args.output_dir, postprocess=args.postprocess)


def inference_folder_multi(args):
    all_imgs = sorted(
        [
            str(p)
            for p in Path(args.input_dir).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
    )
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        inference_worker(0, 1, args, [all_imgs])
        return

    chunk_size = int(np.ceil(len(all_imgs) / world_size))
    img_chunks = [all_imgs[i : i + chunk_size] for i in range(0, len(all_imgs), chunk_size)]

    if len(img_chunks) < world_size:
        img_chunks.extend([[] for _ in range(world_size - len(img_chunks))])

    spawn(
        inference_worker,
        args=(world_size, args, img_chunks),
        nprocs=world_size,
        join=True,
    )


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_ckpt", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--cache_dir", type=str, default="./feature_cache")
    parser.add_argument("--img_size", type=int, default=1536)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument(
        "--dino_type",
        type=str,
        default="dinov3",
        choices=["dinov1", "dinov3"],
    )
    parser.add_argument("--dino_repo", type=str, required=True)
    parser.add_argument("--dino_weights", type=str, default=None)
    parser.add_argument("--dino_model_name", type=str, required=True)
    parser.add_argument("--dino_depth", type=int, default=4)
    parser.add_argument(
        "--postprocess",
        type=str,
        default="none",
        choices=["none", "bs", "crf", "bs+crf"],
    )
    return parser


def main():
    args = build_argparser().parse_args()
    inference_folder_multi(args)


if __name__ == "__main__":
    main()
