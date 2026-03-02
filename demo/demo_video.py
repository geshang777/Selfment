import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference import (
    PatchHead,
    _expected_head_shapes,
    _normalize_dino_type,
    extract_features,
    load_dino_model,
    make_transform,
    visualize_mask_clean,
)


def _build_writer(path: str, fps: float, w: int, h: int):
    import cv2

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, float(fps), (int(w), int(h)))


def _apply_postprocess(postprocess: str, frame_bgr: np.ndarray, sal_prob: np.ndarray) -> np.ndarray:
    postprocess = (postprocess or "none").strip().lower()
    if postprocess in {"none", ""}:
        return sal_prob.astype(np.float32)
    if postprocess in {"crf"}:
        from utils.crf import dense_crf

        refined = dense_crf(frame_bgr, sal_prob)
        return refined.astype(np.float32)
    raise ValueError("demo_video.py only supports postprocess: none|crf for videos")


def _video_meta(video_path: str) -> tuple[float, int, int, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in is None or fps_in <= 0:
        fps_in = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps_in), w, h, total


def _frame_indices(total: int, start_frame: int, max_frames: int, stride: int) -> list[int]:
    start = max(0, int(start_frame))
    stride = max(1, int(stride))
    end = total if total > 0 else None
    if max_frames is not None and int(max_frames) > 0:
        if end is None:
            end = start + int(max_frames) * stride
        else:
            end = min(end, start + int(max_frames) * stride)
    if end is None:
        raise RuntimeError("Unknown total frame count; please pass --max_frames > 0")
    return list(range(start, int(end), stride))


def _resolve_output_paths(
    video: str,
    output_dir: str,
    output_video: str | None,
    output_mask_video: str | None,
    output_bin_mask_video: str | None,
):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    out_vis = output_video or os.path.join(output_dir, f"{stem}_vis.mp4")
    out_mask = output_mask_video or os.path.join(output_dir, f"{stem}_mask.mp4")
    out_mask_bin = output_bin_mask_video or os.path.join(output_dir, f"{stem}_mask_bin.mp4")
    return out_vis, out_mask, out_mask_bin


def _init_head_from_first_frame(
    args,
    device: torch.device,
    frame_bgr: np.ndarray,
):
    import cv2

    args.dino_type = _normalize_dino_type(args.dino_type, args.dino_repo, args.dino_model_name)
    transform = make_transform(args.img_size)
    dino = load_dino_model(args.dino_type, args.dino_repo, args.dino_model_name, args.dino_weights, device)

    first_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tmp_feats, _ = extract_features(dino, first_pil, transform, device, args.dino_type, args.dino_depth)
    in_dim = int(tmp_feats.shape[1])
    del tmp_feats

    ckpt = torch.load(args.head_ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    expected_in_dim, expected_embed_dim = _expected_head_shapes(state if isinstance(state, dict) else {})
    embed_dim = int(args.embed_dim)
    if expected_embed_dim is not None and embed_dim != expected_embed_dim:
        embed_dim = expected_embed_dim

    head = PatchHead(in_dim=in_dim, embed_dim=embed_dim).to(device)
    if expected_in_dim is not None and expected_in_dim != in_dim:
        raise RuntimeError(
            f"Checkpoint head expects in_dim={expected_in_dim}, but current DINO features are in_dim={in_dim}."
        )
    head.load_state_dict(state)
    head.eval()
    return transform, dino, head


@torch.inference_mode()
def _infer_one_frame(
    args,
    device: torch.device,
    transform,
    dino,
    head,
    frame_bgr: np.ndarray,
    out_size_hw: tuple[int, int],
):
    import cv2

    h, w = out_size_hw
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    feats, (fh, fw) = extract_features(dino, pil, transform, device, args.dino_type, args.dino_depth)
    feats = F.normalize(feats, dim=1)
    logits, _ = head(feats)
    sal = logits[:, 1].reshape(fh, fw)
    sal_up = F.interpolate(
        sal[None, None],
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    sal_prob = torch.sigmoid(sal_up).clamp(0, 1).detach().cpu().numpy().astype(np.float32)

    mask_for_vis = _apply_postprocess(args.postprocess, frame_bgr, sal_prob)
    vis_pil = visualize_mask_clean(pil, mask_for_vis)
    vis_bgr = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)

    mask_bin_u8 = ((mask_for_vis > 0.6).astype(np.uint8) * 255)
    if args.mask_mode == "bin":
        mask_u8 = mask_bin_u8
    else:
        mask_u8 = (np.clip(mask_for_vis, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    mask_bin_bgr = cv2.cvtColor(mask_bin_u8, cv2.COLOR_GRAY2BGR)
    return vis_bgr, mask_bgr, mask_bin_bgr


def _run_single_gpu(args, fps_out: float, w: int, h: int, indices: list[int]):
    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read first frame: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_vis, out_mask, out_mask_bin = _resolve_output_paths(
        args.video,
        args.output_dir,
        args.output_video,
        args.output_mask_video,
        args.output_bin_mask_video,
    )
    vis_writer = _build_writer(out_vis, fps_out, w, h)
    mask_writer = _build_writer(out_mask, fps_out, w, h)
    mask_bin_writer = _build_writer(out_mask_bin, fps_out, w, h)

    device = torch.device(args.device)
    transform, dino, head = _init_head_from_first_frame(args, device, first)

    pbar = tqdm(total=len(indices), desc="video inference", ncols=100)
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                break
            vis_bgr, mask_bgr, mask_bin_bgr = _infer_one_frame(
                args, device, transform, dino, head, frame_bgr, (h, w)
            )
            vis_writer.write(vis_bgr)
            mask_writer.write(mask_bgr)
            mask_bin_writer.write(mask_bin_bgr)
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        vis_writer.release()
        mask_writer.release()
        mask_bin_writer.release()

    print(f"vis_video: {out_vis}")
    print(f"mask_video: {out_mask}")
    print(f"mask_bin_video: {out_mask_bin}")


def _worker_video(rank: int, world_size: int, args, indices: list[int], w: int, h: int, queue):
    import cv2

    device = torch.device(f"cuda:{rank}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        queue.put(("__error__", rank, f"Failed to open video: {args.video}"))
        queue.put(("__done__", rank, None))
        return

    ok, first = cap.read()
    if not ok:
        cap.release()
        queue.put(("__error__", rank, f"Failed to read first frame: {args.video}"))
        queue.put(("__done__", rank, None))
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        transform, dino, head = _init_head_from_first_frame(args, device, first)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                break
            vis_bgr, mask_bgr, mask_bin_bgr = _infer_one_frame(
                args, device, transform, dino, head, frame_bgr, (h, w)
            )
            queue.put((int(idx), vis_bgr, mask_bgr, mask_bin_bgr))
    except Exception as e:
        queue.put(("__error__", rank, repr(e)))
    finally:
        cap.release()
        queue.put(("__done__", rank, None))


def _run_multi_gpu(args, fps_out: float, w: int, h: int, indices: list[int]):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run multi-GPU inference")
    world_size = int(args.num_gpus) if args.num_gpus is not None else int(torch.cuda.device_count())
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found")

    out_vis, out_mask, out_mask_bin = _resolve_output_paths(
        args.video,
        args.output_dir,
        args.output_video,
        args.output_mask_video,
        args.output_bin_mask_video,
    )
    vis_writer = _build_writer(out_vis, fps_out, w, h)
    mask_writer = _build_writer(out_mask, fps_out, w, h)
    mask_bin_writer = _build_writer(out_mask_bin, fps_out, w, h)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=int(args.queue_size))

    chunks = [indices[i::world_size] for i in range(world_size)]
    procs: list[mp.Process] = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker_video, args=(rank, world_size, args, chunks[rank], w, h, queue))
        p.daemon = True
        p.start()
        procs.append(p)

    done = 0
    buffer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    next_pos = 0
    errors: list[str] = []

    pbar = tqdm(total=len(indices), desc=f"video inference (x{world_size})", ncols=100)
    try:
        while done < world_size or buffer:
            msg = queue.get()
            if not msg:
                continue
            key = msg[0]
            if key == "__done__":
                done += 1
                continue
            if key == "__error__":
                _, r, err = msg
                errors.append(f"[GPU {r}] {err}")
                continue

            idx, vis_bgr, mask_bgr, mask_bin_bgr = msg
            buffer[int(idx)] = (vis_bgr, mask_bgr, mask_bin_bgr)

            while next_pos < len(indices) and indices[next_pos] in buffer:
                vis, mask, mask_bin = buffer.pop(indices[next_pos])
                vis_writer.write(vis)
                mask_writer.write(mask)
                mask_bin_writer.write(mask_bin)
                next_pos += 1
                pbar.update(1)
    finally:
        pbar.close()
        vis_writer.release()
        mask_writer.release()
        mask_bin_writer.release()
        for p in procs:
            p.join(timeout=0.2)
            if p.is_alive():
                p.terminate()

    if errors:
        raise RuntimeError("\n".join(errors))

    print(f"vis_video: {out_vis}")
    print(f"mask_video: {out_mask}")
    print(f"mask_bin_video: {out_mask_bin}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--head_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./demo_outputs_video")
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
    parser.add_argument("--img_size", type=int, default=2560)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--postprocess", type=str, default="none", choices=["none", "crf"])
    parser.add_argument("--output_video", type=str, default=None)
    parser.add_argument("--output_mask_video", type=str, default=None)
    parser.add_argument("--output_bin_mask_video", type=str, default=None)
    parser.add_argument("--mask_mode", type=str, default="prob", choices=["prob", "bin"])
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--queue_size", type=int, default=8)
    args = parser.parse_args()

    fps_in, w, h, total = _video_meta(args.video)
    fps_out = float(args.fps) if args.fps is not None else float(fps_in)
    indices = _frame_indices(total, args.start_frame, args.max_frames, args.stride)
    if not indices:
        raise RuntimeError("No frames selected for inference")

    if args.multi_gpu:
        _run_multi_gpu(args, fps_out, w, h, indices)
    else:
        _run_single_gpu(args, fps_out, w, h, indices)


if __name__ == "__main__":
    main()
