import argparse
import hashlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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


@torch.no_grad()
def extract_features_from_pil(model, pil_image: Image.Image, transform, device, dino_type, depth):
    inputs = transform(pil_image)[None].to(device)

    dino_type = dino_type.lower()
    if "dinov3" in dino_type:
        feats, _ = model.get_intermediate_layers(
            inputs, n=depth, reshape=True, return_class_token=True
        )[-1]
        _, dim, h, w = feats.shape
        feats = feats.squeeze(0).permute(1, 2, 0).reshape(h * w, dim)
        return feats, (h, w)

    if "dinov2" in dino_type:
        outputs = model.forward_features(inputs)
        feats = outputs["x_norm_patchtokens"].squeeze(0)
        h = w = int(math.sqrt(feats.shape[0]))
        return feats, (h, w)

    if "dinov1" in dino_type:
        outputs = model.get_intermediate_layers(inputs, n=1)
        feats = outputs[-1][0].squeeze(0)
        h = w = int(math.sqrt(feats.shape[0]))
        return feats[1:, :], (h, w)

    raise ValueError(f"Unknown dino_type: {dino_type}")


class ImageFolderDinoDataset(Dataset):
    def __init__(
        self,
        folder: str,
        cache_dir: str,
        transform,
        model,
        device,
        dino_type: str,
        depth: int,
        cache_tag: str,
        exts=(".jpg", ".jpeg", ".png", ".bmp"),
        max_images: int | None = None,
        seed: int = 42,
    ):
        self.folder = Path(folder)
        self.paths = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in exts])
        if max_images is not None and len(self.paths) > max_images:
            rng = random.Random(seed)
            self.paths = rng.sample(self.paths, max_images)

        self.transform = transform
        self.model = model
        self.device = device
        self.dino_type = dino_type
        self.depth = depth
        self.cache_dir = Path(cache_dir)
        self.cache_tag = cache_tag
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(str(p)).convert("RGB")
        cache_path = self.cache_dir / f"{p.stem}_feats_{self.cache_tag}.pt"
        if cache_path.exists():
            cache_data = torch.load(cache_path, map_location=self.device)
            feats = cache_data["feats"].to(self.device)
            grid = cache_data["shape"]
        else:
            feats, grid = extract_features_from_pil(
                self.model,
                pil,
                self.transform,
                self.device,
                self.dino_type,
                self.depth,
            )
            torch.save({"feats": feats.cpu(), "shape": grid}, cache_path)
        return {"path": str(p), "feats": feats, "grid": grid, "size": pil.size}


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


class PatchHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.proj(x)
        logits = self.classifier(emb)
        return logits, emb


def _unpack_hw(grid_value):
    if torch.is_tensor(grid_value):
        if grid_value.numel() == 2:
            return int(grid_value.view(-1)[0].item()), int(grid_value.view(-1)[1].item())
        if grid_value.numel() == 1:
            raise ValueError("grid must have 2 values")
    if isinstance(grid_value, (list, tuple)) and len(grid_value) == 2:
        a, b = grid_value
        if torch.is_tensor(a):
            a = a.view(-1)[0].item()
        if torch.is_tensor(b):
            b = b.view(-1)[0].item()
        return int(a), int(b)
    raise ValueError(f"Unsupported grid type: {type(grid_value)}")


def _tokencut_init_labels(
    feats: torch.Tensor,
    h: int,
    w: int,
    tau: float = 0.2,
    eps: float = 1e-5,
    no_binary_graph: bool = False,
) -> torch.Tensor:
    from scipy import ndimage
    from scipy.linalg import eigh

    feats = F.normalize(feats, dim=1)
    a = (feats @ feats.transpose(1, 0)).detach().float().cpu().numpy()
    if no_binary_graph:
        a[a < tau] = eps
    else:
        a = a > tau
        a = np.where(a.astype(float) == 0, eps, a)

    d_i = np.sum(a, axis=1)
    d = np.diag(d_i)
    _, eigenvectors = eigh(d - a, d, subset_by_index=[1, 2])
    second_smallest_vec = np.copy(eigenvectors[:, 0])
    avg = float(np.sum(second_smallest_vec) / len(second_smallest_vec))
    bipartition = second_smallest_vec > avg
    seed = int(np.argmax(np.abs(second_smallest_vec)))

    if not bipartition[seed]:
        bipartition = np.logical_not(bipartition)

    bip_2d = bipartition.reshape(h, w)
    objects, _ = ndimage.label(bip_2d)
    seed_xy = np.unravel_index(seed, (h, w))
    cc_id = objects[seed_xy]
    if cc_id == 0:
        mask = bip_2d.astype(np.uint8)
    else:
        mask = (objects == cc_id).astype(np.uint8)

    labels = torch.from_numpy(mask.reshape(-1)).to(device=feats.device)
    return labels.long()


def _unsupervised_saliency_labels(
    feats: torch.Tensor,
    h: int,
    w: int,
    n_iter: int = 20,
    sample_size: int = 10,
    tau: float = 0.2,
    no_binary_graph: bool = False,
) -> torch.Tensor:
    feats = F.normalize(feats, dim=1)
    labels = _tokencut_init_labels(
        feats,
        h,
        w,
        tau=tau,
        eps=1e-5,
        no_binary_graph=no_binary_graph,
    )

    if labels.sum().item() == 0 or labels.sum().item() == labels.numel():
        return labels

    fg_anchor = feats[labels == 1].mean(dim=0)
    bg_anchor = feats[labels == 0].mean(dim=0)
    ref_vector = fg_anchor - bg_anchor

    for _ in range(n_iter):
        fg_idx = (labels == 1).nonzero(as_tuple=True)[0]
        bg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        if len(fg_idx) < sample_size or len(bg_idx) < sample_size:
            break
        fg_mean = feats[fg_idx].mean(dim=0)
        bg_mean = feats[bg_idx].mean(dim=0)
        sim_fg = feats @ fg_mean
        sim_bg = feats @ bg_mean
        labels = (sim_fg > sim_bg).long()
        if labels.sum().item() == 0 or labels.sum().item() == labels.numel():
            break
        cur_mean_fg = feats[labels == 1].mean(dim=0)
        cur_mean_bg = feats[labels == 0].mean(dim=0)
        if torch.dot(cur_mean_fg - cur_mean_bg, ref_vector) < 0:
            labels = 1 - labels
    return labels


def contrastive_loss_full_patch(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
    device = emb.device
    idx_fg = (labels == 1).nonzero(as_tuple=True)[0]
    idx_bg = (labels == 0).nonzero(as_tuple=True)[0]

    if len(idx_fg) < 2 or len(idx_bg) < 2:
        return torch.tensor(0.0, device=device)

    ef = F.normalize(emb[idx_fg], dim=1)
    eb = F.normalize(emb[idx_bg], dim=1)
    all_emb = torch.cat([ef, eb], dim=0)
    all_labels = torch.cat(
        [torch.ones(len(ef), device=device), torch.zeros(len(eb), device=device)]
    )

    sim_matrix = all_emb @ all_emb.T / temperature
    labels_matrix = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
    mask_self = torch.eye(labels_matrix.size(0), dtype=torch.bool, device=device)
    labels_matrix = labels_matrix & (~mask_self)
    log_probs = F.log_softmax(sim_matrix, dim=1)

    loss = 0.0
    for i in range(all_emb.size(0)):
        pos_idx = labels_matrix[i].nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            continue
        loss = loss + (-log_probs[i, pos_idx].mean())
    return loss / all_emb.size(0)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + eps) / (union + eps)


def save_checkpoint(state: dict, outdir: str, epoch: int):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(outdir, f"checkpoint_epoch{epoch}.pth"))


def train_ddp(args):
    try:
        import wandb
    except Exception:
        wandb = None

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank = int(os.environ.get("RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if rank == 0 and wandb is not None and args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    transform = make_transform(args.img_size)
    dino = load_dino_model(
        args.dino_type,
        args.dino_repo,
        args.dino_model_name,
        args.dino_weights,
        device,
    )

    dataset = ImageFolderDinoDataset(
        args.input_dir,
        args.cache_dir,
        transform,
        dino,
        device,
        args.dino_type,
        args.dino_depth,
        cache_tag=_cache_tag(args),
        max_images=args.max_images,
        seed=args.seed,
    )
    if len(dataset) == 0:
        if rank == 0:
            print(f"No images found in {args.input_dir}")
        return

    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    sample0 = dataset[0]
    _, dim = sample0["feats"].shape
    del sample0

    model_head = PatchHead(in_dim=dim, embed_dim=args.embed_dim).to(device)
    model_head = torch.nn.parallel.DistributedDataParallel(
        model_head, device_ids=[local_rank], output_device=local_rank
    )

    optimizer = torch.optim.Adam(
        model_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model_head.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, item in enumerate(loader):
            feats = F.normalize(item["feats"][0], dim=1)
            h, w = _unpack_hw(item["grid"])
            labels = _unsupervised_saliency_labels(
                feats,
                h,
                w,
                n_iter=args.pseudo_iter,
                sample_size=args.pseudo_sample_size,
                tau=args.tau,
                no_binary_graph=args.no_binary_graph,
            )

            logits, emb = model_head(feats)
            ce_loss = F.binary_cross_entropy_with_logits(logits[:, 1], labels.float())
            sim_loss = contrastive_loss_full_patch(emb, labels, temperature=args.temperature)
            pred_probs = F.softmax(logits, dim=1)[:, 1]
            loss_dice = dice_loss(pred_probs, labels.float())

            loss = ce_loss + args.sim_weight * sim_loss + loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if rank == 0 and batch_idx % args.log_interval == 0:
                print(
                    f"Epoch[{epoch+1}/{args.epochs}] Step {batch_idx}/{len(loader)} "
                    f"loss {loss.item():.4f} ce {ce_loss.item():.4f} dice {loss_dice.item():.4f} "
                    f"sim {sim_loss.item():.4f}"
                )
                if wandb is not None and args.wandb_project:
                    wandb.log(
                        {
                            "loss_total": loss.item(),
                            "loss_ce": ce_loss.item(),
                            "loss_sim": sim_loss.item(),
                            "loss_dice": loss_dice.item(),
                            "step": global_step,
                        }
                    )

        if rank == 0:
            avg_loss = epoch_loss / max(1, len(loader))
            print(
                f"Epoch {epoch+1} done. time {time.time()-start_time:.1f}s avg_loss {avg_loss:.4f}"
            )
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model_head.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            }
            save_checkpoint(ckpt, args.output_dir, epoch + 1)

    if rank == 0 and wandb is not None and args.wandb_project:
        wandb.finish()
    torch.distributed.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./saliency_results")
    parser.add_argument("--cache_dir", type=str, default="./feature_cache")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--sim_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--no_binary_graph", action="store_true")
    parser.add_argument("--pseudo_iter", type=int, default=20)
    parser.add_argument("--pseudo_sample_size", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument(
        "--dino_type",
        type=str,
        default="dinov3",
        choices=["dinov1", "dinov2", "dinov3"],
    )
    parser.add_argument("--dino_repo", type=str, required=True)
    parser.add_argument("--dino_model_name", type=str, required=True)
    parser.add_argument("--dino_weights", type=str, default=None)
    parser.add_argument("--dino_depth", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    train_ddp(args)


if __name__ == "__main__":
    main()
