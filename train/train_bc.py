#!/usr/bin/env python3
"""
UR5e minimal Behavior Cloning training on LeRobot-format dataset.

Usage:
  python train/train_bc.py --dataset datasets/ur5e_block_pick
  python train/train_bc.py --dataset datasets/ur5e_block_pick --epochs 50 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from train.dataset import LeRobotBCDataset
    from train.model import BCPolicy
except ModuleNotFoundError:
    from dataset import LeRobotBCDataset
    from model import BCPolicy


def split_episodes(episodes: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    eps = list(episodes)
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * val_ratio)))
    val_eps = sorted(eps[:n_val])
    train_eps = sorted(eps[n_val:])
    return train_eps, val_eps


def train_one_epoch(
    model: BCPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        action_gt = batch["action"].to(device)

        pred = model(image, state)
        loss = criterion(pred, action_gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.size(0)
        n += image.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model: BCPolicy, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        action_gt = batch["action"].to(device)
        pred = model(image, state)
        loss = criterion(pred, action_gt)
        total_loss += loss.item() * image.size(0)
        n += image.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_episode(
    model: BCPolicy,
    dataset: LeRobotBCDataset,
    episode_index: int,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    indices = [i for i in range(len(dataset)) if dataset.episode_indices[i] == episode_index]
    indices.sort(key=lambda i: dataset.frame_indices[i])
    gt_list, pred_list = [], []
    model.eval()
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        images = torch.stack([dataset[i]["image"] for i in chunk]).to(device)
        states = torch.stack([dataset[i]["state"] for i in chunk]).to(device)
        pred = model(images, states).cpu().numpy()
        gt = np.stack([dataset[i]["action"] for i in chunk])
        pred_list.append(pred)
        gt_list.append(gt)
    return np.concatenate(gt_list, axis=0), np.concatenate(pred_list, axis=0)


def plot_action_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    episode_index: int,
    save_path: Path,
    action_dim: int = 7,
) -> None:
    labels = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"][:action_dim]
    t = np.arange(len(gt))
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 2.2 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    fig.suptitle(f"Episode {episode_index}: pred vs GT action")
    for i, ax in enumerate(axes):
        ax.plot(t, gt[:, i], label="gt", linewidth=1.5)
        ax.plot(t, pred[:, i], label="pred", linewidth=1.2, linestyle="--")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {save_path}")


def save_checkpoint(path: Path, model: BCPolicy, optimizer, epoch: int, val_loss: float, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "meta": meta,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e minimal BC training")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ur5e_block_pick"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/checkpoints"))
    parser.add_argument("--plot-dir", type=Path, default=Path("train/plots"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    full_ds = LeRobotBCDataset(args.dataset, image_size=args.image_size)
    all_episodes = full_ds.episodes()
    train_eps, val_eps = split_episodes(all_episodes, args.val_ratio, args.seed)
    print(f"[split] train episodes ({len(train_eps)}): {train_eps}")
    print(f"[split] val episodes   ({len(val_eps)}): {val_eps}")

    train_indices = [i for i in range(len(full_ds)) if int(full_ds.episode_indices[i]) in train_eps]
    val_indices = [i for i in range(len(full_ds)) if int(full_ds.episode_indices[i]) in val_eps]
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    val_ds_ref = full_ds  # for episode prediction on full arrays

    # print one batch shapes
    sample = train_ds[0]
    print(f"[sample] image={tuple(sample['image'].shape)} state={tuple(sample['state'].shape)} action={tuple(sample['action'].shape)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = BCPolicy(
        state_dim=full_ds.state_dim,
        action_dim=full_ds.action_dim,
        pretrained=not args.no_pretrained,
    ).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    meta = {
        "dataset": str(args.dataset.resolve()),
        "image_key": full_ds.image_key,
        "state_dim": full_ds.state_dim,
        "action_dim": full_ds.action_dim,
        "train_episodes": train_eps,
        "val_episodes": val_eps,
        "image_size": args.image_size,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}, f, indent=2)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f"epoch {epoch:03d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, val_loss, meta)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, val_loss, meta)
            print(f"  → new best val_loss={best_val:.6f}")

    # Plot random val episode
    plot_ep = random.choice(val_eps)
    ckpt = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    gt, pred = predict_episode(model, val_ds_ref, plot_ep, device, batch_size=args.batch_size)
    mse = float(np.mean((gt - pred) ** 2))
    print(f"[eval] episode {plot_ep} action MSE={mse:.6f}")
    plot_action_comparison(
        gt,
        pred,
        plot_ep,
        args.plot_dir / f"action_compare_ep{plot_ep:03d}.png",
        action_dim=full_ds.action_dim,
    )
    print(f"Training done. best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
