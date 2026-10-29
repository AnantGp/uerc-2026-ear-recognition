import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from rank1_lib import (
    ArcMarginProduct,
    CANDIDATES,
    UERCTrainDataset,
    append_jsonl,
    configure_torch,
    create_candidate_model,
    evaluate_model_on_pairs,
    gate_status,
    model_stats,
    save_json,
    set_seed,
    train_transform,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a compact UERC 2026 Track 1 candidate.")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/anant/uerc2026/data_raw/data/public"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="mobilenetv4s050")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--val-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--arc-s", type=float, default=30.0)
    parser.add_argument("--arc-m", type=float, default=0.30)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--tta-val", action="store_true")
    return parser.parse_args()


def save_checkpoint(path, model, margin, optimizer, scheduler, epoch, metrics, args):
    payload = {
        "model": model.state_dict(),
        "margin": margin.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_resume(path, model, margin, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    if "margin" in checkpoint:
        margin.load_state_dict(checkpoint["margin"], strict=False)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", -1)) + 1, checkpoint.get("metrics", {})


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.work_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    set_seed(args.seed)
    configure_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_file = args.data_dir / "dataset_split.csv"
    pairs_file = args.data_dir / "pairs_val.csv"

    train_dataset = UERCTrainDataset(args.data_dir, split_file, transform=train_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_dataset.weighted_sampler(),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = create_candidate_model(args.candidate, pretrained=True, embedding_dim=args.embedding_dim).to(device)
    margin = ArcMarginProduct(args.embedding_dim, train_dataset.num_classes, s=args.arc_s, m=args.arc_m).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": margin.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_ver = -1.0
    best_metrics = {}
    if args.resume:
        start_epoch, previous_metrics = load_resume(args.resume, model, margin, optimizer, scheduler, device)
        best_ver = float(previous_metrics.get("VER@0.1%", -1.0))
        best_metrics = previous_metrics
        print(f"Resumed from {args.resume} at epoch {start_epoch}; previous best VER@0.1%={best_ver:.6f}")

    stats = model_stats(model)
    print(json.dumps({"event": "start", "device": str(device), "num_classes": train_dataset.num_classes, "train_images": len(train_dataset), **stats}, indent=2))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        margin.train()
        running = 0.0
        seen = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast("cuda", enabled=device.type == "cuda"):
                embeddings = model(images)
                logits = margin(embeddings, labels)
                loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(margin.parameters()), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch = images.size(0)
            running += float(loss.item()) * batch
            seen += batch
            pbar.set_postfix(loss=running / max(seen, 1), lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            metrics, info = evaluate_model_on_pairs(
                model,
                args.data_dir,
                pairs_file,
                batch_size=args.val_batch_size,
                workers=args.workers,
                device=device,
                max_pairs=args.max_val_pairs,
                tta=args.tta_val,
            )

        ckpt_path = args.work_dir / f"epoch_{epoch + 1:03d}.pt"
        combined = {
            **metrics,
            **model_stats(model, ckpt_path),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "seconds": time.time() - epoch_start,
            "candidate": args.candidate,
            "val_info": info,
        }
        save_checkpoint(ckpt_path, model, margin, optimizer, scheduler, epoch, combined, args)
        checks, gate_ok = gate_status({**combined, "weight_file_size": combined["model_size"]})
        combined["gate_ok"] = gate_ok
        combined["gate_checks"] = checks
        append_jsonl(args.work_dir / "history.jsonl", combined)
        save_json(args.work_dir / "latest_metrics.json", combined)

        if metrics["VER@0.1%"] > best_ver:
            best_ver = metrics["VER@0.1%"]
            best_metrics = combined
            save_checkpoint(args.work_dir / "best.pt", model, margin, optimizer, scheduler, epoch, combined, args)
            save_json(args.work_dir / "best_metrics.json", combined)
            print(f"new best epoch={epoch + 1} VER@0.1%={best_ver:.6f} AUC={metrics['AUC']:.6f} EER={metrics['EER']:.6f}")
        else:
            print(f"epoch={epoch + 1} VER@0.1%={metrics['VER@0.1%']:.6f} AUC={metrics['AUC']:.6f} EER={metrics['EER']:.6f}")

    print("BEST")
    print(json.dumps(best_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
