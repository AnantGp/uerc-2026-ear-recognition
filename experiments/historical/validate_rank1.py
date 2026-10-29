import argparse
import json
from pathlib import Path

import torch

from rank1_lib import (
    CANDIDATES,
    configure_torch,
    create_candidate_model,
    evaluate_model_on_pairs,
    gate_status,
    load_candidate_checkpoint,
    load_submission_model,
    model_stats,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Full-pair validation for UERC Track 1 submissions/candidates.")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/anant/uerc2026/data_raw/data/public"))
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="mobilenetv4s050")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--postprocess", choices=["raw", "center", "train_center", "train_whiten"], default="raw")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch()
    if args.pairs_file is None:
        args.pairs_file = args.data_dir / "pairs_val.csv"
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.submission_dir:
        model, load_info = load_submission_model(args.submission_dir, device)
        weight_file = args.submission_dir / "ConvNeXt_base.pt"
        if not weight_file.exists():
            weight_file = args.submission_dir / "submitted_model_weights.pt"
        source = str(args.submission_dir)
    elif args.checkpoint:
        model = create_candidate_model(args.candidate, pretrained=False, embedding_dim=args.embedding_dim).to(device)
        checkpoint, missing, unexpected = load_candidate_checkpoint(model, args.checkpoint, device)
        model.eval()
        load_info = {"missing": list(missing), "unexpected": list(unexpected)}
        weight_file = args.checkpoint
        source = str(args.checkpoint)
    else:
        raise SystemExit("Use --submission-dir or --checkpoint.")

    metrics, info = evaluate_model_on_pairs(
        model,
        args.data_dir,
        args.pairs_file,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        max_pairs=args.max_pairs,
        tta=args.tta,
        postprocess=args.postprocess,
    )
    result = {
        "source": source,
        "device": str(device),
        "metrics": metrics,
        "info": info,
        "load_info": load_info,
        **model_stats(model, weight_file),
    }
    if args.checkpoint:
        result["weight_file_size"] = result["model_size"]
    checks, gate_ok = gate_status(result)
    result["gate_ok"] = gate_ok
    result["gate_checks"] = checks

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        save_json(args.out, result)


if __name__ == "__main__":
    main()
