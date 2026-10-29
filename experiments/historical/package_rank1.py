import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from rank1_lib import (
    CANDIDATES,
    candidate_spec_dict,
    create_candidate_model,
    embed_images,
    gate_status,
    load_candidate_checkpoint,
    model_stats,
    postprocess_features,
    safe_replace_submission_dir,
    save_json,
    split_image_names,
    zip_submission,
)


MODEL_TEMPLATE = r'''import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


BACKBONE_NAME = "{backbone}"
EMBEDDING_DIM = {embedding_dim}


class EmbeddingBackbone(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            probe = self.backbone(torch.zeros(2, 3, 224, 224))
            if probe.ndim > 2:
                probe = probe.flatten(1)
            in_features = int(probe.shape[1])
        self.backbone.train(was_training)
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        if x.ndim > 2:
            x = x.flatten(1)
        x = self.neck(x)
        return F.normalize(x, p=2, dim=1)


class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EmbeddingBackbone()

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True):
        clean = {{}}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
            clean[key] = value
        return self.model.load_state_dict(clean, strict=strict)
'''


SOLUTION_TEMPLATE = r'''import gc
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

sys.path.append("../../")
from uerc26_dataset import UERCPairwiseDataset
from uerc26_utils import compute_metrics


USE_TTA = __USE_TTA__


class Solution:
    def __init__(self, name, test_data_path, split, image_list_csv="image_list.csv", dataset_split_csv="dataset_split.csv", pairs_csv="pairs_test.csv", batch_size=1, device="cpu", preload_images=False):
        self.name = name
        self.test_data_path = test_data_path
        self.split = split
        self.batch_size = batch_size
        self.device = device
        dataset = UERCPairwiseDataset(split, test_data_path, image_list_csv, dataset_split_csv, pairs_csv=pairs_csv, preload_images=preload_images)
        self.test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model = None

    def load_model(self, model, model_parameters_file_path):
        self.model = model
        state = torch.load(model_parameters_file_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def get_model_stats(self):
        if self.model is None:
            return {"num_parameters": 0, "model_size": 0}
        return {
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_size": sum(p.numel() * p.element_size() for p in self.model.parameters()),
        }

    def __reset_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def warmup(self, runs=10, evaluations_per_run=0):
        if self.model is None:
            raise ValueError("Model not loaded.")
        self.__reset_cache()
        self.model.eval()
        if evaluations_per_run == 0:
            evaluations_per_run = len(self.test_dataloader)
        with torch.no_grad():
            for _ in tqdm(range(runs), desc="Warmup", unit="run"):
                for batch_idx, ((img1, img2), _) in enumerate(self.test_dataloader):
                    if batch_idx >= evaluations_per_run:
                        break
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    _ = self._embed(img1)
                    _ = self._embed(img2)

    def _embed(self, images):
        feats = self.model(images)
        if USE_TTA:
            feats = feats + self.model(torch.flip(images, dims=[3]))
        return F.normalize(feats, p=2, dim=1)

    def evaluate(self, runs=1):
        if self.model is None:
            raise ValueError("Model not loaded.")
        self.__reset_cache()
        self.model.eval()
        run_data = {}
        with torch.no_grad():
            for i in tqdm(range(runs), desc="Evaluating", unit="run"):
                scores, labels, times = [], [], []
                for (img1, img2), label in tqdm(self.test_dataloader, desc="Processing batches", unit="batch"):
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    start = time.time()
                    e1 = self._embed(img1)
                    e2 = self._embed(img2)
                    end = time.time()
                    times.append(end - start)
                    scores.extend(F.cosine_similarity(e1, e2).cpu().numpy())
                    labels.extend(label.cpu().numpy())
                run_data[str(i)] = compute_metrics(np.array(labels), np.array(scores), times=np.array(times))
        return run_data
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Package a validated compact UERC Track 1 candidate.")
    parser.add_argument("--data-raw", type=Path, default=Path("/home/anant/uerc2026/data_raw"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="mobilenetv4s050")
    parser.add_argument("--name", default="vv_mnv4s050_arcface_t1")
    parser.add_argument("--track", choices=["T1", "T2"], default="T1")
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extract-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--postprocess", choices=["raw", "center", "train_center", "train_whiten"], default="raw")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--solution-tta", action="store_true")
    parser.add_argument("--force-gamble", action="store_true")
    return parser.parse_args()


def load_metrics(path: Path | None):
    if path is None or not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    merged = dict(metrics)
    for key in ("num_parameters", "model_size", "weight_file_size"):
        if key in data:
            merged[key] = data[key]
    return merged


def write_submission_files(submission_dir: Path, name: str, candidate: str, embedding_dim: int, track: str, solution_tta: bool):
    spec = CANDIDATES[candidate]
    (submission_dir / "config.ini").write_text(
        "\n".join(
            [
                "[SUBMISSION]",
                f"track = {track}",
                f"name = {name}",
                "",
                "[MODEL]",
                f"model_class = {spec.class_name}",
                "weights = submitted_model_weights.pt",
                "",
                "[DATA]",
                "data_path = ../../data/sequestered_anonymized",
                "data_split = test",
                "",
            ]
        )
    )
    (submission_dir / "model.py").write_text(
        MODEL_TEMPLATE.format(
            backbone=spec.backbone,
            embedding_dim=embedding_dim,
            class_name=spec.class_name,
        )
    )
    (submission_dir / "solution.py").write_text(
        SOLUTION_TEMPLATE.replace("__USE_TTA__", "True" if solution_tta else "False")
    )


def image_list(data_dir: Path):
    names = []
    with (data_dir / "image_list.csv").open("r") as f:
        for line in f:
            name = line.strip().split("\t")[0]
            if name:
                names.append(name)
    return names


def main():
    args = parse_args()
    if args.name in {"final_baseline_t1", "final_baseline_t1_centered"}:
        raise SystemExit("Refusing to package over protected baseline names.")

    device = torch.device("cpu")
    if args.extract_device == "cuda":
        device = torch.device("cuda")
    elif args.extract_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_candidate_model(args.candidate, pretrained=False, embedding_dim=args.embedding_dim).to(device)
    checkpoint, missing, unexpected = load_candidate_checkpoint(model, args.checkpoint, device)
    stats = model_stats(model)
    stats["weight_file_size"] = stats["model_size"]
    metrics = load_metrics(args.metrics_json)
    gate_input = {**metrics, **stats}
    checks, gate_ok = gate_status(gate_input)
    if not gate_ok and not args.force_gamble:
        print(json.dumps({"gate_ok": gate_ok, "gate_checks": checks, "metrics": metrics, "stats": stats}, indent=2, sort_keys=True))
        raise SystemExit("Candidate did not pass rank-1 gate. Use --force-gamble only for a clearly labeled gamble package.")

    submissions = args.data_raw / "SUBMISSIONS"
    submission_dir = submissions / args.name
    safe_replace_submission_dir(submission_dir)
    write_submission_files(submission_dir, args.name, args.candidate, args.embedding_dim, args.track, args.solution_tta)

    state = checkpoint.get("model", checkpoint)
    clean = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        clean[key] = value.cpu()
    torch.save(clean, submission_dir / "submitted_model_weights.pt")

    seq_dir = args.data_raw / "data" / "sequestered_anonymized"
    names = image_list(seq_dir)
    ordered, features = embed_images(
        model,
        seq_dir,
        names,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        image_size=224,
        tta=args.tta,
    )
    reference_features = None
    if args.postprocess in {"train_center", "train_whiten"}:
        public_dir = args.data_raw / "data" / "public"
        reference_names = split_image_names(public_dir, "train")
        _, reference_features = embed_images(
            model,
            public_dir,
            reference_names,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            image_size=224,
            tta=args.tta,
        )
    features, postprocess_info = postprocess_features(
        features,
        mode=args.postprocess,
        reference_features=reference_features,
    )
    feature_dict = {name: features[i].astype("float32") for i, name in enumerate(ordered)}
    np.save(submission_dir / "sequestered_features.npy", feature_dict)

    first_key = next(iter(feature_dict))
    feature_check = {
        "exists": (submission_dir / "sequestered_features.npy").exists(),
        "dict_entries": len(feature_dict),
        "first_key": first_key,
        "first_vec_shape": list(feature_dict[first_key].shape),
        "first_vec_finite": bool(np.isfinite(feature_dict[first_key]).all()),
    }
    if feature_check["dict_entries"] != 1670 or not feature_check["first_vec_finite"]:
        raise SystemExit(f"Feature check failed: {feature_check}")

    package_info = {
        "candidate": candidate_spec_dict(args.candidate),
        "name": args.name,
        "track": args.track,
        "checkpoint": str(args.checkpoint),
        "load_missing": list(missing),
        "load_unexpected": list(unexpected),
        "metrics": metrics,
        "stats": model_stats(model, submission_dir / "submitted_model_weights.pt"),
        "gate_ok": gate_ok,
        "gate_checks": checks,
        "postprocess": postprocess_info,
        "tta": bool(args.tta),
        "solution_tta": bool(args.solution_tta),
        "feature_check": feature_check,
    }
    report_path = submissions / f"{args.name}_package_report.json"
    save_json(report_path, package_info)

    zip_info = zip_submission(submission_dir)
    package_info["zip"] = zip_info
    package_info["report_path"] = str(report_path)
    save_json(report_path, package_info)
    print(json.dumps(package_info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
