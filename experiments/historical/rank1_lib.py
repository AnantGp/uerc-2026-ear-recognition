import configparser
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    backbone: str
    class_name: str
    embedding_dim: int = 256
    image_size: int = 224


CANDIDATES = {
    "mobilenetv4s050": CandidateSpec(
        key="mobilenetv4s050",
        backbone="mobilenetv4_conv_small_050.e3000_r224_in1k",
        class_name="VVMobileNetV4Small050ArcFaceT1",
    ),
    "edgenext_xxs": CandidateSpec(
        key="edgenext_xxs",
        backbone="edgenext_xx_small.in1k",
        class_name="VVEdgeNeXtXXSmallArcFaceT1",
    ),
    "convnext_zepto": CandidateSpec(
        key="convnext_zepto",
        backbone="convnext_zepto_rms.ra4_e3600_r224_in1k",
        class_name="VVConvNeXtZeptoArcFaceT1",
    ),
    "convnext_nano_in12k": CandidateSpec(
        key="convnext_nano_in12k",
        backbone="convnext_nano.in12k_ft_in1k",
        class_name="VVConvNeXtNanoIn12kArcFaceT2",
    ),
    "convnext_tiny_in12k": CandidateSpec(
        key="convnext_tiny_in12k",
        backbone="convnext_tiny.in12k_ft_in1k",
        class_name="VVConvNeXtTinyIn12kArcFaceT2",
    ),
    "convnext_tiny_dinov3": CandidateSpec(
        key="convnext_tiny_dinov3",
        backbone="convnext_tiny.dinov3_lvd1689m",
        class_name="VVConvNeXtTinyDinoV3ArcFaceT2",
    ),
    "convnext_small_in12k": CandidateSpec(
        key="convnext_small_in12k",
        backbone="convnext_small.in12k_ft_in1k",
        class_name="VVConvNeXtSmallIn12kArcFaceT2",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class EmbeddingBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int = 256,
        pretrained: bool = False,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            probe = torch.zeros(2, 3, 224, 224)
            probe_out = self.backbone(probe)
            if probe_out.ndim > 2:
                probe_out = probe_out.flatten(1)
            in_features = int(probe_out.shape[1])
        self.backbone.train(was_training)

        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        nn.init.normal_(self.neck[0].weight, std=0.01)
        nn.init.ones_(self.neck[1].weight)
        nn.init.zeros_(self.neck[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if x.ndim > 2:
            x = x.flatten(1)
        x = self.neck(x)
        return F.normalize(x, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s


def create_candidate_model(candidate_key: str, pretrained: bool, embedding_dim: int = 256) -> EmbeddingBackbone:
    spec = CANDIDATES[candidate_key]
    return EmbeddingBackbone(spec.backbone, embedding_dim=embedding_dim, pretrained=pretrained)


def train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.82, 1.18),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03)],
                p=0.35,
            ),
            transforms.RandomGrayscale(p=0.04),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.04),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
        ]
    )


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def read_split(split_file: Path):
    rows = []
    with split_file.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0], row[1]))
    return rows


class UERCTrainDataset(Dataset):
    def __init__(self, data_dir: Path, split_file: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        train_rows = [(img, split) for img, split in read_split(split_file) if split == "train"]
        subjects = sorted({img.split("/")[0] for img, _ in train_rows})
        self.subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
        self.samples = []
        self.class_counts = np.zeros(len(subjects), dtype=np.int64)
        for img, _ in train_rows:
            path = self.data_dir / img
            if path.exists():
                label = self.subject_to_idx[img.split("/")[0]]
                self.samples.append((img, label))
                self.class_counts[label] += 1
        self.num_classes = len(subjects)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, label = self.samples[idx]
        image = Image.open(self.data_dir / img_rel).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def weighted_sampler(self) -> WeightedRandomSampler:
        class_weights = 1.0 / np.maximum(self.class_counts, 1)
        sample_weights = [float(class_weights[label]) for _, label in self.samples]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


class ImagePathDataset(Dataset):
    def __init__(self, data_dir: Path, image_names, transform=None):
        self.data_dir = Path(data_dir)
        self.image_names = list(image_names)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(self.data_dir / image_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name


def read_pairs(pairs_file: Path, max_pairs: int = 0):
    pairs = []
    with Path(pairs_file).open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            try:
                label = int(row[2])
            except ValueError:
                continue
            pairs.append((row[0], row[1], label))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def split_image_names(data_dir: Path, split: str = "train"):
    split_file = Path(data_dir) / "dataset_split.csv"
    names = []
    for image_name, row_split in read_split(split_file):
        if row_split == split and (Path(data_dir) / image_name).exists():
            names.append(image_name)
    return names


def normalize_features(features: np.ndarray) -> np.ndarray:
    features = features.astype("float32", copy=False)
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)


def postprocess_features(
    features: np.ndarray,
    mode: str = "raw",
    reference_features: np.ndarray | None = None,
):
    features = normalize_features(features)
    details = {"postprocess": mode}
    if mode == "raw":
        return features, details
    if mode == "center":
        mean = features.mean(axis=0, keepdims=True)
        details["mean_source"] = "evaluation_images"
        return normalize_features(features - mean), details
    if mode in {"train_center", "train_whiten"}:
        if reference_features is None:
            raise ValueError(f"{mode} requires reference_features")
        reference_features = normalize_features(reference_features)
        mean = reference_features.mean(axis=0, keepdims=True)
        centered = features - mean
        details["mean_source"] = "reference_images"
        details["reference_count"] = int(reference_features.shape[0])
        if mode == "train_center":
            return normalize_features(centered), details
        ref_centered = reference_features - mean
        cov = np.cov(ref_centered, rowvar=False).astype("float32")
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-5)
        whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        details["min_eig"] = float(eigvals.min())
        details["max_eig"] = float(eigvals.max())
        return normalize_features(centered @ whitening.astype("float32")), details
    raise ValueError(f"Unknown postprocess mode: {mode}")


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_candidate_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint_state_dict(ckpt)
    clean = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        clean[key] = value
    missing, unexpected = model.load_state_dict(clean, strict=False)
    return ckpt, missing, unexpected


def load_submission_model(submission_dir: Path, device: torch.device):
    submission_dir = Path(submission_dir)
    config = configparser.ConfigParser()
    config.read(submission_dir / "config.ini")
    model_class_name = config.get("MODEL", "model_class")
    weights_name = config.get("MODEL", "weights")

    module_name = f"uerc_model_{submission_dir.name}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, submission_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model = getattr(module, model_class_name)()
    state = torch.load(submission_dir / weights_name, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    load_target = "wrapper"
    if len(missing) > 0 and len(unexpected) > 0 and hasattr(model, "model"):
        inner_missing, inner_unexpected = model.model.load_state_dict(state, strict=False)
        if len(inner_missing) + len(inner_unexpected) < len(missing) + len(unexpected):
            model = model.model
            missing, unexpected = inner_missing, inner_unexpected
            load_target = "inner_model"
    model.to(device)
    model.eval()
    return model, {"missing": list(missing), "unexpected": list(unexpected), "load_target": load_target}


@torch.no_grad()
def embed_images(
    model: nn.Module,
    data_dir: Path,
    image_names,
    batch_size: int,
    workers: int,
    device: torch.device,
    image_size: int = 224,
    tta: bool = False,
):
    dataset = ImagePathDataset(data_dir, image_names, transform=eval_transform(image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    features = []
    ordered_names = []
    model.eval()
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        feats = model(images)
        if tta:
            flipped = torch.flip(images, dims=[3])
            feats = F.normalize(feats + model(flipped), p=2, dim=1)
        else:
            feats = F.normalize(feats, p=2, dim=1)
        features.append(feats.float().cpu().numpy())
        ordered_names.extend(names)
    feature_matrix = np.concatenate(features, axis=0).astype("float32")
    feature_matrix /= np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-9
    return ordered_names, feature_matrix


def compute_metrics(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    interp = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
    fnr = 1.0 - tpr
    return {
        "VER@0.1%": float(interp(0.001)),
        "VER@1%": float(interp(0.01)),
        "EER": float(fpr[np.argmin(np.abs(fnr - fpr))]),
        "AUC": float(roc_auc_score(labels, scores)),
    }


def evaluate_pairs_from_features(pairs, names, features, chunk_size: int = 1_000_000):
    index = {name: i for i, name in enumerate(names)}
    n = len(pairs)
    left = np.empty(n, dtype=np.int32)
    right = np.empty(n, dtype=np.int32)
    labels = np.empty(n, dtype=np.int8)
    missing = 0
    for i, (img1, img2, label) in enumerate(pairs):
        try:
            left[i] = index[img1]
            right[i] = index[img2]
        except KeyError:
            left[i] = -1
            right[i] = -1
            missing += 1
        labels[i] = label
    if missing:
        keep = (left >= 0) & (right >= 0)
        left = left[keep]
        right = right[keep]
        labels = labels[keep]

    scores = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), chunk_size):
        end = min(start + chunk_size, len(left))
        scores[start:end] = np.sum(features[left[start:end]] * features[right[start:end]], axis=1)
    return compute_metrics(labels, scores), {"num_pairs": int(len(labels)), "missing_pairs": int(missing)}


def evaluate_model_on_pairs(
    model: nn.Module,
    data_dir: Path,
    pairs_file: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    image_size: int = 224,
    max_pairs: int = 0,
    tta: bool = False,
    postprocess: str = "raw",
    reference_image_names=None,
):
    pairs = read_pairs(pairs_file, max_pairs=max_pairs)
    unique = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    names, features = embed_images(model, data_dir, unique, batch_size, workers, device, image_size=image_size, tta=tta)
    reference_features = None
    if postprocess in {"train_center", "train_whiten"}:
        if reference_image_names is None:
            reference_image_names = split_image_names(data_dir, "train")
        _, reference_features = embed_images(
            model,
            data_dir,
            reference_image_names,
            batch_size,
            workers,
            device,
            image_size=image_size,
            tta=tta,
        )
    features, postprocess_info = postprocess_features(features, mode=postprocess, reference_features=reference_features)
    metrics, info = evaluate_pairs_from_features(pairs, names, features)
    info["num_images"] = len(names)
    info["embedding_dim"] = int(features.shape[1])
    info.update(postprocess_info)
    return metrics, info


def model_stats(model: nn.Module, weight_file: Path | None = None):
    params = sum(p.numel() for p in model.parameters())
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    stats = {"num_parameters": int(params), "model_size": int(size)}
    if weight_file is not None and Path(weight_file).exists():
        stats["weight_file_size"] = int(Path(weight_file).stat().st_size)
    return stats


def gate_status(metrics):
    checks = {
        "VER@0.1% >= 0.50": metrics.get("VER@0.1%", 0.0) >= 0.50,
        "AUC >= 0.965": metrics.get("AUC", 0.0) >= 0.965,
        "EER <= 0.105": metrics.get("EER", 1.0) <= 0.105,
        "num_parameters <= 1.8M": metrics.get("num_parameters", 10**12) <= 1_800_000,
        "weight_file_size <= 8MB": metrics.get("weight_file_size", 10**12) <= 8_000_000,
    }
    return checks, all(checks.values())


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_replace_submission_dir(target: Path):
    target = Path(target)
    if target.name in {"final_baseline_t1", "final_baseline_t1_centered"}:
        raise ValueError(f"Refusing to overwrite protected submission: {target}")
    if not target.name.startswith("vv_"):
        raise ValueError(f"Refusing to replace non-rank1 candidate directory: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)


def zip_submission(submission_dir: Path):
    submission_dir = Path(submission_dir)
    zip_path = submission_dir.parent / f"{submission_dir.name}.zip"
    required_files = [
        "config.ini",
        "model.py",
        "solution.py",
        "submitted_model_weights.pt",
        "sequestered_features.npy",
    ]
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in required_files:
            zf.write(submission_dir / file_name, arcname=f"{submission_dir.name}/{file_name}")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = set(zf.namelist())
    required = {
        f"{submission_dir.name}/config.ini",
        f"{submission_dir.name}/model.py",
        f"{submission_dir.name}/solution.py",
        f"{submission_dir.name}/submitted_model_weights.pt",
        f"{submission_dir.name}/sequestered_features.npy",
    }
    return {
        "zip_path": str(zip_path),
        "size_mb": zip_path.stat().st_size / 1e6,
        "sha256": sha256_file(zip_path),
        "testzip": bad,
        "required_present": required <= names,
        "missing_required": sorted(required - names),
        "num_files": len(names),
    }


def candidate_spec_dict(candidate_key: str):
    return asdict(CANDIDATES[candidate_key])
