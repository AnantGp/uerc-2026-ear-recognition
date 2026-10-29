import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/anant/uerc2026/rank1_sprint")
SUBMISSIONS = Path("/home/anant/uerc2026/data_raw/SUBMISSIONS")
PY = "python3"
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = (
    "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:"
    "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:"
    "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:"
    "/usr/local/cuda-12.9/targets/x86_64-linux/lib:"
    "/usr/lib/x86_64-linux-gnu:/usr/local/gib/lib64"
)


RUNS = [
    {
        "candidate": "convnext_tiny_in12k",
        "work_dir": ROOT / "runs/convnext_tiny_t2",
        "name": "vv_convnext_tiny_in12k_t2_rawtta",
        "epochs": 24,
        "batch_size": 256,
        "val_batch_size": 384,
    },
    {
        "candidate": "convnext_tiny_dinov3",
        "work_dir": ROOT / "runs/convnext_tiny_dinov3_t2",
        "name": "vv_convnext_tiny_dinov3_t2_rawtta",
        "epochs": 18,
        "batch_size": 256,
        "val_batch_size": 384,
    },
    {
        "candidate": "convnext_small_in12k",
        "work_dir": ROOT / "runs/convnext_small_t2",
        "name": "vv_convnext_small_in12k_t2_rawtta",
        "epochs": 14,
        "batch_size": 128,
        "val_batch_size": 256,
    },
]


def log(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_existing(work_dir):
    pid_file = work_dir / "train.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return
    if not is_alive(pid):
        return
    log(f"waiting for existing training pid={pid} in {work_dir}")
    while is_alive(pid):
        time.sleep(300)


def run_command(cmd, log_file, pid_file=None):
    log(f"running: {' '.join(str(c) for c in cmd)}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        if pid_file:
            pid_file.write_text(str(proc.pid))
        rc = proc.wait()
    log(f"finished rc={rc}: {' '.join(str(c) for c in cmd[:4])}")
    return rc


def load_json(path):
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def flatten(data):
    if not data:
        return {}
    if "metrics" in data:
        merged = dict(data["metrics"])
        for key in ("num_parameters", "model_size", "weight_file_size"):
            if key in data:
                merged[key] = data[key]
        return merged
    return data


def train_if_needed(run):
    work_dir = run["work_dir"]
    if (work_dir / "best_metrics.json").exists():
        log(f"best metrics already exist for {run['candidate']}; not retraining")
        return 0
    wait_existing(work_dir)
    if (work_dir / "best_metrics.json").exists():
        return 0
    cmd = [
        PY,
        "-u",
        "train_rank1.py",
        "--candidate",
        run["candidate"],
        "--work-dir",
        str(work_dir),
        "--epochs",
        str(run["epochs"]),
        "--batch-size",
        str(run["batch_size"]),
        "--val-batch-size",
        str(run["val_batch_size"]),
        "--workers",
        "8",
    ]
    return run_command(cmd, work_dir / "train.log", work_dir / "train.pid")


def validate_raw_tta(run):
    out = run["work_dir"] / "validate_raw_tta.json"
    cmd = [
        PY,
        "validate_rank1.py",
        "--checkpoint",
        str(run["work_dir"] / "best.pt"),
        "--candidate",
        run["candidate"],
        "--batch-size",
        str(run["val_batch_size"]),
        "--workers",
        "8",
        "--postprocess",
        "raw",
        "--out",
        str(out),
        "--tta",
    ]
    rc = run_command(cmd, run["work_dir"] / "validate_raw_tta.log")
    if rc != 0:
        return None
    return load_json(out)


def package_t2(run, metrics_path):
    cmd = [
        PY,
        "package_rank1.py",
        "--track",
        "T2",
        "--checkpoint",
        str(run["work_dir"] / "best.pt"),
        "--candidate",
        run["candidate"],
        "--name",
        f"{run['name']}_gamble",
        "--metrics-json",
        str(metrics_path),
        "--postprocess",
        "raw",
        "--tta",
        "--solution-tta",
        "--extract-device",
        "cuda",
        "--force-gamble",
    ]
    return run_command(cmd, run["work_dir"] / "package_t2_rawtta.log")


def main():
    best_ver = 0.502175304987976
    target_ver = 0.65
    log("Track 2 orchestrator started")
    for run in RUNS:
        rc = train_if_needed(run)
        if rc != 0:
            log(f"training failed for {run['candidate']}; continuing")
            continue
        raw = load_json(run["work_dir"] / "best_metrics.json")
        raw_m = flatten(raw)
        log(f"raw best {run['candidate']}: VER={raw_m.get('VER@0.1%')} AUC={raw_m.get('AUC')} EER={raw_m.get('EER')}")
        tta = validate_raw_tta(run)
        tta_m = flatten(tta)
        log(f"raw+TTA {run['candidate']}: VER={tta_m.get('VER@0.1%')} AUC={tta_m.get('AUC')} EER={tta_m.get('EER')}")
        if tta_m.get("VER@0.1%", 0.0) > best_ver:
            best_ver = tta_m["VER@0.1%"]
            package_t2(run, run["work_dir"] / "validate_raw_tta.json")
        if best_ver >= target_ver:
            log(f"target reached with VER={best_ver}; stopping")
            break
    log(f"Track 2 orchestrator finished; best_ver={best_ver}")
    for path in sorted(SUBMISSIONS.glob("vv_*t2*zip")):
        log(f"zip {path} {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
