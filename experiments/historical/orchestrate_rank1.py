import json
import os
import argparse
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/anant/uerc2026/rank1_sprint")
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
        "candidate": "mobilenetv4s050",
        "work_dir": ROOT / "runs/mnv4s050",
        "name": "vv_mnv4s050_arcface_t1",
        "epochs": 24,
    },
    {
        "candidate": "edgenext_xxs",
        "work_dir": ROOT / "runs/edgenext_xxs",
        "name": "vv_edgenext_xxs_arcface_t1",
        "epochs": 24,
    },
    {
        "candidate": "convnext_zepto",
        "work_dir": ROOT / "runs/convnext_zepto",
        "name": "vv_convnext_zepto_arcface_t1_gamble",
        "epochs": 18,
        "accuracy_gamble": True,
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


def run_command(cmd, log_file):
    log(f"running: {' '.join(str(c) for c in cmd)}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        (log_file.parent / "train.pid").write_text(str(proc.pid))
        rc = proc.wait()
    log(f"finished rc={rc}: {' '.join(str(c) for c in cmd[:3])}")
    return rc


def gate(metrics):
    metrics = flatten_metrics(metrics)
    return (
        metrics.get("VER@0.1%", 0.0) >= 0.50
        and metrics.get("AUC", 0.0) >= 0.965
        and metrics.get("EER", 1.0) <= 0.105
        and metrics.get("num_parameters", 10**12) <= 1_800_000
        and metrics.get("model_size", 10**12) <= 8_000_000
    )


def close_accuracy(metrics):
    metrics = flatten_metrics(metrics)
    return (
        metrics.get("VER@0.1%", 0.0) >= 0.42
        and metrics.get("AUC", 0.0) >= 0.955
        and metrics.get("EER", 1.0) <= 0.12
    )


def flatten_metrics(data):
    if not data:
        return {}
    if "metrics" in data:
        merged = dict(data["metrics"])
        for key in ("num_parameters", "model_size", "weight_file_size"):
            if key in data:
                merged[key] = data[key]
        return merged
    return data


def load_best(work_dir):
    path = work_dir / "best_metrics.json"
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


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
        "384",
        "--val-batch-size",
        "512",
        "--workers",
        "8",
    ]
    return run_command(cmd, work_dir / "train.log")


def validate_variant(run, postprocess="raw", tta=False):
    suffix = postprocess + ("_tta" if tta else "")
    out = run["work_dir"] / f"validate_{suffix}.json"
    cmd = [
        PY,
        "validate_rank1.py",
        "--checkpoint",
        str(run["work_dir"] / "best.pt"),
        "--candidate",
        run["candidate"],
        "--batch-size",
        "512",
        "--workers",
        "8",
        "--postprocess",
        postprocess,
        "--out",
        str(out),
    ]
    if tta:
        cmd.append("--tta")
    rc = run_command(cmd, run["work_dir"] / f"validate_{suffix}.log")
    if rc != 0 or not out.exists():
        return None, out
    with out.open("r") as f:
        return json.load(f), out


def sweep_postprocess(run, metrics):
    candidates = [(metrics, run["work_dir"] / "best_metrics.json", "raw", False)]
    flat = flatten_metrics(metrics)
    if flat.get("VER@0.1%", 0.0) < 0.36 and not run.get("accuracy_gamble", False):
        return candidates[0]

    for postprocess, tta in [("center", False), ("raw", True), ("center", True)]:
        result, path = validate_variant(run, postprocess=postprocess, tta=tta)
        if result:
            candidates.append((result, path, postprocess, tta))

    best = max(candidates, key=lambda item: flatten_metrics(item[0]).get("VER@0.1%", 0.0))
    best_flat = flatten_metrics(best[0])
    if best_flat.get("AUC", 0.0) >= 0.94:
        for postprocess in ["train_center", "train_whiten"]:
            result, path = validate_variant(run, postprocess=postprocess, tta=False)
            if result:
                candidates.append((result, path, postprocess, False))
        best = max(candidates, key=lambda item: flatten_metrics(item[0]).get("VER@0.1%", 0.0))
    best_flat = flatten_metrics(best[0])
    log(
        f"best variant {run['candidate']} postprocess={best[2]} tta={best[3]} "
        f"VER@0.1={best_flat.get('VER@0.1%')} AUC={best_flat.get('AUC')} EER={best_flat.get('EER')}"
    )
    return best


def package(run, metrics_path, postprocess="raw", tta=False, force=False):
    name = run["name"]
    if force and not name.endswith("_gamble"):
        name = f"{name}_gamble"
    cmd = [
        PY,
        "package_rank1.py",
        "--checkpoint",
        str(run["work_dir"] / "best.pt"),
        "--candidate",
        run["candidate"],
        "--name",
        name,
        "--metrics-json",
        str(metrics_path),
        "--postprocess",
        postprocess,
    ]
    if tta:
        cmd.append("--tta")
    if force:
        cmd.append("--force-gamble")
    return run_command(cmd, run["work_dir"] / "package.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", choices=[run["candidate"] for run in RUNS], default=RUNS[0]["candidate"])
    args = parser.parse_args()
    started = False
    log("rank1 orchestrator started")
    for run in RUNS:
        if run["candidate"] == args.start_at:
            started = True
        if not started:
            continue
        wait_existing(run["work_dir"])
        rc = train_if_needed(run)
        if rc != 0:
            log(f"training failed for {run['candidate']}; moving to next candidate")
            continue
        metrics = load_best(run["work_dir"])
        if not metrics:
            log(f"no best metrics for {run['candidate']}; moving on")
            continue
        log(f"best {run['candidate']}: VER@0.1={metrics.get('VER@0.1%')} AUC={metrics.get('AUC')} EER={metrics.get('EER')}")
        best_variant, metrics_path, postprocess, tta = sweep_postprocess(run, metrics)
        if gate(best_variant):
            log(f"gate passed for {run['candidate']}; packaging final candidate")
            package(run, metrics_path, postprocess=postprocess, tta=tta, force=False)
            break
        if close_accuracy(best_variant) or run.get("accuracy_gamble", False) and close_accuracy(best_variant):
            log(f"candidate is close on accuracy but not full gate; packaging gamble")
            package(run, metrics_path, postprocess=postprocess, tta=tta, force=True)
        else:
            log(f"gate failed for {run['candidate']}; continuing")
    log("rank1 orchestrator finished")


if __name__ == "__main__":
    main()
