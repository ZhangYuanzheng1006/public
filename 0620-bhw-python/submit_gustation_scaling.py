from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests


TASK_NAME = "0620 bHW scaling exponents Python CPU direct"
DEFAULT_IMAGE = "docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_OUT_DIR = "/home/magnus/data/0620_bhw_scaling_exponents"
DEFAULT_REPO_ZIP = "https://github.com/ZhangYuanzheng1006/public/archive/refs/heads/main.zip"


def load_secret(path: Path, site: str) -> tuple[str, str]:
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    site = site.lower()
    address = data.get(f"magnus_address-{site}") or data.get("magnus_address")
    token = data.get(f"magnus_token-{site}") or data.get("magnus_token")
    if not isinstance(address, str) or not address.strip():
        raise SystemExit(f"Missing Magnus address for site={site} in {path}")
    if not isinstance(token, str) or not token.strip():
        raise SystemExit(f"Missing Magnus token for site={site} in {path}")
    return address.rstrip("/"), token.strip()


def api_url(address: str, path: str) -> str:
    if address.endswith("/api"):
        address = address[:-4]
    return f"{address}/api{path}"


def request_json(method: str, address: str, token: str, path: str, *, verify_ssl: bool, **kwargs: Any) -> Any:
    response = requests.request(
        method,
        api_url(address, path),
        headers={"Authorization": f"Bearer {token}"},
        timeout=kwargs.pop("timeout", 60),
        verify=verify_ssl,
        **kwargs,
    )
    if response.status_code >= 400:
        detail = response.text[:1000]
        raise SystemExit(f"Magnus API error {response.status_code} on {path}: {detail}")
    if not response.text.strip():
        return None
    return response.json()


def active_existing_job(address: str, token: str, task_name: str, verify_ssl: bool) -> dict[str, Any] | None:
    data = request_json(
        "GET",
        address,
        token,
        "/jobs",
        verify_ssl=verify_ssl,
        params={"limit": 20, "search": task_name},
        timeout=30,
    )
    for job in data.get("items", []):
        if job.get("task_name") != task_name:
            continue
        if job.get("status") in {"Preparing", "Pending", "Queued", "Running"}:
            return job
    return None


def build_entry_command(repo_zip: str, out_dir: str, workers: int, fftw_threads: int) -> str:
    return f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export WORKERS={workers}
export FFTW_THREADS={fftw_threads}
export OUT_DIR={out_dir}
mkdir -p "$OUT_DIR"
if [ ! -w "$OUT_DIR" ]; then
  echo "Output directory is not writable by the Magnus container user: $OUT_DIR" >&2
  exit 3
fi
echo "Starting bHW scaling exponent run at $(date -Is)" | tee "$OUT_DIR/run_status.log"
python3 - <<'PY'
import pathlib
import shutil
import urllib.request
import zipfile

url = {repo_zip!r}
zip_path = pathlib.Path('/tmp/public-main.zip')
extract_root = pathlib.Path('/tmp')
shutil.rmtree('/tmp/public-main', ignore_errors=True)
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(extract_root)
PY
cd /tmp/public-main/0620-bhw-python
python3 -m pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple 'numpy>=1.24' 'matplotlib>=3.7'
python3 -m pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple 'pyfftw>=0.13' || echo 'pyfftw install failed; falling back to numpy.fft' | tee -a "$OUT_DIR/run_status.log"
python3 compute_scaling_exponents_py.py --mode production --workers "$WORKERS" --fftw-threads "$FFTW_THREADS" --out-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/run_status.log"
echo "Finished bHW scaling exponent run at $(date -Is)" | tee -a "$OUT_DIR/run_status.log"
echo '{{"status":"success","out_dir":"{out_dir}","task":"compute_scaling_exponents"}}' > "$MAGNUS_RESULT"
""".strip()


def build_system_entry_command() -> str:
    return """
mounts=("/home/magnus/data:/home/magnus/data")
export APPTAINER_BIND=$(IFS=,; echo "${mounts[*]}")
export MAGNUS_HOME=/magnus
mkdir -p /home/magnus/data 2>/dev/null || true
unset -f nvidia-smi || true
unset VIRTUAL_ENV SSL_CERT_FILE
""".strip()


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_name": args.task_name,
        "entry_command": build_entry_command(args.repo_zip, args.out_dir, args.workers, args.fftw_threads),
        "repo_name": "magnus",
        "branch": "main",
        "commit_sha": "9019705e964d728c461b8e8e8a771d5a53ea8c62",
        "gpu_type": "cpu",
        "gpu_count": 0,
        "namespace": "Rise-AGI",
        "job_type": args.job_type,
        "description": (
            "Direct requests POST to Gustation /api/jobs/submit. CPU-only optimized Python run "
            "for compute_scaling_exponents.m from GitHub zip. Outputs persist under " + args.out_dir
        ),
        "container_image": args.container_image,
        "cpu_count": args.cpu_count,
        "memory_demand": args.memory_demand,
        "ephemeral_storage": args.ephemeral_storage,
        "runner": "magnus",
        "system_entry_command": build_system_entry_command(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit bHW scaling-exponent job to Gustation via requests.")
    parser.add_argument("--secret-json", type=Path, default=Path(__file__).resolve().parents[1] / "secret.json")
    parser.add_argument("--site", default="gu")
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--repo-zip", default=DEFAULT_REPO_ZIP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cpu-count", type=int, default=48)
    parser.add_argument("--memory-demand", default="200G")
    parser.add_argument("--ephemeral-storage", default="80G")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fftw-threads", type=int, default=12)
    parser.add_argument("--job-type", default="B2")
    parser.add_argument("--verify-ssl", action="store_true", help="Enable TLS verification. Default is disabled for GU compatibility.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Submit even if an active same-name job exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    address, token = load_secret(args.secret_json, args.site)
    verify_ssl = bool(args.verify_ssl)
    if not verify_ssl:
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    payload = build_payload(args)
    print(f"Target: {address}")
    print(f"Endpoint: {api_url(address, '/jobs/submit')}")
    print(
        "Payload summary: "
        f"task={payload['task_name']!r}, job_type={payload['job_type']}, "
        f"gpu={payload['gpu_count']}, cpu={payload['cpu_count']}, mem={payload['memory_demand']}, "
        f"image={payload['container_image']}, out_dir={args.out_dir}"
    )

    if args.dry_run:
        print(json.dumps({k: v for k, v in payload.items() if k != "entry_command"}, indent=2))
        return 0

    if not args.force:
        existing = active_existing_job(address, token, args.task_name, verify_ssl)
        if existing:
            print(f"Active same-name job exists: {existing.get('id')} status={existing.get('status')}")
            return 0

    job = request_json("POST", address, token, "/jobs/submit", verify_ssl=verify_ssl, json=payload, timeout=180)
    print(f"Submitted job: {job['id']}")
    print(f"View: {address}/jobs/{job['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
