#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

WORKDIR="${WORKDIR:-/tmp/0620-bhw-python}"
OUT_DIR="${OUT_DIR:-/magnus/workspace/0620_bhw_scaling_exponents}"
REPO_URL="${REPO_URL:-https://github.com/ZhangYuanzheng1006/public.git}"
SUBDIR="${SUBDIR:-0620-bhw-python}"

rm -rf "$WORKDIR"
git clone --depth 1 "$REPO_URL" "$WORKDIR"
cd "$WORKDIR/$SUBDIR"

export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install --user "magnus-sdk>=0.6.0" "numpy>=1.24" "matplotlib>=3.7"
python3 -m pip install --user "pyfftw>=0.13" || echo "pyfftw install failed; falling back to numpy.fft"
python3 compute_scaling_exponents_py.py \
  --mode production \
  --workers "${WORKERS:-4}" \
  --fftw-threads "${FFTW_THREADS:-14}" \
  --out-dir "$OUT_DIR"

SECRET=$(magnus custody "$OUT_DIR" --expire-minutes 1440 --max-downloads 10 | grep -o 'magnus-secret:[^ ]*' | tail -n 1)
echo "magnus receive $SECRET --output ./0620_bhw_scaling_exponents" > "$MAGNUS_ACTION"
echo "{\"status\":\"success\",\"output_secret\":\"$SECRET\",\"download_command\":\"magnus receive $SECRET --output ./0620_bhw_scaling_exponents\"}" > "$MAGNUS_RESULT"
