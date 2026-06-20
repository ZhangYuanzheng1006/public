# bHW Scaling Exponents Python Runner

This is a CPU-only optimized Python port for `compute_scaling_exponents.m`.

## Local quick validation

```bash
python compute_scaling_exponents_py.py --mode quick --workers 1 --fftw-threads 1 --out-dir results/scaling_exponents_py_quick
```

## Gustation production command

```bash
WORKERS=4 FFTW_THREADS=14 bash run_magnus_scaling_exponents.sh
```

The production run executes eight cases: four alpha values and four kappa values. Outputs are CSV, JSON and PNG files under `OUT_DIR`; on Magnus use `/data/magnus/0620_bhw_scaling_exponents` because the container user can write there.
