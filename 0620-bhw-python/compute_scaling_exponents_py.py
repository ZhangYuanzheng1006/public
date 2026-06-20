from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bhw_solver_py import BHWParams, BHWSolver, write_case_outputs


ALPHA_VALUES = [0.01, 0.1, 0.5, 5.0]
KAPPA_VALUES = [0.2, 0.5, 1.0, 2.0]


def fit_exponent(k: np.ndarray, ek: np.ndarray, k_min: float, k_max: float) -> float:
    idx = (k >= k_min) & (k <= k_max) & (ek > 0.0)
    if int(np.sum(idx)) < 5:
        return float("nan")
    slope, _ = np.polyfit(np.log(k[idx]), np.log(ek[idx]), 1)
    return float(-slope)


def run_case(case: dict) -> dict:
    # Keep BLAS from multiplying process-level parallelism unexpectedly.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    params = BHWParams(**case["params"])
    solver = BHWSolver(params)
    solver.run()
    k, ek = solver.spectrum()
    exponent = fit_exponent(k, ek, case["k_min"], case["k_max"])
    out_prefix = Path(case["out_dir"]) / case["tag"]
    write_case_outputs(out_prefix, solver, k, ek, exponent)
    return {
        "scan": case["scan"],
        "value": case["value"],
        "k": k,
        "Ek": ek,
        "exponent": exponent,
        "tag": case["tag"],
    }


def make_cases(base: BHWParams, out_dir: Path, k_min: float, k_max: float) -> list[dict]:
    cases = []
    for alpha in ALPHA_VALUES:
        params = replace(base, alpha=alpha, kappa=0.5)
        cases.append(
            {
                "scan": "alpha",
                "value": alpha,
                "tag": f"alpha_{alpha:.4f}",
                "params": params.__dict__,
                "out_dir": str(out_dir),
                "k_min": k_min,
                "k_max": k_max,
            }
        )
    for kappa in KAPPA_VALUES:
        params = replace(base, alpha=0.5, kappa=kappa)
        cases.append(
            {
                "scan": "kappa",
                "value": kappa,
                "tag": f"kappa_{kappa:.2f}",
                "params": params.__dict__,
                "out_dir": str(out_dir),
                "k_min": k_min,
                "k_max": k_max,
            }
        )
    return cases


def plot_spectra(results: list[dict], scan: str, out_dir: Path, k_min: float, k_max: float) -> None:
    rows = sorted([r for r in results if r["scan"] == scan], key=lambda x: x["value"])
    fig, ax = plt.subplots(figsize=(9, 7))
    param_label = r"\alpha" if scan == "alpha" else r"\kappa"
    for row in rows:
        ax.loglog(row["k"], row["Ek"], lw=1.5, label=fr"${param_label}={row['value']:.3g}$")
    k_ref = np.logspace(np.log10(0.3), np.log10(3.0), 20)
    ax.loglog(k_ref, 1e-3 * k_ref ** -3, "k--", lw=1.2, label=r"$k^{-3}$")
    ax.axvspan(k_min, k_max, color="0.85", alpha=0.35, label="fit range")
    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel(r"$E(k)$")
    if scan == "alpha":
        ax.set_title(r"Energy spectra at different $\alpha$ ($\kappa=0.5$)")
    else:
        ax.set_title(r"Energy spectra at different $\kappa$ ($\alpha=0.5$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"spectra_{scan}_comparison.png", dpi=300)
    plt.close(fig)


def plot_exponents(results: list[dict], out_dir: Path, k_min: float, k_max: float) -> None:
    alpha_rows = sorted([r for r in results if r["scan"] == "alpha"], key=lambda x: x["value"])
    kappa_rows = sorted([r for r in results if r["scan"] == "kappa"], key=lambda x: x["value"])
    alpha_vals = np.array([r["value"] for r in alpha_rows])
    nu_alpha = np.array([r["exponent"] for r in alpha_rows])
    kappa_vals = np.array([r["value"] for r in kappa_rows])
    nu_kappa = np.array([r["exponent"] for r in kappa_rows])

    np.savetxt(out_dir / "alpha_exponents.csv", np.column_stack([alpha_vals, nu_alpha]), delimiter=",", header="Alpha,ScalingExponent", comments="")
    np.savetxt(out_dir / "kappa_exponents.csv", np.column_stack([kappa_vals, nu_kappa]), delimiter=",", header="Kappa,ScalingExponent", comments="")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].semilogx(alpha_vals, nu_alpha, "o-", lw=2)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"Scaling exponent $\nu$")
    axes[0].set_title(r"$\nu$ vs $\alpha$ ($\kappa=0.5$)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(kappa_vals, nu_kappa, "s-", lw=2)
    axes[1].set_xlabel(r"$\kappa$")
    axes[1].set_ylabel(r"Scaling exponent $\nu$")
    axes[1].set_title(r"$\nu$ vs $\kappa$ ($\alpha=0.5$)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(fr"Energy spectrum scaling exponent, fit range $k \in [{k_min:g}, {k_max:g}]$")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_exponent_summary.png", dpi=300)
    plt.close(fig)

    for name, x, y, xlabel in (
        ("nu_vs_alpha.png", alpha_vals, nu_alpha, r"$\alpha$"),
        ("nu_vs_kappa.png", kappa_vals, nu_kappa, r"$\kappa$"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        if name == "nu_vs_alpha.png":
            ax.semilogx(x, y, "o-", lw=2)
        else:
            ax.plot(x, y, "s-", lw=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Scaling exponent $\nu$")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized Python compute_scaling_exponents for the bHW model.")
    parser.add_argument("--mode", choices=["quick", "debug", "production"], default="production")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "scaling_exponents_py")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fftw-threads", type=int, default=1)
    parser.add_argument("--k-min", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.mode == "quick":
        base = BHWParams(N=32, T=0.05, dt=0.01, save_every=1, verbose=not args.quiet, fft_threads=args.fftw_threads)
    elif args.mode == "debug":
        base = BHWParams(N=64, T=2.0, dt=0.01, save_every=20, verbose=not args.quiet, fft_threads=args.fftw_threads)
    else:
        base = BHWParams(N=256, T=500.0, dt=0.01, save_every=20, verbose=not args.quiet, fft_threads=args.fftw_threads)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = make_cases(base, args.out_dir, args.k_min, args.k_max)
    print(f"Running {len(cases)} cases: mode={args.mode}, workers={args.workers}, fft_threads={args.fftw_threads}", flush=True)

    results = []
    if args.workers == 1:
        for case in cases:
            results.append(run_case(case))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_case, c) for c in cases]
            for fut in as_completed(futs):
                row = fut.result()
                print(f"case done: {row['scan']}={row['value']} nu={row['exponent']:.6g}", flush=True)
                results.append(row)

    plot_exponents(results, args.out_dir, args.k_min, args.k_max)
    plot_spectra(results, "alpha", args.out_dir, args.k_min, args.k_max)
    plot_spectra(results, "kappa", args.out_dir, args.k_min, args.k_max)
    print(f"All outputs saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
