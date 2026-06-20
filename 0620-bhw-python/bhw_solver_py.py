from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time

import numpy as np


try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fftw_np

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
except Exception:  # pragma: no cover - exercised when pyfftw is unavailable
    pyfftw = None
    fftw_np = None


@dataclass
class BHWParams:
    L: float = 40.0
    N: int = 256
    kappa: float = 0.5
    alpha: float = 0.1
    mu: float = 5e-4
    D: float = 5e-4
    C: float = 0.0
    nu: float = 7e-21
    s: int = 8
    dt: float = 0.01
    T: float = 500.0
    save_every: int = 20
    verbose: bool = True
    fft_threads: int = 1


class BHWSolver:
    """CPU-optimized pseudospectral bHW solver for scaling-exponent runs."""

    def __init__(self, params: BHWParams):
        self.p = params
        self.nsteps = int(round(params.T / params.dt))
        self.nsaves = self.nsteps // params.save_every + 1
        self._setup_grid()
        self._setup_spectral()
        self._setup_fft()
        self._initialize_fields()
        self.t = 0.0
        self.diag_idx = 0
        self.t_vals = np.zeros(self.nsaves, dtype=np.float64)
        self.e_tot = np.zeros(self.nsaves, dtype=np.float64)
        self.e_zonal = np.zeros(self.nsaves, dtype=np.float64)
        self.gamma = np.zeros(self.nsaves, dtype=np.float64)

    def _setup_grid(self) -> None:
        p = self.p
        dx = p.L / p.N
        self.x = np.arange(p.N, dtype=np.float64) * dx
        self.y = np.arange(p.N, dtype=np.float64) * dx

    def _setup_spectral(self) -> None:
        p = self.p
        k = 2.0 * np.pi * np.fft.fftfreq(p.N, d=p.L / p.N)
        self.kx = k.reshape(1, p.N)
        self.ky = k.reshape(p.N, 1)
        self.KX, self.KY = np.meshgrid(k, k)
        self.iKX = 1j * self.KX
        self.iKY = 1j * self.KY
        self.k2 = self.KX * self.KX + self.KY * self.KY
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0] = 1.0
        self.k2s = self.k2 ** p.s
        kx_max = np.max(np.abs(self.kx))
        ky_max = np.max(np.abs(self.ky))
        self.dealias_mask = (np.abs(self.KX) <= (2.0 / 3.0) * kx_max) & (
            np.abs(self.KY) <= (2.0 / 3.0) * ky_max
        )
        self.filter_hyper_half = np.exp(-p.nu * self.k2s * p.dt / 2.0)

    def _setup_fft(self) -> None:
        p = self.p
        self._fftn_threads = max(1, int(p.fft_threads))
        if fftw_np is None:
            self.fft2 = np.fft.fft2
            self.ifft2 = np.fft.ifft2
            return

        def fft2(a):
            return fftw_np.fft2(a, threads=self._fftn_threads, planner_effort="FFTW_MEASURE")

        def ifft2(a):
            return fftw_np.ifft2(a, threads=self._fftn_threads, planner_effort="FFTW_MEASURE")

        self.fft2 = fft2
        self.ifft2 = ifft2

    def _initialize_fields(self) -> None:
        p = self.p
        rng = np.random.default_rng(42)
        phi = 1e-4 * rng.standard_normal((p.N, p.N))
        phi_hat = self.fft2(phi)
        phi_hat[0, 0] = 0.0
        m_seed = max(1, min(round(math.sqrt(2.0) * p.L / (2.0 * math.pi)), p.N // 2))
        amp = 1e-3 * p.N * p.N
        phi_hat[0, m_seed] = amp * (1.0 + 1.0j) / math.sqrt(2.0)
        if m_seed < p.N // 2:
            phi_hat[0, p.N - m_seed] = np.conj(phi_hat[0, m_seed])
        self.n_hat = 0.5 * phi_hat
        self.zeta_hat = -self.k2 * phi_hat
        self.zeta_hat[0, 0] = 0.0

    def compute_phi_from_zeta(self, zeta_hat: np.ndarray) -> np.ndarray:
        phi_hat = -zeta_hat / self.k2_safe
        phi_hat[0, 0] = 0.0
        return phi_hat

    def compute_rhs(self, zeta_hat: np.ndarray, n_hat: np.ndarray):
        p = self.p
        phi_hat = self.compute_phi_from_zeta(zeta_hat)
        dphidx = self.ifft2(self.iKX * phi_hat).real
        dphidy = self.ifft2(self.iKY * phi_hat).real
        dzetadx = self.ifft2(self.iKX * zeta_hat).real
        dzetady = self.ifft2(self.iKY * zeta_hat).real
        dndx = self.ifft2(self.iKX * n_hat).real
        dndy = self.ifft2(self.iKY * n_hat).real

        j_phi_zeta = self.fft2(dphidx * dzetady - dphidy * dzetadx) * self.dealias_mask
        j_phi_n = self.fft2(dphidx * dndy - dphidy * dndx) * self.dealias_mask

        phi_tilde_hat = phi_hat.copy()
        phi_tilde_hat[0, :] = 0.0
        n_tilde_hat = n_hat.copy()
        n_tilde_hat[0, :] = 0.0
        adiabatic = p.alpha * (phi_tilde_hat - n_tilde_hat)

        rz = -j_phi_zeta + adiabatic - p.mu * self.k2 * zeta_hat + p.C * phi_hat
        rn = -j_phi_n - 1j * p.kappa * self.KY * phi_tilde_hat + adiabatic - p.D * self.k2 * n_hat
        return rz, rn

    def step(self) -> None:
        p = self.p
        self.zeta_hat *= self.filter_hyper_half
        self.n_hat *= self.filter_hyper_half
        z0 = self.zeta_hat
        n0 = self.n_hat
        dt = p.dt
        k1z, k1n = self.compute_rhs(z0, n0)
        k2z, k2n = self.compute_rhs(z0 + 0.5 * dt * k1z, n0 + 0.5 * dt * k1n)
        k3z, k3n = self.compute_rhs(z0 + 0.5 * dt * k2z, n0 + 0.5 * dt * k2n)
        k4z, k4n = self.compute_rhs(z0 + dt * k3z, n0 + dt * k3n)
        self.zeta_hat = z0 + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        self.n_hat = n0 + (dt / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
        self.zeta_hat *= self.filter_hyper_half
        self.n_hat *= self.filter_hyper_half

    def run(self) -> None:
        p = self.p
        if p.verbose:
            print(
                f"bHW N={p.N} T={p.T:g} dt={p.dt:g} steps={self.nsteps} "
                f"alpha={p.alpha:g} kappa={p.kappa:g} fft_threads={p.fft_threads}",
                flush=True,
            )
        t0 = time.time()
        self.save_diagnostics()
        next_report = 0.0
        for n in range(1, self.nsteps + 1):
            self.step()
            self.t = n * p.dt
            if n % p.save_every == 0:
                self.save_diagnostics()
            if p.verbose:
                progress = n / self.nsteps
                if progress >= next_report:
                    elapsed = time.time() - t0
                    eta = elapsed / max(progress, 1e-12) * (1.0 - progress)
                    print(
                        f"  t={self.t:.1f} step={n}/{self.nsteps} "
                        f"({100*progress:.0f}%) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                        flush=True,
                    )
                    next_report += 0.05

    def save_diagnostics(self) -> None:
        idx = self.diag_idx
        if idx >= self.nsaves:
            return
        self.t_vals[idx] = self.t
        phi_hat = self.compute_phi_from_zeta(self.zeta_hat)
        cell_area = (self.p.L / self.p.N) ** 2
        dphidx = self.ifft2(self.iKX * phi_hat).real
        dphidy = self.ifft2(self.iKY * phi_hat).real
        self.e_tot[idx] = 0.5 * cell_area * np.sum(dphidx * dphidx + dphidy * dphidy)
        phi_bar_hat = phi_hat.copy()
        phi_bar_hat[1:, :] = 0.0
        dphi_bar_dx = self.ifft2(self.iKX * phi_bar_hat).real
        self.e_zonal[idx] = 0.5 * cell_area * np.sum(dphi_bar_dx * dphi_bar_dx)
        phi_tilde_hat = phi_hat - phi_bar_hat
        n_tilde_hat = self.n_hat.copy()
        n_tilde_hat[0, :] = 0.0
        u_tilde = self.ifft2(-self.iKY * phi_tilde_hat).real
        n_tilde = self.ifft2(n_tilde_hat).real
        self.gamma[idx] = np.mean(u_tilde * n_tilde)
        self.diag_idx += 1

    def spectrum(self):
        p = self.p
        phi_hat = self.compute_phi_from_zeta(self.zeta_hat)
        kr = np.sqrt(self.k2)
        dk = 2.0 * np.pi / p.L
        k_edges = np.arange(0, math.ceil(float(np.max(kr)) / dk) + 1, dtype=np.float64) * dk
        k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
        e_modes = 0.5 * (p.L * p.L / p.N**4) * self.k2 * np.abs(phi_hat) ** 2
        shell = np.digitize(kr.ravel(), k_edges) - 1
        valid_shell = (shell >= 0) & (shell < len(k_centers))
        ek = np.bincount(shell[valid_shell], weights=e_modes.ravel()[valid_shell], minlength=len(k_centers))
        valid = (k_centers > 0.0) & (ek > 0.0)
        return k_centers[valid], ek[valid]

    def diagnostics(self):
        last = min(self.diag_idx, self.nsaves)
        return self.t_vals[:last], self.e_tot[:last], self.e_zonal[:last], self.gamma[:last]


def write_case_outputs(out_prefix: Path, solver: BHWSolver, k: np.ndarray, ek: np.ndarray, exponent: float) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    t, e_tot, e_zonal, gamma = solver.diagnostics()
    diag = np.column_stack([t, e_tot, e_zonal, gamma])
    np.savetxt(
        out_prefix.with_name(out_prefix.name + "_diagnostics.csv"),
        diag,
        delimiter=",",
        header="Time,TotalKineticEnergy,ZonalFlowEnergy,ParticleFlux",
        comments="",
    )
    np.savetxt(
        out_prefix.with_name(out_prefix.name + "_spectrum.csv"),
        np.column_stack([k, ek]),
        delimiter=",",
        header="Wavenumber_k,Spectrum_Ek",
        comments="",
    )
    payload = {
        "parameters": solver.p.__dict__,
        "scaling_exponent": exponent,
        "diagnostics_tail": {
            "time": t[-10:].tolist(),
            "E_tot": e_tot[-10:].tolist(),
            "E_zonal": e_zonal[-10:].tolist(),
            "Gamma": gamma[-10:].tolist(),
        },
    }
    out_prefix.with_name(out_prefix.name + "_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
