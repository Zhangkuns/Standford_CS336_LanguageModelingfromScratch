#!/usr/bin/env python3
"""
Reproduce the IsoFLOPs method (Hoffmann et al., 2022-style) for fitting scaling laws.

Input: data/isoflops_curves.json
Each run: {"parameters": int, "compute_budget": float, "final_loss": float}

Method (as instructed):
  1) For each compute budget Ci, choose the run with the lowest final_loss.
     This gives points <Ci, Nopt(Ci)>.
  2) Fit power law: Nopt = k * C^a.
  3) Infer dataset size using the standard transformer training FLOPs proxy:
        C ≈ 6 * N * D   =>   D ≈ C / (6N)
     Then for each Ci, Dopt(Ci) = Ci / (6 * Nopt(Ci))
     Fit power law: Dopt = k2 * C^b.
  4) Plot points and fitted curves, extrapolate to at least 1e24 FLOPs.
  5) Print predicted Nopt and Dopt at 1e23 and 1e24 FLOPs.

Usage:
  python isoflops_scaling.py --json data/isoflops_curves.json --outdir out

Outputs:
  out/model_size_scaling.png
  out/dataset_size_scaling.png
  Printed one-line predictions.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(C, k, a):
    # C is float or np.array
    return k * np.power(C, a)


def fit_power_law(C_vals, Y_vals):
    """
    Fit Y = k * C^a robustly by fitting in log-space (equivalent to power law).
    Returns (k, a).
    """
    C = np.asarray(C_vals, dtype=float)
    Y = np.asarray(Y_vals, dtype=float)

    if np.any(C <= 0) or np.any(Y <= 0):
        raise ValueError("All C and Y must be > 0 for log-space power-law fitting.")

    x = np.log(C)
    y = np.log(Y)

    # Linear regression in log space: y = log(k) + a * x
    a, logk = np.polyfit(x, y, 1)
    k = float(np.exp(logk))
    return k, float(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="data/isoflops_curves.json", help="Path to isoflops_curves.json")
    parser.add_argument("--outdir", default="out", help="Directory to save plots")
    parser.add_argument("--targets", nargs="*", type=float, default=[1e23, 1e24], help="Target FLOPs to predict")
    parser.add_argument("--flops_coef", type=float, default=6.0,
                        help="Compute proxy coefficient in C ≈ flops_coef * N * D (default 6)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load runs ---
    with open(args.json, "r") as f:
        runs = json.load(f)

    # --- Group by compute_budget and pick min-loss run (IsoFLOPs minimum) ---
    by_C = defaultdict(list)
    for r in runs:
        C = float(r["compute_budget"])
        N = float(r["parameters"])
        L = float(r["final_loss"])
        by_C[C].append((N, L))

    C_points = []
    Nopt_points = []
    Lopt_points = []
    for C, nl_list in by_C.items():
        # pick lowest loss
        N_best, L_best = min(nl_list, key=lambda x: x[1])
        C_points.append(C)
        Nopt_points.append(N_best)
        Lopt_points.append(L_best)

    # Sort by compute
    order = np.argsort(C_points)
    C_points = np.array(C_points)[order]
    Nopt_points = np.array(Nopt_points)[order]
    Lopt_points = np.array(Lopt_points)[order]

    # --- Infer Dopt for each Ci using D ≈ C / (coef * N) ---
    coef = float(args.flops_coef)
    Dopt_points = C_points / (coef * Nopt_points)

    # --- Fit power laws ---
    kN, a = fit_power_law(C_points, Nopt_points)
    kD, b = fit_power_law(C_points, Dopt_points)

    # --- Predictions at target budgets ---
    targets = np.array(args.targets, dtype=float)
    N_pred = power_law(targets, kN, a)
    D_pred = power_law(targets, kD, b)

    # --- Build x-grid for smooth curves up to at least 1e24 FLOPs ---
    C_min = float(np.min(C_points))
    C_max = max(float(np.max(C_points)), 1e24)
    C_grid = np.logspace(math.log10(C_min), math.log10(C_max), 300)

    N_fit = power_law(C_grid, kN, a)
    D_fit = power_law(C_grid, kD, b)

    # --- Plot 1: Model size scaling ---
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(C_points, Nopt_points, label="IsoFLOPs minima points ⟨C, Nopt(C)⟩")
    plt.plot(C_grid, N_fit, label=f"Fit: Nopt = k·C^{a:.3f}")
    for t, n in zip(targets, N_pred):
        plt.scatter([t], [n], marker="x")
        plt.annotate(f"{t:.0e}: {n:.3g}",
                     (t, n),
                     textcoords="offset points",
                     xytext=(8, 8))
    plt.xlabel("Compute budget C (FLOPs)")
    plt.ylabel("Optimal model size Nopt (parameters)")
    plt.title("IsoFLOPs Scaling Law: Model Size vs Compute")
    plt.legend()
    plt.tight_layout()
    model_plot_path = os.path.join(args.outdir, "model_size_scaling.png")
    plt.savefig(model_plot_path, dpi=200)
    plt.close()

    # --- Plot 2: Dataset size scaling ---
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(C_points, Dopt_points, label="Derived points ⟨C, Dopt(C)⟩ with D≈C/(coef·Nopt)")
    plt.plot(C_grid, D_fit, label=f"Fit: Dopt = k·C^{b:.3f}")
    for t, d in zip(targets, D_pred):
        plt.scatter([t], [d], marker="x")
        plt.annotate(f"{t:.0e}: {d:.3g}",
                     (t, d),
                     textcoords="offset points",
                     xytext=(8, 8))
    plt.xlabel("Compute budget C (FLOPs)")
    plt.ylabel("Optimal dataset size Dopt (tokens, proxy)")
    plt.title("IsoFLOPs Scaling Law: Dataset Size vs Compute")
    plt.legend()
    plt.tight_layout()
    data_plot_path = os.path.join(args.outdir, "dataset_size_scaling.png")
    plt.savefig(data_plot_path, dpi=200)
    plt.close()

    # --- One-sentence predictions (as requested) ---
    # (We print both targets in one line each.)
    def fmt_int_like(x):
        # Parameters/tokens can be huge; show in scientific notation with 3 sig figs.
        return f"{x:.3g}"

    # Model size prediction line
    pred_parts_N = [f"C={t:.0e}: Nopt≈{fmt_int_like(n)} params" for t, n in zip(targets, N_pred)]
    print(f"Predicted optimal model size: {', '.join(pred_parts_N)}.  (fit exponent a={a:.3f})")

    # Dataset size prediction line
    pred_parts_D = [f"C={t:.0e}: Dopt≈{fmt_int_like(d)} tokens" for t, d in zip(targets, D_pred)]
    print(f"Predicted optimal dataset size: {', '.join(pred_parts_D)}.  (fit exponent b={b:.3f}, using C≈{coef}·N·D)")

    print(f"\nSaved plots:\n- {model_plot_path}\n- {data_plot_path}")


if __name__ == "__main__":
    main()
