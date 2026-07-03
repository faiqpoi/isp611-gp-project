"""
Sensitivity Analysis for Warehouse PSO – Part A only
=====================================================
Runs a 7×7 grid of (α, β) pairs and a constrained sweep (α + β = 1).
Saves two PNG figures and a summary CSV.

  PART A – α / β weight sweep
      A 7×7 grid of (α, β) pairs is evaluated. For each pair the PSO
      runs once and the best fitness is recorded. The heatmap shows
      which weighting regime produces the highest fitness and how quickly
      fitness degrades as you move away from the optimum.

      Reference: Coello Coello et al. (2004) "Handling Multiple Objectives
      with Particle Swarm Optimization", IEEE TEC 8(3).
      https://doi.org/10.1109/TEVC.2004.826067

Usage
-----
  1. Place logistics_warehouse_dataset.csv in the same directory.
  2. python pso_sensitivity.py

  Output files:
      sensitivity_alpha_beta_heatmap.png
      sensitivity_alpha_beta_constrained.png
      sensitivity_summary.csv
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Make pso_warehouse importable regardless of working directory ─────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pso_warehouse as pso

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATASET_PATH  = "logistics_dataset.csv"
SA_MAX_ITER   = 100   # iterations per PSO run inside sensitivity
SA_PARTICLES  = 30    # swarm size for sensitivity runs

# Clerc-Kennedy defaults held fixed during the sweep
W_DEFAULT     = 0.729
C_DEFAULT     = 1.494

# ═════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_arrays(path: str):
    df = pso.load_data(path)
    df = pso.normalise_columns(df)
    E  = df["layout_efficiency_score"].values.astype(float)
    P  = df["picking_time_seconds"].values.astype(float)
    C  = df["handling_cost_per_unit"].values.astype(float)
    return E, P, C


def quick_pso(E, P, C, alpha=0.5, beta=0.5, seed=42) -> float:
    """Run PSO with fixed hyper-parameters and return best fitness."""
    _, best_f, _ = pso.run_pso(
        E, P, C,
        n_particles=SA_PARTICLES,
        max_iter=SA_MAX_ITER,
        w=W_DEFAULT, c1=C_DEFAULT, c2=C_DEFAULT,
        alpha=alpha, beta=beta,
        seed=seed
    )
    return best_f


# ══════════════════════════════════════════════════════════════════════════════
# PART A – α / β GRID SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_alpha_beta_sweep(E, P, C):
    """
    Two sub-analyses:
      (i)  Free 7×7 grid: α, β ∈ [0.1, 0.9] independently → heatmap
      (ii) Constrained:   α + β = 1 → 1-D line plot
    """
    alphas = np.round(np.linspace(0.1, 0.9, 7), 2)
    betas  = np.round(np.linspace(0.1, 0.9, 7), 2)

    print("\n[PART A] α/β free grid sweep …")
    total_runs = len(alphas) * len(betas)
    grid = np.zeros((len(alphas), len(betas)))

    run = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            run += 1
            f = quick_pso(E, P, C, alpha=a, beta=b)
            grid[i, j] = f
            print(f"  [{run:>3d}/{total_runs}]  α={a:.1f}  β={b:.1f}  →  f={f:.5f}")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   cmap="RdYlGn",
                   extent=[betas[0]-0.05, betas[-1]+0.05,
                           alphas[0]-0.05, alphas[-1]+0.05])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best fitness  f", fontsize=10)
    ax.set_xlabel("β  (cost + time weight)", fontsize=11)
    ax.set_ylabel("α  (efficiency weight)", fontsize=11)
    ax.set_title("Sensitivity to α / β weighting\n(PSO best fitness – higher is better)",
                 fontsize=12)
    ax.set_xticks(betas);  ax.set_xticklabels(betas, fontsize=8)
    ax.set_yticks(alphas); ax.set_yticklabels(alphas, fontsize=8)

    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(betas[j], alphas[i], f"{grid[i,j]:.3f}",
                    ha="center", va="center", fontsize=6.5, color="black")

    best_idx = np.unravel_index(np.argmax(grid), grid.shape)
    ax.plot(betas[best_idx[1]], alphas[best_idx[0]],
            marker="*", color="blue", markersize=14, label="Best")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out1 = "sensitivity_alpha_beta_heatmap.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  → Saved: {out1}")

    # ── Constrained line (α + β = 1) ─────────────────────────────────────────
    alphas_c = np.round(np.linspace(0.05, 0.95, 15), 3)
    print("\n  Constrained sweep (α + β = 1) …")
    f_constrained = []
    for a in alphas_c:
        b = round(1.0 - a, 3)
        f = quick_pso(E, P, C, alpha=a, beta=b)
        f_constrained.append(f)
        print(f"    α={a:.3f}  β={b:.3f}  →  f={f:.5f}")

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(alphas_c, f_constrained, marker="o", color="steelblue", linewidth=1.8)
    ax2.axvline(alphas_c[np.argmax(f_constrained)], color="red",
                linestyle="--", linewidth=1.2,
                label=f"Best α={alphas_c[np.argmax(f_constrained)]:.3f}")
    ax2.set_xlabel("α  (β = 1 − α)", fontsize=11)
    ax2.set_ylabel("Best fitness  f", fontsize=11)
    ax2.set_title("Constrained sensitivity: α + β = 1", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    out2 = "sensitivity_alpha_beta_constrained.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  → Saved: {out2}")

    return grid, alphas, betas, alphas_c, f_constrained


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(grid, alphas, betas, alphas_c, f_constrained):
    rows = []

    # Full grid results
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            rows.append({
                "sweep":     "free_grid",
                "alpha":     a,
                "beta":      b,
                "fitness":   grid[i, j],
            })

    # Constrained results
    for a, f in zip(alphas_c, f_constrained):
        rows.append({
            "sweep":   "constrained_alpha+beta=1",
            "alpha":   a,
            "beta":    round(1.0 - a, 3),
            "fitness": f,
        })

    df = pd.DataFrame(rows)
    out = "sensitivity_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n  Summary CSV saved → {out}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Warehouse PSO – Sensitivity Analysis (Part A: α / β)")
    print("=" * 65)
    print(f"\n  Dataset   : {DATASET_PATH}")
    print(f"  PSO iters : {SA_MAX_ITER}")
    print(f"  PSO swarm : {SA_PARTICLES}")
    print(f"  Total PSO runs: 7×7 grid (49) + constrained sweep (15) = 64")

    t0 = time.time()

    print(f"\nLoading dataset …")
    E, P, C = prepare_arrays(DATASET_PATH)
    print(f"  {len(E)} items loaded and normalised.")

    grid, alphas, betas, alphas_c, f_constr = run_alpha_beta_sweep(E, P, C)

    write_summary(grid, alphas, betas, alphas_c, f_constr)

    elapsed = time.time() - t0
    print(f"\n  Total analysis time: {elapsed/60:.1f} min")
    print("\n  Output files:")
    print("    sensitivity_alpha_beta_heatmap.png")
    print("    sensitivity_alpha_beta_constrained.png")
    print("    sensitivity_summary.csv")
    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()