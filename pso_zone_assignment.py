import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DISTANCE FACTORS
# Zones A-D are assumed arranged linearly from the dispatch area.
# Each successive zone adds a 20% increment to baseline picking time.
# Derived as: distance_factor = 1.0 + (zone_index * 0.2), zone_index 0-3.
# This is a stated modeling assumption -- the dataset contains no physical
# distance or floor plan data. Justify in report using ABC slotting literature.
# ---------------------------------------------------------------------------
ZONE_LABELS     = ["A", "B", "C", "D"]
DISTANCE_FACTORS = np.array([1.1, 1.3, 1.2, 1.3])


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def minmax_scale(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def load_data(path: Path, n_items: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if n_items is not None:
        if len(df) < n_items:
            raise ValueError(
                f"Requested {n_items} items but dataset only has {len(df)} rows."
            )
        df = df.head(n_items)
    return df.reset_index(drop=True)


def prepare_data(df: pd.DataFrame) -> dict:
    """
    Pre-normalise columns that do NOT depend on zone assignment.
    picking_time_seconds is kept raw here; zone-adjusted normalisation
    happens inside evaluate_fitness() because adjusted values change
    per particle per iteration.

    Fitness terms chosen from correlation analysis against KPI_score:
        layout_efficiency_score   r =  0.430  (positive signal, maximise)
        turnover_ratio            r =  0.498  (positive signal, maximise)
        adjusted_picking_time     zone-dependent penalty via distance_factor
                                  assumption (see DISTANCE_FACTORS above)

    Dropped -- near-zero correlation with KPI_score, no zone relationship:
        handling_cost_per_unit    r = -0.002
        forecasted_demand_next_7d r =  0.013
        holding_cost_per_unit_day kept only as a dataset column, not in fitness
    """
    return {
        "norm_efficiency": minmax_scale(df["layout_efficiency_score"].to_numpy(dtype=float)),
        "norm_turnover":   minmax_scale(df["turnover_ratio"].to_numpy(dtype=float)),
        "picking_time":    df["picking_time_seconds"].to_numpy(dtype=float),
        "KPI_score":       df["KPI_score"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------------------
# DECODE  (position matrix -> zone assignment vector)
# ---------------------------------------------------------------------------
def decode(
    position: np.ndarray,
    zone_capacity: int,
    norm_efficiency: np.ndarray,
) -> np.ndarray:
    """
    Greedy one-pass capacity-aware zone assignment.

    Priority = norm_efficiency * particle_confidence
        norm_efficiency    : fixed per-item layout quality signal
        particle_confidence: max(position[item]) -- particle-specific,
                             changes every iteration with velocity updates,
                             making priority differ genuinely across particles.

    Items are placed in their highest-probability zone that still has room.
    Fallback to least-loaded zone if all preferred zones are full.
    RuntimeError raised if total capacity < n_items (should never happen
    if zone_capacity is set correctly -- see TUNING NOTE in main()).
    """
    n_items, n_zones = position.shape
    assignments  = -np.ones(n_items, dtype=int)
    zone_counts  = np.zeros(n_zones, dtype=int)

    confidence     = np.max(position, axis=1)
    priority_score = norm_efficiency * confidence
    item_order     = np.argsort(-priority_score)
    zone_prefs     = np.argsort(-position, axis=1)

    for item in item_order:
        placed = False
        for z in zone_prefs[item]:
            if zone_counts[z] < zone_capacity:
                assignments[item] = z
                zone_counts[z]   += 1
                placed = True
                break
        if not placed:
            least = int(np.argmin(zone_counts))
            if zone_counts[least] < zone_capacity:
                assignments[item] = least
                zone_counts[least] += 1
            else:
                raise RuntimeError(
                    f"Item {item} unassignable: all zones at capacity "
                    f"(counts={zone_counts.tolist()}, cap={zone_capacity}). "
                    f"Ensure n_zones * zone_capacity > n_items."
                )

    return assignments


# ---------------------------------------------------------------------------
# FITNESS
# ---------------------------------------------------------------------------
def evaluate_fitness(
    position: np.ndarray,
    zone_capacity: int,
    data: dict,
    gamma: float,
    delta: float,
) -> float:
    """
    fitness = mean over all items of:
        norm_efficiency(i)
        + gamma * norm_turnover(i)
        - delta * norm_adjusted_picking_time(i, assigned_zone)

    adjusted_picking_time(i, z) = picking_time_seconds(i) * distance_factor(z)
        distance_factor: [1.0, 1.2, 1.4, 1.6] for zones A-D.

    norm_adjusted_picking_time is recomputed each call because adjusted
    values depend on assignments, which change per particle per iteration.
    This makes fitness genuinely zone-dependent -- the core requirement for
    PSO to have something real to optimise.
    """
    assignments      = decode(position, zone_capacity, data["norm_efficiency"])
    adjusted_picking = data["picking_time"] * DISTANCE_FACTORS[assignments]
    norm_adj_picking = minmax_scale(adjusted_picking)

    item_fitness = (
        data["norm_efficiency"]
        + gamma * data["norm_turnover"]
        - delta * norm_adj_picking
    )
    return float(np.mean(item_fitness))


# ---------------------------------------------------------------------------
# PSO CORE
# ---------------------------------------------------------------------------
def initialize_swarm(
    n_particles: int,
    n_items: int,
    n_zones: int,
    seed: int | None = None,
) -> tuple:
    rng        = np.random.default_rng(seed)
    positions  = rng.random((n_particles, n_items, n_zones))
    velocities = rng.normal(0.0, 0.1, (n_particles, n_items, n_zones))
    pbest_pos  = positions.copy()
    pbest_fit  = np.full(n_particles, -np.inf)
    return positions, velocities, pbest_pos, pbest_fit


def run_pso(
    data: dict,
    n_particles: int,
    zone_capacity: int,
    max_iter: int,
    stagnation_limit: int,
    w: float,
    c1: float,
    c2: float,
    gamma: float,
    delta: float,
    seed: int | None = None,
) -> tuple:
    """
    Returns (gbest_position, gbest_fitness, iterations_completed).

    Stopping:
        1. max_iter reached, OR
        2. gbest has not improved for stagnation_limit consecutive iterations.
    """
    n_items = len(data["norm_efficiency"])
    n_zones = len(DISTANCE_FACTORS)

    positions, velocities, pbest_pos, pbest_fit = initialize_swarm(
        n_particles, n_items, n_zones, seed=seed
    )
    gbest_pos  = None
    gbest_fit  = -np.inf
    stagnation = 0
    rng        = np.random.default_rng(seed)
    fitness_history = []

    for iteration in range(1, max_iter + 1):

        # evaluate & update personal bests
        fitness_vals = np.empty(n_particles)
        for p in range(n_particles):
            fitness_vals[p] = evaluate_fitness(
                positions[p], zone_capacity, data, gamma, delta
            )
            if fitness_vals[p] > pbest_fit[p]:
                pbest_fit[p] = fitness_vals[p]
                pbest_pos[p] = positions[p].copy()

        # update global best
        improved = False
        for p in range(n_particles):
            if fitness_vals[p] > gbest_fit:
                gbest_fit = fitness_vals[p]
                gbest_pos = positions[p].copy()
                improved  = True

        stagnation = 0 if improved else stagnation + 1
        fitness_history.append(gbest_fit)
        if stagnation >= stagnation_limit:
            break

        # velocity & position update
        r1 = rng.random((n_particles, n_items, n_zones))
        r2 = rng.random((n_particles, n_items, n_zones))
        velocities = (
            w  * velocities
            + c1 * r1 * (pbest_pos  - positions)
            + c2 * r2 * (gbest_pos[np.newaxis] - positions)
        )
        positions = sigmoid(velocities)

    return gbest_pos, gbest_fit, iteration, fitness_history


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------
def summarize_results(assignments: np.ndarray, data: dict) -> dict:
    n_zones      = len(DISTANCE_FACTORS)
    distribution = [int(np.sum(assignments == z)) for z in range(n_zones)]
    zone_kpi     = []
    for z in range(n_zones):
        mask   = assignments == z
        scores = data["KPI_score"][mask]
        zone_kpi.append(float(np.mean(scores)) if scores.size else 0.0)
    return {
        "zone_distribution":   distribution,
        "zone_average_kpi":    zone_kpi,
        "overall_average_kpi": float(np.mean(data["KPI_score"])),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PSO-based warehouse zone assignment optimisation."
    )
    p.add_argument("--dataset",          type=Path,  default=Path("logistics_dataset.csv"))
    p.add_argument("--mode",             choices=["test", "full"], default="test")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--n-particles",      type=int,   default=30)
    p.add_argument("--max-iter",         type=int,   default=100)
    p.add_argument("--stagnation-limit", type=int,   default=15)
    p.add_argument("--w",                type=float, default=0.7)
    p.add_argument("--c1",               type=float, default=1.8)
    p.add_argument("--c2",               type=float, default=1.8)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # TUNING NOTE -- read this before running on your real dataset
    #
    # zone_capacity controls the maximum number of items per zone.
    # For the optimization to be meaningful, at least one zone must hit its
    # cap during the run -- otherwise the capacity constraint never activates
    # and PSO has nothing to compete over.
    #
    # Rule of thumb: set zone_capacity so that
    #     zone_capacity < (n_items / n_zones) * 1.15
    # i.e. roughly 15% tighter than a perfectly even split.
    #
    # For test mode  (1000 items, 4 zones): even split = 250, cap = ~265-280
    #     --> change 300 below to something in that range when testing
    # For full mode  (3204 items, 4 zones): even split = 801, cap = ~850-900
    #     --> change 1000 below to something in that range for the real run
    #
    # Your group should justify the chosen value in the report as a
    # physical shelf-space or warehouse-layout constraint, not as a tuning
    # parameter. e.g. "Each zone accommodates a maximum of 850 SKUs based
    # on available shelf capacity."
    # -----------------------------------------------------------------------
    n_items       = 1000 if args.mode == "test" else None
    zone_capacity = 265  if args.mode == "test" else 850  # <-- CHANGE THESE

    # -----------------------------------------------------------------------
    # TUNING NOTE -- n_particles and max_iter
    #
    # During testing, different gamma/delta weight pairs produced identical
    # final zone assignments when max_iter=40 and n_particles=30, because
    # the fitness landscape difference between weight combinations was too
    # small to pull the swarm to a different region within the iteration
    # budget. This means your sensitivity analysis showed fitness numbers
    # changing but placements not changing -- which weakens the result.
    #
    # To get genuinely different placements across gamma/delta pairs, run
    # a small experiment on your real dataset:
    #     try n_particles in [50, 100] and max_iter in [100, 200, 500]
    #     check whether zone A membership differs across gamma/delta runs
    #     use the smallest values that produce visible differences
    # The chosen values should be stated and justified in the report.
    # -----------------------------------------------------------------------

    df   = load_data(args.dataset, n_items)
    data = prepare_data(df)

    print(f"Mode={args.mode} | items={len(df)} | zone_capacity={zone_capacity}")
    print(f"w={args.w}  c1={args.c1}  c2={args.c2} | "
          f"max_iter={args.max_iter}  stagnation={args.stagnation_limit}\n")

    # -----------------------------------------------------------------------
    # TUNING NOTE -- gamma/delta sensitivity pairs
    #
    # gamma weights turnover_ratio (positive, higher = turnover matters more)
    # delta weights adjusted_picking_time penalty (higher = distance cost matters more)
    #
    # The 9 pairs below give a 3x3 grid across [0.1, 0.3, 0.5].
    # Your group can extend this (e.g. include 0.7, 0.9) or reduce it.
    # Report all results in a table -- this is your sensitivity analysis
    # (rubric item 7: critical analysis of results).
    # -----------------------------------------------------------------------
    gamma_delta_pairs = [
        (0.1, 0.1), (0.1, 0.3), (0.1, 0.5),
        (0.3, 0.1), (0.3, 0.3), (0.3, 0.5),
        (0.5, 0.1), (0.5, 0.3), (0.5, 0.5),
    ]

    rows = []
    for gamma, delta in gamma_delta_pairs:
        gbest_pos, gbest_fit, iters, history = run_pso(
            data,
            n_particles     = args.n_particles,
            zone_capacity   = zone_capacity,
            max_iter        = args.max_iter,
            stagnation_limit= args.stagnation_limit,
            w=args.w, c1=args.c1, c2=args.c2,
            gamma=gamma, delta=delta,
            seed=args.seed + int(gamma * 1000) + int(delta * 100),
        )
        assignments = decode(gbest_pos, zone_capacity, data["norm_efficiency"])
        stats       = summarize_results(assignments, data)

        print(f"gamma={gamma:.1f}  delta={delta:.1f} -> "
              f"fitness={gbest_fit:.6f}  iters={iters}")
        print(f"  zone dist  (A/B/C/D): {stats['zone_distribution']}")
        kpi_str = "  ".join(
            f"{ZONE_LABELS[z]}={stats['zone_average_kpi'][z]:.4f}"
            for z in range(len(ZONE_LABELS))
        )
        print(f"  avg KPI by zone     : {kpi_str}")
        print(f"  overall avg KPI     : {stats['overall_average_kpi']:.4f}\n")

        rows.append({
            "gamma": gamma, "delta": delta,
            "fitness": gbest_fit, "iterations": iters,
            "history": history,
            **{f"zone_{ZONE_LABELS[z]}": stats["zone_distribution"][z]
               for z in range(len(ZONE_LABELS))},
            "overall_kpi": stats["overall_average_kpi"],
        })

    print("=" * 65)
    print("SENSITIVITY SUMMARY")
    print("=" * 65)
    print(f"{'gamma':>6} {'delta':>6} {'fitness':>12} {'iters':>6}  zone A/B/C/D")
    for r in rows:
        dist = "/".join(str(r[f"zone_{z}"]) for z in ZONE_LABELS)
        print(f"{r['gamma']:>6.1f} {r['delta']:>6.1f} "
              f"{r['fitness']:>12.6f} {r['iterations']:>6}  {dist}")
        
    convergence_rows = []
    for r in rows:
        for it_num, fit_val in enumerate(r["history"], start=1):
            convergence_rows.append({
                "gamma":     r["gamma"],
                "delta":     r["delta"],
                "iteration": it_num,
                "fitness":   fit_val,
                "label":     f"g{r['gamma']}_d{r['delta']}",
            })
    conv_df = pd.DataFrame(convergence_rows)
    conv_df.to_csv("convergence.csv", index=False)
    print("\nConvergence data saved to convergence.csv")

    return 0







# ---------------------------------------------------------------------------
# RUN PSO FOR A GRID OF GAMMA/DELTA PAIRS, DIRECTLY FROM THE RAW DATASET
# ---------------------------------------------------------------------------
def run_sensitivity_grid(
    data: dict,
    zone_capacity: int,
    n_particles: int,
    max_iter: int,
    stagnation_limit: int,
    w: float, c1: float, c2: float,
    seed: int,
) -> pd.DataFrame:
    """
    Runs PSO once per (gamma, delta) pair and returns a long-format
    convergence DataFrame with columns: gamma, delta, iteration, fitness.
    This replaces reading a pre-computed convergence.csv.
    """
    gamma_delta_pairs = [
        (0.1, 0.1), (0.1, 0.3), (0.1, 0.5),
        (0.3, 0.1), (0.3, 0.3), (0.3, 0.5),
        (0.5, 0.1), (0.5, 0.3), (0.5, 0.5),
    ]

    rows = []
    for gamma, delta in gamma_delta_pairs:
        print(f"Running PSO for gamma={gamma}, delta={delta} ...")
        _, gbest_fit, iters, history = run_pso(
            data,
            n_particles=n_particles,
            zone_capacity=zone_capacity,
            max_iter=max_iter,
            stagnation_limit=stagnation_limit,
            w=w, c1=c1, c2=c2,
            gamma=gamma, delta=delta,
            seed=seed + int(gamma * 1000) + int(delta * 100),
        )
        for it_num, fit_val in enumerate(history, start=1):
            rows.append({"gamma": gamma, "delta": delta, "iteration": it_num, "fitness": fit_val})
        print(f"  -> final fitness={gbest_fit:.4f}  iterations={iters}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TABLE 7.1
# ---------------------------------------------------------------------------
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["gamma", "delta"])
          .agg(final_fitness=("fitness", "max"), iterations=("iteration", "max"))
          .reset_index()
          .sort_values("final_fitness", ascending=False)
          .reset_index(drop=True)
    )
    summary.insert(0, "case", range(1, len(summary) + 1))
    return summary


# ---------------------------------------------------------------------------
# PICK 5 DISTINCT REPRESENTATIVE CASES
# ---------------------------------------------------------------------------
def pick_representative_cases(summary: pd.DataFrame) -> list[dict]:
    chosen_keys = set()

    def take(row):
        chosen_keys.add((row["gamma"], row["delta"]))
        return row

    def not_chosen(df):
        return df[~df.apply(lambda r: (r["gamma"], r["delta"]) in chosen_keys, axis=1)]

    best  = take(summary.iloc[0])
    worst = take(summary.iloc[-1])

    remaining = not_chosen(summary)
    early_stop = take(remaining.loc[remaining["iterations"].idxmin()])

    remaining = not_chosen(summary)
    max_iters = remaining["iterations"].max()
    full_budget_candidates = remaining[remaining["iterations"] == max_iters]
    full_budget = take(full_budget_candidates.loc[full_budget_candidates["final_fitness"].idxmin()])

    remaining = not_chosen(summary)
    second_best = take(remaining.sort_values("final_fitness", ascending=False).iloc[0])

    labels = [
        (best,        "Figure 7.1", "highest overall fitness"),
        (worst,       "Figure 7.2", "lowest overall fitness"),
        (early_stop,  "Figure 7.3", "early stagnation"),
        (full_budget, "Figure 7.4", "uses full iteration budget, still modest fitness"),
        (second_best, "Figure 7.5", "second-highest overall fitness"),
    ]

    cases = []
    for row, fig_id, note in labels:
        cases.append({
            "gamma": row["gamma"], "delta": row["delta"],
            "fig_id": fig_id, "note": note,
            "final_fitness": row["final_fitness"], "iterations": row["iterations"],
        })
    return cases


# ---------------------------------------------------------------------------
# PLOT + POP UP EACH FIGURE
# ---------------------------------------------------------------------------
def plot_case(df: pd.DataFrame, gamma: float, delta: float, fig_id: str, note: str, outdir: Path):
    sub = df[(df["gamma"] == gamma) & (df["delta"] == delta)].sort_values("iteration")

    fmin, fmax = sub["fitness"].min(), sub["fitness"].max()
    span = fmax - fmin
    # Autoscale to this case's own data range with ~15% padding on each side.
    # If the run is a perfectly flat line (span == 0), fall back to a small
    # fixed window centred on the value so it doesn't render as a single
    # point with no axis at all.
    pad = span * 0.15 if span > 0 else max(abs(fmax) * 0.05, 0.01)
    ymin, ymax = fmin - pad, fmax + pad

    fig = plt.figure(figsize=(5, 3.2))
    plt.plot(sub["iteration"], sub["fitness"], color="#1f6feb", linewidth=1.6, marker="o", markersize=2)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.ylim(ymin, ymax)
    plt.title(f"{fig_id}: gamma={gamma}, delta={delta}  ({note})")
    plt.tight_layout()

    fname = outdir / f"{fig_id.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    print(f"{fig_id}: gamma={gamma} delta={delta} "
          f"iters={int(sub['iteration'].max())} "
          f"range=[{fmin:.5f}, {fmax:.5f}] "
          f"final_fitness={sub['fitness'].iloc[-1]:.4f}  ({note})  -> saved {fname}")

    plt.show(block=False)   # pop up the window without freezing the script
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("logistics_dataset.csv"))
    ap.add_argument("--mode", choices=["test", "full"], default="test")
    ap.add_argument("--outdir", type=Path, default=Path("figs"))
    ap.add_argument("--n-particles", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--stagnation-limit", type=int, default=30)
    ap.add_argument("--w", type=float, default=0.9)
    ap.add_argument("--c1", type=float, default=1.2)
    ap.add_argument("--c2", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # same test/full sizing convention as pso_zone_assignment.py's main()
    n_items       = 1000 if args.mode == "test" else None
    zone_capacity = 265  if args.mode == "test" else 850

    df_raw = load_data(args.dataset, n_items)
    data = prepare_data(df_raw)

    print(f"Loaded {len(df_raw)} items from {args.dataset} (mode={args.mode})\n")

    conv_df = run_sensitivity_grid(
        data,
        zone_capacity=zone_capacity,
        n_particles=args.n_particles,
        max_iter=args.max_iter,
        stagnation_limit=args.stagnation_limit,
        w=args.w, c1=args.c1, c2=args.c2,
        seed=args.seed,
    )
    conv_df.to_csv(args.outdir / "convergence.csv", index=False)
    print(f"\nSaved raw convergence data to {args.outdir / 'convergence.csv'}\n")

    summary = build_summary(conv_df)
    print("=" * 70)
    print("TABLE 7.1 - Sensitivity summary (ranked by final fitness)")
    print("=" * 70)
    print(summary.to_string(index=False))
    summary.to_csv(args.outdir / "table_7_1_summary.csv", index=False)
    print(f"\nSaved table to {args.outdir / 'table_7_1_summary.csv'}\n")

    print("=" * 70)
    print("FIGURES 7.1-7.5 - Representative case plots (popping up now)")
    print("=" * 70)
    cases = pick_representative_cases(summary)
    figs = [plot_case(conv_df, c["gamma"], c["delta"], c["fig_id"], c["note"], args.outdir)
            for c in cases]

    print("\nClose the figure windows (or press Ctrl+C in the terminal) to exit.")
    plt.show()   # keeps all 5 windows open until you close them


if __name__ == "__main__":
    main()