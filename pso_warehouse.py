"""
Warehouse Storage Allocation Optimization using Particle Swarm Optimization (PSO)
===================================================================================
Dataset : Logistics Warehouse Dataset (Kaggle - ziya07)
          https://www.kaggle.com/datasets/ziya07/logistics-warehouse-dataset
Problem : Assign N items to M storage locations to
              MAXIMIZE  layout_efficiency_score  (weighted by α)
              MINIMIZE  picking_time_seconds + handling_cost_per_unit  (weighted by β)

Fitness function (from assignment spec):
    f = α * Σ Ei  -  β * Σ (Pi + Ci)     for i = 1..N

Solution representation: ROV (Ranked-Order-Value) mapping of a continuous PSO
position vector to a discrete slot permutation (Section 5.1 of the assignment).

Usage
-----
1. Download the dataset CSV from Kaggle and place it as:
       logistics_warehouse_dataset.csv
   in the same directory as this script.

2. Run:
       python pso_warehouse.py

3. Optionally tune hyper-parameters in the CONFIG section below.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG – tune these without touching the algorithm
# ══════════════════════════════════════════════════════════════════════════════
DATASET_PATH    = "logistics_dataset.csv"

# PSO hyper-parameters
N_PARTICLES     = 50        # swarm size
MAX_ITER        = 200       # maximum iterations
W               = 0.729     # inertia weight  (Clerc & Kennedy constriction)
C1              = 1.494     # cognitive coefficient
C2              = 1.494     # social coefficient
V_MAX_RATIO     = 0.20      # |v| ≤ V_MAX_RATIO * N  (N = number of items)

# Objective weights (α for efficiency, β for cost+time)
ALPHA           = 0.9
BETA            = 0.1

# Normalise P and C before computing fitness so the scale is comparable to E
NORMALISE       = True

# Random seed for reproducibility
SEED            = 42

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and keep only the columns needed by the fitness function."""
    df = pd.read_csv(path)

    required = [
        "item_id",
        "layout_efficiency_score",
        "picking_time_seconds",
        "handling_cost_per_unit",
        # extra columns kept for reporting / binary encoding
        "category",
        "zone",
        "item_popularity_score",
        "daily_demand",
        "order_fulfillment_rate",
        "turnover_ratio",
        "lead_time_days",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[required].copy()
    df.dropna(subset=["layout_efficiency_score",
                      "picking_time_seconds",
                      "handling_cost_per_unit"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise P and C to [0,1] so they are on the same scale as E."""
    for col in ["picking_time_seconds", "handling_cost_per_unit"]:
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn + 1e-9)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROV – Continuous → Discrete slot permutation
# ══════════════════════════════════════════════════════════════════════════════

def rov_decode(position: np.ndarray) -> np.ndarray:
    """
    Ranked-Order-Value (ROV) decoding.

    Given a continuous position vector of length N, return an integer
    permutation [0, N) that maps each item index j to a storage slot index.

    The item with the smallest position value gets slot 0, the next gets
    slot 1, and so on.  This follows the standard ROV rule used in
    combinatorial PSO (Bean 1994; Tasgetiren et al. 2007).

    Parameters
    ----------
    position : np.ndarray, shape (N,)
        Continuous PSO position vector for one particle.

    Returns
    -------
    slot_assignment : np.ndarray, shape (N,), dtype int
        slot_assignment[j] = warehouse slot index assigned to item j.
    """
    return np.argsort(np.argsort(position))   # double argsort = rank


# ══════════════════════════════════════════════════════════════════════════════
# 3. FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def fitness(position: np.ndarray,
            E: np.ndarray,
            P: np.ndarray,
            C: np.ndarray,
            W: np.ndarray | None = None,
            alpha: float = ALPHA,
            beta: float  = BETA) -> float:
    """
    Evaluate the fitness of one particle.

    f = α * Σ_i E[slot_i]  -  β * Σ_i (P[slot_i] + C[slot_i])

    where slot_i is the warehouse slot assigned to item i by the ROV rule.

    Higher is better (we maximise f).

    Parameters
    ----------
    position : np.ndarray, shape (N,)
    E, P, C  : np.ndarray, shape (N,)  – per-slot attribute arrays
    alpha, beta : float – weighting factors

    Returns
    -------
    f : float
    """
    slots = rov_decode(position)          # item j → slot slots[j]
    # If no item weights provided, treat all items equally (ones)
    if W is None:
        W = np.ones_like(E)
    return alpha * np.sum(E[slots] * W) - beta * np.sum((P[slots] + C[slots]) * W)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PSO ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

class Particle:
    """One particle in the swarm."""

    def __init__(self, n: int, v_max: float, rng: np.random.Generator):
        # Continuous position – uniform in [0, n)
        self.position  = rng.uniform(0.0, float(n), size=n)
        # Velocity – uniform in [-v_max, v_max]
        self.velocity  = rng.uniform(-v_max, v_max, size=n)
        self.pbest_pos = self.position.copy()
        self.pbest_val = -np.inf

    def update_velocity(self,
                        gbest_pos: np.ndarray,
                        w: float, c1: float, c2: float,
                        v_max: float,
                        rng: np.random.Generator) -> None:
        r1 = rng.random(len(self.position))
        r2 = rng.random(len(self.position))
        cognitive = c1 * r1 * (self.pbest_pos - self.position)
        social    = c2 * r2 * (gbest_pos       - self.position)
        self.velocity = w * self.velocity + cognitive + social
        # Clamp velocity
        self.velocity = np.clip(self.velocity, -v_max, v_max)

    def update_position(self, n: int) -> None:
        self.position = self.position + self.velocity
        # Keep position in [0, n) to avoid extremely skewed rank values
        self.position = np.clip(self.position, 0.0, float(n - 1) + 0.9999)


def run_pso(E: np.ndarray,
            P: np.ndarray,
            C: np.ndarray,
            item_weights: np.ndarray | None = None,
            n_particles: int = N_PARTICLES,
            max_iter:    int = MAX_ITER,
            w:         float = W,
            c1:        float = C1,
            c2:        float = C2,
            v_max_ratio: float = V_MAX_RATIO,
            alpha:     float = ALPHA,
            beta:      float = BETA,
            seed:        int = SEED):
    """
    Run PSO and return the best slot assignment found.

    Parameters
    ----------
    E, P, C      : per-slot arrays (layout_efficiency, picking_time, handling_cost)
    n_particles  : swarm size
    max_iter     : iteration budget
    w            : inertia weight
    c1, c2       : acceleration coefficients
    v_max_ratio  : |v| ≤ v_max_ratio * N
    alpha, beta  : fitness weights
    seed         : RNG seed

    Returns
    -------
    gbest_slots  : np.ndarray  – best permutation found (item → slot)
    gbest_val    : float       – best fitness value
    history      : list[float] – gbest fitness per iteration
    """
    rng   = np.random.default_rng(seed)
    N     = len(E)
    v_max = v_max_ratio * N

    # ── Initialise swarm ──────────────────────────────────────────────────────
    swarm = [Particle(N, v_max, rng) for _ in range(n_particles)]

    gbest_pos = None
    gbest_val = -np.inf
    history   = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(1, max_iter + 1):
        for particle in swarm:
            f = fitness(particle.position, E, P, C, W=item_weights, alpha=alpha, beta=beta)

            # Update personal best
            if f > particle.pbest_val:
                particle.pbest_val = f
                particle.pbest_pos = particle.position.copy()

            # Update global best
            if f > gbest_val:
                gbest_val = f
                gbest_pos = particle.position.copy()

        history.append(gbest_val)

        # Velocity & position update
        for particle in swarm:
            particle.update_velocity(gbest_pos, w, c1, c2, v_max, rng)
            particle.update_position(N)

        # Progress report every 50 iterations
        if iteration % 50 == 0 or iteration == 1:
            print(f"  Iter {iteration:>4d}/{max_iter}  |  gbest fitness = {gbest_val:.6f}")

    gbest_slots = rov_decode(gbest_pos)
    return gbest_slots, gbest_val, history


# ══════════════════════════════════════════════════════════════════════════════
# 5. BINARY ENCODING (Table 5.1 mapping)
# ══════════════════════════════════════════════════════════════════════════════

def encode_category(cat: str) -> str:
    mapping = {
        "pharma":       "0000",
        "automotive":   "0001",
        "groceries":    "0010",
        "apparel":      "0011",
        "electronics":  "0100",
    }
    return mapping.get(str(cat).lower().strip(), "1111")


def encode_zone(zone: str) -> str:
    mapping = {"a": "0000", "b": "0001", "c": "0010", "d": "0011"}
    return mapping.get(str(zone).lower().strip(), "1111")


def encode_popularity(score: float) -> str:
    if score < 0.2:   return "0000"
    if score < 0.4:   return "0001"
    if score < 0.6:   return "0010"
    if score < 0.8:   return "0011"
    return "0100"


def encode_demand(demand: float) -> str:
    if demand < 10:   return "0000"
    if demand < 25:   return "0001"
    if demand < 40:   return "0010"
    return "0011"


def encode_picking_time(pt: float) -> str:
    if pt < 50:       return "0000"
    if pt < 120:      return "0001"
    if pt < 170:      return "0010"
    return "0011"


def encode_handling_cost(hc: float) -> str:
    if hc < 1.5:      return "0000"
    if hc < 3.0:      return "0001"
    if hc < 4.5:      return "0010"
    return "0011"


def encode_fulfillment(rate: float) -> str:
    if rate < 0.75:   return "0000"
    if rate < 0.85:   return "0001"
    if rate < 0.95:   return "0010"
    return "0011"


def encode_turnover(tr: float) -> str:
    if tr < 5:        return "0000"
    if tr < 10:       return "0001"
    if tr < 14:       return "0010"
    return "0011"


def encode_efficiency(eff: float) -> str:
    if eff < 0.3:     return "0000"
    if eff < 0.6:     return "0001"
    if eff < 0.8:     return "0010"
    return "0011"


def encode_lead_time(lt: float) -> str:
    if lt < 5:        return "0000"
    if lt < 7:        return "0001"
    if lt < 9:        return "0010"
    return "0011"


def build_chromosome_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame matching Table 5.2 structure."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "Item_ID":       row["item_id"],
            "Category":      encode_category(row.get("category", "")),
            "Zone":          encode_zone(row.get("zone", "")),
            "Popularity":    encode_popularity(row.get("item_popularity_score", 0)),
            "Demand":        encode_demand(row.get("daily_demand", 0)),
            "Picking_Time":  encode_picking_time(row.get("picking_time_seconds", 0)),
            "Handling_Cost": encode_handling_cost(row.get("handling_cost_per_unit", 0)),
            "Fulfillment":   encode_fulfillment(row.get("order_fulfillment_rate", 0)),
            "Turnover":      encode_turnover(row.get("turnover_ratio", 0)),
            "Efficiency":    encode_efficiency(row.get("layout_efficiency_score", 0)),
            "Lead_Time":     encode_lead_time(row.get("lead_time_days", 0)),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6. RESULTS & PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(history: list, save_path: str = "pso_convergence.png"):
    plt.figure(figsize=(9, 4))
    plt.plot(history, color="steelblue", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Global-best fitness  f")
    plt.title("PSO Convergence – Warehouse Storage Allocation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConvergence plot saved → {save_path}")


def summarise_allocation(df: pd.DataFrame,
                         slots: np.ndarray,
                         E: np.ndarray,
                         P: np.ndarray,
                         C: np.ndarray,
                         W: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build a result DataFrame: for each item, show its assigned slot and
    the attribute values at that slot.
    """
    result = df[["item_id"]].copy()
    result["assigned_slot"]           = slots
    result["slot_efficiency"]         = E[slots]
    result["slot_picking_time"]       = P[slots]
    result["slot_handling_cost"]      = C[slots]
    if W is None:
        W = np.ones_like(E)
    result["item_contribution"]       = (
        (ALPHA * E[slots] - BETA * (P[slots] + C[slots])) * W
    )
    return result.sort_values("item_contribution", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Warehouse Storage Allocation – PSO Optimiser")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset from '{DATASET_PATH}' …")
    df = load_data(DATASET_PATH)
    print(f"      {len(df)} items loaded.")

    # ── Build binary chromosome table (Table 5.2) ─────────────────────────────
    print("\n[2/5] Building binary chromosome table (Table 5.2) …")
    chrom_df = build_chromosome_table(df)
    chrom_df.to_csv("chromosome_table.csv", index=False)
    print("      Chromosome table saved → chromosome_table.csv")
    print(chrom_df.head(6).to_string(index=False))

    # ── Prepare attribute arrays ───────────────────────────────────────────────
    print("\n[3/5] Preparing fitness-function arrays …")
    if NORMALISE:
        df = normalise_columns(df)
        print("      P and C min-max normalised to [0, 1].")

    # These are the slot attribute arrays (indexed by slot ID = row index)
    E = df["layout_efficiency_score"].values.astype(float)
    P = df["picking_time_seconds"].values.astype(float)
    C = df["handling_cost_per_unit"].values.astype(float)

    # Select item weights for the fitness (prefer demand, else popularity)
    if "daily_demand" in df.columns:
        W = df["daily_demand"].values.astype(float)
    else:
        W = df["item_popularity_score"].values.astype(float)
    # Normalise weights to [0,1] to avoid large scale effects
    W = (W - W.min()) / (W.max() - W.min() + 1e-9)

    print(f"      E  range: [{E.min():.4f}, {E.max():.4f}]")
    print(f"      P  range: [{P.min():.4f}, {P.max():.4f}]")
    print(f"      C  range: [{C.min():.4f}, {C.max():.4f}]")

    # ── Run PSO ───────────────────────────────────────────────────────────────
    print(f"\n[4/5] Running PSO  (particles={N_PARTICLES}, iters={MAX_ITER}, "
          f"α={ALPHA}, β={BETA}) …\n")
    t0 = time.time()
    best_slots, best_fitness, history = run_pso(E, P, C, item_weights=W)
    elapsed = time.time() - t0
    print(f"\n  ✓ PSO finished in {elapsed:.1f}s")
    print(f"  ✓ Best fitness f = {best_fitness:.6f}")

    # ── Report results ────────────────────────────────────────────────────────
    print("\n[5/5] Generating results …")
    result_df = summarise_allocation(df, best_slots, E, P, C, W=W)
    result_df.to_csv("pso_allocation_result.csv", index=False)
    print("      Full allocation saved → pso_allocation_result.csv")

    print("\n  Top 10 items by fitness contribution:")
    print(result_df.head(10).to_string(index=False))

    # Aggregate stats
    total_E = ALPHA * np.sum(E[best_slots] * W)
    total_PC = BETA  * np.sum((P[best_slots] + C[best_slots]) * W)
    print(f"\n  α·ΣEi            = {total_E:.4f}")
    print(f"  β·Σ(Pi+Ci)       = {total_PC:.4f}")
    print(f"  Net fitness f    = {best_fitness:.4f}")

    plot_convergence(history, save_path="pso_convergence.png")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
