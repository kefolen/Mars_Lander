from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from terrain import Terrain
from physics import State, Outcome, Gene, simulate_chromosome


@dataclass(frozen=True)
class FitnessWeights:
    """
    Weights for cost and fitness computation.
    Defaults align with GA_overview.md and enforce: any crash is strictly worse
    than any valid landing (via base_crash).
    """
    # Success (landing) terms
    w_fuel: float = 1.0
    w_time: float = 0.0  # optional; not used unless time provided

    # Crash guidance terms
    w_dist: float = 1.0
    w_h: float = 5.0
    w_v: float = 1.0
    w_ang: float = 2.0
    w_off: float = 3000.0

    # Strict dominance: any crash cost >= base_crash
    base_crash: float = 100000.0


def compute_cost(out: Outcome, terrain: Terrain, weights: FitnessWeights = FitnessWeights()) -> float:
    """
    Convert a simulation Outcome to a scalar cost.
    Lower is better. Any successful landing strictly dominates any crash.
    """
    if out.landed:
        # Primary objective among landings: minimize fuel used (optionally time)
        cost = weights.w_fuel * float(out.fuel_used)
        # If time term is desired in the future, it can be added here when Outcome carries t
        return cost

    # Crash: guide towards feasible touchdown on flat with safe speeds and angle
    st = out.final_state
    d = terrain.distance_to_flat(st.x, st.y)
    hpen = max(0.0, abs(st.vx) - 20.0)
    vpen = max(0.0, abs(st.vy) - 40.0)
    apen = abs(st.angle)
    off = 0.0 if out.on_flat else 1.0

    cost = (
        weights.base_crash
        + weights.w_dist * d
        + weights.w_h * hpen
        + weights.w_v * vpen
        + weights.w_ang * apen
        + weights.w_off * off
    )
    return float(cost)


def fitness_from_cost(cost: float) -> float:
    """Map cost to positive fitness for roulette selection."""
    return 1.0 / (1.0 + float(cost))


def evaluate_sequence(
    s0: State,
    chromosome: List[Gene],
    terrain: Terrain,
    weights: FitnessWeights = FitnessWeights(),
) -> Tuple[Outcome, float, float]:
    """
    Convenience function: simulate a chromosome, compute cost and fitness.
    Returns: (outcome, cost, fitness)
    """
    out = simulate_chromosome(s0, chromosome, terrain)
    cost = compute_cost(out, terrain, weights)
    fit = fitness_from_cost(cost)
    return out, cost, fit
