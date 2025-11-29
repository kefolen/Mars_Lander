from __future__ import annotations

"""
Online receding-horizon controller (Milestone 4)

- Reads CodinGame-style input from stdin
- Runs a time-bounded GA loop each turn
- Outputs the first command of the current best chromosome
- Uses a basic emergency fallback near ground if best looks unsafe

This module relies on:
- terrain.Terrain
- physics.State
- ga.GA
- evaluation.FitnessWeights (optional tuning)

All printing is to stdout per CodinGame protocol: two integers per line: angle power
"""

import sys
import time
import math
from typing import List, Tuple, Optional

from terrain import Terrain
from physics import State, Gene
from ga import GA


def _read_ints() -> List[int]:
    return list(map(int, sys.stdin.readline().strip().split()))


def _shift_seed(chrom: List[Gene], H: int) -> List[Gene]:
    """Drop the first gene and append a copy of the last; ensure length H."""
    if not chrom:
        return [(0.0, 0.0) for _ in range(H)]
    shifted = list(chrom[1:])
    if not shifted:
        shifted = [chrom[-1]]
    while len(shifted) < H:
        shifted.append(shifted[-1])
    if len(shifted) > H:
        shifted = shifted[:H]
    return shifted


def _clamp_cmd(angle: int, power: int) -> Tuple[int, int]:
    if angle < -90:
        angle = -90
    elif angle > 90:
        angle = 90
    if power < 0:
        power = 0
    elif power > 4:
        power = 4
    return angle, power


def _need_emergency(y: int, vy: int, y_flat: float, best_is_safe: bool, fuel: int) -> bool:
    # Robust safety policy: trigger a vertical burn slightly earlier and for steeper descents
    # to ensure rate limits allow recovery before touchdown.
    altitude = y - y_flat
    near_ground = altitude < 1200
    too_fast = vy < -37
    has_fuel = fuel > 0
    return (not best_is_safe) and near_ground and too_fast and has_fuel


def _adaptive_ga_params(y: float, y_flat: float, vy: float) -> Tuple[int, int]:
    """Choose (population, horizon) based on altitude and descent rate."""
    altitude = max(0.0, y - y_flat)
    # Horizon scales with altitude; ensure within [60, 120]
    base_h = int(min(120, max(60, math.ceil(altitude / 35.0) + 30)))
    # Population size modest near ground to keep time; larger when high
    if altitude < 900:
        pop = 90
    elif altitude < 1800:
        pop = 110
    else:
        pop = 130
    return pop, base_h


def main() -> None:
    # --- Initialization: parse terrain ---
    first = sys.stdin.readline()
    if not first:
        return
    try:
        surface_n = int(first.strip())
    except ValueError:
        # Allow running in local mode without CodinGame I/O
        return

    points: List[Tuple[float, float]] = []
    for _ in range(surface_n):
        xi, yi = _read_ints()
        points.append((float(xi), float(yi)))

    terrain = Terrain.from_points(points)

    # --- GA configuration (tuned defaults) ---
    ga = GA(pop_size=110, horizon=90, elite_frac=0.16, pm=0.01)
    prev_best_chrom: Optional[List[Gene]] = None

    # --- Per-turn loop ---
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        vals = list(map(int, line.strip().split()))
        if len(vals) != 7:
            # Ignore malformed line
            continue
        X, Y, hSpeed, vSpeed, fuel, rotate, power = vals

        # Invert angle sign on input to align internal physics convention
        s0 = State(
            x=float(X),
            y=float(Y),
            vx=float(hSpeed),
            vy=float(vSpeed),
            fuel=int(fuel),
            angle=float(-rotate),
            power=int(power),
        )

        # Deadline with small safety margin for I/O
        start = time.perf_counter()
        deadline = start + 0.095

        # Adapt GA params to current altitude
        pop, horizon = _adaptive_ga_params(s0.y, terrain.y_flat, s0.vy)
        ga.update_params(pop_size=pop, horizon=horizon)

        # Seed population from previous best by shifting one step
        seed = _shift_seed(prev_best_chrom, ga.H) if prev_best_chrom else None
        seed_copies = max(6, ga.N // 12)  # scale with population size
        ga.init_population(seed=seed, seed_copies=seed_copies)

        # Evaluate initial population
        ga.evaluate(s0, terrain)

        # Iterate generations until deadline with annealing
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            # Anneal mutation scale as we approach deadline
            total = max(1e-6, deadline - start)
            progress = min(1.0, max(0.0, (now - start) / total))
            ga.set_anneal_progress(progress)
            ga.next_generation()
            ga.evaluate(s0, terrain)

        best = ga.best

        # Determine output command
        if best is None or not best.chrom:
            out_angle, out_power = 0, 4  # emergency fallback if GA failed
            best_is_safe = False
        else:
            # Round the first gene to integer commands
            out_angle = int(round(best.chrom[0][0]))
            out_power = int(round(best.chrom[0][1]))
            out_angle, out_power = _clamp_cmd(out_angle, out_power)
            best_is_safe = bool(best.outcome and best.outcome.landed)

        # Emergency burn near ground if descending too fast and best not safe
        if _need_emergency(Y, vSpeed, terrain.y_flat, best_is_safe, fuel):
            out_angle, out_power = 0, 4

        # Emit command (invert angle back to game convention)
        sys.stdout.write(f"{-out_angle} {out_power}\n")
        sys.stdout.flush()

        # Roll horizon: keep shifted best chromosome as seed for next turn
        prev_best_chrom = _shift_seed(best.chrom, ga.H) if best and best.chrom else None


if __name__ == "__main__":
    main()
