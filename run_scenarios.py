from __future__ import annotations

"""
Headless GA scenario runner (Milestone 6)

- Runs the GA/controller logic without CodinGame I/O
- Simulates the environment by applying the selected command using our physics step
- Reports whether we landed, crashed, steps taken, and fuel used

Usage:
  python run_scenarios.py

You can adjust SCENARIOS to add custom terrains and initial states.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

from terrain import Terrain
from physics import State, Gene, simulate_chromosome, step
from ga import GA


@dataclass
class Scenario:
    name: str
    points: List[Tuple[float, float]]  # terrain points
    start: State
    max_turns: int = 400


def make_default_scenarios() -> List[Scenario]:
    # Simple flat ground at y=1000, lander above and slightly to the left
    points1 = [(-1000.0, 1000.0), (2000.0, 1000.0)]
    start1 = State(x=0.0, y=2500.0, vx=0.0, vy=-40.0, fuel=800, angle=0.0, power=0)

    # A flat plateau to the right
    points2 = [(0.0, 1500.0), (1000.0, 1500.0), (2000.0, 1000.0), (3500.0, 1000.0), (5000.0, 1800.0)]
    start2 = State(x=500.0, y=2600.0, vx=30.0, vy=-20.0, fuel=900, angle=0.0, power=0)

    return [
        Scenario("Flat1000_Center", points1, start1, 260),
        Scenario("Plateau_Right", points2, start2, 300),
    ]


def run_one(scn: Scenario) -> None:
    terrain = Terrain.from_points(scn.points)
    ga = GA(pop_size=110, horizon=90, elite_frac=0.16, pm=0.01)
    prev_best: Optional[List[Gene]] = None

    s = scn.start
    total_steps = 0
    t0 = time.perf_counter()

    while total_steps < scn.max_turns:
        # Time-bounded loop similar to controller.py
        start = time.perf_counter()
        deadline = start + 0.09

        # Seed and evaluate
        seed = (prev_best[1:] + [prev_best[-1]]) if prev_best else None
        ga.init_population(seed=seed, seed_copies=max(6, ga.N // 12))
        ga.evaluate(s, terrain)
        while time.perf_counter() < deadline:
            # optional lightweight annealing based on time
            elapsed = time.perf_counter() - start
            total = max(1e-6, deadline - start)
            ga.set_anneal_progress(min(1.0, elapsed / total))
            ga.next_generation()
            ga.evaluate(s, terrain)

        best = ga.best
        if best is None or not best.chrom:
            # emergency: command vertical burn
            cmd = (0.0, 4.0)
        else:
            cmd = best.chrom[0]

        # Apply one second using simulate_chromosome for robust collision detection
        out = simulate_chromosome(s, [cmd], terrain)
        total_steps += 1

        if out.landed:
            dt = time.perf_counter() - t0
            print(f"[OK] {scn.name}: Landed in {total_steps} s, fuel used {out.fuel_used}, time {dt*1000:.1f} ms")
            return
        if out.crashed:
            print(f"[X] {scn.name}: Crashed at step {total_steps}, pos={out.crash_pos}, vx={out.final_state.vx:.1f}, vy={out.final_state.vy:.1f}")
            return

        # Continue with the updated actual state
        s = out.final_state
        # Seed for next turn: shift chromosome
        prev_best = best.chrom if best else None

    print(f"[?] {scn.name}: Reached max_turns without landing (treat as failure)")


def main() -> None:
    scenarios = make_default_scenarios()
    for scn in scenarios:
        run_one(scn)


if __name__ == "__main__":
    main()
