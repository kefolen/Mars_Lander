# GA Milestones — Sequential Implementation Plan

This document breaks the Mars Lander GA project into large, sequential milestones. Each milestone has a goal, key tasks, deliverables, and acceptance criteria. Complete them in order.


## Milestone 1 — Terrain + Physics Simulator Parity
- Goal: Recreate CodinGame’s per‑second physics and terrain representation with collision detection that matches the game.
- Key tasks:
  - Terrain: parse surface points, build segments, find the unique flat segment.
  - Physics step: apply angle/power rate limits (±15°/s, ±1/s), compute ax, ay, update velocities and positions, subtract fuel.
  - Segment intersection: detect collision of path segment (prev→next pos) with any terrain segment.
  - Landing validator: check on‑flat, angle==0°, |vx|≤20, |vy|≤40 at touchdown.
- Deliverables: `terrain.py`, `physics.py` with `State`, `step()`, `simulate_chromosome()`.
- Acceptance: Given fixed command sequences, simulator matches known CodinGame traces within negligible tolerance; intersection works on edge cases.


## Milestone 2 — Evaluation/Scoring Function
- Goal: Turn simulation outcomes into a scalar cost/fitness that strictly prefers safe landings and guides the search when crashing.
- Key tasks:
  - Cost for landed: prioritize minimizing fuel (optional small time term).
  - Cost for crash: distance to flat segment, penalties for |vx|, |vy|, angle at impact; heavy penalty for off‑flat crashes.
  - Fitness mapping: fitness = 1/(1+cost); optional rank scaling.
- Deliverables: Cost/fitness utilities integrated into simulation outcomes.
- Acceptance: Any valid landing scores strictly better than any crash. Sanity tests show intuitive ordering (safer, closer, slower → better).


## Milestone 3 — GA Core Engine
- Goal: Implement the GA with continuous genes and modern operators.
- Key tasks:
  - Data: Gene=(angle_cmd, power_cmd), Chromosome=list[Gene], Individual with fitness/cost/outcome.
  - Initialization: random/smoothed chromosomes of length H.
  - Selection: roulette with cumulative fitness and bisect lookup.
  - Crossover: arithmetic (per‑gene α∈[0,1]).
  - Mutation: per‑gene probability pm≈0.01 with small Gaussian noise; clamp after mutation.
  - Elitism: keep top 10–20%.
- Deliverables: `ga.py` with `GA.init_population()`, `GA.evaluate()`, `GA.next_generation()`.
- Acceptance: On a fixed seed, fitness improves over generations on sample maps; evaluation and replacement run without errors.


## Milestone 4 — Online Receding‑Horizon Controller
- Goal: Time‑bounded, per‑turn GA loop that outputs the first command of the best chromosome.
- Key tasks:
  - Deadline handling: run within ~90–95 ms/turn.
  - Seeding: shift previous best chromosome by one step, append one gene; add a few mutated copies.
  - Loop: evaluate population, iterate GA generations until deadline, keep global best.
  - Output: round first gene to ints and print per CodinGame protocol.
- Deliverables: `controller.py` wired to I/O; basic safety fallback (angle=0, power=4 when near ground and descending too fast).
- Acceptance: Runs within time budget on official Level 2 cases; emits plausible commands; no I/O timeouts.


## Milestone 5 — Tuning, Robustness, and Performance
- Goal: Make landings reliable and reduce fuel usage while staying within the time budget.
- Key tasks:
  - Parameter tuning: N, H, elite_frac, pm, σ values, fitness weights.
  - Heuristics: initialization bias, mild smoothing, mutation annealing.
  - Micro‑optimizations: early termination on collision, single cumulative fitness build, reduce allocations.
  - Safety: robust fallback triggers and thresholds.
- Deliverables: Tuned defaults in GA and controller; optional profiling notes.
- Acceptance: High success rate across official maps; consistent ≤100 ms per turn; noticeable reduction in fuel vs. untuned baseline.


## Milestone 6 — Testing, Docs, and Packaging
- Goal: Solidify quality and provide usage guidance.
- Key tasks:
  - Unit tests for simulator parity and intersection edge cases.
  - Scenario scripts to run GA headless on maps and collect metrics.
  - Documentation: update README with how to run, parameters, and tips; link GA_spec and this milestone plan.
- Deliverables: Tests (where applicable), updated README, notes on results; submission packaging as a single .py file (e.g., mars_lander.py) containing the entire solution.
- Acceptance: Tests pass locally; README instructs how to run and tweak; reproducible landing results on sample maps.


Dependencies
- M2 depends on M1.
- M3 depends on M2.
- M4 depends on M3.
- M5 depends on M4.
- M6 depends on all previous milestones.

Notes
- This plan aligns with GA_spec.md and GA_workflow.txt. It intentionally groups work into large, sequential chunks to de‑risk the implementation and enable iterative progress.