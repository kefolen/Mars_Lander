# Mars Lander GA — Genetic Algorithm Controller (Level 2)

This repository contains a from‑scratch Genetic Algorithm (GA) solution for CodinGame “Mars Lander” Level 2. It follows the plan in GA_overview.md and the milestones in GA_milestones.md.

Milestone 6 adds tests, a headless scenario runner, and packaging into a single‐file submission.

## Quick start
- Run tests:
  - `python -m unittest -v`
- Run headless scenarios (local, no CodinGame I/O):
  - `python run_scenarios.py`
- Run online controller on CodinGame (copy/paste the single file built below, or run locally and feed input):
  - `python controller.py`

## Project layout
- terrain.py — terrain segments, flat detection, intersection, distances
- physics.py — rate‑limited controls, gravity, per‑second integrator, chromosome simulation
- evaluation.py — landing/crash cost and fitness mapping
- ga.py — GA core: init, evaluation, roulette selection, arithmetic crossover, mutation, elitism
- controller.py — time‑bounded online loop using GA (receding horizon, safety fallback)
- run_scenarios.py — simple headless runner for deterministic local checks
- tests/ — unit tests for terrain, physics, and evaluation
- pack_submission.py — packs modules into one file for CodinGame

## Packaging: single‑file submission (.py only)
Per GA_milestones.md, the final runnable submission must be a single .py file. Build it with:

```
python pack_submission.py
```

The script writes `dist/submission_ga.py`. Submit that single file to CodinGame. It contains terrain, physics, evaluation, GA, and controller sections with intra‑project imports stripped.

## Notes & tips
- Time budget is ~100 ms per turn. The controller adapts population and horizon to altitude and uses annealed mutation as the deadline approaches.
- Safety fallback engages near ground if descent is too fast.
- Tuning knobs: GA parameters in ga.py and adaptive logic in controller.py; fitness weights in evaluation.FitnessWeights.

See GA_overview.md and GA_milestones.md for the full design and milestone breakdown.