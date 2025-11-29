Break # GA_spec — Genetic Algorithm implementation plan for Mars Lander L2

This document proposes a concrete, from‑scratch implementation plan for a time‑bounded Genetic Algorithm (GA) controller that safely lands the CodinGame “Mars Lander” (Level 2) while minimizing fuel. It aligns with:
- GA_workflow.txt (simulator first; collision vs. landing; evaluation; roulette selection; continuous GA; elitism; time constraints)
- Task.md (physics, constraints, I/O, runtime ≤ 100 ms/turn)

The scope is a Python solution intended for CodinGame, but the structure is modular and can be adapted.


## 1. Goals and success criteria
- Hard constraints on touchdown:
  - Land on the unique flat segment
  - Angle = 0° (upright)
  - |horizontal speed| ≤ 20 m/s; |vertical speed| ≤ 40 m/s
- Primary objective among valid landings: minimize total fuel used
- Controller must operate on‑line with ≤ 100 ms each turn
- Robustness: provide safe fallback commands when the best GA candidate is unsafe

Acceptance criteria:
- Simulator parity: given identical command sequences, our simulator reproduces CodinGame’s end states within negligible tolerance
- During typical maps (official Level 2 cases), the controller finds and executes safe landings; many runs converge to low fuel usage
- Per‑turn runtime within budget on a modest machine; no frame drops due to GA


## 2. High‑level architecture (modules)
- terrain.py
  - Parse surface points; build segments; locate the flat segment index
  - Segment intersection test between the lander path segment (per second) and terrain segments
  - Utilities: flat_xrange, distance to flat segment
- physics.py
  - State, integrator step with rate limits and gravity
  - Per‑second update ordering matching game: apply rate‑limits → compute accel → update velocities → update positions → update fuel
  - Single‑step function and chromosome simulation with early termination on collision
- ga.py
  - Individual/Population data structures; GA engine (evaluate, select, crossover, mutate, elitism, next generation)
  - Continuous representation, roulette selection, arithmetic crossover, per‑gene mutation
- controller.py (CodinGame loop)
  - Receding‑horizon on‑line GA loop every turn
  - Seed population from previous best (shifted), add noise, run generations until deadline, output first gene
  - Safety fallback when best is unsafe near ground


## 3. Data model
- State: x, y, vx, vy, fuel, angle_actual, power_actual
- Gene (continuous GA): (angle_cmd: float in [−90,90], power_cmd: float in [0,4])
  - Stored unclamped; simulator enforces legal ranges and rate limits during application
- Chromosome: list[Gene] of horizon length H (seconds)
- Population: list[Chromosome] of size N
- Outcome: landed?, crashed?, on_flat?, final_state, fuel_used, crash_pos(optional)


## 4. Physics and simulator (must mirror game)
- Gravity g = 3.711 m/s²
- Controls per second; rate limits on actuals:
  - |Δangle| ≤ 15°/s; |Δpower| ≤ 1/s, clamped to [−90,90] and [0,4]
- Acceleration (θ in radians):
  - ax = P · sin(θ)
  - ay = P · cos(θ) − g
- Update (Δt = 1 s):
  - Update actual angle/power toward commanded values (respecting rate limits)
  - Compute accel from actuals
  - vx += ax ; vy += ay
  - x += vx_prev + 0.5 · ax ; y += vy_prev + 0.5 · ay
  - fuel = max(0, fuel − P)
- Collision: after each step, test segment [(x_prev, y_prev) → (x, y)] vs every terrain segment; on first hit, stop and evaluate landing
- Landing validity on flat segment: angle == 0°, |vx| ≤ 20, |vy| ≤ 40
- Note: The simulator does not need sub‑step integration as long as we test the path segment against terrain segments every second as in GA_workflow.txt

Unit tests:
- Replay known traces; verify parity of end state and collision detection
- Edge cases: endpoints, grazing the flat segment, steep walls


## 5. Fitness/evaluation design
- Strict ordering: any successful landing has better score than any crash
- If landed: cost = w_fuel · fuel_used + w_time · t (default w_time ~ 0 to avoid loitering)
- If crashed: cost =
  - w_dist · distance_to_flat
  - + w_h · max(0, |vx| − 20) + w_v · max(0, |vy| − 40)
  - + w_ang · |angle| at impact
  - + w_off · 1{impact not on flat}
- Turn cost into fitness for roulette selection:
  - fitness = 1 / (1 + cost)
  - Optionally use rank scaling when cost ranges vary widely

Default weights (tunable):
- w_fuel = 1.0; w_time = 0.0–0.2; w_dist = 1.0; w_h = 5.0; w_v = 1.0; w_ang = 2.0; w_off = 3000


## 6. Genetic operators and GA loop
- Selection: Roulette wheel
  - Normalize positive fitnesses so they sum to 1
  - Build cumulative array once per generation; select with bisect on uniform [0,1)
- Crossover: Arithmetic (continuous GA)
  - For each gene i, sample α∈[0,1]; children are convex combinations of parents’ gene values
  - Option: single α per chromosome (exploit) vs per gene (explore). Start per gene
- Mutation: Per gene with prob pm ≈ 0.01
  - angle_cmd += Normal(0, σ_angle≈5–10°)
  - power_cmd += Normal(0, σ_power≈0.5–1.0)
  - Clamp to legal ranges after mutation
  - Optional annealing across generations in a turn: decay σ_*
- Elitism: copy top 10–20% directly to next population
- Population replacement: elitism + offspring until N reached


## 7. Online receding‑horizon control (per turn)
1. Set deadline now + 90–95 ms (safety margin for I/O)
2. Build/refresh population:
   - If prev best exists: seed by shifting its chromosome (drop first gene), append reasonable last gene; create several mutated copies
   - Fill remainder with randomized/smoothed chromosomes biased toward feasible rates
3. Evaluate unevaluated individuals; update global best
4. While time remains:
   - next_generation(); evaluate(); keep best
5. Output the first command of the best chromosome (rounded to ints)
6. Roll horizon forward for next turn: prev_best = best shifted
7. Safety fallback: if near ground and vy too negative and best is unsafe → command angle=0, power=4


## 8. Initialization and heuristics
- Random gene sampling ranges initially:
  - angle in [−45, +45] (expand if far lateral travel needed)
  - power in [0, 4] biased to 0–2 to encourage fuel saving
- Optional smoothing of random walks to better match rate limits
- Adaptive horizon H ≈ expected time to land from current altitude; start at 90
- Population size N ≈ 100 (tune vs time budget)


## 9. Parameters and starting defaults
- pop_size N = 100
- horizon H = 90 (60–120 depending on altitude)
- elite_frac = 0.15
- pm = 0.01
- σ_angle = 7°, σ_power = 0.7
- Fitness weights as per Section 5


## 10. Performance considerations
- Use lists of tuples for genes; avoid heavy objects in inner loops
- Trig only once per simulated second per individual; acceptable for N×H within budget
- Early termination when collision occurs
- Build cumulative fitness once per generation; select by bisect (log N)
- Keep outcomes minimal; store full trajectory only for current best if needed for visualization/debug


## 11. Interfaces (proposed file structure)
- terrain.py
  - class Segment; class Terrain(segments, flat_idx)
  - methods: flat_xrange(), intersect(p1, p2) → (hit:bool, seg_idx:int, hit_point:(x,y))
- physics.py
  - dataclass State; dataclass Outcome
  - step(state, angle_cmd, power_cmd) → State
  - simulate_chromosome(s0, chromosome, terrain) → Outcome
- ga.py
  - Gene = tuple[float, float]; Chromosome = list[Gene]
  - class Individual; class GA(pop_size, horizon, ...)
  - init_population(seed, seed_copies=K); evaluate(s0, terrain); next_generation()
- controller.py
  - Integrates with CodinGame I/O; maintains prev_best; emits commands each turn


## 12. Testing strategy
- Simulator parity tests: fixed inputs produce identical outputs vs known references
- Terrain collision tests: segments intersection correctness including edge cases
- Fitness monotonicity: any valid landing scores strictly better than any crash
- On‑line runtime test: measure time per turn; ensure ≤ 100 ms on worst test
- End‑to‑end regression: run GA for multiple seeds on official maps; record success rate and fuel usage distribution


## 13. Milestones and deliverables
1. Physics + terrain utilities complete; unit tests for parity and intersection
2. Basic GA engine with random population; able to simulate and score
3. Online controller loop with deadline; emits plausible commands; fallback present
4. Tuning pass: parameters; mutation schedule; seeding; weights
5. Documentation updates: README usage section; parameters and tips

Deliverable for this task: this GA_spec.md with an actionable plan consistent with GA_workflow.txt and Task.md.


## 14. Notes and rationale (mapping to GA_workflow.txt)
- Simulator first: avoid misleading fitness (GA_workflow lines 1–7)
- Clean representation: genes/chromosomes/population (lines 7–31)
- Collision vs landing separation (lines 34–37)
- Evaluation focused on distance, speeds, angle, off‑flat penalty (lines 38–43)
- Selection (roulette), crossover, mutation (lines 44–77)
- Continuous GA with arithmetic crossover, pm ≈ 0.01; elitism 10–20% (lines 85–101)
- Time constraints: cumulative fitness prep and bisect; on‑line, per‑turn evolution (lines 102–171)


## 15. Risk management and fallbacks
- If GA fails to produce a safe candidate near ground, enforce emergency burn (angle=0, power=4)
- If terrain intersection imprecision causes early/late hits, increase geometric robustness (e.g., compute exact intersection point instead of approximating at step end)
- If time budget is tight, reduce N or H, or lower mutation to speed selection/crossover


## 16. Future improvements (optional)
- Rank‑based or tournament selection to stabilize scaling
- Adaptive mutation/σ schedule driven by population diversity
- Two‑phase objective: once landings become common, increase w_fuel to focus on fuel minimization
- Cache sin/cos for integer angles −90..90 if profiling shows trig as hotspot
