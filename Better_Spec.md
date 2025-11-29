# Better_Spec.md — Unified Controller Spec (≤ 5 chunks)

## Purpose
Safely land on the flat segment, upright, with minimal fuel. Combine a lightweight candidate generator and waypoint follower for directional guidance with control‑point Simulated Annealing (SA) refinement under a 100 ms/turn budget. Keep a conservative emergency fallback.

## Models and Constraints
- State S = {x, y, vx, vy, fuel, angle, power}; terrain is a polyline with a unique flat [xL, xR] at y_flat.
- Physics per second: ax = P · sin(θ), ay = P · cos(θ) − g, with g = 3.711.
- Rate limits: |Δθ| ≤ 15°/s, |ΔP| ≤ 1/s; θ ∈ [−90°, +90°], P ∈ [0..4].
- Success: on flat, θ = 0°, |vx| ≤ 20, |vy| ≤ 40 at touchdown.
- Objective: among successful landings, minimize fuel used; all crashes are strictly worse than any success.
- All code should be contained in 1 file
## Core Interfaces
- ramp_enforce(plan, angle0, power0) -> plan_realizable (per‑sec commands)
- simulate(state0, plan_realizable, terrain) -> result {success, touch_state, fuel_used, cost_aux}
- tilt_allow(state, terrain) -> theta_max_abs
- lateral_stopping(vx, theta_allow) -> dx_stop
- vertical_stopping(vy) -> dy_stop
- generate_candidates(state, terrain, H) -> List[ControlPointPlan]
- expand_cp(cp_plan, H) -> per_sec_plan
- follow_waypoints(state, terrain, waypoints, H) -> per_sec_plan
- score(state0, plan) -> cost
- mutate_cp(cp_plan) -> cp_plan'
- optimize_multi_seed(state0, seeds, budget_ms) -> best_cp_plan
- emergency_policy(state) -> optional (θ, P)

## Cost Function (shaped for capture reliability and fuel economy)
- If success: cost = fuel_used + w_time · T_land
- Else: cost = C_CRASH + w_zone · dist_to_center + w_edge · outside_flat + w_ang · |θ_touch| + w_vx · excess_vx^2 + w_vy · excess_vy^2 + w_y · max(0, y_flat − y_touch)
- Mid‑flight penalties (applied throughout sim):
  - w_oob · (distance_outside [0,6999]) if any
  - w_ground · max(0, h_ground_margin − altitude_local)
  - small w_ang to not over‑penalize banking
  - bang‑bang bias: tiny per‑step penalty when P ∈ {1,2,3}
  - ETA synchronization: small w_eta · |eta_x − eta_y| (when both defined)
- Suggested starts: C_CRASH ≫ 1e5; w_oob 800–1200; w_ground 60–100; w_zone 100–160; w_edge 160–220; w_ang 20–40; w_vx, w_vy 1–10; w_time 0–0.1; w_eta 10–30.

---

## Implementation Chunk 1 — Simulator, Safety Primitives, and Guards
Scope:
- Physics‑correct simulator, collision with segment sweep, flat touchdown check.
- Ramping constraints.
- tilt_allow(state, terrain): compute max |θ| such that vertical feasibility remains: Pmax · cos(θ) − g ≥ −a_down_allow(h, vy), with a_down_allow(h, vy) increasing with altitude and decreasing with faster descent. Clamp to ±90° and respect rotate‑time margin for returning to 0° by flare time.
- Stopping helpers: lateral_stopping(vx, θ_allow), vertical_stopping(vy).
- Safety fallback: if altitude is low versus vertical_stopping(vy) plus rotate‑time margin, override with (θ=0°, P=4) until safe; enforce OOB early exit.
- Lightweight runtime guards that don’t fight the planner: edge pushback, corridor lock at low altitude, and early horizontal bias when far and |vx| is very small. All use tilt_allow().

Acceptance:
- Kinematics match expectations; no false flat success at edges; OOB classified early.
- With fallback only, lander avoids late crashes in low‑altitude emergencies.

---

## Implementation Chunk 2 — Candidate Library and Waypoint Follower
Scope:
- Control‑point plan structure: Cp = (θ, P, dur); K ≤ 10.
- Candidate types (6–12 total):
  - Two‑burn (push → horizontal brake → flare)
  - Glide‑slope capture → flare
  - Three‑impulse bang‑bang (push, coast, counter‑push, flare)
  - Corridor‑first (ensure entry into [xL+margin, xR−margin] before h_corr_enter)
- Expand candidates to per‑sec using expand_cp, then ramp_enforce, clamp angles via tilt_allow().
- Waypoints (flip_point, flare_point), and a follower:
  - Lateral: pick x_target (center or inside corridor), desired vx using proportional–damping: vx_des = clamp(kx · (x_target − x) − kv · vx, vmax(y)).
  - Compute minimal P for vertical feasibility, then set θ = −arcsin(clamp(vx_dot / P, −1, 1)).
  - Vertical: during flare, track vy_des(y) toward −40 near y_flat with minimal P.
- Evaluate all candidates with score; keep top few for refinement.

Acceptance:
- From diverse starts, a candidate reliably moves toward platform/corridor; follower tracks without long hovering.

---

## Implementation Chunk 3 — SA with Control Points, Multi‑Seed, and Warm Start
Scope:
- Plan representation upgrade: K control points, mutate_cp operators (±5..15° on θ, ±1 on P, ±1..3 s on dur, occasional neighbor swaps).
- Plan repair before scoring: force last N seconds upright; verify vertical stopping distance is reachable with ramped P.
- Multi‑seed refinement: select top 2–3 seeds from Chunk 2; split the per‑turn budget round‑robin (e.g., 3 × 25 ms), keep best incumbent.
- Temperature schedule: T0 = 0.05..0.2 × best_cost, alpha 0.98..0.995.
- Rolling horizon: shift previous best by one step each turn; append neutral tail; ramp_enforce before SA; output first command.
- Constant‑time mini beam pre‑filter (optional): evaluate 3–5 ultra‑short (2–3 s) discrete action stubs for initial direction sanity before candidate generation.
- Low‑fuel mode policy: reduce K and candidate count; increase bang‑bang bias.

Acceptance:
- Best candidate’s cost improves within budget versus no SA; visible fuel savings compared to follower‑only.
- Iterations stay within ~80–90 ms total, no timeouts.

---

## Implementation Chunk 4 — Cost Shaping and Runtime Guards Tuning
Scope:
- Integrate mid‑flight penalties (w_oob, w_ground, ETA sync, bang‑bang bias) and corridor decay with altitude.
- Synchronize eta_x and eta_y via small penalty; verify it reduces dithering/hover.
- Tune guards so they rarely interrupt good plans but reliably avoid OOB and edge misses.
- Parameter baselines: corr_margin 80–120, alt_corr_thresh ≈ 1200 m, a_down_allow(h): 1.0 low alt → 3.0 high alt.

Acceptance:
- Fewer near‑edge crashes; earlier coasts and shorter burns on successes; median fuel improves ≥ 10% vs baseline.

---

## Implementation Chunk 5 — Testing, Profiling, and Validation
Scope:
- Deterministic tests: official Level 2 maps + adversarial starts (far left/right, high |vx|, low altitude over slopes, low fuel).
- Logging (stderr): candidate type, SA iterations, best cost, emergency triggers, OOB events.
- Profiling: count simulate() steps/turn; target thousands of evaluations within ~80–90 ms.
- Validation split: keep a held‑out set of terrains/seeds; avoid overfitting weights.
- Tuning loop: adjust K, alpha, seed count, weights; verify no regression in landing rate.

Acceptance:
- Meets time budget; landing rate ≥ 98% on feasible starts; improved fuel metrics vs baseline across maps.

---

## Notes and Practical Limits
- Hard caps: candidates ≤ 12; control points K ≤ 12.
- Always clamp via tilt_allow() and re‑check vertical_stopping() with upright margin before committing to plans.
- Keep emergency fallback as last resort, not a primary controller.

## Why this spec is better
- Merges the clarity of the simple SA spec (simulator, SA loop) with the operational sophistication of improvements (candidate library, follower, control‑point SA, cost shaping, runtime guards).
- Keeps the implementation plan to five digestible chunks, each with acceptance criteria, so you can deliver iteratively while respecting the 100 ms budget.
- Adds ideas like ETA synchronization, mini beam pre‑filter, and low‑fuel mode to improve reliability and fuel economy without heavy complexity.
