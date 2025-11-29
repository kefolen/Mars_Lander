# Mars Lander — Fuel‑Minimizing Landing (Task Brief)

This project targets CodinGame “Mars Lander” Level 2: safely land on the unique flat segment while minimizing total fuel usage.

Key points
- World: 7000 m wide (x ∈ [0,6999]), 3000 m high (y ∈ [0,2999])
- Surface: polyline with a single flat segment of width ≥ 1000 m at height y_flat
- Physics: no atmosphere; gravity g = 3.711 m/s²
  - Thrust power P ∈ {0,1,2,3,4} produces acceleration magnitude P m/s²
  - Fuel consumption equals thrust power per second
- Controls (per second):
  - Angle θ ∈ [−90°, +90°]
  - Power P ∈ [0, 4]
  - Rate limits: |Δθ| ≤ 15°/s, |ΔP| ≤ 1/s (applied to actual values each step)
- Success (on touchdown):
  - Lander is on the flat segment
  - Angle = 0° (upright)
  - |horizontal speed| ≤ 20 m/s
  - |vertical speed| ≤ 40 m/s
- Objective: minimize total fuel used subject to a successful landing (hard constraint). If a crash happens, it is considered strictly worse than any successful landing.
- I/O (game loop):
  - Initialization: surface point count and coordinates forming the terrain polyline
  - Each turn: X Y hSpeed vSpeed fuel rotate power
  - Output each turn: desired rotate (angle°), desired power [0..4]; the simulator applies rate limits vs. the previous actual values
- Runtime budget: ≤ 100 ms per turn

Notes
- Angle is measured from vertical; θ = 0° means thrust is purely vertical. In the simulator, horizontal and vertical accelerations are typically:
  - ax = P · sin(θ)
  - ay = P · cos(θ) − g
  (Use radians inside the implementation.)
- Because of rate limits, planning must ensure angle returns to 0° before touchdown and power can ramp to needed values in time.
- A practical controller uses receding‑horizon optimization (e.g., simulated annealing) with a safety fallback (emergency burn) for robustness.
