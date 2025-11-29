"""
Chunk 1 implementation from Better_Spec.md — Simulator, Safety Primitives, and Guards

All functionality is contained in this single file, as requested in the spec.

Provided primitives:
- ramp_enforce(plan, angle0, power0)
- simulate(state0, plan_realizable, terrain)
- tilt_allow(state, terrain)
- lateral_stopping(vx, theta_allow)
- vertical_stopping(vy)
- emergency_policy(state, terrain)
- apply_runtime_guards(state, desired_cmd, terrain)

Notes:
- Angles are in degrees for inputs/outputs; internal trig uses radians.
- This module focuses on correctness and robustness for Chunk 1; optimization layers
  (candidate generation, SA, cost shaping) are out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import random
import time
import sys
import os

# --- Lightweight logging/profiling (Chunk 5) ---
_PROFILE = {
    'simulate_calls': 0,
    'sim_steps': 0,
    'oob_events': 0,
}
_LAST_STATS = {
    'sa_iters': 0,
    'best_cost': None,
    'picked': [],
    'low_fuel': False,
    'best_name': None,
    'emerg_used': False,
}

def profile_reset():
    _PROFILE['simulate_calls'] = 0
    _PROFILE['sim_steps'] = 0
    _PROFILE['oob_events'] = 0


def profile_note(res=None):
    try:
        if res is None:
            _PROFILE['simulate_calls'] += 1
        else:
            _PROFILE['sim_steps'] += int(getattr(res, 'steps_simulated', 0) or 0)
            if getattr(res, 'oob', False):
                _PROFILE['oob_events'] += 1
    except Exception:
        pass


def log_err(msg: str):
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass

# --- Constants (CodinGame Mars Lander Level 2) ---
G = 3.711  # m/s^2
P_MIN = 0
P_MAX = 4
THETA_MIN = -90.0
THETA_MAX = 90.0
DTHETA_MAX = 15.0  # deg per second
DPOWER_MAX = 1  # per second
WORLD_X_MIN = 0.0
WORLD_X_MAX = 6999.0
WORLD_Y_MIN = 0.0
WORLD_Y_MAX = 2999.0

# Success constraints
MAX_VX_LAND = 20.0
MAX_VY_LAND = 40.0

# Landing proximity error window for |vx| near touchdown (do NOT change success limit)
VX_LAND_PROX_MARGIN = 2.5  # m/s margin below MAX_VX_LAND for proximity checks
VX_NEAR_LAND = max(0.0, MAX_VX_LAND - VX_LAND_PROX_MARGIN)
# Distance-based near-landing thresholds (to center of flat)
DIST_NEAR_LAND = 1500.0       # start near-landing behavior within this horizontal distance to flat center
DIST_NEAR_LAND_TIGHT = 800.0  # tighter behavior very close to center

# --- Chunk 4: Cost shaping weights and guard tuning params ---
# Crash shaping (touchdown)
C_CRASH = 1_000_000.0
W_ZONE = 130.0      # distance to center of flat
W_EDGE = 190.0      # off-flat distance
W_ANG_TOUCH = 30.0  # |theta| at touch
W_VX = 5.0          # excess vx^2
W_VY = 5.0          # excess vy^2
W_Y_BELOW = 100.0   # below-flat height
OOB_HARD_PEN = 200_000.0

# Success shaping
W_TIME = 0.0        # no time penalty on successful landings

# Mid-flight penalties (summed over steps)
W_OOB = 1000.0      # distance outside world bounds
W_GROUND = 80.0     # deficit to ground clearance margin
W_ANG_MID = 0.3     # small angle bias mid-flight (per degree-second)
W_BANG = 0.08       # small bias against hovering at intermediate powers
W_ETA = 30.0        # stronger sync between vertical/horizontal ETAs
W_CORR = 110.0      # corridor penalty with altitude decay
W_VX_MID = 8.0      # mid-flight |vx| above envelope penalty near flat altitude
W_EDGE_BAND_MID = 160.0  # penalty for flying inside a 0-20 m edge band over the flat at low altitude

# Geometry/guard baselines
CORR_MARGIN = 100.0
ALT_CORR_THRESH = 1200.0  # altitude below which corridor weight rises
H_GROUND_MARGIN = 80.0    # desired clearance to ground


# --- Data classes ---
@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    fuel: float
    angle: float  # degrees (actual)
    power: int    # actual [0..4]


@dataclass
class Command:
    angle: float  # desired angle for this second (deg)
    power: int    # desired power for this second [0..4]


@dataclass
class SimulationResult:
    success: bool
    touch_state: Optional[State]
    fuel_used: float
    steps_simulated: int
    oob: bool
    crash_reason: Optional[str]
    cost_aux: dict = None  # mid-flight penalty aggregates for cost shaping


@dataclass
class FlatInfo:
    xL: float
    xR: float
    y: float
    seg_index: int  # index of segment (start point index) that is flat


class Terrain:
    """Terrain as polyline points [(x0,y0),(x1,y1),...]. Exactly one flat segment exists."""
    def __init__(self, points: List[Tuple[float, float]]):
        assert len(points) >= 2
        self.points: List[Tuple[float, float]] = points
        self._flat = self._find_flat()

    @property
    def flat(self) -> FlatInfo:
        return self._flat

    def _find_flat(self) -> FlatInfo:
        best = None
        for i in range(len(self.points) - 1):
            (x0, y0) = self.points[i]
            (x1, y1) = self.points[i + 1]
            if abs(y1 - y0) < 1e-9:
                # flat segment
                xL, xR = (x0, x1) if x0 <= x1 else (x1, x0)
                best = FlatInfo(xL=xL, xR=xR, y=y0, seg_index=i)
                break
        if best is None:
            raise ValueError("Terrain must have a flat segment")
        return best

    def ground_y_at(self, x: float) -> float:
        """Returns ground y at given x by locating containing segment and interpolating."""
        # Clamp x inside world bounds
        x = max(min(x, WORLD_X_MAX), WORLD_X_MIN)
        for i in range(len(self.points) - 1):
            (x0, y0) = self.points[i]
            (x1, y1) = self.points[i + 1]
            if x0 <= x <= x1 or x1 <= x <= x0:
                # interpolate
                if abs(x1 - x0) < 1e-9:
                    return max(y0, y1)
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        # If outside provided segments due to malformed data, fallback to last segment y
        return self.points[-1][1]

    def iter_segments(self):
        for i in range(len(self.points) - 1):
            yield i, self.points[i], self.points[i + 1]


# --- Utility functions ---
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def sign(v: float) -> int:
    return -1 if v < 0 else 1 if v > 0 else 0


# --- Ramping constraints ---
def ramp_enforce(plan: List[Command], angle0: float, power0: int) -> List[Command]:
    """Apply rate limits to desired plan, yielding realizable per-second commands.

    - |Δθ| ≤ 15°/s, |ΔP| ≤ 1/s
    - θ ∈ [−90°, +90°], P ∈ [0..4]
    """
    out: List[Command] = []
    theta = clamp(angle0, THETA_MIN, THETA_MAX)
    power = int(clamp(power0, P_MIN, P_MAX))
    for cmd in plan:
        target_theta = clamp(cmd.angle, THETA_MIN, THETA_MAX)
        dtheta = clamp(target_theta - theta, -DTHETA_MAX, DTHETA_MAX)
        theta = clamp(theta + dtheta, THETA_MIN, THETA_MAX)

        target_power = int(clamp(cmd.power, P_MIN, P_MAX))
        dp = target_power - power
        if dp > DPOWER_MAX:
            dp = DPOWER_MAX
        elif dp < -DPOWER_MAX:
            dp = -DPOWER_MAX
        power = int(clamp(power + dp, P_MIN, P_MAX))

        out.append(Command(angle=theta, power=power))
    return out


# --- Stopping helpers ---
def vertical_stopping(vy: float) -> float:
    """Return vertical stopping distance needed to nullify vertical velocity vy (m).

    Assumes best-case vertical deceleration using P=4 and angle=0.
    ay_max = Pmax - g = 4 - 3.711 = 0.289 m/s^2 upward.
    If vy is downward (negative), distance needed is vy^2 / (2*ay_max).
    If ay_max <= 0, return +inf.
    """
    ay_max = P_MAX - G
    if ay_max <= 0:
        return float('inf')
    if vy >= 0:
        return 0.0
    return (vy * vy) / (2.0 * ay_max)


def lateral_stopping(vx: float, theta_allow: float) -> float:
    """Return horizontal stopping distance needed to nullify vx using max thrust and tilt limit.

    ax_max = Pmax * sin(theta_allow)
    """
    th = abs(theta_allow) * math.pi / 180.0
    ax_max = P_MAX * math.sin(th)
    if ax_max <= 1e-9:
        return float('inf')
    return (vx * vx) / (2.0 * ax_max)


# --- Tilt allowance ---
def _a_down_allow(altitude: float, vy: float) -> float:
    """Allowable net downward acceleration magnitude (m/s^2).

    Heuristic for Chunk 4 tuning: ~1.0 near ground → ~3.0 high up; reduce allowance when |vy| large.
    """
    # Base on altitude (linear 300→1800m)
    if altitude <= 300:
        base = 1.0
    elif altitude >= 1800:
        base = 3.0
    else:
        base = 1.0 + (altitude - 300) * (3.0 - 1.0) / (1800 - 300)

    # Adjust for fast descent: reduce allowance (demand more vertical thrust)
    speed_factor = clamp(1.0 - abs(vy) / 120.0, 0.5, 1.0)
    return clamp(base * speed_factor, 1.0, 3.2)


def tilt_allow(state: State, terrain: Terrain) -> float:
    """Compute maximum absolute tilt (degrees) allowed by vertical feasibility and rotate-time margin.

    Condition: Pmax * cos(theta) - g >= -a_down_allow(h, vy)
      -> cos(theta) >= (g - a_allow)/Pmax

    Also reserve enough time to return to 0° before flare by limiting |theta| <= 15° * t_flare_est.
    """
    flat = terrain.flat
    # Corridor-aware altitude basis: use flat altitude inside corridor; local ground elsewhere
    margin = max(60.0, CORR_MARGIN * 0.6)
    in_corridor = (flat.xL + margin) <= state.x <= (flat.xR - margin)
    h_basis = max(0.0, state.y - (flat.y if in_corridor else terrain.ground_y_at(state.x)))

    a_allow = _a_down_allow(h_basis, state.vy)
    rhs = (G - a_allow) / P_MAX
    rhs = clamp(rhs, -1.0, 1.0)
    # For safety, do not allow negative RHS to inflate theta (cos(theta) >= negative means up to 180°)
    rhs = max(rhs, 0.0)
    theta_vert_limit_rad = math.acos(rhs)
    theta_vert_limit_deg = math.degrees(theta_vert_limit_rad)
    theta_vert_limit_deg = clamp(theta_vert_limit_deg, 0.0, 90.0)

    # Rotate-time margin: estimate time to flare (time to need to be upright)
    # Use stopping distance and current descent rate
    dy_stop = vertical_stopping(state.vy)
    vdown = max(1.0, abs(state.vy))
    t_flare_est = dy_stop / vdown  # seconds until we likely must start flare

    # Allow no more than what we can rotate back to 0 within t_flare_est at 15°/s
    theta_rotate_cap = DTHETA_MAX * t_flare_est

    theta_allow_abs = min(theta_vert_limit_deg, theta_rotate_cap)
    theta_allow_abs = clamp(theta_allow_abs, 0.0, 90.0)

    # Additionally, never allow more than world constraints and ability to rotate from current angle to 0
    t_to_upright = abs(state.angle) / DTHETA_MAX
    # Reserve a small buffer (0.3s)
    theta_time_cap = max(0.0, (t_flare_est - t_to_upright - 0.3) * DTHETA_MAX)
    if theta_time_cap < theta_allow_abs:
        theta_allow_abs = max(0.0, theta_time_cap)

    return theta_allow_abs


# --- Collision / intersection helpers ---
def _seg_intersection(p1: Tuple[float, float], p2: Tuple[float, float],
                      p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float, float, float]]:
    """Segment intersection. Returns (x, y, t, u) where point = p1 + t*(p2-p1) = p3 + u*(p4-p3), or None.
    t,u in [0,1] for proper intersection.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return x, y, t, u
    return None


# --- Simulator ---
def simulate(state0: State, plan_realizable: List[Command], terrain: Terrain) -> SimulationResult:
    """Simulate applying a realizable plan to the lander for up to len(plan) seconds.

    - Applies physics per second.
    - Checks for OOB early.
    - Sweeps each second's movement segment against terrain segments to detect collision.
    - On collision, determines success if on the flat segment with upright angle and within speed limits.
    - Accumulates mid-flight penalty auxiliaries for cost shaping (Chunk 4).
    """
    s = State(**vars(state0))
    steps = 0
    profile_note()
    # Mid-flight aggregates
    flat = terrain.flat
    x_center = 0.5 * (flat.xL + flat.xR)
    aux = {
        'oob_dist': 0.0,
        'ground_def': 0.0,
        'ang_sum': 0.0,
        'bang_steps': 0.0,
        'eta_mismatch': 0.0,
        'corr_penalty': 0.0,
        'vx_excess_mid': 0.0,
        'edge_band': 0.0,
    }

    def is_oob(st: State) -> bool:
        return not (WORLD_X_MIN <= st.x <= WORLD_X_MAX and WORLD_Y_MIN <= st.y <= WORLD_Y_MAX)

    for cmd in plan_realizable:
        # Apply actual commanded angle/power for this second
        s.angle = clamp(cmd.angle, THETA_MIN, THETA_MAX)
        s.power = int(clamp(cmd.power, P_MIN, P_MAX))

        # Acceleration from thrust and gravity
        th_rad = math.radians(s.angle)
        ax = s.power * math.sin(th_rad)
        ay = s.power * math.cos(th_rad) - G

        # Integrate with constant accel over 1 second
        x_prev, y_prev = s.x, s.y
        vx_prev, vy_prev = s.vx, s.vy

        # Update velocities
        s.vx = s.vx + ax
        s.vy = s.vy + ay
        # Update positions: x = x0 + v0*dt + 0.5*a*dt^2, dt=1
        s.x = s.x + vx_prev + 0.5 * ax
        s.y = s.y + vy_prev + 0.5 * ay

        # Fuel usage = power per second
        s.fuel = max(0.0, s.fuel - s.power)

        # Mid-flight shaping aggregates after integration (approximate per-second sample)
        # OOB distance (x only per spec wording), accumulate overshoot
        if s.x < WORLD_X_MIN:
            aux['oob_dist'] += (WORLD_X_MIN - s.x)
        elif s.x > WORLD_X_MAX:
            aux['oob_dist'] += (s.x - WORLD_X_MAX)
        # Ground clearance deficit
        ground_y = terrain.ground_y_at(s.x)
        h_local = s.y - ground_y
        if h_local < H_GROUND_MARGIN:
            aux['ground_def'] += (H_GROUND_MARGIN - max(0.0, h_local))
        # Mid-flight small angle bias
        aux['ang_sum'] += abs(s.angle)
        # Bang-bang bias when using intermediate powers
        if 1 <= s.power <= 3:
            aux['bang_steps'] += 1.0
        # ETA synchronization estimate
        theta_cap = max(1e-3, tilt_allow(s, terrain))
        ax_max = P_MAX * math.sin(math.radians(theta_cap))
        eta_x = abs(s.vx) / max(1e-6, ax_max)
        ay_max_up = max(1e-6, P_MAX - G)
        # time to flare based on stopping distance divided by speed as in tilt_allow
        if s.vy != 0:
            eta_y = vertical_stopping(s.vy) / max(1.0, abs(s.vy))
        else:
            eta_y = 0.0
        aux['eta_mismatch'] += abs(eta_x - eta_y)
        # Corridor penalty with altitude decay (stronger near ground below ALT_CORR_THRESH)
        x_left = flat.xL + CORR_MARGIN
        x_right = flat.xR - CORR_MARGIN
        if s.x < x_left:
            dist_out = x_left - s.x
        elif s.x > x_right:
            dist_out = s.x - x_right
        else:
            dist_out = 0.0
        h = max(0.0, s.y - flat.y)
        if ALT_CORR_THRESH > 1e-6:
            factor = clamp(1.0 - h / ALT_CORR_THRESH, 0.0, 1.0)
            factor = factor * factor  # sharper near ground
        else:
            factor = 0.0
        aux['corr_penalty'] += dist_out * factor

        # Edge band penalty: discourage flying very near flat edges at low altitude
        h_flat = s.y - flat.y
        if h_flat < 200.0:
            edge_band = 20.0
            if flat.xL <= s.x <= flat.xR:
                d_edge = min(abs(s.x - flat.xL), abs(s.x - flat.xR))
                if d_edge < edge_band:
                    aux['edge_band'] += (edge_band - d_edge) / max(1.0, edge_band)

        # Mid-flight horizontal speed excess near flat altitude (discourage huge |vx| close to flat)
        if h_flat < 1500.0:
            try:
                vx_env_mid = vmax_by_alt(s.y, flat.y)
            except Exception:
                vx_env_mid = 40.0
            vx_excess = max(0.0, abs(s.vx) - vx_env_mid)
            aux['vx_excess_mid'] += vx_excess

        # Early OOB detection using swept segment
        # If final position goes OOB, still sweep collision first to allow ground collision at boundary
        if is_oob(s):
            # But check if we crossed ground before going OOB using sweep
            hit = _sweep_ground_hit((x_prev, y_prev), (s.x, s.y), terrain)
            if hit is not None:
                x_hit, y_hit, t_hit, seg_idx = hit
                # Touch state at hit: interpolate velocities linearly in time
                s_hit = _interp_state(state0=s, x0=x_prev, y0=y_prev, vx0=vx_prev, vy0=vy_prev,
                                      ax=ax, ay=ay, t=t_hit)
                ok, reason = _is_success_touch(s_hit, seg_idx, terrain)
                if ok:
                    _res = SimulationResult(True, s_hit, fuel_used=state0.fuel - s.fuel,
                                            steps_simulated=steps + 1, oob=False, crash_reason=None, cost_aux=aux)
                    profile_note(_res)
                    return _res
                else:
                    _res = SimulationResult(False, s_hit, fuel_used=state0.fuel - s.fuel,
                                            steps_simulated=steps + 1, oob=False, crash_reason=reason, cost_aux=aux)
                    profile_note(_res)
                    return _res
                _res = SimulationResult(False, s, fuel_used=state0.fuel - s.fuel,
                                        steps_simulated=steps + 1, oob=True, crash_reason="OOB", cost_aux=aux)
                profile_note(_res)
                return _res

        # Sweep collision with terrain
        hit = _sweep_ground_hit((x_prev, y_prev), (s.x, s.y), terrain)
        if hit is not None:
            x_hit, y_hit, t_hit, seg_idx = hit
            s_hit = _interp_state(state0=s, x0=x_prev, y0=y_prev, vx0=vx_prev, vy0=vy_prev, ax=ax, ay=ay, t=t_hit)
            ok, reason = _is_success_touch(s_hit, seg_idx, terrain)
            if ok:
                _res = SimulationResult(True, s_hit, fuel_used=state0.fuel - s.fuel,
                                        steps_simulated=steps + 1, oob=False, crash_reason=None, cost_aux=aux)
                profile_note(_res)
                return _res
            else:
                _res = SimulationResult(False, s_hit, fuel_used=state0.fuel - s.fuel,
                                        steps_simulated=steps + 1, oob=False, crash_reason=reason, cost_aux=aux)
                profile_note(_res)
                return _res

        steps += 1

    # If plan ends without touchdown, return last state (no success)
    _res = SimulationResult(False, s, fuel_used=state0.fuel - s.fuel,
                            steps_simulated=steps, oob=False, crash_reason="Plan ended", cost_aux=aux)
    profile_note(_res)
    return _res


def _sweep_ground_hit(p0: Tuple[float, float], p1: Tuple[float, float], terrain: Terrain) -> Optional[Tuple[float, float, float, int]]:
    """Check if movement from p0 to p1 intersects any terrain segment.
    Return (x, y, t, seg_index) for earliest intersection by t (0..1) or None.
    """
    best_t = None
    best_hit = None
    for idx, a, b in terrain.iter_segments():
        inter = _seg_intersection(p0, p1, a, b)
        if inter is not None:
            x, y, t, u = inter
            if best_t is None or t < best_t:
                best_t = t
                best_hit = (x, y, t, idx)
    return best_hit


def _interp_state(state0: State, x0: float, y0: float, vx0: float, vy0: float, ax: float, ay: float, t: float) -> State:
    """Interpolate state at time t within the current 1-second interval using constant acceleration.
    Note: angle and power remain as currently applied over the second.
    """
    x = x0 + vx0 * t + 0.5 * ax * t * t
    y = y0 + vy0 * t + 0.5 * ay * t * t
    vx = vx0 + ax * t
    vy = vy0 + ay * t
    return State(x=x, y=y, vx=vx, vy=vy, fuel=state0.fuel, angle=state0.angle, power=state0.power)


def _is_success_touch(s: State, seg_idx: int, terrain: Terrain) -> Tuple[bool, Optional[str]]:
    """Check touchdown success criteria when we collide with a terrain segment at index seg_idx.

    Additional rule: success only if touchdown x is at least 10 m inside the flat borders.
    """
    flat = terrain.flat
    # Require an interior margin of 10 m from each edge for a valid touchdown
    inner_margin = 10.0
    xL_in = flat.xL + inner_margin
    xR_in = flat.xR - inner_margin
    on_flat_interior = (seg_idx == flat.seg_index) and (xL_in - 1e-6 <= s.x <= xR_in + 1e-6) and (abs(s.y - flat.y) < 1.5)
    if on_flat_interior and abs(s.angle) <= 1e-6 and abs(s.vx) <= MAX_VX_LAND and abs(s.vy) <= MAX_VY_LAND:
        return True, None
    # Provide a more specific reason when touching the flat but too close to the edge
    if (seg_idx == flat.seg_index) and not (xL_in - 1e-6 <= s.x <= xR_in + 1e-6):
        reason = "Too close to edge"
    else:
        reason = "Not on flat" if seg_idx != flat.seg_index else "Bad pose/speed"
    return False, reason


# --- Emergency policy and runtime guards ---
def emergency_policy(state: State, terrain: Terrain) -> Optional[Command]:
    """Emergency override when altitude is critically low relative to vertical stopping.

    Changes from previous behavior:
    - Only triggers if we are descending faster than the altitude-aware target (vy_des_by_alt) by a margin.
    - When it triggers and horizontal speed is large, allow a small tilt (≤12° and ≤tilt_allow) that opposes vx
      to bleed lateral speed while still commanding full power. This avoids riding long 0°,4 plateaus that
      ignore horizontal braking and lead to high-|vx| touchdowns on high flats.
    - Additionally, if local clearance is very small, force strictly upright (0°) to guarantee vertical touchdown.
    """
    flat = terrain.flat
    # Altitude versus the flat (conservative for the high-flat edge case)
    h = max(0.0, state.y - flat.y)
    h_local = state.y - terrain.ground_y_at(state.x)

    # Only consider emergency if we're actually too fast vertically relative to profile
    vy_target = vy_des_by_alt(state.y, flat.y)
    overspeed_margin = 6.0  # m/s faster (more negative) than target before we call it emergency
    if state.vy >= (vy_target - overspeed_margin):
        return None

    # Classical stopping check (to upright) with a small buffer
    dy_stop = vertical_stopping(state.vy)
    t_to_upright = abs(state.angle) / DTHETA_MAX
    # Additional height lost while rotating to upright under gravity
    h_rotate = abs(state.vy) * t_to_upright + 0.5 * G * (t_to_upright ** 2)
    buffer = 50.0
    if h > dy_stop + h_rotate + buffer:
        return None

    # If extremely low to ground, enforce strictly upright landing posture
    if h_local < 80.0:
        return Command(angle=0.0, power=4)

    # We are in the critical zone. Command full power and a small feasible tilt to oppose current vx.
    theta_cap = tilt_allow(state, terrain)
    # Max emergency tilt: small, to preserve vertical authority yet bleed |vx|
    theta_emax = min(theta_cap, 12.0)
    angle = 0.0
    if abs(state.vx) > MAX_VX_LAND and theta_emax > 0.5:
        # Oppose horizontal velocity: vx<0 (moving left) -> push right (positive angle)
        angle = (12.0 if state.vx < 0 else -12.0)
        # Ensure within feasible cap
        angle = clamp(angle, -theta_emax, theta_emax)
    return Command(angle=angle, power=4)


def apply_runtime_guards(state: State, desired_cmd: Command, terrain: Terrain) -> Command:
    """Lightweight guards that gently nudge commands toward safety without fighting the planner.

    - Clamp angle within tilt_allow().
    - Edge pushback near world bounds.
    - Corridor lock at low altitude with ALT_CORR_THRESH.
    - Early horizontal bias when far and |vx| is small.
    - Shared vertical-lateral feasibility: ensure vertical demand is met by adjusting |theta| and power, not by zeroing tilt.
    """
    theta_cap = tilt_allow(state, terrain)
    angle = clamp(desired_cmd.angle, -theta_cap, theta_cap)
    power = int(clamp(desired_cmd.power, P_MIN, P_MAX))

    # Edge pushback
    margin = 120.0
    push_tilt = 20.0
    if state.x < WORLD_X_MIN + margin:
        angle = max(angle, min(theta_cap, push_tilt))  # push right (positive ax)
    elif state.x > WORLD_X_MAX - margin:
        angle = min(angle, max(-theta_cap, -push_tilt))  # push left (negative ax)

    # Corridor lock strength increases as altitude decreases below ALT_CORR_THRESH
    h = max(0.0, state.y - terrain.flat.y)
    flat = terrain.flat
    if h < ALT_CORR_THRESH:
        # If outside corridor + margin, bias toward center; otherwise gently reduce tilt when quite low
        if state.x < flat.xL - CORR_MARGIN:
            angle = max(angle, min(theta_cap, 15.0))
        elif state.x > flat.xR + CORR_MARGIN:
            angle = min(angle, max(-theta_cap, -15.0))
        else:
            # Only tighten inside the corridor when h < 600 m; allow larger tilt higher up for efficient lateral moves
            if h < 600.0:
                # tighter clamp when very low, but allow a bit more if |vx| is above the altitude envelope
                tighter = 8.0 if h < 300.0 else 10.0
                try:
                    vx_env = vmax_by_alt(state.y, flat.y) + 10.0
                except Exception:
                    vx_env = 35.0
                if abs(state.vx) > vx_env and h >= 250.0:
                    tighter = min(18.0, theta_cap)
                angle = clamp(angle, -min(theta_cap, tighter), min(theta_cap, tighter))

    # Ground-clearance aware tilt tightening (no forced zero-tilt)
    h_local = state.y - terrain.ground_y_at(state.x)
    speed_down = max(0.0, -state.vy)
    clear_soft = H_GROUND_MARGIN + 20.0 + 0.6 * speed_down
    clear_crit = H_GROUND_MARGIN + 10.0 + 0.3 * speed_down
    if h_local < clear_crit:
        # Critical: limit tilt to preserve vertical authority; if |vx| is excessive and we still have flat altitude
        # headroom, allow a bit more tilt to bleed horizontal speed.
        limit = 6.0
        try:
            vx_env = vmax_by_alt(state.y, flat.y) + 5.0
        except Exception:
            vx_env = 35.0
        h_flat = max(0.0, state.y - flat.y)
        if abs(state.vx) > vx_env and h_flat >= 250.0:
            limit = min(12.0, theta_cap)
        angle = clamp(angle, -min(theta_cap, limit), min(theta_cap, limit))
        power = max(power, 3)
    elif h_local < clear_soft:
        # Soft: allow limited tilt toward corridor and avoid needless hover
        limit = 12.0 if h < 600.0 else 15.0
        x_center = 0.5 * (flat.xL + flat.xR)
        dir_right = 1.0 if (x_center - state.x) > 0 else -1.0
        desired = clamp(angle, -limit, limit)
        if abs(desired) < 3.0:
            desired = dir_right * min(limit, 6.0)
        angle = clamp(desired, -theta_cap, theta_cap)
        # do not cap power here; feasibility step below will ensure minimum needed

    # Flat-aware braking override: if near/over the flat and |vx| exceeds envelope, flip angle to oppose vx
    try:
        vx_env_flat = vmax_by_alt(state.y, flat.y)
    except Exception:
        vx_env_flat = 40.0
    h_flat = max(0.0, state.y - flat.y)
    # Determine if current commanded angle would accelerate horizontally in the same direction as current vx
    def would_accelerate(vx: float, ang: float) -> bool:
        # Positive angle -> ax>0 (push RIGHT); Negative -> ax<0 (push LEFT)
        if abs(ang) < 1e-3:
            return False
        return (vx > 0 and ang > 0) or (vx < 0 and ang < 0)

    if h_flat < 1400.0 and abs(state.vx) > (vx_env_flat - 2.0) and would_accelerate(state.vx, angle):
        # Flip to braking direction within feasible tilt
        brake_sign = 1.0 if state.vx < 0 else -1.0
        angle = clamp(brake_sign * min(theta_cap, 15.0), -theta_cap, theta_cap)
        # Apply small lateral power floor; refined by vertical feasibility later
        gate_open, _hg, _eta = burn_gate_params(state, terrain)
        power = max(power, 3 if gate_open else 2)

    # Predictive edge nudge (avoid small-miss at flat edges)
    x_left = flat.xL + CORR_MARGIN
    x_right = flat.xR - CORR_MARGIN
    if h_flat < 900.0:
        # Predict x in a short horizon using current vx (conservative)
        t_h = clamp(h_flat / max(1.0, abs(state.vy)), 0.5, 3.0)
        x_pred = state.x + state.vx * t_h
        tol = 80.0  # meters tolerance to treat as "near edge" (was 40)
        need_push_right = (state.x < x_left and (x_left - state.x) <= 120.0) or (x_pred < x_left and (x_left - x_pred) <= tol)
        need_push_left = (state.x > x_right and (state.x - x_right) <= 120.0) or (x_pred > x_right and (x_pred - x_right) <= tol)
        if need_push_right or need_push_left:
            dir_in = 1.0 if need_push_right else -1.0
            # Nudge inward stronger and earlier; still limited by feasibility later
            nudge_abs = 12.0 if h_flat > 300.0 else 10.0
            angle = clamp(dir_in * min(theta_cap, nudge_abs), -theta_cap, theta_cap)
            # Provide a small power floor to realize the lateral nudge
            gate_open, _hg, _eta = burn_gate_params(state, terrain)
            power = max(power, 3 if gate_open else 2)
        else:
            # If very close to edge and moving outward, do not allow an outward-accelerating angle
            if state.x < x_left and state.vx < -0.5 and would_accelerate(state.vx, angle):
                angle = clamp(min(theta_cap, 12.0), -theta_cap, theta_cap)  # push RIGHT
            elif state.x > x_right and state.vx > 0.5 and would_accelerate(state.vx, angle):
                angle = clamp(max(-theta_cap, -12.0), -theta_cap, theta_cap)  # push LEFT

    # Early horizontal bias when far and |vx| small (only when reasonably high)
    x_center = 0.5 * (flat.xL + flat.xR)
    far = abs(state.x - x_center) > 1000.0
    slow = abs(state.vx) < 5.0
    if far and slow and h > ALT_CORR_THRESH:
        dir_right = 1 if (x_center - state.x) > 0 else -1
        angle_bias = dir_right * 10.0
        angle = clamp(angle + angle_bias, -theta_cap, theta_cap)

    # Late centering upward burn near edges (far-high plateau case): when low over/near flat and close to edges, burn more up and nudge inward
    h_flat = max(0.0, state.y - flat.y)
    if h_flat < 220.0:
        # Determine edge proximity now and in a short horizon
        over_flat = (flat.xL <= state.x <= flat.xR)
        edge_band = 40.0
        d_edge_now = (min(abs(state.x - flat.xL), abs(state.x - flat.xR)) if over_flat else float('inf'))
        # Predict near-term x using current vx for a short horizon based on altitude and vy
        t_h = clamp(h_flat / max(1.0, abs(state.vy)), 0.4, 2.5)
        x_pred = state.x + state.vx * t_h
        near_edge_pred = False
        if flat.xL <= x_pred <= flat.xR:
            d_edge_pred = min(abs(x_pred - flat.xL), abs(x_pred - flat.xR))
            near_edge_pred = d_edge_pred < edge_band
        # Trigger when close to edge now or likely soon; only when already within corridor/over flat
        x_left = flat.xL + CORR_MARGIN
        x_right = flat.xR - CORR_MARGIN
        in_corr_or_flat = over_flat or (x_left <= state.x <= x_right)
        if in_corr_or_flat and (d_edge_now < edge_band or near_edge_pred):
            # Inward small tilt and stronger upward burn to buy time to translate away from edge
            dir_in = 1.0 if (x_center - state.x) > 0 else -1.0
            tilt_abs = 12.0 if h_flat >= 120.0 else 9.0
            angle = clamp(dir_in * min(theta_cap, tilt_abs), -theta_cap, theta_cap)
            gate_open, _hg, _eta = burn_gate_params(state, terrain)
            power = max(power, 4 if gate_open else 3)

    # Inside-corridor anti-drift damping: oppose residual vx when safely within corridor
    x_left = flat.xL + CORR_MARGIN
    x_right = flat.xR - CORR_MARGIN
    if x_left <= state.x <= x_right and h < 1000.0:
        if abs(state.vx) > 2.0:
            damp_lim = min(theta_cap, 14.0 if h >= 250.0 else 10.0)
            angle_damp = clamp(-0.5 * state.vx, -damp_lim, damp_lim)
            # Blend with requested angle to preserve planner’s intent
            angle = clamp(0.5 * angle + 0.5 * angle_damp, -theta_cap, theta_cap)

    # Corridor/flat-aware terminal braking vs upright: when horizontally near the flat center, brake residual |vx| before forcing upright
    over_flat = (flat.xL <= state.x <= flat.xR)
    x_center = 0.5 * (flat.xL + flat.xR)
    d_center = abs(state.x - x_center)
    if (over_flat or (x_left <= state.x <= x_right)) and d_center <= DIST_NEAR_LAND_TIGHT:
        vx_tol = VX_NEAR_LAND  # demand braking until safely under limit (proximity window)
        if abs(state.vx) > vx_tol:
            brake_sign = 1.0 if state.vx < 0 else -1.0
            # Keep tilt modest; use altitude to slightly scale but gate by feasibility
            tilt_abs = 14.0 if h_flat >= 90.0 else 10.0
            angle = clamp(brake_sign * min(theta_cap, tilt_abs), -theta_cap, theta_cap)
            # Provide a small power floor to realize braking nudge
            gate_open, _hg, _eta = burn_gate_params(state, terrain)
            power = max(power, 3 if gate_open else 2)
        else:
            angle = 0.0
    # Strict upright near ground with residual-|vx|-aware exception: allow a small braking tilt until ~35 m AGL
    if h_local < 60.0:
        if (over_flat or (x_left <= state.x <= x_right)) and h_flat >= 40.0 and abs(state.vx) > VX_NEAR_LAND and h_local >= 35.0:
            brake_sign = 1.0 if state.vx < 0 else -1.0
            angle = clamp(brake_sign * min(theta_cap, 8.0), -theta_cap, theta_cap)
            gate_open, _hg, _eta = burn_gate_params(state, terrain)
            power = max(power, 3 if gate_open else 2)
        else:
            angle = 0.0

    # Final shared-authority feasibility: ensure vertical demand can be met with this tilt
    vy_tgt = vy_des_by_alt(state.y, flat.y)
    # Faster response near ground
    tau_y = 2.0 if h_local < clear_soft else 3.0
    ay_need_net = clamp((vy_tgt - state.vy) / max(1.0, tau_y), -5.0, 5.0)
    # Require: P*cos(theta) - G >= ay_need_net  -> cos(theta) >= (ay_need_net + G)/P_MAX
    rhs = (ay_need_net + G) / max(1e-6, P_MAX)
    rhs = clamp(rhs, 0.0, 1.0)
    theta_vert_cap = math.degrees(math.acos(rhs)) if rhs <= 1.0 else 0.0
    # apply vertical feasibility on top of theta_cap
    theta_allow_final = min(theta_cap, theta_vert_cap)
    angle = clamp(angle, -theta_allow_final, theta_allow_final)
    # minimal power needed at this angle
    cos_th = max(1e-3, math.cos(math.radians(angle)))
    P_needed = (ay_need_net + G) / cos_th
    P_needed = clamp(P_needed, 0.0, 4.0)
    # Round up to ensure feasibility against rate/quantization
    power = int(max(power, math.ceil(P_needed - 1e-6)))
    power = int(clamp(power, P_MIN, P_MAX))

    return Command(angle=angle, power=power)


# --- Chunk 2: Candidate Library and Waypoint Follower ---
from dataclasses import field


@dataclass
class ControlPoint:
    angle: float  # degrees
    power: int    # 0..4
    dur: int      # seconds (>=1)


@dataclass
class ControlPointPlan:
    cps: List[ControlPoint]
    name: str = "candidate"
    meta: dict = field(default_factory=dict)


def expand_cp(cp_plan: ControlPointPlan, H: int) -> List[Command]:
    """Expand control points into per-second desired commands, length <= H.
    Does not apply ramping or guards. Use ramp_enforce and guards afterwards.
    """
    seq: List[Command] = []
    for cp in cp_plan.cps:
        dur = max(0, int(cp.dur))
        if dur <= 0:
            continue
        for _ in range(dur):
            seq.append(Command(angle=cp.angle, power=int(clamp(cp.power, P_MIN, P_MAX))))
            if len(seq) >= H:
                return seq[:H]
    return seq[:H]


def _preview_apply_guards(state0: State, plan: List[Command], terrain: Terrain) -> List[Command]:
    """Apply emergency + runtime guards step-by-step while preview-integrating state.
    This mirrors the pattern used in __main__ to clamp commands by tilt_allow that depends on the evolving state.
    """
    out: List[Command] = []
    s_tmp = State(**vars(state0))
    for cmd in plan:
        emerg = emergency_policy(s_tmp, terrain)
        if emerg is not None:
            guarded = emerg
        else:
            guarded = apply_runtime_guards(s_tmp, cmd, terrain)
        out.append(guarded)
        # integrate preview (no collision checks)
        th_rad = math.radians(guarded.angle)
        ax = guarded.power * math.sin(th_rad)
        ay = guarded.power * math.cos(th_rad) - G
        vx_prev, vy_prev = s_tmp.vx, s_tmp.vy
        s_tmp.vx = s_tmp.vx + ax
        s_tmp.vy = s_tmp.vy + ay
        s_tmp.x = s_tmp.x + vx_prev + 0.5 * ax
        s_tmp.y = s_tmp.y + vy_prev + 0.5 * ay
        s_tmp.fuel = max(0.0, s_tmp.fuel - guarded.power)
        s_tmp.angle = guarded.angle
        s_tmp.power = guarded.power
    return out


def vmax_by_alt(y: float, y_flat: float) -> float:
    """Maximum desired |vx| based on altitude; more permissive high up."""
    h = max(0.0, y - y_flat)
    if h >= 2000:
        return 80.0
    if h <= 300:
        return 25.0
    # linear interpolation between 25 and 80
    return 25.0 + (h - 300.0) * (80.0 - 25.0) / (2000.0 - 300.0)


def vy_des_by_alt(y: float, y_flat: float) -> float:
    """Desired vertical speed profile; approach -40 near flat height."""
    h = max(0.0, y - y_flat)
    if h > 1500:
        return -55.0
    if h > 600:
        # smooth ramp from -55 to -45
        t = (h - 600.0) / (1500.0 - 600.0)
        return -45.0 - 10.0 * t
    if h > 90:
        # ramp from -45 to -40 (keep faster descent a bit longer to reduce early hover)
        t = (h - 90.0) / (600.0 - 90.0)
        return -40.0 - 5.0 * t
    return -38.0


def predictive_brake_gate(state: State, terrain: Terrain, h_low: float = 600.0) -> Tuple[bool, float, float, int]:
    """Predict if we must begin lateral braking now to meet |vx| limits before reaching the near‑landing zone.

    Returns (must_brake_now, t_need, t_avail, brake_dir), where brake_dir is the desired
    angle sign for braking (+1 => positive angle/push RIGHT to reduce vx<0; -1 => push LEFT).
    Now uses distance to the flat center to define the near‑landing threshold instead of altitude.
    """
    flat = terrain.flat
    # Corridor geometry
    M = CORR_MARGIN
    x_left = flat.xL + M
    x_right = flat.xR - M
    x_center = 0.5 * (flat.xL + flat.xR)
    in_corridor = (x_left <= state.x <= x_right)

    # Altitude bases (still used for vertical feasibility), but not for near-landing gating
    h_flat = max(0.0, state.y - flat.y)
    h_local = max(0.0, state.y - terrain.ground_y_at(state.x))
    h_basis = h_flat if in_corridor else h_local

    # Feasible tilt and effective lateral authority
    theta_cap = min(tilt_allow(state, terrain), 25.0)
    gate_open, _h_gate, _eta_y = burn_gate_params(state, terrain)
    far = abs(state.x - x_center) > 2000.0
    # When gate is closed, still allow notable lateral push if far from target
    P_eff = 4.0 if gate_open else (2.2 if far else 1.0)
    a_h = max(1e-6, P_eff * math.sin(math.radians(max(0.0, min(89.0, theta_cap)))))

    # |vx| envelope by altitude (unchanged), but landing-limit clamp is distance-based to center
    vx_env_hi = vmax_by_alt(state.y, flat.y)
    d_center = abs(state.x - x_center)
    near_center = (d_center <= DIST_NEAR_LAND)
    near_center_tight = (d_center <= DIST_NEAR_LAND_TIGHT)
    # Only clamp to near-landing limit when close to the center horizontally
    if near_center:
        vx_goal = VX_NEAR_LAND
    else:
        vx_goal = 0.85 * vx_env_hi

    dv = max(0.0, abs(state.vx) - vx_goal)
    t_need = dv / a_h if a_h > 0 else float('inf')

    # Time available: vertical by altitude basis, horizontal by distance to near-center ring
    if state.vy < -1e-3:
        t_avail_alt = max(0.0, (h_basis - h_low) / (-state.vy)) + 0.7
    else:
        t_avail_alt = float('inf')

    # Time to reach the near-center boundary (outer ring)
    d_to_ring = max(0.0, d_center - DIST_NEAR_LAND)
    v_nonzero = max(1.0, abs(state.vx))
    t_to_center = d_to_ring / v_nonzero + 0.5

    t_avail = min(t_avail_alt, t_to_center)
    must_brake = (near_center_tight and abs(state.vx) > VX_NEAR_LAND * 1.05) or (t_need > max(0.0, t_avail - 1.2))

    brake_dir = 1 if state.vx < 0 else -1 if state.vx > 0 else 0
    return must_brake, t_need, t_avail, brake_dir


def burn_gate_params(state: State, terrain: Terrain):
    """Compute a conservative vertical burn gate and a rough vertical ETA.

    Uses flat-based altitude when we're vertically above the flat segment (no obstacles),
    otherwise uses local ground-based altitude for safety.

    Returns:
      gate_open (bool): whether we should start vertical braking now
      h_gate (float): altitude threshold at which the gate opens
      eta_y (float): rough seconds until flare end used to coordinate horizontal
    """
    flat = terrain.flat
    # Inside the flat corridor (expanded slightly) we can treat descent as vertical to the flat.
    margin = max(60.0, CORR_MARGIN * 0.6)
    in_corridor = (flat.xL + margin) <= state.x <= (flat.xR - margin)
    # Altitude: to flat when inside corridor, otherwise to local ground for safety
    h_flat = state.y - flat.y
    h_local = state.y - terrain.ground_y_at(state.x)
    h = h_flat if in_corridor else h_local

    vy = state.vy
    # Max upward net acceleration when upright at P=4
    ay_max = 4.0 - G
    # Altitude-aware target vertical speed to avoid early hovering
    vy_target = vy_des_by_alt(state.y, flat.y)
    # Time to brake to vy_target (if descending faster than target)
    t_stop = 0.0
    if vy < vy_target:
        t_stop = (vy_target - vy) / max(1e-6, ay_max)
    # Rotation time to ensure we can be upright before flare
    t_rot = (abs(state.angle) + 15.0) / 15.0
    # Vertical stopping distance to reach vy_target with max burn
    dy_stop = 0.0
    if vy < vy_target:
        dy_stop = vy * t_stop + 0.5 * ay_max * t_stop * t_stop
        dy_stop = abs(dy_stop)
    # Allowance grows with speed; smaller buffer when directly above flat
    vy_allow = max(0.0, abs(vy) - abs(vy_target)) * (0.4 if in_corridor else 0.5)
    # Slightly later burn over flat (reduced buffer) to avoid early flare/hover
    h_buf = 35.0 if in_corridor else 110.0
    h_gate = dy_stop + vy_allow + t_rot * max(0.0, abs(vy) - abs(vy_target)) + h_buf
    gate_open = h <= h_gate
    # Rough ETA used by horizontal planner
    eta_y = (h / max(1.0, abs(vy))) + (t_stop if gate_open else 0.0) + t_rot
    eta_y = max(3.0, min(eta_y, 25.0))
    return gate_open, h_gate, eta_y


def follow_waypoints(state0: State, terrain: Terrain, waypoints: Optional[dict], H: int) -> List[Command]:
    """Fuel-aware follower using a vertical burn gate and ETA-aligned lateral braking.

    waypoints: optional dict that can contain keys:
      - x_target: desired horizontal target (defaults to flat center)
      - flare_alt: altitude to begin aggressive flare bias (defaults 250m)
    """
    flat = terrain.flat
    x_center = 0.5 * (flat.xL + flat.xR)
    x_target = x_center if waypoints is None else waypoints.get('x_target', x_center)
    flare_alt = 250.0 if waypoints is None else waypoints.get('flare_alt', 250.0)

    plan: List[Command] = []
    s = State(**vars(state0))
    # Hysteresis mode for horizontal profile to avoid accelerate/brake oscillations within the horizon
    mode = "neutral"  # one of: neutral, accelerate, brake
    prev_pow = int(clamp(s.power, P_MIN, P_MAX))
    for _ in range(H):
        # 1) Compute vertical burn gate and rough vertical ETA
        gate_open, _h_gate, eta_y = burn_gate_params(s, terrain)

        # Predictive lateral brake gate: do we need to start braking now to meet |vx| limits?
        must_pred_brake, t_need_b, t_avail_b, brake_dir = predictive_brake_gate(s, terrain)

        # 2) Horizontal: be assertive high up; only ensure we can meet the limit by ETA near flare
        theta_cap = tilt_allow(s, terrain)
        # Altitude-based cap (fast when high, gentle when low)
        vx_cap = vmax_by_alt(s.y, flat.y)
        # Targeting and deadband around the flat center to avoid drift accumulation
        dx = x_target - s.x
        deadband = 200.0
        dir_to_target = 0.0 if abs(dx) < 1e-6 else (1.0 if dx > 0 else -1.0)
        inside_band = abs(dx) < deadband
        # Precompute lateral authority estimate for braking-distance and hysteresis
        th_cap_rad = math.radians(max(0.0, min(89.0, theta_cap)))
        a_h_max = P_MAX * math.sin(th_cap_rad)
        # Use an effective lateral decel that respects near-term power caps (gate) for braking distance
        if not gate_open:
            far = abs(s.x - x_target) > 2000.0
            P_cap_preview = 1.6 if far else (0.8 if inside_band else 1.2)
        else:
            P_cap_preview = 4.0
        a_h_eff = max(1e-3, P_cap_preview * math.sin(th_cap_rad))
        # Desired end-of-ETA horizontal speed toward target, with hysteresis to avoid accel/brake oscillations
        margin_brake = 150.0
        if must_pred_brake:
            mode = "brake"
        elif inside_band:
            # Inside band: prefer braking mode to damp residual vx
            if mode != "accelerate":
                mode = "brake" if abs(s.vx) > 2.0 else "neutral"
        else:
            # Outside band
            if mode == "neutral":
                if dir_to_target != 0.0 and (s.vx * dx) <= 0:
                    mode = "accelerate"
                else:
                    # If we are headed toward target and within stopping distance, choose brake
                    d_brake = (s.vx * s.vx) / (2.0 * a_h_eff)
                    if abs(dx) <= d_brake + margin_brake and (s.vx * dir_to_target) > 0:
                        mode = "brake"
            elif mode == "accelerate":
                # Switch to brake when within stopping distance
                d_brake = (s.vx * s.vx) / (2.0 * a_h_eff)
                if abs(dx) <= d_brake + margin_brake:
                    mode = "brake"
            elif mode == "brake":
                # If we drift far again, allow accelerate
                if abs(dx) >= deadband * 1.2 and dir_to_target != 0.0 and (s.vx * dx) <= 0:
                    mode = "accelerate"

        # Hard |vx| envelope by altitude; force braking only when horizontally near the landing center
        vx_env = max(10.0, vmax_by_alt(s.y, flat.y) - 5.0)
        h_flat_now = max(0.0, s.y - flat.y)
        # Corridor-aware envelope brake: only force braking near corridor AND near center (distance-based)
        flat_left_nb = flat.xL - CORR_MARGIN
        flat_right_nb = flat.xR + CORR_MARGIN
        near_corridor = (flat_left_nb <= s.x <= flat_right_nb)
        d_center = abs(s.x - x_center)
        near_center = (d_center <= DIST_NEAR_LAND)
        near_center_tight = (d_center <= DIST_NEAR_LAND_TIGHT)
        if abs(s.vx) > vx_env and near_corridor and near_center:
            mode = "brake"
        # Extra safeguard: demand braking when |vx| is near landing limit, but only near corridor/flat by distance
        over_flat_now = (flat.xL <= s.x <= flat.xR)
        if (abs(s.vx) > VX_NEAR_LAND) and (over_flat_now or (near_corridor and near_center_tight)):
            mode = "brake"

        # Compute vx_goal_end based on mode
        if mode == "accelerate":
            vx_goal_end = dir_to_target * min(vx_cap, abs(dx) / max(3.0, eta_y))
        else:
            vx_goal_end = 0.0
        
        # Move toward that goal over a few seconds; if already moving in the correct direction faster than goal, keep it (don't brake early)
        tau_x = 4.0
        vx_goal_now = vx_goal_end
        if mode == "brake":
            tau_x = 3.0
        elif mode == "accelerate" and dir_to_target != 0.0 and (s.vx * dir_to_target) < 0:
            # moving away from target: correct quickly
            tau_x = 3.0
        elif dir_to_target != 0.0 and (s.vx * dir_to_target) > abs(vx_goal_end) and mode == "accelerate":
            # already faster in the right direction: don't demand braking yet
            vx_goal_now = s.vx
        # Desired horizontal acceleration
        vx_dot_des = (vx_goal_now - s.vx) / max(1.0, tau_x)
        # Predict reachable tilt next second given rate limit and clamp by tilt_allow
        if vx_dot_des > 0:
            theta_sign = 1.0  # positive desired ax -> positive tilt (push RIGHT)
        elif vx_dot_des < 0:
            theta_sign = -1.0  # negative desired ax -> negative tilt (push LEFT)
        else:
            theta_sign = 0.0
        if theta_sign != 0.0:
            theta_next = clamp(s.angle + DTHETA_MAX * theta_sign, -theta_cap, theta_cap)
        else:
            theta_next = clamp(s.angle, -theta_cap, theta_cap)
        sin_next = abs(math.sin(math.radians(max(0.0, min(89.0, abs(theta_next))))))
        # If we can barely tilt next second, be less aggressive horizontally
        if sin_next < 0.2:
            tau_x = max(tau_x, 6.0)
            vx_dot_des = (vx_goal_now - s.vx) / max(1.0, tau_x)
            sin_next = max(sin_next, 0.1)
        # Thrust needed to realize desired horizontal acceleration with reachable tilt
        P_x_raw = abs(vx_dot_des) / max(1e-3, sin_next)
        # While burn gate is closed, cap thrust to encourage gravity-first coasting; allow stronger lateral when far
        if not gate_open:
            far = abs(s.x - x_target) > 2000.0
            P_cap = (1.6 if far else (0.8 if inside_band else 1.2))
        else:
            P_cap = 4.0
        P_x = min(P_x_raw, P_cap)
        
        # Low-altitude horizontal power cap when inside corridor even if gate is open (reduce oscillatory accel)
        flat_left = flat.xL + CORR_MARGIN
        flat_right = flat.xR - CORR_MARGIN
        inside_corr = (flat_left <= s.x <= flat_right)
        h_flat = max(0.0, s.y - flat.y)
        if gate_open and inside_corr and h_flat < 1000.0:
            # Allow stronger lateral decel when actively braking near ground to kill residual |vx|
            if mode == "brake" or abs(s.vx) > VX_NEAR_LAND:
                P_x = min(P_x, 3.4 if h_flat >= 600.0 else 2.6)
            else:
                P_x = min(P_x, 2.5 if h_flat >= 600.0 else 1.8)
        
        # Ensure minimum lateral authority when far/outside band or with large |vx|
        if (not inside_band and (abs(dx) > 800.0 or abs(s.vx) > 50.0)) and P_x >= 0.35:
            P_x = max(P_x, 1.0)
        # When forced into braking by the |vx| envelope near ground, keep a small lateral thrust floor
        if mode == "brake" and abs(s.vx) > vx_env and h_flat_now < 900.0 and P_x >= 0.25:
            P_x = max(P_x, 1.2)

        # 3) Vertical: only burn when gate opens; otherwise coast
        P_y = 0.0
        theta_cmd = 0.0
        if gate_open:
            vy_target_touch = vy_des_by_alt(s.y, flat.y)
            vy_err = vy_target_touch - s.vy
            tau_y = 3.0
            ay_need = clamp(vy_err / max(1.0, tau_y), -5.0, 5.0)
            P_y = clamp(ay_need + G, 0.0, 4.0)

        # 4) Combine minimal thrust that satisfies both needs; apply quantization with hysteresis to avoid oscillations
        P_cont = max(P_x, P_y)
        P_cont = clamp(P_cont, 0.0, 4.0)
        # Sticky bins: only change desired power if underlying demand moves enough
        if P_cont >= prev_pow + 0.6:
            P = min(4, prev_pow + 1)
        elif P_cont <= prev_pow - 0.6:
            P = max(0, prev_pow - 1)
        else:
            P = prev_pow

        # 5) Command angle to realize horizontal demand (respect tilt)
        if P > 0:
            s_ratio = clamp(vx_dot_des / max(P, 1e-3), -1.0, 1.0)
            # Positive desired horizontal acceleration -> positive angle (push RIGHT)
            theta_cmd = math.degrees(math.asin(s_ratio))
        theta_cmd = clamp(theta_cmd, -theta_cap, theta_cap)

        cmd = Command(angle=theta_cmd, power=int(P))
        plan.append(cmd)
        prev_pow = cmd.power

        # Preview integrate with guards for next step stability
        guarded = apply_runtime_guards(s, cmd, terrain)
        th = math.radians(guarded.angle)
        ax = guarded.power * math.sin(th)
        ay = guarded.power * math.cos(th) - G
        vx_prev, vy_prev = s.vx, s.vy
        s.vx = s.vx + ax
        s.vy = s.vy + ay
        s.x = s.x + vx_prev + 0.5 * ax
        s.y = s.y + vy_prev + 0.5 * ay
        s.fuel = max(0.0, s.fuel - guarded.power)
        s.angle = guarded.angle
        s.power = guarded.power

        # 6) Late flare safety bias: when very low and still too fast vertically, go upright full
        if s.y - flat.y < flare_alt and s.vy < -42:
            plan[-1] = Command(angle=0.0, power=max(plan[-1].power, 4))

    # Apply ramping and guards over the whole horizon
    plan = ramp_enforce(plan, state0.angle, state0.power)
    plan = _preview_apply_guards(state0, plan, terrain)
    return plan


def build_plan_from_cp(state0: State, terrain: Terrain, cp_plan: ControlPointPlan, H: int) -> List[Command]:
    raw = expand_cp(cp_plan, H)
    ramped = ramp_enforce(raw, state0.angle, state0.power)
    guarded = _preview_apply_guards(state0, ramped, terrain)
    return guarded


def generate_candidates(state: State, terrain: Terrain, H: int) -> List[ControlPointPlan]:
    """Generate a small library of control-point plans as seeds (6–12 total)."""
    flat = terrain.flat
    x_center = 0.5 * (flat.xL + flat.xR)
    dx = x_center - state.x
    dir_right = 1 if dx >= 0 else -1
    h = max(0.0, state.y - flat.y)
    theta_cap = max(5.0, min(35.0, tilt_allow(state, terrain)))

    # Helper to make cp
    def cp(a, p, d):
        return ControlPoint(angle=clamp(dir_right * a, -theta_cap, theta_cap), power=int(clamp(p, 0, 4)), dur=int(d))

    # Estimate durations heuristically
    t_push = int(clamp(abs(dx) / 350.0 + 2, 2, 8))
    t_brake = int(clamp(abs(state.vx) / 8.0 + 2, 2, 8))
    t_flare = int(clamp(vertical_stopping(state.vy) / 30.0 + 3, 3, 12))
    t_coast = int(clamp(abs(dx) / 500.0, 0, 6))

    seeds: List[ControlPointPlan] = []

    # 1) Two-burn variants
    seeds.append(ControlPointPlan([
        cp(30, 4, t_push), cp(-25, 4, t_brake), ControlPoint(0.0, 4, t_flare)
    ], name="two_burn_A"))
    seeds.append(ControlPointPlan([
        cp(20, 4, t_push + 1), cp(-20, 3, t_brake + 1), ControlPoint(0.0, 4, t_flare)
    ], name="two_burn_B"))

    # 2) Glide-slope capture → flare
    seeds.append(ControlPointPlan([
        cp(12, 3, max(3, t_push + t_coast)), ControlPoint(0.0, 4, t_flare)
    ], name="glide_slope_A"))
    seeds.append(ControlPointPlan([
        cp(15, 2, max(4, t_push + t_coast + 2)), ControlPoint(0.0, 4, t_flare)
    ], name="glide_slope_B"))

    # 3) Three-impulse bang-bang
    seeds.append(ControlPointPlan([
        cp(35, 4, max(2, t_push - 1)), ControlPoint(0.0, 0, max(1, t_coast)), cp(-30, 4, max(2, t_brake)), ControlPoint(0.0, 4, t_flare)
    ], name="bangbang_A"))
    seeds.append(ControlPointPlan([
        cp(30, 4, max(2, t_push)), ControlPoint(0.0, 0, max(0, t_coast - 1)), cp(-25, 4, max(2, t_brake + 1)), ControlPoint(0.0, 4, t_flare)
    ], name="bangbang_B"))

    # 4) Corridor-first: aim to enter corridor before altitude threshold
    corr_margin = 100.0
    x_left = flat.xL + corr_margin
    x_right = flat.xR - corr_margin
    want_inside = not (x_left <= state.x <= x_right)
    t_corr = int(clamp(h / 300.0, 2, 8))
    if want_inside:
        seeds.append(ControlPointPlan([
            cp(25, 4, t_corr + 2), ControlPoint(0.0, 3, max(2, t_coast)), ControlPoint(0.0, 4, t_flare)
        ], name="corridor_first"))

    # Ensure total K limit and each plan lengths within H estimate
    out: List[ControlPointPlan] = []
    for s in seeds:
        K = len(s.cps)
        if K <= 10:
            out.append(s)
    return out[:10]


def score(state0: State, plan: List[Command], terrain: Terrain) -> float:
    """Evaluate plan by simulation, returning a shaped cost (Chunk 4).
    Lower is better. Integrates success/crash shaping and mid‑flight penalties.
    """
    res = simulate(state0, plan, terrain)
    flat = terrain.flat
    x_center = 0.5 * (flat.xL + flat.xR)

    # Mid-flight penalties
    aux = res.cost_aux or {}
    oob_dist = float(aux.get('oob_dist', 0.0))
    ground_def = float(aux.get('ground_def', 0.0))
    ang_sum = float(aux.get('ang_sum', 0.0))
    bang_steps = float(aux.get('bang_steps', 0.0))
    eta_m = float(aux.get('eta_mismatch', 0.0))
    corr_p = float(aux.get('corr_penalty', 0.0))

    vx_excess_mid = float(aux.get('vx_excess_mid', 0.0))
    edge_band = float(aux.get('edge_band', 0.0))
    mid_pen = (W_OOB * oob_dist + W_GROUND * ground_def + W_ANG_MID * ang_sum
               + W_BANG * bang_steps + W_ETA * eta_m + W_CORR * corr_p
               + W_VX_MID * vx_excess_mid + W_EDGE_BAND_MID * edge_band)

    if res.success and res.touch_state is not None:
        fuel_used = res.fuel_used
        time_pen = W_TIME * res.steps_simulated
        return fuel_used + time_pen + mid_pen

    # Crash cost shaping
    s = res.touch_state if res.touch_state is not None else res.touch_state
    if s is None:
        # No touch state, rely on OOB flag and mid-flight pen
        return C_CRASH + (OOB_HARD_PEN if res.oob else 50_000.0) + mid_pen

    # Distance to center and off-flat penalty
    dist_center = abs(s.x - x_center)
    outside_flat = 0.0
    if not (flat.xL <= s.x <= flat.xR):
        outside_flat = min(abs(s.x - flat.xL), abs(s.x - flat.xR))
    ang_pen_touch = abs(s.angle)
    vx_excess = max(0.0, abs(s.vx) - MAX_VX_LAND)
    vy_excess = max(0.0, abs(s.vy) - MAX_VY_LAND)
    y_below = max(0.0, flat.y - s.y)

    return (C_CRASH + W_ZONE * dist_center + W_EDGE * outside_flat + W_ANG_TOUCH * ang_pen_touch
            + W_VX * vx_excess * vx_excess + W_VY * vy_excess * vy_excess + W_Y_BELOW * y_below
            + (OOB_HARD_PEN if res.oob else 0.0) + mid_pen)


def evaluate_candidates(state0: State, terrain: Terrain, H: int, k_top: int = 4):
    """Build, expand, ramp, guard, and score candidates including a follower seed.
    Returns a list of tuples (cost, name, plan, result) sorted by cost.
    """
    cps = generate_candidates(state0, terrain, H)
    evals = []
    for cp in cps:
        plan = build_plan_from_cp(state0, terrain, cp, H)
        c = score(state0, plan, terrain)
        evals.append((c, cp.name, plan))

    # Add waypoint follower seed
    follower_plan = follow_waypoints(state0, terrain, waypoints=None, H=H)
    c_f = score(state0, follower_plan, terrain)
    evals.append((c_f, "follower", follower_plan))

    evals.sort(key=lambda x: x[0])
    return evals[:k_top]


# --- Chunk 3: SA with Control Points, Multi-Seed, and Warm Start ---
# Module-level memory for rolling horizon
_PREV_BEST_CP: Optional[ControlPointPlan] = None


def _cp_clone(plan: ControlPointPlan) -> ControlPointPlan:
    return ControlPointPlan(cps=[ControlPoint(cp.angle, cp.power, cp.dur) for cp in plan.cps], name=plan.name, meta=dict(plan.meta))


def _cp_total_seconds(plan: ControlPointPlan) -> int:
    return sum(max(0, int(cp.dur)) for cp in plan.cps)


def _cp_trim_to_H(plan: ControlPointPlan, H: int) -> ControlPointPlan:
    cps: List[ControlPoint] = []
    total = 0
    for cp in plan.cps:
        dur = int(max(0, cp.dur))
        if dur <= 0:
            continue
        if total + dur <= H:
            cps.append(ControlPoint(cp.angle, cp.power, dur))
            total += dur
        else:
            left = max(0, H - total)
            if left > 0:
                cps.append(ControlPoint(cp.angle, cp.power, left))
                total += left
            break
    if total < H:
        cps.append(ControlPoint(0.0, 0, H - total))
    return ControlPointPlan(cps=cps, name=plan.name, meta=dict(plan.meta))


def _cp_shift_one_second(plan: ControlPointPlan) -> ControlPointPlan:
    if not plan.cps:
        return ControlPointPlan([ControlPoint(0.0, 0, 1)], name="warm_shift")
    cps = [ControlPoint(cp.angle, cp.power, cp.dur) for cp in plan.cps]
    # Remove 1 sec from head
    while cps and cps[0].dur <= 0:
        cps.pop(0)
    if not cps:
        cps = [ControlPoint(0.0, 0, 1)]
    else:
        cps[0].dur -= 1
        if cps[0].dur <= 0:
            cps.pop(0)
    if not cps:
        cps = [ControlPoint(0.0, 0, 1)]
    return ControlPointPlan(cps=cps, name=plan.name + "_shift", meta=dict(plan.meta))


def _ensure_k_limit(plan: ControlPointPlan, Kmax: int = 12) -> ControlPointPlan:
    cps = [ControlPoint(cp.angle, cp.power, cp.dur) for cp in plan.cps]
    while len(cps) > Kmax:
        # Merge the shortest CP into its neighbor to reduce K
        idx = min(range(len(cps)), key=lambda i: cps[i].dur)
        if idx > 0:
            cps[idx - 1].dur += cps[idx].dur
            cps.pop(idx)
        elif idx + 1 < len(cps):
            cps[idx + 1].dur += cps[idx].dur
            cps.pop(idx)
        else:
            cps.pop()
    return ControlPointPlan(cps=cps, name=plan.name, meta=dict(plan.meta))


def repair_cp_plan(plan: ControlPointPlan, state0: State, terrain: Terrain, H: int) -> ControlPointPlan:
    # Ensure durations and K limit
    plan = _cp_trim_to_H(plan, H)
    plan = _ensure_k_limit(plan, 12)

    # Tail upright requirement
    t_rot = abs(state0.angle) / DTHETA_MAX
    N_tail = max(3, int(math.ceil(t_rot)) + 2)
    cps = [ControlPoint(cp.angle, cp.power, cp.dur) for cp in plan.cps]
    total = _cp_total_seconds(plan)
    # Ensure last CP upright and power high
    if cps:
        last = cps[-1]
        if abs(last.angle) > 1e-6 or last.power < 3:
            # Replace/append an upright flare tail
            tail_needed = max(N_tail, min(12, int(0.25 * H)))
            if last.dur >= tail_needed:
                cps[-1] = ControlPoint(0.0, 4, last.dur)
            else:
                # Extend tail if possible by borrowing from previous cp
                borrow = 0
                if len(cps) >= 2 and cps[-2].dur > 1:
                    borrow = min(cps[-2].dur - 1, tail_needed - last.dur)
                    cps[-2].dur -= borrow
                cps[-1] = ControlPoint(0.0, 4, last.dur + borrow)
                if cps[-1].dur < tail_needed:
                    cps.append(ControlPoint(0.0, 4, tail_needed - cps[-1].dur))
    else:
        cps = [ControlPoint(0.0, 4, max(N_tail, min(12, int(0.25 * H))))]

    # Vertical stopping feasibility check (coarse)
    h = max(0.0, state0.y - terrain.flat.y)
    dy_stop = vertical_stopping(state0.vy)
    if h <= dy_stop + 80.0:
        # Increase tail early by converting the penultimate CP to upright 4 for a few seconds
        if len(cps) >= 2:
            add = min(4, max(0, cps[-2].dur - 1))
            cps[-2].dur -= add
            cps.insert(-1, ControlPoint(0.0, 4, add))
        else:
            cps[0].dur += 3

    repaired = ControlPointPlan(cps=cps, name=plan.name + "_repaired", meta=dict(plan.meta))
    repaired = _cp_trim_to_H(repaired, H)
    repaired = _ensure_k_limit(repaired, 12)
    return repaired


def mutate_cp(plan: ControlPointPlan, low_fuel_mode: bool = False) -> ControlPointPlan:
    cps = [ControlPoint(cp.angle, cp.power, cp.dur) for cp in plan.cps]
    if not cps:
        cps = [ControlPoint(0.0, 0, 1)]
    choice = random.random()

    def snap_power(p: int) -> int:
        if low_fuel_mode and random.random() < 0.7:
            return 4 if random.random() < 0.6 else 0
        if random.random() < 0.25:
            return 4 if random.random() < 0.5 else 0
        return int(clamp(p, P_MIN, P_MAX))

    if choice < 0.35:
        # Angle tweak on a random CP (avoid last tail half the time)
        idx = random.randrange(len(cps))
        if idx == len(cps) - 1 and random.random() < 0.5:
            idx = max(0, idx - 1)
        delta = random.choice([-15, -10, -5, 5, 10, 15])
        cps[idx].angle = clamp(cps[idx].angle + delta, -45.0, 45.0)
    elif choice < 0.6:
        # Power tweak
        idx = random.randrange(len(cps))
        if idx == len(cps) - 1 and random.random() < 0.5:
            idx = max(0, idx - 1)
        cps[idx].power = snap_power(cps[idx].power + random.choice([-1, 1]))
    elif choice < 0.8:
        # Duration tweak
        idx = random.randrange(len(cps))
        d = random.choice([-3, -2, -1, 1, 2, 3])
        cps[idx].dur = max(1, cps[idx].dur + d)
    else:
        # Swap neighbors or split/merge
        if len(cps) >= 2 and random.random() < 0.6:
            i = random.randrange(len(cps) - 1)
            cps[i], cps[i + 1] = cps[i + 1], cps[i]
        else:
            # Insert a neutral or merge two small ones
            if len(cps) < 12 and random.random() < 0.5:
                i = random.randrange(len(cps) + 1)
                cps.insert(i, ControlPoint(0.0, 0 if not low_fuel_mode else 4, random.randint(1, 3)))
            elif len(cps) >= 2:
                i = random.randrange(len(cps) - 1)
                cps[i].dur += cps[i + 1].dur
                cps.pop(i + 1)

    mutated = ControlPointPlan(cps=cps, name=plan.name + "_m", meta=dict(plan.meta))
    return mutated


def cp_score(state0: State, terrain: Terrain, cp_plan: ControlPointPlan, H: int) -> float:
    cp_repaired = repair_cp_plan(cp_plan, state0, terrain, H)
    plan = build_plan_from_cp(state0, terrain, cp_repaired, H)
    return score(state0, plan, terrain)


def cp_score_repaired(state0: State, terrain: Terrain, cp_plan_repaired: ControlPointPlan, H: int) -> float:
    """Score a plan that is already repaired; avoids an extra repair call during SA iterations."""
    plan = build_plan_from_cp(state0, terrain, cp_plan_repaired, H)
    return score(state0, plan, terrain)


# --- Macro-chunk planning (Chunked MPC) ---
@dataclass
class MacroChunk:
    angle: float   # deg
    power: int     # 0..4
    dur: int       # seconds (>=1)


def _roll_chunk_preview(state0: State, terrain: Terrain, angle: float, power: int, dur: int) -> Tuple[State, List[Command]]:
    """Preview-execute a constant-thrust chunk under ramp+guards and return the end-state and realized commands.
    The execution mirrors runtime: ramp -> guards per second -> integrate physics -> fuel burn.
    """
    dur = max(1, int(dur))
    desired = [Command(angle=angle, power=int(clamp(power, P_MIN, P_MAX))) for _ in range(dur)]
    ramped = ramp_enforce(desired, state0.angle, state0.power)
    realized = _preview_apply_guards(state0, ramped, terrain)

    # Integrate to end-state
    s_tmp = State(**vars(state0))
    for cmd in realized:
        th = math.radians(cmd.angle)
        ax = cmd.power * math.sin(th)
        ay = cmd.power * math.cos(th) - G
        vx_prev, vy_prev = s_tmp.vx, s_tmp.vy
        s_tmp.vx = s_tmp.vx + ax
        s_tmp.vy = s_tmp.vy + ay
        s_tmp.x = s_tmp.x + vx_prev + 0.5 * ax
        s_tmp.y = s_tmp.y + vy_prev + 0.5 * ay
        s_tmp.fuel = max(0.0, s_tmp.fuel - cmd.power)
        s_tmp.angle = cmd.angle
        s_tmp.power = cmd.power
    return s_tmp, realized


def _cp_from_chunks(chunks: List[MacroChunk]) -> ControlPointPlan:
    cps = [ControlPoint(angle=c.angle, power=int(clamp(c.power, 0, 4)), dur=int(max(1, c.dur))) for c in chunks]
    return ControlPointPlan(cps=cps, name="macro")


def _macro_chunk_options(state: State, terrain: Terrain, x_target: float) -> List[Tuple[float, int]]:
    # Conservative angle set bounded by tilt_allow
    cap = max(5.0, min(35.0, tilt_allow(state, terrain)))
    angles = [-min(cap, 25.0), -15.0, -5.0, 0.0, 5.0, 15.0, min(cap, 25.0)]

    # Discrete powers. Near ground, include more bins for finer control; higher up keep it sparse.
    h = max(0.0, state.y - terrain.flat.y)
    powers = [0, 2, 4] if h > 700 else [0, 1, 3, 4]

    # Ordering bias toward target direction helps the beam keep better nodes
    dx = x_target - state.x
    if abs(dx) > 1e-6:
        sign_to_target = 1.0 if dx > 0 else -1.0
        angles.sort(key=lambda a: -sign_to_target * a)

    return [(a, p) for a in angles for p in powers]


def build_chunked_plan(state0: State, terrain: Terrain, H: int = 60,
                       chunk_len_hi: int = 6, chunk_len_lo: int = 4,
                       depth_max: int = 3, beam_k: int = 6,
                       x_target: Optional[float] = None) -> ControlPointPlan:
    """Tiny beam search over constant-thrust chunks that preview-runs ramp+guards and picks a smooth macro-plan.
    Returns a `ControlPointPlan` (sequence of CPs) suitable for `build_plan_from_cp`.
    """
    flat = terrain.flat
    x_center = 0.5 * (flat.xL + flat.xR)
    if x_target is None:
        x_target = x_center

    @dataclass
    class _Node:
        s: State
        cps: List[MacroChunk]
        T: int
        hcost: float

    def heuristic(s: State) -> float:
        # Cheap smoothness + guidance: distance to center, vx envelope excess, vertical profile mismatch, corridor
        h = max(0.0, s.y - flat.y)
        dx = abs(s.x - x_target)
        vx_env = vmax_by_alt(s.y, flat.y)
        vx_term = max(0.0, abs(s.vx) - 0.5 * vx_env)
        vy_tgt = vy_des_by_alt(s.y, flat.y)
        vy_term = max(0.0, abs(abs(s.vy) - abs(vy_tgt)) - 5.0)
        # Corridor penalty near ground
        corr = 0.0
        if h < ALT_CORR_THRESH:
            left = flat.xL + CORR_MARGIN
            right = flat.xR - CORR_MARGIN
            if s.x < left:
                corr = (left - s.x) * (1.0 - h / ALT_CORR_THRESH)
            elif s.x > right:
                corr = (s.x - right) * (1.0 - h / ALT_CORR_THRESH)
        # Predictive braking pressure: penalize if we won't have time to bleed |vx|
        must_b, t_need, t_avail, _ = predictive_brake_gate(s, terrain)
        pb = max(0.0, t_need - t_avail)
        # Low-altitude |vx| over limit penalty
        low_vx_pen = 0.0
        if h < 600.0:
            low_vx_pen = max(0.0, abs(s.vx) - MAX_VX_LAND)
        return 0.001 * dx + 0.02 * vx_term + 0.02 * vy_term + 0.001 * corr + 0.35 * pb + 0.05 * low_vx_pen

    def chunk_len_for(s: State) -> int:
        h = max(0.0, s.y - flat.y)
        return chunk_len_hi if h > 900 else chunk_len_lo

    start_node = _Node(s=State(**vars(state0)), cps=[], T=0, hcost=heuristic(state0))
    best_nodes: List[_Node] = [start_node]

    # Keep beam runtime small; leave time for SA
    deadline = time.perf_counter() + 0.015  # ~15 ms (further reduced for runtime)

    for _depth in range(depth_max):
        if time.perf_counter() > deadline:
            break
        next_nodes: List[_Node] = []
        for node in best_nodes:
            L = min(chunk_len_for(node.s), H - node.T)
            if L <= 0:
                next_nodes.append(node)
                continue
            # Predictive brake filter
            must_b, _tn, _ta, _dir = predictive_brake_gate(node.s, terrain)
            opts = _macro_chunk_options(node.s, terrain, x_target)
            if must_b:
                vx = node.s.vx
                # Keep only braking actions (angle sign opposes vx) with some authority
                filtered = []
                for ang, pw in opts:
                    if abs(ang) < 5.0 or pw <= 0:
                        continue
                    if (vx < 0 and ang > 0) or (vx > 0 and ang < 0):
                        if pw >= 2:
                            filtered.append((ang, pw))
                if filtered:
                    opts = filtered
            for ang, pw in opts:
                s1, _realized = _roll_chunk_preview(node.s, terrain, ang, pw, L)
                cps1 = node.cps + [MacroChunk(angle=ang, power=pw, dur=L)]
                T1 = node.T + L
                next_nodes.append(_Node(s=s1, cps=cps1, T=T1, hcost=heuristic(s1)))
        next_nodes.sort(key=lambda n: n.hcost)
        best_nodes = next_nodes[:beam_k]
        if not best_nodes:
            break

    # Finalize: add a short upright tail, then score with exact simulator cost and cost shaping
    finalists: List[Tuple[float, ControlPointPlan]] = []
    for node in best_nodes[:min(3, beam_k)]:  # limit finalist scoring to at most 3 to save time
        cps = list(node.cps)
        tail = max(6, min(12, int(0.25 * H)))
        if node.T < H:
            cps.append(MacroChunk(0.0, 4, min(tail, H - node.T)))
        cp_plan = _cp_from_chunks(cps)
        plan = build_plan_from_cp(state0, terrain, cp_plan, H)
        c = score(state0, plan, terrain)
        finalists.append((c, cp_plan))

    if not finalists:
        return ControlPointPlan([ControlPoint(0.0, 4, H)], name="macro_fallback")

    finalists.sort(key=lambda t: t[0])
    best_cp = finalists[0][1]
    best_cp.name = "macro_chunk"
    return best_cp


def optimize_multi_seed(state0: State, terrain: Terrain, H: int, budget_ms: float = 70.0,
                        max_seeds: int = 2) -> ControlPointPlan:
    global _PREV_BEST_CP
    low_fuel_mode = state0.fuel < 200 or (state0.fuel / max(1.0, H)) < 2.0
    _LAST_STATS['low_fuel'] = bool(low_fuel_mode)

    # Build seeds from candidates
    seeds = generate_candidates(state0, terrain, H)

    # Warm start: include shifted previous best if available
    if _PREV_BEST_CP is not None:
        warm = _cp_shift_one_second(_PREV_BEST_CP)
        warm = _cp_trim_to_H(warm, H)
        seeds.insert(0, ControlPointPlan(cps=warm.cps, name="warm", meta={}))

    # Add chunked-MPC macro candidate as an extra seed
    try:
        macro_cp = build_chunked_plan(state0, terrain, H=H, depth_max=2, beam_k=4)
        seeds.append(macro_cp)
    except Exception:
        pass

    # Score seeds quickly and pick top few
    scored = []
    for cp in seeds:
        c = cp_score(state0, terrain, cp, H)
        scored.append((c, cp))
    scored.sort(key=lambda t: t[0])
    seed_cost_by_id = {id(cp): c for c, cp in scored}
    if low_fuel_mode:
        max_seeds = min(max_seeds, 2)
    picked = [cp for _, cp in scored[:max_seeds]]
    _LAST_STATS['picked'] = [cp.name for cp in picked]

    # Initialize SA states per seed
    start = time.perf_counter()
    deadline = start + budget_ms / 1000.0

    best_global_cost = float('inf')
    best_global_cp = picked[0] if picked else ControlPointPlan([ControlPoint(0.0, 4 if low_fuel_mode else 0, H)], name="fallback")

    # Establish initial best cost for T0 (reuse precomputed seed scores)
    if picked:
        best_initial_cost = min(seed_cost_by_id.get(id(cp), cp_score(state0, terrain, cp, H)) for cp in picked)
    else:
        best_initial_cost = cp_score(state0, terrain, best_global_cp, H)

    # Temperature schedule
    T0 = max(1e-6, 0.1 * best_initial_cost)
    alpha = 0.992 if not low_fuel_mode else 0.995

    # Per-seed states
    per_seed = []
    for cp in picked:
        cur = repair_cp_plan(cp, state0, terrain, H)
        cur_cost = seed_cost_by_id.get(id(cp), cp_score_repaired(state0, terrain, cur, H))
        per_seed.append({
            'cur': cur,
            'cur_cost': cur_cost,
            'best': cur,
            'best_cost': cur_cost,
            'T': T0,
        })
        if cur_cost < best_global_cost:
            best_global_cost = cur_cost
            best_global_cp = cur

    # Round-robin SA steps until deadline
    idx = 0
    sa_iters = 0
    while time.perf_counter() < deadline and per_seed:
        s = per_seed[idx]
        # Propose mutation
        cand = mutate_cp(s['cur'], low_fuel_mode=low_fuel_mode)
        cand = repair_cp_plan(cand, state0, terrain, H)
        cost_cand = cp_score_repaired(state0, terrain, cand, H)
        d = cost_cand - s['cur_cost']
        if d <= 0 or random.random() < math.exp(-d / max(1e-9, s['T'])):
            s['cur'], s['cur_cost'] = cand, cost_cand
        if cost_cand < s['best_cost']:
            s['best'], s['best_cost'] = cand, cost_cand
            if cost_cand < best_global_cost:
                best_global_cost = cost_cand
                best_global_cp = cand
        # Cool down a bit
        s['T'] *= alpha
        sa_iters += 1
        # Next seed
        idx = (idx + 1) % len(per_seed)

    _LAST_STATS['sa_iters'] = sa_iters
    _LAST_STATS['best_cost'] = float(best_global_cost) if best_global_cost != float('inf') else None

    # Save for warm start next turn
    _PREV_BEST_CP = best_global_cp
    return best_global_cp


def build_and_get_first_command(state0: State, terrain: Terrain, H: int = 60, budget_ms: float = 70.0) -> Command:
    """Rolling-horizon driver: optimize within budget and return the first command of the best plan.
    Applies ramp_enforce and runtime guards via build_plan_from_cp.
    """
    best_cp = optimize_multi_seed(state0, terrain, H=H, budget_ms=budget_ms)
    _LAST_STATS['best_name'] = best_cp.name if hasattr(best_cp, 'name') else None
    final_plan = build_plan_from_cp(state0, terrain, best_cp, H)
    if not final_plan:
        _LAST_STATS['emerg_used'] = True
        return Command(angle=0.0, power=4)
    # Emergency override for the immediate command if needed
    emerg = emergency_policy(state0, terrain)
    if emerg is not None:
        _LAST_STATS['emerg_used'] = True
        return emerg
    _LAST_STATS['emerg_used'] = False
    return apply_runtime_guards(state0, final_plan[0], terrain)


# --- CodinGame I/O loop ---

def run_codingame():
    # Read surface
    try:
        line = sys.stdin.readline()
        if not line:
            return
        surfaceN = int(line.strip())
        pts = []
        for _ in range(surfaceN):
            l = sys.stdin.readline()
            if not l:
                return
            x_str, y_str = l.strip().split()
            pts.append((float(x_str), float(y_str)))
        terrain = Terrain(pts)

        # Game loop
        H = 60
        while True:
            row = sys.stdin.readline()
            if not row:
                break
            parts = row.strip().split()
            if len(parts) < 7:
                # CodinGame may call an extra read at end; exit quietly
                break
            X, Y, hSpeed, vSpeed, fuel, rotate, power = map(int, parts)
            # Map CodinGame rotate (positive = thrust to the LEFT) to internal angle (positive = thrust to the RIGHT)
            state = State(x=float(X), y=float(Y), vx=float(hSpeed), vy=float(vSpeed),
                          fuel=float(fuel), angle=-float(rotate), power=int(power))

            t0 = time.perf_counter()
            profile_reset()
            cmd = build_and_get_first_command(state, terrain, H=H, budget_ms=70.0)
            t1 = time.perf_counter()

            # Output command (stdout)
            # Map internal angle (positive = thrust to RIGHT) to CodinGame rotate (positive = thrust to LEFT)
            ang_out = int(round(clamp(-cmd.angle, THETA_MIN, THETA_MAX)))
            pow_out = int(clamp(cmd.power, P_MIN, P_MAX))
            print(f"{ang_out} {pow_out}")
            # Debug to stderr
            elapsed_ms = (t1 - t0) * 1000.0
            log_err(f"ms={elapsed_ms:.1f} sim_calls={_PROFILE['simulate_calls']} sim_steps={_PROFILE['sim_steps']} oob={_PROFILE['oob_events']} seeds={_LAST_STATS['picked']} iters={_LAST_STATS['sa_iters']} best={_LAST_STATS['best_cost']}")
            sys.stdout.flush()
    except Exception as e:
        log_err(f"Error in run_codingame: {e}")


if __name__ == "__main__":
    # Default to CodinGame loop for actual runs; set DEMO=1 to run the internal demo
    if sys.argv[1:] and sys.argv[1] == "--demo" or (os.environ.get("DEMO", "0") == "1"):
        # Minimal demo terrain and a single command print for sanity
        terrain = Terrain([
            (0.0, 2500.0), (1000.0, 2000.0), (2000.0, 1500.0), (3500.0, 1500.0), (5000.0, 2000.0), (6999.0, 2500.0)
        ])
        s0 = State(x=1000.0, y=2500.0, vx=0.0, vy=-20.0, fuel=1000.0, angle=0.0, power=0)
        cmd = build_and_get_first_command(s0, terrain, H=60, budget_ms=50.0)
        print(int(round(cmd.angle)), cmd.power)
    else:
        run_codingame()



