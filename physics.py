from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from terrain import Terrain

G = 3.711  # gravity (m/s^2)


@dataclass
class State:
    """Lander dynamic state using actual (rate-limited) controls from last step."""
    x: float
    y: float
    vx: float
    vy: float
    fuel: int
    angle: float  # degrees, measured from vertical; 0 = upright
    power: int    # 0..4, integer


@dataclass
class Outcome:
    landed: bool
    crashed: bool
    on_flat: bool
    final_state: State
    fuel_used: int
    crash_pos: Optional[Tuple[float, float]]
    hit_seg_idx: int = -1


def _approach(curr: float, target: float, dmax: float) -> float:
    """Move curr toward target by at most dmax, without overshoot."""
    if curr < target:
        return min(target, curr + dmax)
    if curr > target:
        return max(target, curr - dmax)
    return curr


def step(state: State, angle_cmd: float, power_cmd: float) -> State:
    """
    Advance the state by 1 second, applying rate limits to actual angle/power
    toward the commanded values and updating physics with gravity.
    """
    # Apply rate limits toward commands
    ang = _approach(state.angle, angle_cmd, 15.0)
    powf = _approach(float(state.power), float(power_cmd), 1.0)

    # Clamp to legal ranges and quantize power to integer as in game
    ang = max(-90.0, min(90.0, ang))
    powi = int(max(0, min(4, round(powf))))

    # Acceleration from actuals (angle in radians)
    theta = math.radians(ang)
    # Note: In CodinGame, positive rotation (counter-clockwise, left tilt) accelerates left (negative X).
    # Therefore horizontal acceleration uses a negative sine.
    ax = -powi * math.sin(theta)
    ay = powi * math.cos(theta) - G

    # Update velocities
    vx_prev, vy_prev = state.vx, state.vy
    vx = vx_prev + ax
    vy = vy_prev + ay

    # Update positions using previous velocities and half-accel term
    x = state.x + vx_prev + 0.5 * ax
    y = state.y + vy_prev + 0.5 * ay

    # Fuel consumption equals thrust power per second
    fuel = max(0, state.fuel - powi)

    return State(x=x, y=y, vx=vx, vy=vy, fuel=fuel, angle=ang, power=powi)


Gene = Tuple[float, float]  # (angle_cmd, power_cmd)


def simulate_chromosome(s0: State, chromosome: List[Gene], terrain: Terrain) -> Outcome:
    """
    Simulate a sequence of commands (chromosome) from starting state s0 over
    1-second steps until a collision with terrain is detected or the chromosome
    ends. Returns an Outcome with whether it was a valid landing.
    """
    fuel_start = s0.fuel
    prev = s0
    for ang_cmd, pow_cmd in chromosome:
        p1 = (prev.x, prev.y)
        nxt = step(prev, ang_cmd, pow_cmd)
        p2 = (nxt.x, nxt.y)
        hit, seg_idx, hit_pt = terrain.intersect(p1, p2)
        if hit:
            on_flat = (seg_idx == terrain.flat_idx)
            # Landing validity per Task.md: on flat, angle==0°, |vx|<=20, |vy|<=40
            landed = False
            if on_flat:
                if abs(nxt.angle) == 0 and abs(nxt.vx) <= 20.0 and abs(nxt.vy) <= 40.0:
                    landed = True
            return Outcome(
                landed=landed,
                crashed=not landed,
                on_flat=on_flat,
                final_state=nxt,
                fuel_used=fuel_start - nxt.fuel,
                crash_pos=hit_pt,
                hit_seg_idx=seg_idx,
            )
        prev = nxt

    # No collision within horizon — treat as unfinished (count as crash proxy)
    return Outcome(
        landed=False,
        crashed=True,
        on_flat=False,
        final_state=prev,
        fuel_used=fuel_start - prev.fuel,
        crash_pos=None,
        hit_seg_idx=-1,
    )
