from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Terrain:
    segments: List[Segment]
    flat_idx: int  # index of the unique flat (horizontal) segment

    @property
    def flat_xrange(self) -> Tuple[float, float]:
        s = self.segments[self.flat_idx]
        return (min(s.x1, s.x2), max(s.x1, s.x2))

    @property
    def y_flat(self) -> float:
        s = self.segments[self.flat_idx]
        return s.y1  # flat, so y1 == y2

    def nearest_point_on_flat(self, x: float, y: float) -> Tuple[float, float]:
        """Project (x,y) to the closest point on the flat segment and return it."""
        fx1, fx2 = self.flat_xrange
        nx = min(max(x, fx1), fx2)
        ny = self.y_flat
        return (nx, ny)

    def distance_to_flat(self, x: float, y: float) -> float:
        """Euclidean distance from (x,y) to the flat segment."""
        nx, ny = self.nearest_point_on_flat(x, y)
        dx = x - nx
        dy = y - ny
        return (dx * dx + dy * dy) ** 0.5

    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> "Terrain":
        if len(points) < 2:
            raise ValueError("Need at least two surface points to build terrain")
        segs: List[Segment] = []
        flat_indices: List[int] = []
        for i in range(len(points) - 1):
            (x1, y1) = points[i]
            (x2, y2) = points[i + 1]
            segs.append(Segment(float(x1), float(y1), float(x2), float(y2)))
            if y1 == y2:
                flat_indices.append(i)
        if not flat_indices:
            raise ValueError("No flat segment found in terrain points")
        if len(flat_indices) > 1:
            # CodinGame Mars Lander has exactly one flat segment.
            raise ValueError("Multiple flat segments found; expected exactly one")
        return cls(segs, flat_indices[0])

    @staticmethod
    def _cross(ax: float, ay: float, bx: float, by: float) -> float:
        return ax * by - ay * bx

    def intersect(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[bool, int, Tuple[float, float]]:
        """
        Compute the first intersection (if any) between the path segment p1->p2 and any terrain segment.
        Returns: (hit: bool, seg_idx: int, hit_point: (x, y)). If no hit, seg_idx = -1 and point = (0.0, 0.0).
        """
        px, py = p1
        rx, ry = p2[0] - px, p2[1] - py
        best_t: Optional[float] = None
        best_idx: int = -1
        best_pt: Tuple[float, float] = (0.0, 0.0)

        for idx, s in enumerate(self.segments):
            qx, qy = s.x1, s.y1
            sx_, sy_ = s.x2 - s.x1, s.y2 - s.y1
            rxs = self._cross(rx, ry, sx_, sy_)
            q_p_x, q_p_y = qx - px, qy - py
            q_pxr = self._cross(q_p_x, q_p_y, rx, ry)
            if rxs == 0.0:
                # Parallel (including colinear). For our purposes, ignore colinear overlaps as rare in maps
                continue
            t = self._cross(q_p_x, q_p_y, sx_, sy_) / rxs
            u = q_pxr / rxs
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                # Intersection at p = p1 + t*r
                if best_t is None or t < best_t:
                    ix = px + t * rx
                    iy = py + t * ry
                    best_t = t
                    best_idx = idx
                    best_pt = (ix, iy)

        if best_t is None:
            return False, -1, (0.0, 0.0)
        return True, best_idx, best_pt
