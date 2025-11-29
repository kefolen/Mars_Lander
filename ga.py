from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from terrain import Terrain
from physics import State, Gene
from evaluation import FitnessWeights, evaluate_sequence


Chromosome = List[Gene]


@dataclass
class Individual:
    chrom: Chromosome
    fitness: float = 0.0
    cost: float = float("inf")
    outcome: Optional[object] = None  # physics.Outcome, kept as object to avoid import cycles


class GA:
    """
    Genetic Algorithm core engine using continuous genes for Mars Lander L2.

    - Gene: (angle_cmd, power_cmd) as floats; simulator enforces rate limits and clamps.
    - Selection: roulette wheel with cumulative fitness + bisect lookup.
    - Crossover: arithmetic per gene with random alpha in [0,1].
    - Mutation: per-gene probability with Gaussian noise and clamping.
    - Elitism: keep top elite_frac (by lowest cost) into next generation.

    Milestone 5 tuning hooks:
    - Biased initialization toward low thrust to promote fuel efficiency
    - Mutation annealing within a turn via set_anneal_progress(progress)
    - Runtime parameter updates via update_params()
    """

    def __init__(
        self,
        pop_size: int = 100,
        horizon: int = 90,
        elite_frac: float = 0.15,
        pm: float = 0.01,
        sigma_angle: float = 7.0,
        sigma_power: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.N = int(pop_size)
        self.H = int(horizon)
        self.pm = float(pm)
        # Store base and current sigmas for annealing
        self.base_sigma_angle = float(sigma_angle)
        self.base_sigma_power = float(sigma_power)
        self.min_sigma_angle = 2.5
        self.min_sigma_power = 0.25
        self.sigma_angle = float(sigma_angle)
        self.sigma_power = float(sigma_power)
        self.elite_frac = float(elite_frac)
        self.elite = max(1, int(round(self.elite_frac * self.N)))
        self.rng = rng if rng is not None else random.Random()

        self.pop: List[Individual] = []
        self.best: Optional[Individual] = None
        # Reusable cumulative array to reduce allocations (micro-optimization)
        self._cumulative: List[float] = []

    # ---------- population initialization ----------
    def _rand_gene(self) -> Gene:
        # Angles biased to [-45, 45] initially; powers biased toward low thrust (fuel saving)
        a = self.rng.uniform(-45.0, 45.0)
        # Discrete bias for power levels with slight noise
        levels = [0.0, 1.0, 2.0, 3.0, 4.0]
        weights = [0.30, 0.30, 0.20, 0.15, 0.05]
        r = self.rng.random()
        cum = 0.0
        idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                idx = i
                break
        p = levels[idx] + self.rng.uniform(-0.2, 0.2)
        if p < 0.0:
            p = 0.0
        if p > 4.0:
            p = 4.0
        return (a, p)

    def _smooth(self, chrom: Chromosome) -> Chromosome:
        # Mild EMA smoothing to encourage rate-limit-compatible changes
        if not chrom:
            return chrom
        out: Chromosome = [chrom[0]]
        for i in range(1, len(chrom)):
            a_prev, p_prev = out[-1]
            a, p = chrom[i]
            out.append((0.7 * a_prev + 0.3 * a, 0.7 * p_prev + 0.3 * p))
        return out

    # ---------- tuning hooks (Milestone 5) ----------
    def set_anneal_progress(self, progress: float) -> None:
        """
        Set annealing progress in [0,1], where 0 keeps base sigmas and 1 reaches min sigmas.
        This is intended to be called repeatedly within a turn as the deadline approaches.
        """
        if progress is None:
            return
        try:
            p = max(0.0, min(1.0, float(progress)))
        except Exception:
            p = 0.0
        # Linear interpolation from base -> min
        self.sigma_angle = self.min_sigma_angle + (self.base_sigma_angle - self.min_sigma_angle) * (1.0 - p)
        self.sigma_power = self.min_sigma_power + (self.base_sigma_power - self.min_sigma_power) * (1.0 - p)
        # Optionally reduce mutation probability slightly as we anneal
        self.pm = max(0.004, min(self.pm, 0.012))  # keep within reasonable bounds

    def update_params(
        self,
        pop_size: Optional[int] = None,
        horizon: Optional[int] = None,
        elite_frac: Optional[float] = None,
        pm: Optional[float] = None,
        sigma_angle: Optional[float] = None,
        sigma_power: Optional[float] = None,
    ) -> None:
        """Update GA parameters (applies to subsequent init/population cycles)."""
        if pop_size is not None and pop_size > 0:
            self.N = int(pop_size)
            self.elite = max(1, int(round(self.elite_frac * self.N)))
        if horizon is not None and horizon > 0:
            self.H = int(horizon)
        if elite_frac is not None and elite_frac > 0:
            self.elite_frac = float(elite_frac)
            self.elite = max(1, int(round(self.elite_frac * self.N)))
        if pm is not None and pm > 0:
            self.pm = float(pm)
        if sigma_angle is not None and sigma_angle > 0:
            self.base_sigma_angle = float(sigma_angle)
            self.sigma_angle = float(sigma_angle)
        if sigma_power is not None and sigma_power > 0:
            self.base_sigma_power = float(sigma_power)
            self.sigma_power = float(sigma_power)

    def _mutated_copy(self, chrom: Chromosome) -> Chromosome:
        out = list(chrom)
        for i in range(len(out)):
            if self.rng.random() < self.pm:
                a, p = out[i]
                a += self.rng.gauss(0.0, self.sigma_angle)
                p += self.rng.gauss(0.0, self.sigma_power)
                # Clamp to legal command ranges (sim will also clamp actuals)
                a = max(-90.0, min(90.0, a))
                p = max(0.0, min(4.0, p))
                out[i] = (a, p)
        return out

    def init_population(self, seed: Optional[Chromosome], seed_copies: int = 8) -> None:
        self.pop.clear()
        self.best = None

        # Prepare base from seed if provided
        if seed:
            base = list(seed)
            # Ensure length exactly H by repeating the last gene or trimming
            if not base:
                base = [self._rand_gene() for _ in range(self.H)]
            if len(base) < self.H:
                last = base[-1]
                base = base + [last] * (self.H - len(base))
            elif len(base) > self.H:
                base = base[: self.H]
            # Add mutated copies of the seed
            for _ in range(max(1, seed_copies)):
                self.pop.append(Individual(self._mutated_copy(base)))

        # Fill the rest randomly with mild smoothing
        while len(self.pop) < self.N:
            chrom = [self._rand_gene() for _ in range(self.H)]
            chrom = self._smooth(chrom)
            self.pop.append(Individual(chrom))

    # ---------- evaluation ----------
    def evaluate(
        self,
        s0: State,
        terrain: Terrain,
        weights: Optional[FitnessWeights] = None,
    ) -> None:
        if weights is None:
            weights = FitnessWeights()

        sum_fit = 0.0
        best: Optional[Individual] = self.best

        for ind in self.pop:
            if ind.outcome is None:
                out, cost, fit = evaluate_sequence(s0, ind.chrom, terrain, weights)
                ind.outcome = out
                ind.cost = float(cost)
                ind.fitness = float(fit)
            sum_fit += ind.fitness
            if best is None or ind.cost < best.cost:
                best = ind

        # Normalize fitnesses for roulette selection; protect against zeros
        if sum_fit <= 0.0 or not math.isfinite(sum_fit):
            # Assign uniform positive fitness
            unif = 1.0 / float(len(self.pop)) if self.pop else 1.0
            for ind in self.pop:
                ind.fitness = unif
        else:
            inv_sum = 1.0 / sum_fit
            for ind in self.pop:
                ind.fitness *= inv_sum

        self.best = best

    # ---------- selection / crossover / mutation ----------
    def _build_cumulative(self) -> List[float]:
        # Reuse internal buffer to reduce allocations
        cumulative = self._cumulative
        cumulative.clear()
        s = 0.0
        for ind in self.pop:
            s += max(0.0, ind.fitness)  # ensure non-negative
            cumulative.append(s)
        # If all fitness were zero (shouldn't happen after normalize), fix to uniform
        if s <= 0.0:
            cumulative.clear()
            step = 1.0 / float(len(self.pop)) if self.pop else 1.0
            acc = 0.0
            for _ in self.pop:
                acc += step
                cumulative.append(acc)
        else:
            # Normalize cumulative to end at 1.0
            inv_s = 1.0 / s
            for i in range(len(cumulative)):
                cumulative[i] *= inv_s
        return cumulative

    def _select_idx(self, cumulative: List[float]) -> int:
        u = self.rng.random()
        return bisect.bisect_left(cumulative, u)

    def _crossover(self, a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        c0: Chromosome = []
        c1: Chromosome = []
        for (ang0, pow0), (ang1, pow1) in zip(a, b):
            alpha = self.rng.random()
            c0.append((alpha * ang0 + (1.0 - alpha) * ang1, alpha * pow0 + (1.0 - alpha) * pow1))
            c1.append(((1.0 - alpha) * ang0 + alpha * ang1, (1.0 - alpha) * pow0 + alpha * pow1))
        return c0, c1

    # ---------- generational step ----------
    def next_generation(self) -> None:
        if not self.pop:
            return

        # Prepare roulette cumulative array
        cumulative = self._build_cumulative()

        # Elitism: copy top-K by lowest cost
        elites = sorted(self.pop, key=lambda ind: ind.cost)[: self.elite]
        new_pop: List[Individual] = [Individual(chrom=list(e.chrom)) for e in elites]

        # Refill via selection + crossover + mutation
        while len(new_pop) < self.N:
            i = self._select_idx(cumulative)
            j = self._select_idx(cumulative)
            # ensure two distinct parents (fallback to different index if possible)
            if j == i and len(self.pop) > 1:
                j = (j + 1) % len(self.pop)
            p0 = self.pop[i].chrom
            p1 = self.pop[j].chrom
            c0, c1 = self._crossover(p0, p1)
            c0 = self._mutated_copy(c0)
            c1 = self._mutated_copy(c1)
            new_pop.append(Individual(c0))
            if len(new_pop) < self.N:
                new_pop.append(Individual(c1))

        # Replace population and invalidate outcomes (must be re-evaluated)
        self.pop = new_pop
        for ind in self.pop:
            ind.outcome = None
            ind.fitness = 0.0
            ind.cost = float("inf")
        # Best will be recomputed on next evaluate()
        self.best = None

    # ---------- utility ----------
    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)
