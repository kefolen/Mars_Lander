import unittest

from terrain import Terrain
from physics import State
from evaluation import FitnessWeights, evaluate_sequence
from ga import GA


class TestEvaluation(unittest.TestCase):
    def test_landing_better_than_crash(self):
        # Flat ground at y=0
        t = Terrain.from_points([(0.0, 0.0), (2000.0, 0.0)])
        weights = FitnessWeights()
        # A gentle landing within limits
        s_landing = State(x=1000.0, y=5.0, vx=0.0, vy=-10.0, fuel=100, angle=0.0, power=0)
        out1, cost1, fit1 = evaluate_sequence(s_landing, [(0.0, 0.0)], t, weights)
        self.assertTrue(out1.landed)
        # A fast crash (way too fast and tilted off the flat)
        s_crash = State(x=2500.0, y=50.0, vx=50.0, vy=-100.0, fuel=100, angle=45.0, power=0)
        out2, cost2, fit2 = evaluate_sequence(s_crash, [(45.0, 0.0)], t, weights)
        self.assertTrue(out2.crashed)
        self.assertLess(cost1, cost2)
        self.assertGreater(fit1, fit2)


if __name__ == "__main__":
    unittest.main()