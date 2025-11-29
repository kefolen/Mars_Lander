import unittest

from terrain import Terrain
from physics import State, step, simulate_chromosome


class TestPhysics(unittest.TestCase):
    def make_flat_ground(self) -> Terrain:
        # Flat ground from x=-1000 to x=1000 at y=0
        pts = [(-1000.0, 0.0), (1000.0, 0.0)]
        return Terrain.from_points(pts)

    def test_rate_limits_angle_power(self):
        t = self.make_flat_ground()
        s0 = State(x=0.0, y=2500.0, vx=0.0, vy=0.0, fuel=100, angle=0.0, power=0)
        # Command a big change; should be limited to ±15 deg and ±1 power per step
        s1 = step(s0, angle_cmd=50.0, power_cmd=4.0)
        self.assertEqual(s1.angle, 15.0)  # limited by 15 deg/s
        self.assertIn(s1.power, (1,))      # power increases by at most 1 and quantized to int

        # Next step toward command again
        s2 = step(s1, angle_cmd=50.0, power_cmd=4.0)
        self.assertEqual(s2.angle, 30.0)
        self.assertIn(s2.power, (2,))

    def test_one_step_kinematics(self):
        t = self.make_flat_ground()
        s0 = State(x=0.0, y=2500.0, vx=10.0, vy=-20.0, fuel=100, angle=0.0, power=2)
        s1 = step(s0, angle_cmd=0.0, power_cmd=2.0)
        # With angle=0, ax=0, ay=2*cos(0)-g = 2-3.711 = -1.711
        # vx1 = 10, vy1 = -21.711 approximately; x1 = x0 + vx0 + 0.5*ax = 0 + 10 + 0 = 10
        self.assertAlmostEqual(s1.vx, 10.0, places=6)
        self.assertAlmostEqual(s1.vy, -21.711, places=3)
        self.assertAlmostEqual(s1.x, 10.0, places=6)
        # y1 = 2500 + (-20) + 0.5*(-1.711) ≈ 2479.1445
        self.assertAlmostEqual(s1.y, 2479.1445, places=3)

    def test_valid_landing_detection(self):
        t = self.make_flat_ground()
        # Start just above the flat with gentle downward speed and upright
        s0 = State(x=0.0, y=10.0, vx=0.0, vy=-10.0, fuel=50, angle=0.0, power=0)
        # One gene is enough to cross the flat line; with angle=0 and power=0, vy increases by -g
        chrom = [(0.0, 0.0)]
        out = simulate_chromosome(s0, chrom, t)
        self.assertTrue(out.landed, "Should be a valid landing on flat with safe speeds and angle")
        self.assertFalse(out.crashed)
        self.assertTrue(out.on_flat)


if __name__ == "__main__":
    unittest.main()