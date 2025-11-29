import math
import unittest

from terrain import Terrain, Segment


class TestTerrain(unittest.TestCase):
    def make_simple_terrain(self) -> Terrain:
        # Flat from x=0..2000 at y=1000, with simple slopes up/down around
        points = [
            ( -500.0, 1500.0),
            (    0.0, 1000.0),  # start of flat
            ( 2000.0, 1000.0),  # end of flat
            ( 3000.0, 1500.0),
        ]
        return Terrain.from_points(points)

    def test_flat_xrange_and_y(self):
        t = self.make_simple_terrain()
        self.assertEqual(t.flat_xrange, (0.0, 2000.0))
        self.assertEqual(t.y_flat, 1000.0)

    def test_intersect_hits_flat_when_crossing(self):
        t = self.make_simple_terrain()
        # Path from above through the flat: should report a hit on the flat segment (index 1)
        p1 = (1000.0, 1500.0)
        p2 = (1000.0,  900.0)
        hit, idx, pt = t.intersect(p1, p2)
        self.assertTrue(hit)
        self.assertEqual(idx, t.flat_idx)
        self.assertAlmostEqual(pt[0], 1000.0, places=6)
        self.assertTrue(990.0 <= pt[1] <= 1010.0)  # around y=1000

    def test_intersect_earliest_hit(self):
        # Build a terrain with one flat and one angled segment; ensure earliest is chosen
        # Terrain: seg 0 flat at y=900; seg 1 angled up to the right
        points = [
            (0.0, 900.0), (1000.0, 900.0),  # seg 0 (flat)
            (2000.0, 1100.0),               # seg 1 (angled)
            (3000.0, 800.0)                 # seg 2 (angled)
        ]
        t = Terrain.from_points(points)
        # Vertical drop from y=2000 at x=500 crosses seg 0 first due to higher y
        p1 = (500.0, 2000.0)
        p2 = (500.0,  700.0)
        hit, idx, pt = t.intersect(p1, p2)
        self.assertTrue(hit)
        # Should hit the flat (index 0) first in path order
        self.assertEqual(idx, 0)

    def test_no_intersection_parallel(self):
        t = self.make_simple_terrain()
        # Segment far above, parallel to flat; choose y above any terrain y (max is 1500)
        p1 = (-1000.0, 1600.0)
        p2 = ( 3000.0, 1600.0)
        hit, idx, _ = t.intersect(p1, p2)
        self.assertFalse(hit)
        self.assertEqual(idx, -1)


if __name__ == "__main__":
    unittest.main()