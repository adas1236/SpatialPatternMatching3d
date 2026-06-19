import unittest

from espm3d import (
    EdgeSign,
    ESPM3DMatcher,
    InvertedOctreeIndex,
    PatternEdge,
    SpatialObject,
    SpatialPattern,
    brute_force_match,
)


def canonical(matches):
    """Turn a list of assignment dicts into an order-insensitive set."""
    return {tuple(sorted(m.items(), key=lambda kv: str(kv[0]))) for m in matches}


class TestESPM3D(unittest.TestCase):
    def test_residential_style_source_exclusion_in_3d(self):
        objects = [
            SpatialObject("h1", 0.0, 0.0, 0.0, {"house"}),
            SpatialObject("p1", 0.2, 0.0, 0.0, {"park"}),
            SpatialObject("s1", 0.7, 0.0, 0.0, {"station"}),
            SpatialObject("h2", 5.0, 0.0, 0.0, {"house"}),
            SpatialObject("p2", 5.1, 0.0, 0.0, {"park"}),
            SpatialObject("s_near_h2", 5.2, 0.0, 0.0, {"station"}),
            SpatialObject("s_candidate_h2", 5.8, 0.0, 0.0, {"station"}),
        ]
        pattern = SpatialPattern(
            vertices={"H": "house", "P": "park", "S": "station"},
            edges=[
                PatternEdge("H", "P", 0.0, 0.5, "--"),
                PatternEdge("H", "S", 0.5, 1.0, "->"),
            ],
        )
        index = InvertedOctreeIndex(objects, capacity=1, min_level=1, max_level=5)
        matcher = ESPM3DMatcher(index)
        matches = matcher.match(pattern)
        self.assertEqual(canonical(matches), canonical([{"H": "h1", "P": "p1", "S": "s1"}]))

    def test_true_3d_distance_interval(self):
        objects = [
            SpatialObject("a1", 0.0, 0.0, 0.0, {"A"}),
            SpatialObject("a2", 10.0, 10.0, 10.0, {"A"}),
            SpatialObject("b1", 1.0, 1.0, 1.0, {"B"}),
            SpatialObject("b2", 10.0, 10.0, 12.0, {"B"}),
        ]
        pattern = SpatialPattern(
            vertices={"A": "A", "B": "B"},
            edges=[PatternEdge("A", "B", 1.70, 1.75, EdgeSign.INCLUSION)],
        )
        matcher = ESPM3DMatcher(InvertedOctreeIndex(objects, capacity=1, min_level=1, max_level=5))
        matches = matcher.match(pattern)
        self.assertEqual(canonical(matches), canonical([{"A": "a1", "B": "b1"}]))

    def test_triangle_skip_edge_filters_bad_tuple(self):
        objects = [
            SpatialObject("a", 0.0, 0.0, 0.0, {"A"}),
            SpatialObject("b", 1.0, 0.0, 0.0, {"B"}),
            SpatialObject("c_good", 1.0, 1.0, 0.0, {"C"}),
            SpatialObject("c_bad", 2.0, 0.0, 0.0, {"C"}),
        ]
        pattern = SpatialPattern(
            vertices={"A": "A", "B": "B", "C": "C"},
            edges=[
                PatternEdge("A", "B", 0.9, 1.1, "--"),
                PatternEdge("B", "C", 0.9, 1.1, "--"),
                PatternEdge("A", "C", 1.3, 1.5, "--"),
            ],
        )
        matcher = ESPM3DMatcher(InvertedOctreeIndex(objects, capacity=1, min_level=1, max_level=5))
        matches = matcher.match(pattern)
        self.assertEqual(canonical(matches), canonical([{"A": "a", "B": "b", "C": "c_good"}]))
        self.assertIn(2, matcher.last_stats.skip_edges)

    def test_matches_bruteforce_on_small_exclusion_pattern(self):
        objects = [
            SpatialObject("a1", 0.0, 0.0, 0.0, {"A"}),
            SpatialObject("a2", 4.0, 0.0, 0.0, {"A"}),
            SpatialObject("b_near_a1", 0.5, 0.0, 0.0, {"B"}),
            SpatialObject("b1", 2.0, 0.0, 0.0, {"B"}),
            SpatialObject("b2", 6.0, 0.0, 0.0, {"B"}),
            SpatialObject("c1", 0.0, 2.0, 0.0, {"C"}),
            SpatialObject("c2", 4.0, 2.0, 1.0, {"C"}),
        ]
        pattern = SpatialPattern(
            vertices={"A": "A", "B": "B", "C": "C"},
            edges=[
                PatternEdge("A", "B", 1.0, 3.0, "->"),
                PatternEdge("A", "C", 0.0, 3.0, "--"),
                PatternEdge("B", "C", 0.0, 3.0, "--"),
            ],
        )
        matcher = ESPM3DMatcher(InvertedOctreeIndex(objects, capacity=1, min_level=1, max_level=5))
        indexed = matcher.match(pattern)
        brute = brute_force_match(objects, pattern)
        self.assertEqual(canonical(indexed), canonical(brute))


if __name__ == "__main__":
    unittest.main()
