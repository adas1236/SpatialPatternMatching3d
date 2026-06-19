import json
import tempfile
import unittest
from pathlib import Path

from espm3d import ESPM3DMatcher, InvertedOctreeIndex
from espm3d.generate_synthetic_data import (
    assignment_is_valid,
    default_pattern_suite,
    generate_synthetic_3d_dataset,
    load_objects_jsonl,
    load_patterns_json,
    save_objects_jsonl,
    save_patterns_json,
)


class SyntheticESPM3DTests(unittest.TestCase):
    def test_default_pattern_suite_has_20_valid_patterns(self):
        patterns = default_pattern_suite()
        self.assertEqual(len(patterns), 20)
        self.assertTrue(all(p.pattern.edges for p in patterns))
        self.assertTrue(all(p.template_points for p in patterns))

    def test_generator_controls_exact_size_and_plants_matches(self):
        patterns = default_pattern_suite()[:4]
        dataset = generate_synthetic_3d_dataset(
            400,
            seed=7,
            patterns=patterns,
            ensure_patterns=True,
            planted_matches_per_pattern=2,
            n_noise_keywords=32,
            validate_planted=True,
        )
        self.assertEqual(len(dataset.objects), 400)
        self.assertEqual(len(dataset.planted_matches), 8)
        by_name = {p.name: p.pattern for p in dataset.patterns}
        for planted in dataset.planted_matches:
            self.assertTrue(assignment_is_valid(dataset.objects, by_name[planted.pattern_name], planted.assignment))

    def test_matcher_finds_planted_patterns(self):
        patterns = default_pattern_suite()[:5]
        dataset = generate_synthetic_3d_dataset(
            600,
            seed=42,
            patterns=patterns,
            ensure_patterns=True,
            planted_matches_per_pattern=1,
            n_noise_keywords=64,
            validate_planted=True,
        )
        index = InvertedOctreeIndex(dataset.objects, capacity=64, min_level=1, max_level=8)
        matcher = ESPM3DMatcher(index, require_distinct_objects=True)
        for named in patterns:
            with self.subTest(pattern=named.name):
                self.assertGreaterEqual(len(matcher.match(named.pattern)), 1)

    def test_json_roundtrip(self):
        patterns = default_pattern_suite()[:2]
        dataset = generate_synthetic_3d_dataset(
            150,
            seed=99,
            patterns=patterns,
            ensure_patterns=True,
            planted_matches_per_pattern=1,
            n_noise_keywords=8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            objects_path = tmp / "objects.jsonl"
            patterns_path = tmp / "patterns.json"
            save_objects_jsonl(dataset.objects, objects_path)
            save_patterns_json(dataset.patterns, patterns_path)
            loaded_objects = load_objects_jsonl(objects_path)
            loaded_patterns = load_patterns_json(patterns_path)
            self.assertEqual(len(loaded_objects), len(dataset.objects))
            self.assertEqual(len(loaded_patterns), len(dataset.patterns))
            self.assertEqual(loaded_patterns[0].name, dataset.patterns[0].name)


if __name__ == "__main__":
    unittest.main()
