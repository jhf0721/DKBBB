"""CPU checks that cached evaluation labels cannot affect KCS selection."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_bbdm.kcs import choose


class KCSReplayTests(unittest.TestCase):
    def setUp(self):
        self.weights = {
            "edge_weight": 1.0,
            "artifact_weight": 0.5,
            "illumination_weight": 0.25,
            "consistency_weight": 0.5,
            "area_weight": 0.0,
        }

    def candidate(self, name, edge, dice):
        return {
            "name": name,
            "edge_reward": edge,
            "artifact_penalty": 0.0,
            "illumination_penalty": 0.0,
            "outside_disagreement": 0.0,
            "oracle_dice": dice,
        }

    def test_ground_truth_dice_never_changes_selection_or_scores(self):
        candidates = [self.candidate("primary", 0.2, 0.99), self.candidate("bridge_sample_1", 0.4, 0.01)]
        selected, scores = choose(candidates, self.weights, 0.0)
        changed = copy.deepcopy(candidates)
        changed[0]["oracle_dice"], changed[1]["oracle_dice"] = 0.01, 0.99
        self.assertEqual(selected, 1)
        self.assertEqual((selected, scores), choose(changed, self.weights, 0.0))

    def test_primary_margin_is_enforced(self):
        candidates = [self.candidate("primary", 0.2, 0.1), self.candidate("bridge_sample_1", 0.2005, 0.99)]
        self.assertEqual(choose(candidates, self.weights, 0.0)[0], 1)
        self.assertEqual(choose(candidates, self.weights, 0.001)[0], 0)

    def test_first_score_tie_matches_native_candidate_order(self):
        candidates = [self.candidate("primary", 0.2, 0.1), self.candidate("primary_hflip", 0.2, 0.99)]
        self.assertEqual(choose(candidates, self.weights, 0.0)[0], 0)

    def test_four_term_score_signs(self):
        candidate = self.candidate("primary", 0.8, 0.1)
        candidate.update(artifact_penalty=0.2, illumination_penalty=0.4, outside_disagreement=0.6)
        self.assertAlmostEqual(choose([candidate], self.weights, 0.0)[1][0], 0.3)


if __name__ == "__main__":
    unittest.main()
