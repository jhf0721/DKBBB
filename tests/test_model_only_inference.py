"""CPU tests for backbone-independent refinement; no real models or data."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_bbdm import refine_masks
from dual_bbdm.kcs import lesion_prob_to_binary
from dual_bbdm.priors import build_prior_stack


class FakeBridge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_priors = []
        self.samples = []

    def sample_mask(self, image, prior, steps):
        self.seen_priors.append(prior.clone())
        self.asserted_steps = steps
        output = torch.rand((image.shape[0], 1, *image.shape[-2:]), device=image.device)
        self.samples.append(output.clone())
        return output


class ModelOnlyInferenceTests(unittest.TestCase):
    def setUp(self):
        self.system = FakeBridge().eval()
        self.image = torch.full((2, 3, 32, 32), 0.5)
        self.primary = torch.zeros((2, 1, 32, 32))
        self.primary[:, :, 8:24, 10:22] = 0.9

    def test_output_shape_and_candidates(self):
        output, analyses = refine_masks(self.system, self.image, self.primary)
        self.assertEqual(output.shape, self.primary.shape)
        self.assertEqual(output.device, self.primary.device)
        self.assertEqual(len(analyses), 2)
        for row in analyses:
            self.assertEqual([item["name"] for item in row["candidates"]],
                             ["primary", "bridge_sample_1", "bridge_sample_2", "bridge_mean"])
            self.assertNotIn("oracle_dice", row["candidates"][0])

    def test_pseudo_prior_comes_from_external_prediction(self):
        refine_masks(self.system, self.image, self.primary, num_bridge_samples=1)
        rgb = (self.image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = lesion_prob_to_binary(self.primary[0, 0].numpy())
        expected = build_prior_stack(rgb, mask)["prior"]
        np.testing.assert_allclose(self.system.seen_priors[0][0].numpy(), expected)

    def test_reproducible_sampling_and_preserved_rng(self):
        state = torch.random.get_rng_state().clone()
        first, first_rows = refine_masks(self.system, self.image, self.primary, seed=123)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second, second_rows = refine_masks(self.system, self.image, self.primary, seed=123)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_rows, second_rows)
        self.assertTrue(torch.equal(self.system.samples[0], self.system.samples[2]))
        self.assertFalse(torch.equal(self.system.samples[0], self.system.samples[1]))

    def test_external_candidates_do_not_require_a_backbone(self):
        output, rows = refine_masks(self.system, self.image, self.primary,
                                   additional_candidates={"primary_hflip": self.primary.clone()},
                                   num_bridge_samples=0)
        self.assertEqual(self.system.samples, [])
        self.assertTrue(torch.equal(output, self.primary))
        self.assertTrue(all(row["selected_candidate"] == "primary" for row in rows))

    def test_rejects_training_mode(self):
        with self.assertRaisesRegex(ValueError, "eval"):
            refine_masks(self.system.train(), self.image, self.primary)

    def test_rejects_misaligned_probabilities(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            refine_masks(self.system, self.image, self.primary[:, :, :-1])

    def test_rejects_invalid_probability_values(self):
        for value in (float("nan"), -0.1, 1.1):
            probability = self.primary.clone()
            probability[0, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite lesion"):
                refine_masks(self.system, self.image, probability)

    def test_rejects_reserved_candidate_names_and_unimplemented_schedule(self):
        for name in ("primary", "bridge_sample_1", ""):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "reserved"):
                refine_masks(self.system, self.image, self.primary,
                             additional_candidates={name: self.primary})
        with self.assertRaisesRegex(ValueError, "NFE=3"):
            refine_masks(self.system, self.image, self.primary, bridge_steps=12)


if __name__ == "__main__":
    unittest.main()
