"""Backbone-independent DKB³ refinement of externally supplied predictions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from .kcs import lesion_prob_to_binary, select_candidate
from .models import DualBridgeSystem
from .priors import build_prior_stack


@torch.no_grad()
def refine_masks(
    system: DualBridgeSystem,
    image: torch.Tensor,
    primary_probability: torch.Tensor,
    *,
    additional_candidates: Mapping[str, torch.Tensor] | None = None,
    num_bridge_samples: int = 2,
    bridge_steps: int = 3,
    include_bridge_mean: bool = True,
    pseudo_prior_radius: int = 5,
    score_band_radius: int = 5,
    threshold: float = 0.5,
    empty_fallback: float = 0.3,
    weights: dict | None = None,
    primary_score_margin: float = 0.0,
    amp: bool = True,
    seed: int = 42,
) -> tuple[torch.Tensor, list[dict]]:
    """Return selected probabilities and per-image scores, without GT labels.

    Images are RGB floats [B, 3, H, W] in [0, 1]. External predictions are
    lesion probabilities [B, 1, H, W] in the same unflipped image coordinates.
    Additional candidates may include externally computed TTA predictions.
    The bridge must already be loaded, on the image device, and in eval mode.
    This function does not run or bundle the upstream segmentation network.
    """
    if system.training:
        raise ValueError("Call system.eval() before refinement")
    if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError("image must be a floating-point [B, 3, H, W] tensor")
    if image.shape[-2] < 1 or image.shape[-1] < 1:
        raise ValueError("Image dimensions must be positive")
    if not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1):
        raise ValueError("Image values must be finite and in [0, 1]")
    expected = (image.shape[0], 1, image.shape[-2], image.shape[-1])
    predictions = {"primary": primary_probability}
    for name, probability in (additional_candidates or {}).items():
        if not isinstance(name, str) or not name or name == "primary" or name.startswith("bridge_"):
            raise ValueError("Additional candidate names must be nonempty and not reserved")
        predictions[name] = probability
    for probability in predictions.values():
        if tuple(probability.shape) != expected or not probability.is_floating_point():
            raise ValueError(f"Predictions must be floating-point tensors with shape {expected}")
        if not torch.isfinite(probability).all() or torch.any(probability < 0) or torch.any(probability > 1):
            raise ValueError("Predictions must be finite lesion probabilities in [0, 1]")
    if not isinstance(num_bridge_samples, int) or num_bridge_samples < 0:
        raise ValueError("num_bridge_samples must be a non-negative integer")
    if not isinstance(bridge_steps, int) or not 1 <= bridge_steps <= 3:
        raise ValueError("Bundled bridge uses NFE=3; request 1, 2, or 3 steps")
    if not isinstance(pseudo_prior_radius, int) or pseudo_prior_radius < 0:
        raise ValueError("pseudo_prior_radius must be a non-negative integer")
    if not 0 <= empty_fallback <= threshold <= 1:
        raise ValueError("Require 0 <= empty_fallback <= threshold <= 1")
    image_arrays = (image.detach().cpu().float().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    candidate_arrays = {name: probability.detach().cpu().float().numpy()[:, 0]
                        for name, probability in predictions.items()}
    if num_bridge_samples:
        priors = [
            build_prior_stack(rgb, lesion_prob_to_binary(prob, threshold, empty_fallback),
                              radius=pseudo_prior_radius)["prior"]
            for rgb, prob in zip(image_arrays, candidate_arrays["primary"])
        ]
        prior_tensor = torch.from_numpy(np.stack(priors)).to(image.device)
        samples = []
        devices = [image.device.index if image.device.index is not None else torch.cuda.current_device()] if image.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            for index in range(num_bridge_samples):
                # Seed only the CPU and active CUDA generator; preserve caller RNG state.
                torch.random.default_generator.manual_seed(seed + index)
                if image.is_cuda:
                    torch.cuda.default_generators[devices[0]].manual_seed(seed + index)
                with torch.autocast(device_type=image.device.type, enabled=amp and image.is_cuda):
                    probability = system.sample_mask(image, prior_tensor, steps=bridge_steps)
                array = probability.detach().cpu().float().numpy()[:, 0]
                samples.append(array)
                candidate_arrays[f"bridge_sample_{index + 1}"] = array
        if include_bridge_mean:
            candidate_arrays["bridge_mean"] = np.mean(samples, axis=0).astype(np.float32)
    selected = []
    analyses = []
    for index, rgb in enumerate(image_arrays):
        candidates = [{"name": name, "prob": values[index]} for name, values in candidate_arrays.items()]
        winner, rows = select_candidate(
            rgb, candidates, threshold=threshold, empty_fallback=empty_fallback,
            band_radius=score_band_radius, weights=weights,
            primary_score_margin=primary_score_margin,
        )
        selected.append(candidates[winner]["prob"])
        analyses.append({"selected_candidate": candidates[winner]["name"], "candidates": rows})
    output = torch.from_numpy(np.stack(selected)[:, None]).to(primary_probability)
    return output, analyses
