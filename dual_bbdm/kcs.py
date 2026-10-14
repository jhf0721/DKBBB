"""Knowledge-guided selection using image evidence, never ground-truth labels.

The score, strict probability threshold, empty-prediction fallback, tie order,
and primary margin are retained from the original candidate-reranking code.
No segmentation backbone is imported or instantiated here.
"""

from __future__ import annotations

import numpy as np

from .priors import (
    build_artifact_mask,
    build_gradient_map,
    build_illumination_map,
    build_uncertainty_band,
)


DEFAULT_WEIGHTS = {
    "edge_weight": 1.0,
    "artifact_weight": 0.5,
    "illumination_weight": 0.25,
    "consistency_weight": 0.5,
    "area_weight": 0.0,
}


def lesion_prob_to_binary(prob_np, threshold=0.5, empty_fallback=0.3):
    prob_np = np.asarray(prob_np, dtype=np.float32)
    pred_bin = (prob_np > threshold).astype(np.uint8)
    if pred_bin.sum() == 0:
        pred_bin = (prob_np > empty_fallback).astype(np.uint8)
    return pred_bin


def masked_mean(values, mask):
    mask = mask.astype(np.float32)
    denom = float(mask.sum())
    if denom <= 0.0:
        return 0.0
    return float((values.astype(np.float32) * mask).sum() / denom)


def masked_std(values, mask):
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return 0.0
    return float(np.std(values.astype(np.float32)[mask_bool]))


def score_candidate(
    *, candidate_mask, reference_mask, gradient_map, artifact_map,
    illumination_map, reference_band, band_radius, edge_weight,
    artifact_weight, illumination_weight, consistency_weight, area_weight,
):
    candidate_band = build_uncertainty_band(candidate_mask.astype(np.uint8), radius=band_radius)[2].astype(np.float32)
    edge_reward = masked_mean(gradient_map, candidate_band)
    artifact_penalty = masked_mean(artifact_map, candidate_band)
    illumination_penalty = masked_std(illumination_map, candidate_band)
    outside_mask = 1.0 - reference_band.astype(np.float32)
    outside_disagreement = float(np.mean(np.abs(candidate_mask.astype(np.float32) - reference_mask.astype(np.float32)) * outside_mask))
    area_penalty = abs(float(candidate_mask.mean()) - float(reference_mask.mean()))
    score = (
        edge_weight * edge_reward
        - artifact_weight * artifact_penalty
        - illumination_weight * illumination_penalty
        - consistency_weight * outside_disagreement
        - area_weight * area_penalty
    )
    components = {
        "edge_reward": edge_reward,
        "artifact_penalty": artifact_penalty,
        "illumination_penalty": illumination_penalty,
        "outside_disagreement": outside_disagreement,
        "area_penalty": area_penalty,
        "score": float(score),
    }
    return float(score), components


def choose(candidates: list[dict], weights: dict, margin: float = 0.0):
    """Select cached score components; unrelated fields such as Dice are ignored."""
    if not candidates:
        raise ValueError("At least one candidate is required")
    scores = [
        weights["edge_weight"] * item["edge_reward"]
        - weights["artifact_weight"] * item["artifact_penalty"]
        - weights["illumination_weight"] * item["illumination_penalty"]
        - weights["consistency_weight"] * item["outside_disagreement"]
        - weights.get("area_weight", 0.0) * item.get("area_penalty", 0.0)
        for item in candidates
    ]
    if not np.isfinite(scores).all() or not np.isfinite(margin):
        raise ValueError("Scores and primary margin must be finite")
    primary_indices = [index for index, item in enumerate(candidates) if item["name"] == "primary"]
    if len(primary_indices) != 1:
        raise ValueError("Exactly one primary candidate is required")
    winner = max(range(len(scores)), key=scores.__getitem__)
    primary = primary_indices[0]
    if winner != primary and scores[winner] < scores[primary] + margin:
        winner = primary
    return winner, scores


def select_candidate(
    image_rgb: np.ndarray,
    candidates: list[dict],
    *, threshold=0.5, empty_fallback=0.3, band_radius=5,
    weights: dict | None = None, primary_score_margin=0.0,
):
    """Rank aligned lesion-probability maps on one uint8 RGB image.

    Each candidate supplies ``name`` and a 2D ``prob`` array in [0, 1].
    Exactly one must be named ``primary``. Returns the winning index and
    image-derived score rows. There is deliberately no ground-truth argument.
    """
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be an HxWx3 uint8 RGB array")
    if not 0 <= empty_fallback <= threshold <= 1:
        raise ValueError("Require 0 <= empty_fallback <= threshold <= 1")
    if not isinstance(band_radius, int) or band_radius < 0:
        raise ValueError("band_radius must be a non-negative integer")
    names = [item["name"] for item in candidates]
    if len(set(names)) != len(names) or names.count("primary") != 1:
        raise ValueError("Candidate names must be unique with exactly one primary")
    probs = [np.asarray(item["prob"], dtype=np.float32) for item in candidates]
    for prob in probs:
        if prob.shape != image_rgb.shape[:2] or not np.isfinite(prob).all():
            raise ValueError("Every probability map must be finite and aligned with the image")
        if np.any(prob < 0) or np.any(prob > 1):
            raise ValueError("Lesion probabilities must be in [0, 1]")
    masks = [lesion_prob_to_binary(prob, threshold, empty_fallback) for prob in probs]
    primary = masks[names.index("primary")]
    reference_band = build_uncertainty_band(primary, radius=band_radius)[2]
    gradient = build_gradient_map(image_rgb)
    artifact = build_artifact_mask(image_rgb)
    illumination = build_illumination_map(image_rgb)
    score_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if set(score_weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("Unknown KCS score weight")
    rows = []
    for name, mask in zip(names, masks):
        _, components = score_candidate(
            candidate_mask=mask, reference_mask=primary,
            gradient_map=gradient, artifact_map=artifact,
            illumination_map=illumination, reference_band=reference_band,
            band_radius=band_radius, **score_weights,
        )
        rows.append({"name": name, **components})
    winner, _ = choose(rows, score_weights, primary_score_margin)
    return winner, rows
