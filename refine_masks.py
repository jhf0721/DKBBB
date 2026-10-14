"""Refine one external lesion-probability map with DKB³; no GT is required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dual_bbdm import DualBridgeSystem, load_config, refine_masks
from dual_bbdm.kcs import DEFAULT_WEIGHTS, lesion_prob_to_binary
from dual_bbdm.priors import resize_rgb
from train_dual_bridge import DEFAULTS as BRIDGE_DEFAULTS


def load_probability(path: Path, image_size: int) -> torch.Tensor:
    """Read a 2D float lesion-probability map, without resizing or inversion."""
    if path.suffix.lower() != ".npy":
        raise ValueError("Supply a .npy probability map, not a thresholded mask")
    probability = np.load(path, allow_pickle=False)
    if probability.shape != (image_size, image_size) or not np.issubdtype(probability.dtype, np.floating):
        raise ValueError(f"Expected a floating-point [{image_size}, {image_size}] probability array: {path}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(f"Expected finite lesion probabilities in [0, 1]: {path}")
    return torch.from_numpy(probability.astype(np.float32))[None, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/isic2018.yaml")
    parser.add_argument("--checkpoint", required=True, help="DKB³ bridge checkpoint; no segmentor checkpoint")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--primary-probability", type=Path, required=True, help="HxW float .npy, foreground=lesion")
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH.npy",
                        help="Optional external candidate aligned to the unflipped image; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="Output .png mask, with .npy probabilities and .json scores")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    cfg = load_config(args.config, defaults=BRIDGE_DEFAULTS)
    inference = dict(cfg.get("inference", {}))
    weights = {name: inference.pop(name, default) for name, default in DEFAULT_WEIGHTS.items()}
    if args.seed is not None:
        inference["seed"] = args.seed
    size = int(cfg["bridge"]["image_size"])
    with Image.open(args.image) as handle:
        rgb = resize_rgb(np.asarray(handle.convert("RGB")), size)
    image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255.0
    primary = load_probability(args.primary_probability, size)
    candidates = {}
    for value in args.candidate:
        name, separator, filename = value.partition("=")
        if not separator or not name or not filename or name in candidates:
            parser.error("Each --candidate requires a unique NAME=PATH.npy")
        candidates[name] = load_probability(Path(filename), size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    system = DualBridgeSystem(cfg["bridge"]).to(device)
    incompatible = system.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if ".lpips_loss." not in key]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(f"Incompatible bridge checkpoint: missing={incompatible.missing_keys}, unexpected={unexpected}")
    system.eval()
    probability, analyses = refine_masks(
        system, image.to(device), primary.to(device),
        additional_candidates={name: value.to(device) for name, value in candidates.items()},
        weights=weights, **inference,
    )
    probability_np = probability[0, 0].cpu().numpy()
    mask = lesion_prob_to_binary(probability_np, inference.get("threshold", 0.5), inference.get("empty_fallback", 0.3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(args.output)
    np.save(args.output.with_suffix(".npy"), probability_np)
    args.output.with_suffix(".json").write_text(json.dumps(analyses[0], indent=2) + "\n", encoding="utf-8")
    print(f"Selected {analyses[0]['selected_candidate']}; outputs: {args.output}, .npy, .json")


if __name__ == "__main__":
    main()
