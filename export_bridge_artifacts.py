from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dual_bbdm import DualBridgeSystem, ISICDualBridgeDataset, load_config


DEFAULTS = {
    "data": {
        "manifest_path": 'data/isic2018_train.jsonl',
        "image_size": 256,
        "band_radius": 5,
        "num_workers": 2,
    },
    "bridge": {
        "dims": 2,
        "image_size": 256,
        "in_channels": 3,
        "num_channels": 64,
        "num_res_blocks": 2,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "num_head_channels": -1,
        "attention_resolutions": "32,16,8",
        "channel_mult": "",
        "dropout": 0.0,
        "class_cond": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_fp16": False,
        "use_new_attention_order": False,
        "attention_type": "att",
        "learn_sigma": False,
        "condition_mode": "concat",
        "pred_mode": "ve_simple",
        "weight_schedule": "bridge_karras",
        "sigma_data": 0.5,
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "beta_d": 2.0,
        "beta_min": 0.1,
        "cov_xy": 0.0,
        "unet_type": "adm",
        "prior_in_channels": 5,
        "prior_hidden_channels": 32,
        "consistency_steps": 6,
    },
    "export": {
        "batch_size": 1,
        "sample_steps": 12,
        "checkpoint": 'outputs/bridge/last.pt',
        "artifact_root": 'outputs/artifacts',
    },
}


def save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.45,
    threshold: float = 0.5,
) -> np.ndarray:
    base = np.clip(image.astype(np.float32), 0.0, 1.0).copy()
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bin = mask >= threshold
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    base[mask_bin] = (1.0 - alpha) * base[mask_bin] + alpha * color_arr
    return np.clip(base, 0.0, 1.0)


def make_random_ellipse_mask(size: int, rng: random.Random) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    cx = rng.uniform(size * 0.3, size * 0.7)
    cy = rng.uniform(size * 0.3, size * 0.7)
    rx = rng.uniform(size * 0.08, size * 0.18)
    ry = rng.uniform(size * 0.08, size * 0.22)
    theta = rng.uniform(0.0, np.pi)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = (xx - cx) * cos_t + (yy - cy) * sin_t
    y = -(xx - cx) * sin_t + (yy - cy) * cos_t
    mask = ((x / max(rx, 1.0)) ** 2 + (y / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/isic2018.yaml')
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum samples to export; 0 means all.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip cases whose candidate and synthetic files already exist.")
    parser.add_argument("--save_ablations", action="store_true", help="Save random-mask/zero-prior probes for leakage checks.")
    parser.add_argument("--ablation_seed", type=int, default=1234)
    parser.add_argument("--prior-disabled-channels", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument(
        "--use-ground-truth-candidate",
        action="store_true",
        help="Save the conditioning ground-truth mask as candidate_masks and skip image-to-mask sampling.",
    )
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    cfg = load_config(args.config, defaults=DEFAULTS)
    if args.prior_disabled_channels:
        cfg["bridge"]["prior_disabled_channels"] = [
            part.strip() for part in args.prior_disabled_channels.split(",") if part.strip()
        ]
    if args.checkpoint:
        cfg["export"]["checkpoint"] = args.checkpoint
    if args.artifact_root:
        cfg["export"]["artifact_root"] = args.artifact_root
    if args.manifest_path:
        cfg["data"]["manifest_path"] = args.manifest_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ISICDualBridgeDataset(
        manifest_path=cfg["data"]["manifest_path"],
        image_size=cfg["data"]["image_size"],
        band_radius=cfg["data"]["band_radius"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["export"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    system = DualBridgeSystem(cfg["bridge"]).to(device)
    checkpoint = torch.load(cfg["export"]["checkpoint"], map_location=device)
    incompatible = system.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if ".lpips_loss." not in key]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "bridge checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    system.eval()

    artifact_root = Path(cfg["export"]["artifact_root"])
    candidate_root = artifact_root / "candidate_masks"
    synthetic_root = artifact_root / "synthetic_images"
    visualization_root = artifact_root / "visualizations"
    ablation_root = artifact_root / "ablation_checks"
    candidate_root.mkdir(parents=True, exist_ok=True)
    synthetic_root.mkdir(parents=True, exist_ok=True)
    visualization_root.mkdir(parents=True, exist_ok=True)
    if args.save_ablations:
        ablation_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.ablation_seed)

    processed = 0
    for batch in loader:
        if args.max_samples > 0 and processed >= args.max_samples:
            break

        case_id = batch["id"][0]
        candidate_path = candidate_root / f"{case_id}.png"
        synthetic_path = synthetic_root / f"{case_id}.png"
        if args.skip_existing and candidate_path.exists() and synthetic_path.exists():
            processed += 1
            print(f"skip existing {case_id} ({processed})")
            continue

        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        mask_rgb = batch["mask_rgb"].to(device, non_blocking=True)
        prior = batch["prior"].to(device, non_blocking=True)

        case_seed = int(args.seed + processed * 1009)
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(case_seed)
        candidate = mask if args.use_ground_truth_candidate else system.sample_mask(
            image, prior, steps=cfg["export"]["sample_steps"]
        )
        random.seed(case_seed + 100_000_003)
        np.random.seed((case_seed + 100_000_003) % (2**32))
        torch.manual_seed(case_seed + 100_000_003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(case_seed + 100_000_003)
        synth = system.sample_image(mask_rgb, prior, steps=cfg["export"]["sample_steps"])

        image_np = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        real_mask_np = mask[0, 0].detach().cpu().numpy()
        candidate_np = candidate[0, 0].detach().cpu().numpy()
        synth_np = synth[0].detach().cpu().numpy().transpose(1, 2, 0)
        overlay_real_with_candidate = overlay_mask_on_image(image_np, candidate_np)
        overlay_synth_with_real = overlay_mask_on_image(synth_np, real_mask_np)

        save_gray(candidate_path, candidate_np)
        save_rgb(synthetic_path, synth_np)
        if not args.no_visualizations:
            case_vis_root = visualization_root / case_id
            case_vis_root.mkdir(parents=True, exist_ok=True)
            save_rgb(case_vis_root / "01_real_image.png", image_np)
            save_gray(case_vis_root / "02_generated_mask.png", candidate_np)
            save_rgb(case_vis_root / "03_real_image_with_generated_mask.png", overlay_real_with_candidate)
            save_gray(case_vis_root / "04_real_mask.png", real_mask_np)
            save_rgb(case_vis_root / "05_generated_image.png", synth_np)
            save_rgb(case_vis_root / "06_generated_image_with_real_mask.png", overlay_synth_with_real)
        if args.save_ablations:
            random_mask_np = make_random_ellipse_mask(cfg["data"]["image_size"], rng)
            random_mask = torch.from_numpy(random_mask_np[None, None, ...]).to(device=device, dtype=mask.dtype)
            random_mask_rgb = random_mask.repeat(1, 3, 1, 1)
            zero_prior = torch.zeros_like(prior)

            synth_random_mask_same_prior = system.sample_image(
                random_mask_rgb,
                prior,
                steps=cfg["export"]["sample_steps"],
            )
            synth_real_mask_zero_prior = system.sample_image(
                mask_rgb,
                zero_prior,
                steps=cfg["export"]["sample_steps"],
            )
            synth_random_mask_zero_prior = system.sample_image(
                random_mask_rgb,
                zero_prior,
                steps=cfg["export"]["sample_steps"],
            )

            ablation_case_root = ablation_root / case_id
            ablation_case_root.mkdir(parents=True, exist_ok=True)
            save_rgb(ablation_case_root / "01_real_image.png", image_np)
            save_gray(ablation_case_root / "02_real_mask.png", real_mask_np)
            save_gray(ablation_case_root / "03_random_mask.png", random_mask_np)
            save_rgb(ablation_case_root / "04_generated_real_mask_real_prior.png", synth_np)
            save_rgb(
                ablation_case_root / "05_generated_random_mask_real_prior.png",
                synth_random_mask_same_prior[0].detach().cpu().numpy().transpose(1, 2, 0),
            )
            save_rgb(
                ablation_case_root / "06_generated_real_mask_zero_prior.png",
                synth_real_mask_zero_prior[0].detach().cpu().numpy().transpose(1, 2, 0),
            )
            save_rgb(
                ablation_case_root / "07_generated_random_mask_zero_prior.png",
                synth_random_mask_zero_prior[0].detach().cpu().numpy().transpose(1, 2, 0),
            )

            for name, arr in [
                ("real_mask_real_prior", synth_np),
                ("random_mask_real_prior", synth_random_mask_same_prior[0].detach().cpu().numpy().transpose(1, 2, 0)),
                ("real_mask_zero_prior", synth_real_mask_zero_prior[0].detach().cpu().numpy().transpose(1, 2, 0)),
                ("random_mask_zero_prior", synth_random_mask_zero_prior[0].detach().cpu().numpy().transpose(1, 2, 0)),
            ]:
                mae = float(np.mean(np.abs(arr - image_np)))
                print(f"{case_id} {name} image_mae={mae:.4f}")
        processed += 1
        print(f"{case_id} ({processed})")


if __name__ == "__main__":
    main()
