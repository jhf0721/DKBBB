from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dual_bbdm import DualBridgeSystem, ISICDualBridgeDataset, load_config, load_manifest


DEFAULTS = {
    "data": {
        "manifest_path": 'data/isic2018_train.jsonl',
        "manifest_paths": [],
        "dedupe_samples": False,
        "dedupe_key": "image",
        "image_size": 256,
        "band_radius": 5,
        "num_workers": 4,
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
        "structure_weight": 1.0,
        "gradient_weight": 0.25,
        "mask_supervision_weight": 1.0,
    },
    "train": {
        "epochs": 40,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "amp": True,
        "log_interval": 20,
        "save_interval": 10,
        "output_dir": 'outputs/bridge',
        "resume": "",
        "seed": 42,
    },
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(output_dir: Path, epoch: int, model: DualBridgeSystem, optimizer: torch.optim.Optimizer) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_dir / f"dual_bridge_epoch_{epoch:03d}.pt")
    torch.save(checkpoint, output_dir / "last.pt")


def resolve_manifest_paths(data_cfg: dict[str, Any]) -> list[str]:
    manifest_paths = data_cfg.get("manifest_paths", [])
    if manifest_paths and not isinstance(manifest_paths, (list, tuple)):
        raise ValueError("data.manifest_paths must be a list when provided")

    resolved = [str(path).strip() for path in manifest_paths if str(path).strip()]
    if not resolved:
        manifest_path = str(data_cfg.get("manifest_path", "")).strip()
        if not manifest_path:
            raise ValueError("either data.manifest_path or data.manifest_paths must be configured")
        resolved = [manifest_path]

    for manifest_path in resolved:
        if not Path(manifest_path).exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return resolved


def build_bridge_dataset(data_cfg: dict[str, Any]) -> ISICDualBridgeDataset:
    manifest_paths = resolve_manifest_paths(data_cfg)
    dataset = ISICDualBridgeDataset(
        manifest_path=manifest_paths[0],
        image_size=data_cfg["image_size"],
        band_radius=data_cfg["band_radius"],
    )

    if len(manifest_paths) == 1 and not bool(data_cfg.get("dedupe_samples", False)):
        return dataset

    dedupe_samples = bool(data_cfg.get("dedupe_samples", False))
    dedupe_key = str(data_cfg.get("dedupe_key", "image")).strip()
    merged_samples: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for manifest_path in manifest_paths:
        for sample in load_manifest(manifest_path):
            if dedupe_samples:
                if dedupe_key not in sample:
                    raise ValueError(f"dedupe key '{dedupe_key}' not found in sample from {manifest_path}")
                sample_key = str(sample[dedupe_key])
                if sample_key in seen_keys:
                    continue
                seen_keys.add(sample_key)
            merged_samples.append(sample)

    dataset.samples = merged_samples
    print(
        f"bridge manifests={len(manifest_paths)} samples={len(dataset)} "
        f"dedupe={dedupe_samples} dedupe_key={dedupe_key}"
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/isic2018.yaml')
    parser.add_argument("--prior-disabled-channels", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, defaults=DEFAULTS)
    if args.prior_disabled_channels:
        cfg["bridge"]["prior_disabled_channels"] = [
            part.strip() for part in args.prior_disabled_channels.split(",") if part.strip()
        ]
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
        cfg["train"]["resume"] = ""
    if args.manifest_path:
        cfg["data"]["manifest_path"] = args.manifest_path
        cfg["data"]["manifest_paths"] = []
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_bridge_dataset(cfg["data"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    system = DualBridgeSystem(cfg["bridge"]).to(device)
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    amp_enabled = torch.cuda.is_available() and cfg["train"]["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir = Path(cfg["train"]["output_dir"])
    if cfg["train"]["resume"]:
        checkpoint = torch.load(cfg["train"]["resume"], map_location=device)
        system.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 1

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        system.train()
        running = {"mask": 0.0, "image": 0.0, "structure": 0.0, "mask_sup": 0.0, "total": 0.0}
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            mask_rgb = batch["mask_rgb"].to(device, non_blocking=True)
            prior = batch["prior"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = system.compute_losses(image=image, mask_rgb=mask_rgb, prior=prior)
                loss = losses.total

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["mask"] += float(losses.mask_loss.detach())
            running["image"] += float(losses.image_loss.detach())
            running["structure"] += float(losses.structure_loss.detach())
            running["mask_sup"] += float(losses.mask_supervision_loss.detach())
            running["total"] += float(loss.detach())

            if step % cfg["train"]["log_interval"] == 0:
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={running['total'] / step:.4f} "
                    f"mask={running['mask'] / step:.4f} "
                    f"image={running['image'] / step:.4f} "
                    f"struct={running['structure'] / step:.4f} "
                    f"mask_sup={running['mask_sup'] / step:.4f}"
                )

        if epoch % cfg["train"]["save_interval"] == 0:
            save_checkpoint(output_dir, epoch, system, optimizer)


if __name__ == "__main__":
    main()
