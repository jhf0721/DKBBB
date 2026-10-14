from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .legacy import load_legacy_factory
from .losses import boundary_gradient_loss, soft_dice_loss


def to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def _extract(buffer: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = buffer.to(t.device).gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class PriorEncoder(nn.Module):
    def __init__(self, in_channels: int = 5, hidden_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, prior: torch.Tensor) -> torch.Tensor:
        return self.encoder(prior)


class ConditionAdapter(nn.Module):
    def __init__(self, source_channels: int = 3, prior_channels: int = 32):
        super().__init__()
        self.source_proj = nn.Sequential(
            nn.Conv2d(source_channels, prior_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(prior_channels * 2, prior_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(prior_channels, source_channels, kernel_size=1),
        )

    def forward(self, source: torch.Tensor, prior_features: torch.Tensor) -> torch.Tensor:
        src = self.source_proj(source)
        fused = self.fuse(torch.cat([src, prior_features], dim=1))
        return torch.tanh(source + fused)


class BranchBridgeModel(nn.Module):
    def __init__(self, base_model: nn.Module, adapter: ConditionAdapter):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        xT: torch.Tensor | None = None,
        prior_features: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if xT is None or prior_features is None:
            raise ValueError("xT and prior_features are required for conditioned bridge inference")
        fused_xT = self.adapter(xT, prior_features)
        return self.base_model(x, timesteps, xT=fused_xT, **kwargs)


@dataclass
class DualBridgeOutputs:
    mask_loss: torch.Tensor
    image_loss: torch.Tensor
    structure_loss: torch.Tensor
    mask_supervision_loss: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.mask_loss + self.image_loss + self.structure_loss + self.mask_supervision_loss


class DualBridgeSystem(nn.Module):
    PRIOR_CHANNEL_NAMES = ("distance", "boundary", "gradient", "artifact", "illumination")
    PRIOR_NEUTRAL_VALUES = (0.5, 0.0, 0.0, 0.0, 1.0)

    def __init__(self, bridge_cfg: Dict):
        super().__init__()
        create_model_and_diffusion, defaults_fn = load_legacy_factory()
        base_defaults = defaults_fn()
        merged = dict(base_defaults)
        merged.update({k: v for k, v in bridge_cfg.items() if k in base_defaults})
        merged["condition_mode"] = "concat"
        merged["in_channels"] = 3

        self.prior_encoder = PriorEncoder(
            in_channels=bridge_cfg.get("prior_in_channels", 5),
            hidden_channels=bridge_cfg.get("prior_hidden_channels", 32),
        )
        self.mask_model, self.mask_diffusion = self._build_branch(create_model_and_diffusion, merged)
        self.image_model, self.image_diffusion = self._build_branch(create_model_and_diffusion, merged)
        self.consistency_steps = bridge_cfg.get("consistency_steps", 6)
        self.structure_weight = bridge_cfg.get("structure_weight", 1.0)
        self.gradient_weight = bridge_cfg.get("gradient_weight", 0.25)
        self.mask_supervision_weight = bridge_cfg.get("mask_supervision_weight", 1.0)
        self.prior_disabled_channels = self._normalize_disabled_prior_channels(
            bridge_cfg.get("prior_disabled_channels", [])
        )

    @classmethod
    def _normalize_disabled_prior_channels(cls, channels) -> tuple[str, ...]:
        if channels is None:
            return ()
        if isinstance(channels, str):
            channels = [part.strip() for part in channels.split(",") if part.strip()]
        normalized = tuple(str(channel).strip().lower() for channel in channels)
        unknown = sorted(set(normalized) - set(cls.PRIOR_CHANNEL_NAMES))
        if unknown:
            raise ValueError(
                f"unknown prior_disabled_channels={unknown}; expected names from "
                f"{list(cls.PRIOR_CHANNEL_NAMES)}"
            )
        return normalized

    def set_prior_disabled_channels(self, channels) -> None:
        """Change channel occlusion for sensitivity analysis without altering weights."""
        self.prior_disabled_channels = self._normalize_disabled_prior_channels(channels)

    def apply_prior_ablation(self, prior: torch.Tensor) -> torch.Tensor:
        if not self.prior_disabled_channels:
            return prior
        if prior.ndim != 4 or prior.shape[1] != len(self.PRIOR_CHANNEL_NAMES):
            raise ValueError(
                f"expected a [B, 5, H, W] prior, got shape={tuple(prior.shape)}"
            )
        result = prior.clone()
        for name in self.prior_disabled_channels:
            index = self.PRIOR_CHANNEL_NAMES.index(name)
            result[:, index] = self.PRIOR_NEUTRAL_VALUES[index]
        return result

    def _build_branch(self, factory, cfg: Dict) -> Tuple[BranchBridgeModel, nn.Module]:
        base_model, diffusion = factory(**cfg)
        adapter = ConditionAdapter(source_channels=3, prior_channels=self.prior_encoder.encoder[0].out_channels)
        return BranchBridgeModel(base_model, adapter), diffusion

    def compute_losses(self, image: torch.Tensor, mask_rgb: torch.Tensor, prior: torch.Tensor) -> DualBridgeOutputs:
        prior = self.apply_prior_ablation(prior)
        prior_features = self.prior_encoder(prior)
        image_n = to_neg_one_to_one(image)
        mask_n = to_neg_one_to_one(mask_rgb)
        gt_mask = mask_rgb[:, :1]

        batch_size = image.shape[0]
        t_mask = torch.randint(0, self.mask_diffusion.num_timesteps, (batch_size,), device=image.device).long()
        t_image = torch.randint(0, self.image_diffusion.num_timesteps, (batch_size,), device=image.device).long()

        mask_terms = self.mask_diffusion.training_bridge_losses(
            self.mask_model,
            mask_n,
            t_mask,
            model_kwargs={"xT": image_n, "prior_features": prior_features},
        )
        image_terms = self.image_diffusion.training_bridge_losses(
            self.image_model,
            image_n,
            t_image,
            model_kwargs={"xT": mask_n, "prior_features": prior_features},
        )

        mask_supervision = self._compute_mask_supervision(
            image_n=image_n,
            mask_n=mask_n,
            gt_mask=gt_mask,
            prior_features=prior_features,
        )

        with torch.no_grad():
            synth_image = self.sample_image(mask_rgb, prior, steps=self.consistency_steps)
        recon_mask = self.sample_mask_from_source(synth_image.detach(), prior, steps=self.consistency_steps)
        structure = soft_dice_loss(recon_mask, gt_mask) + self.gradient_weight * boundary_gradient_loss(recon_mask, gt_mask)

        return DualBridgeOutputs(
            mask_loss=mask_terms["loss"].mean(),
            image_loss=image_terms["loss"].mean(),
            structure_loss=structure * self.structure_weight,
            mask_supervision_loss=mask_supervision * self.mask_supervision_weight,
        )

    def _compute_mask_supervision(
        self,
        image_n: torch.Tensor,
        mask_n: torch.Tensor,
        gt_mask: torch.Tensor,
        prior_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = image_n.shape[0]
        t = torch.randint(0, self.mask_diffusion.num_timesteps, (batch_size,), device=image_n.device).long()
        noise = torch.randn_like(mask_n)
        x_t, _ = self.mask_diffusion.q_sample(mask_n, image_n, t, noise)
        objective_recon = self.mask_model(x_t, t, xT=image_n, prior_features=prior_features)
        x0_recon = self.mask_diffusion.predict_x0_from_objective(x_t, image_n, t, objective_recon).clamp(-1.0, 1.0)
        pred_mask = to_zero_to_one(x0_recon).mean(dim=1, keepdim=True)
        with torch.amp.autocast(device_type=pred_mask.device.type, enabled=False):
            bce = F.binary_cross_entropy(pred_mask.float(), gt_mask.float())
        dice = soft_dice_loss(pred_mask, gt_mask)
        return bce + dice

    def _sample_branch(
        self,
        branch_model: BranchBridgeModel,
        diffusion: nn.Module,
        source: torch.Tensor,
        prior_features: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        x = to_neg_one_to_one(source)
        source_n = x.clone()
        schedule = diffusion.steps.to(source.device)
        if steps is not None and steps < len(schedule):
            positions = torch.linspace(0, len(schedule) - 1, steps, device=source.device).long()
            schedule = schedule[positions]
            if int(schedule[-1]) != 0:
                schedule[-1] = 0

        for idx, step_t in enumerate(schedule):
            t = torch.full((source.shape[0],), int(step_t.item()), device=source.device, dtype=torch.long)
            objective = branch_model(x, t, xT=source_n, prior_features=prior_features)
            x0 = diffusion.predict_x0_from_objective(x, source_n, t, objective)
            x0 = x0.clamp(-1.0, 1.0)
            if int(step_t.item()) == 0:
                x = x0
                break

            next_t = torch.full((source.shape[0],), int(schedule[idx + 1].item()), device=source.device, dtype=torch.long)
            m_t = _extract(diffusion.m_t, t, x.shape)
            m_nt = _extract(diffusion.m_t, next_t, x.shape)
            var_t = _extract(diffusion.variance_t, t, x.shape)
            var_nt = _extract(diffusion.variance_t, next_t, x.shape)
            sigma2_t = (var_t - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2) * var_nt / var_t.clamp_min(1e-6)
            sigma2_t = torch.clamp(sigma2_t, min=0.0)
            sigma_t = torch.sqrt(sigma2_t) * getattr(diffusion, "eta", 1.0)
            noise = torch.randn_like(x)
            bridge_residual = x - (1.0 - m_t) * x0 - m_t * source_n
            x = (1.0 - m_nt) * x0 + m_nt * source_n + torch.sqrt((var_nt - sigma2_t).clamp_min(0.0) / var_t.clamp_min(1e-6)) * bridge_residual
            x = x + sigma_t * noise
        return to_zero_to_one(x)

    @torch.no_grad()
    def sample_mask(self, image: torch.Tensor, prior: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        prior = self.apply_prior_ablation(prior)
        prior_features = self.prior_encoder(prior)
        mask_rgb = self._sample_branch(self.mask_model, self.mask_diffusion, image, prior_features, steps=steps)
        return mask_rgb.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def sample_image(self, mask_rgb: torch.Tensor, prior: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        prior = self.apply_prior_ablation(prior)
        prior_features = self.prior_encoder(prior)
        return self._sample_branch(self.image_model, self.image_diffusion, mask_rgb, prior_features, steps=steps)

    def sample_mask_from_source(self, source_image: torch.Tensor, prior: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        prior = self.apply_prior_ablation(prior)
        prior_features = self.prior_encoder(prior)
        mask_rgb = self._sample_branch(self.mask_model, self.mask_diffusion, source_image, prior_features, steps=steps)
        return mask_rgb.mean(dim=1, keepdim=True)
