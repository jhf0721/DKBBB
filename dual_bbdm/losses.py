from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        loss = F.binary_cross_entropy(pred.float(), target.float(), reduction="none")
    weight = mask.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    if mask is not None:
        pred = pred * mask
        target = target * mask
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def boundary_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))


def build_soft_labels(mask: torch.Tensor, candidate: torch.Tensor, band: torch.Tensor, sdm: torch.Tensor, tau: float) -> torch.Tensor:
    lambda_map = torch.clamp(torch.abs(sdm) / max(tau, 1e-6), max=1.0)
    return mask * (1.0 - band) + band * (lambda_map * mask + (1.0 - lambda_map) * candidate)
