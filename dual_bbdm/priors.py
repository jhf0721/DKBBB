from __future__ import annotations

import cv2
import numpy as np


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    threshold = 127 if resized.max() > 1 else 0.5
    return (resized > threshold).astype(np.uint8)


def build_uncertainty_band(mask: np.ndarray, radius: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    inner = cv2.erode(mask, kernel, iterations=1)
    outer = cv2.dilate(mask, kernel, iterations=1)
    band = np.clip(outer - inner, 0, 1).astype(np.uint8)
    return inner.astype(np.uint8), outer.astype(np.uint8), band


def build_signed_distance_map(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    sdm = inside - outside
    max_abs = max(float(np.abs(sdm).max()), 1.0)
    sdm = np.clip(sdm / max_abs, -1.0, 1.0)
    return sdm.astype(np.float32)


def build_gradient_map(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad /= max(float(grad.max()), 1e-6)
    return grad.astype(np.float32)


def build_illumination_map(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=15, sigmaY=15)
    illum = image / np.maximum(blur, 1e-3)
    illum = cv2.cvtColor(np.clip(illum, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    illum /= max(float(illum.max()), 1e-6)
    return illum.astype(np.float32)


def build_artifact_mask(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    hair = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    hair = (hair > 15).astype(np.uint8)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    reflection = ((hsv[..., 2] > 240) & (hsv[..., 1] < 35)).astype(np.uint8)
    artifact = np.clip(hair + reflection, 0, 1).astype(np.float32)
    return artifact


def build_prior_stack(image_rgb: np.ndarray, mask: np.ndarray, radius: int = 5) -> dict[str, np.ndarray]:
    inner, outer, band = build_uncertainty_band(mask, radius=radius)
    sdm = build_signed_distance_map(mask)
    gradient = build_gradient_map(image_rgb)
    artifact = build_artifact_mask(image_rgb)
    illumination = build_illumination_map(image_rgb)
    prior = np.stack(
        [
            (sdm + 1.0) * 0.5,
            band.astype(np.float32),
            gradient,
            artifact,
            illumination,
        ],
        axis=0,
    )
    return {
        "prior": prior.astype(np.float32),
        "band": band.astype(np.float32)[None, ...],
        "inner": inner.astype(np.float32)[None, ...],
        "outer": outer.astype(np.float32)[None, ...],
        "sdm": sdm.astype(np.float32)[None, ...],
    }
