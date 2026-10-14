from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset

from .priors import build_prior_stack, resize_mask, resize_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _case_id_from_mask(mask_path: str, mask_suffix: str = "_segmentation") -> str:
    stem = Path(mask_path).stem
    if mask_suffix and stem.endswith(mask_suffix):
        return stem[: -len(mask_suffix)]
    return stem


def _normalize_extensions(extensions: List[str] | None, defaults: List[str]) -> set[str]:
    if not extensions:
        return {ext.lower() for ext in defaults}
    normalized = set()
    for extension in extensions:
        ext = str(extension).strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    if not normalized:
        return {ext.lower() for ext in defaults}
    return normalized


def build_manifest_from_pairs(
    image_dir: str,
    mask_dir: str,
    output_path: str,
    *,
    mask_suffix: str = "_segmentation",
    image_extensions: List[str] | None = None,
    mask_extensions: List[str] | None = None,
) -> Path:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_extension_set = _normalize_extensions(image_extensions, [".jpg", ".jpeg", ".png"])
    mask_extension_set = _normalize_extensions(mask_extensions, [".png", ".jpg", ".jpeg"])

    image_map: Dict[str, Path] = {}
    for path in sorted(image_root.glob("*")):
        if path.suffix.lower() not in image_extension_set:
            continue
        image_map[path.stem] = path

    entries: List[Dict[str, str]] = []
    for mask_path in sorted(mask_root.glob("*")):
        if mask_path.suffix.lower() not in mask_extension_set:
            continue
        case_id = _case_id_from_mask(str(mask_path), mask_suffix=mask_suffix)
        image_path = image_map.get(case_id)
        if image_path is None:
            continue
        entries.append(
            {
                "id": case_id,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return output


def build_manifest_from_isic(image_dir: str, mask_dir: str, output_path: str) -> Path:
    return build_manifest_from_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_path=output_path,
        mask_suffix="_segmentation",
    )


def build_manifest_from_ph2(dataset_root: str, output_path: str) -> Path:
    root = Path(dataset_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, str]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        image_candidates = sorted((case_dir / f"{case_id}_Dermoscopic_Image").glob("*"))
        mask_candidates = sorted((case_dir / f"{case_id}_lesion").glob("*"))
        if not image_candidates or not mask_candidates:
            continue
        entries.append(
            {
                "id": case_id,
                "image": str(image_candidates[0].resolve()),
                "mask": str(mask_candidates[0].resolve()),
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return output


def load_case_ids_from_csv(csv_path: str) -> List[str]:
    case_ids: List[str] = []
    with open(csv_path, "r", encoding="gbk", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            case_id = str(row[0]).strip()
            if case_id:
                case_ids.append(case_id)
    return case_ids


def build_manifest_from_case_ids(
    image_dir: str,
    mask_dir: str,
    case_ids: List[str],
    output_path: str,
) -> Path:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, str]] = []
    for case_id in case_ids:
        image_path = None
        for suffix in [".jpg", ".jpeg", ".png"]:
            candidate = image_root / f"{case_id}{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        mask_path = None
        for suffix in [".png", ".jpg", ".jpeg"]:
            candidate = mask_root / f"{case_id}_segmentation{suffix}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            continue

        entries.append(
            {
                "id": case_id,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return output


def load_manifest(path: str) -> List[Dict[str, Any]]:
    manifest_path = Path(path).expanduser()
    if manifest_path.is_dir():
        nearby_jsonl = sorted(manifest_path.glob("*.jsonl")) + sorted(manifest_path.parent.glob("*.jsonl"))
        hint = ""
        if nearby_jsonl:
            candidates = ", ".join(str(candidate) for candidate in nearby_jsonl[:5])
            hint = f" Nearby jsonl files: {candidates}"
        raise IsADirectoryError(
            f"manifest_path must be a jsonl file, got directory: {manifest_path}. "
            "Use data.artifact_root for artifact directories such as .../artifacts, "
            "and data.manifest_path/val_manifest_path/test_manifest_path for jsonl manifests."
            f"{hint}"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"manifest_path must be a regular file: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_rgb(path: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"failed to load image: {path}") from exc


def _load_mask(path: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            mask = np.array(image.convert("L"))
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"failed to load mask: {path}") from exc
    return (mask > 127).astype(np.uint8)


def _resize_rgb_legacy(image: np.ndarray, size: int) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize((size, size), resample=Image.BICUBIC)
    return np.array(resized, dtype=np.uint8)


def _resize_mask_legacy(mask: np.ndarray, size: int) -> np.ndarray:
    pil_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    resized = pil_mask.resize((size, size), resample=Image.BICUBIC)
    return np.array(resized, dtype=np.float32) / 255.0


def _horizontal_flip(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(array, axis=1))


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)


def _single_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32))


class ISICDualBridgeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_size: int = 256,
        band_radius: int = 5,
        train: bool = False,
        random_horizontal_flip: bool = False,
        legacy_preprocess: bool = False,
    ):
        self.samples = load_manifest(manifest_path)
        self.image_size = image_size
        self.band_radius = band_radius
        self.train = train
        self.random_horizontal_flip = random_horizontal_flip
        self.legacy_preprocess = legacy_preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def _should_flip(self) -> bool:
        return self.train and self.random_horizontal_flip and random.random() > 0.5

    def _load_real_arrays(self, sample: Dict[str, Any], flip: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_image = _load_rgb(sample["image"])
        raw_mask = _load_mask(sample["mask"])
        if self.legacy_preprocess:
            image = _resize_rgb_legacy(raw_image, self.image_size)
            mask = _resize_mask_legacy(raw_mask, self.image_size)
            prior_mask = (mask > 0.5).astype(np.uint8)
        else:
            image = resize_rgb(raw_image, self.image_size)
            prior_mask = resize_mask(raw_mask, self.image_size)
            mask = prior_mask.astype(np.float32)
        if flip:
            image = _horizontal_flip(image)
            mask = _horizontal_flip(mask)
            prior_mask = _horizontal_flip(prior_mask)
        return image, mask.astype(np.float32), prior_mask.astype(np.uint8)

    def _build_tensor_sample(
        self,
        sample: Dict[str, Any],
        image: np.ndarray,
        mask: np.ndarray,
        prior_mask: np.ndarray,
        *,
        image_path: str | None = None,
        is_synthetic: bool = False,
    ) -> Dict[str, Any]:
        priors = build_prior_stack(image, prior_mask, radius=self.band_radius)

        image_tensor = _image_to_tensor(image)
        mask_tensor = _single_to_tensor(mask[None, ...].astype(np.float32))
        mask_rgb = mask_tensor.repeat(3, 1, 1)

        return {
            "id": sample["id"],
            "image_path": image_path or sample["image"],
            "mask_path": sample["mask"],
            "image": image_tensor,
            "mask": mask_tensor,
            "mask_rgb": mask_rgb,
            "prior": torch.from_numpy(priors["prior"]),
            "band": torch.from_numpy(priors["band"]),
            "inner": torch.from_numpy(priors["inner"]),
            "outer": torch.from_numpy(priors["outer"]),
            "sdm": torch.from_numpy(priors["sdm"]),
            "is_synthetic": torch.tensor(is_synthetic, dtype=torch.bool),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image, mask, prior_mask = self._load_real_arrays(sample, flip=self._should_flip())
        return self._build_tensor_sample(sample, image=image, mask=mask, prior_mask=prior_mask)
