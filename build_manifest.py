from __future__ import annotations

import argparse

from dual_bbdm.dataset import build_manifest_from_pairs


def _parse_extensions(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-suffix", default="_segmentation")
    parser.add_argument("--image-exts", default=".jpg,.jpeg,.png")
    parser.add_argument("--mask-exts", default=".png,.jpg,.jpeg")
    args = parser.parse_args()

    output = build_manifest_from_pairs(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_path=args.output,
        mask_suffix=args.mask_suffix,
        image_extensions=_parse_extensions(args.image_exts),
        mask_extensions=_parse_extensions(args.mask_exts),
    )
    print(output)


if __name__ == "__main__":
    main()
