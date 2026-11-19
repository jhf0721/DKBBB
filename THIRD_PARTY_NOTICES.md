# Third-party runtime and redistribution status

This model-only release does not bundle third-party segmentation backbones. It does retain the bridge runtime required to construct DKB³. Removing comparison models does not remove the licensing obligations of runtime dependencies.

| Required component | Location | Status |
| --- | --- | --- |
| Locally adapted Brownian bridge runtime | `vendor/diff_bbdm/ddbm/` | Copied from the runtime used by the original project. Complete upstream version and license texts need confirmation. Existing in-file notices are preserved. |
| NVIDIA EDM / SongUNet | `vendor/diff_bbdm/ddbm/edm_unet.py` | Copyright 2022 NVIDIA CORPORATION & AFFILIATES. The file explicitly states [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); do not relicense it as MIT/Apache. |
| k-diffusion-derived portions | `vendor/diff_bbdm/ddbm/karras_diffusion.py` | Original file references [k-diffusion](https://github.com/crowsonkb/k-diffusion); verify the incorporated version and restore applicable license texts. |

PyTorch, torchvision, PIQ, OpenCV, and other separately installed packages retain their own licenses and model-weight terms. Medical images and annotations are not licensed or redistributed by this repository.

The framework figure is from the author-supplied manuscript. Confirm its redistribution rights alongside the author-owned code license.

No blanket license grant is made for this tree. Attribution alone does not substitute for permission or license compliance. See [LICENSE_STATUS.md](LICENSE_STATUS.md) before uploading.
