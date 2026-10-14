# Manuscript–implementation alignment

Reference: the current `tmi/main.tex`, not the earlier AAAI manuscript. This model-only snapshot preserves the existing DKB³ numerical implementation; no checkpoints were retrained.

| Item | Current manuscript | Retained implementation |
| --- | --- | --- |
| Direction-specific geometry | I→M uses erosion/dilation with probability 0.5 and radius 1–5; M→I uses clean geometry | `compute_losses` passes one encoded prior to both branches without this perturbation |
| Structural-cycle guidance | Neutral distance/boundary channels; G/A/L recomputed from the synthesized image | The cycle passes the original paired prior into mask recovery |
| Offline sampling | 12 steps | Bundled schedule `[999, 1, 0]` has NFE=3; requesting 12 does not expand it |
| Split/checkpoint provenance | Specific final training/validation/test settings | Portable templates do not certify historical case lists or checkpoints |
| Third-party redistribution | Public release needs verified author/upstream rights | Required bridge runtime still needs complete licensing confirmation |

The illumination implementation and local diffusion objective also remain as copied; this is not an exhaustive mathematical-equivalence audit of the manuscript.

## Packaging changes

- Removed downstream segmentation backbones, native baseline adapters and paper-analysis scripts.
- Removed baseline configuration sections and unused segmentation helpers.
- Retained the core model, prior construction, training/export entrypoints and indispensable bridge runtime.
- Extracted existing KCS scoring bodies into a backbone-independent module.
- Added a refinement API and CLI that consume external lesion-probability maps. They do not use ground-truth annotations.
- Added portable data/split tooling and CPU tests.

The core model definitions and extracted scoring bodies are checked against the original source by `tools/check_release.py --check-source`. The new inference wrapper makes no claim to reproduce every historical preprocessing, batch ordering or RNG protocol.

Before claiming full manuscript alignment, confirm whether the final text should describe the historical implementation or whether a separately authorized algorithm/experimental update is required. This scope-reduction task does not silently make that scientific decision.
