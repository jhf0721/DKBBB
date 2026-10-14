# Model-only validation record

Date: 2026-10-14. Checks used the existing CT2MR conda environment. Runtime tests had GPU visibility disabled. No packages were installed, feature weights downloaded, real datasets/checkpoints loaded, models trained, or GPU costs remeasured.

This record supersedes the broader snapshot's validation counts. Only DKB³ and its required bridge runtime remain.

## Completed checks

| Check | Result | Scope |
| --- | --- | --- |
| CPU tests in the release and an independent relocated copy | 20 passed | Four KCS tests, eight inference-interface tests, eight portability/prior/split tests |
| Public CLI smoke tests | 4 passed | `build_manifest.py`, `train_dual_bridge.py`, `export_bridge_artifacts.py`, `refine_masks.py`: successful `--help` |
| Bundled bridge factory import after relocation | Passed | Loaded the relocated copy's own `vendor/diff_bbdm/ddbm/script_util.py`, with no segmentation backbones |
| Standalone relocated audit | Passed | No neighboring source folders required |
| Static syntax audit | 29 Python files passed | 45 packaged files in total |
| Original-source provenance | 20 source files verified unchanged | Original files remain intact; release-copy modifications are listed in the audit |
| Core definition equivalence | Passed | AST equality for dual bridges, prior encoder, branch adapters and normalization/sampling helpers |
| Bridge data-loading equivalence | Passed | AST equality for the paired dataset and its retained loading/preprocessing helpers |
| KCS scoring equivalence | Passed | Original thresholding, masked statistics and scoring function bodies are unchanged |

Inference tests use a small fake bridge, not a trained model. They check output shapes, external candidates without a backbone, prediction-derived pseudo priors, reproducible sampling, preserved CPU RNG state, invalid inputs, reserved names and rejection of an unimplemented 12-step request.

KCS tests check score signs, candidate-order ties, the primary margin, and selection independence from cached ground-truth Dice fields.

## Rerun portable checks

From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' DIFF_BBDM_ROOT='' python -m unittest discover -s tests -v
python tools/check_release.py
python train_dual_bridge.py --help
python export_bridge_artifacts.py --help
python refine_masks.py --help
```

The developer-only `--check-source` audit additionally requires the original neighboring source folders. Omit that flag for an independently downloaded copy.

See [RELEASE_AUDIT.json](RELEASE_AUDIT.json) and [SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json) for local audit results and source/extraction provenance.

## Scope limitations

These tests do not verify a fresh installation, real bridge/model construction, downloaded perceptual weights, checkpoint compatibility, complete training/export/refinement, GPU RNG behavior, or manuscript performance reproduction. Importing the factory does not instantiate the bridge.

Historical split/checkpoint/config provenance, [manuscript alignment](MANUSCRIPT_ALIGNMENT.md), and [redistribution rights](../LICENSE_STATUS.md) still need separate confirmation. No claim is made that the new backbone-independent wrapper reproduces every historical evaluation/RNG protocol.
