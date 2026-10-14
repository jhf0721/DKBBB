# Model-only release scope

This snapshot contains DKB³ itself: the shared prior encoder, two conditioned Brownian bridges, training losses, mask-anchored synthesis, and image-guided KCS. The bridge runtime required by these components is bundled.

## Scope

No downstream segmentation backbone, native baseline adapter, comparative experiment implementation, statistical/plotting utility, or generation-quality analysis script is included. Real datasets, original manifests, synthetic images, checkpoints, private logs, and manuscript source files are not redistributed.

The original source project is unchanged. The previous broader snapshot was backed up before scope reduction. Only the release copy was edited. No training, checkpoint modification, GPU benchmark, remote publication, or dependency installation was performed.

## Preserved algorithms and new interfaces

`DualBridgeSystem`, the prior encoder, branch adapters, normalization and sampling helpers retain their original implementations. The unused standalone SimpleUNet helper was removed. The KCS scoring-function bodies were extracted from the existing reranking evaluator without changing their formula.

`dual_bbdm.inference.refine_masks` and the single-case `refine_masks.py` CLI connect these components to external lesion-probability maps instead of constructing a particular segmentation network. The supplied image and probabilities must use the same spatial grid. Any external TTA probabilities must already be inverse-transformed to that grid.

This is a reusable model interface, not a replacement claim for every historical evaluation pipeline, preprocessing recipe, checkpoint or reported result. Seed handling preserves the caller's random-generator state; the new wrapper does not claim bitwise equivalence to all historical batch/seed orchestration.

## Protocol

1. Train both bridges only on the training split. Exclude validation/test cases from augmentation.
2. Retain the source training mask as the label of each mask-anchored synthetic image.
3. If training a downstream predictor, obtain it separately and control its training and model-selection protocol.
4. Online pseudo priors use the image and the supplied prediction. Ground-truth annotations are not accepted by the refinement API.
5. Report the actual network evaluations. The bundled schedule has NFE=3 per bridge sample; two refinement samples use six bridge calls.
6. The new YAML files are portable starter templates, not confirmed historical paper configurations. Exact split/checkpoint provenance still needs author verification.

`SNAPSHOT_MANIFEST.json` records copied-source hashes and extraction provenance. The developer-only audit with `--check-source` verifies original-source hashes, AST equality of core model definitions, and equality of retained KCS scoring bodies. Portable audits omit that flag.

## Before public release

Resolve the [manuscript alignment](MANUSCRIPT_ALIGNMENT.md) items, verify source/checkpoint/split provenance, confirm author and third-party redistribution rights, and test a fresh installation plus a data/checkpoint-backed end-to-end run. This packaging work does not perform those additional experiments or grant a license.
