"""Static packaging audit, with an optional check of original source hashes."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_ANYWHERE = {"__pycache__", ".git", ".pytest_cache", ".venv", "venv"}
IGNORED_AT_ROOT = {"datasets", "outputs", "artifacts", "weights", "results", "logs", "wandb"}
PRIVATE_PATH = re.compile(r"/home/[A-Za-z0-9_.-]+/|/mnt/data/")
UNBUNDLED_SUFFIXES = {".pt", ".pth", ".ckpt", ".xlsx", ".jsonl", ".log"}


def definition_dump(path, names, bodies_only=False):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            target = ast.Module(body=node.body, type_ignores=[]) if bodies_only else node
            definitions[node.name] = ast.dump(target, include_attributes=False)
    return definitions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_path(root, record):
    relative = Path(record["file"])
    if record["source_group"] == "bridge-runtime":
        return root.parents[1] / "2/diff_bbdm" / Path(*relative.parts[2:])
    if record["source_group"] == "tmi":
        return root.parent / "tmi" / relative.name
    return root.parent / "dual-bbdm" / relative


def audit(root, check_source=False):
    files = [path for path in sorted(root.rglob("*")) if path.is_file()
             and not (set(path.relative_to(root).parts) & IGNORED_ANYWHERE)
             and path.relative_to(root).parts[0] not in IGNORED_AT_ROOT]
    errors = []
    if (root / "vendor/seg").exists():
        errors.append("Downstream segmentation backbones must not be bundled in the model-only release")
    python_count = 0
    for path in files:
        relative = str(path.relative_to(root))
        if path.is_symlink():
            errors.append(f"Symlink requires explicit review: {relative}")
        if path.suffix in UNBUNDLED_SUFFIXES:
            errors.append(f"Dataset/checkpoint/private output should not be packaged: {relative}")
        if path.suffix == ".py":
            python_count += 1
            text = path.read_text(encoding="utf-8")
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as exc:
                errors.append(f"Syntax error in {relative}: {exc}")
            # The audit rule itself is intentionally a pattern, not a machine path.
            if path != root / "tools/check_release.py" and PRIVATE_PATH.search(text):
                errors.append(f"Author-machine path in {relative}")
        if path.suffix == ".yaml" and PRIVATE_PATH.search(path.read_text()):
            errors.append(f"Author-machine path in {relative}")
    manifest = json.loads((root / "docs/SNAPSHOT_MANIFEST.json").read_text())
    changed = []
    verified = 0
    for record in manifest["copied_source_files"]:
        release_path = root / record["file"]
        if not release_path.is_file():
            errors.append(f"Missing copied file: {record['file']}")
            continue
        if sha(release_path) != record["source_sha256"]:
            changed.append(record["file"])
        if check_source:
            original = original_path(root, record)
            if not original.is_file() or sha(original) != record["source_sha256"]:
                errors.append(f"Original source missing or changed: {record['file']}")
            else:
                verified += 1
    core_checks = None
    if check_source:
        core_checks = {}
        names = {"to_neg_one_to_one", "to_zero_to_one", "_extract", "PriorEncoder",
                 "ConditionAdapter", "BranchBridgeModel", "DualBridgeOutputs", "DualBridgeSystem"}
        original_models = root.parent / "dual-bbdm/dual_bbdm/models.py"
        if original_models.is_file():
            source_definitions = definition_dump(original_models, names)
            release_definitions = definition_dump(root / "dual_bbdm/models.py", names)
            core_checks["bridge_and_prior_definitions_unchanged"] = (
                source_definitions.keys() == names and source_definitions == release_definitions
            )
        else:
            core_checks["bridge_and_prior_definitions_unchanged"] = False
        original_dataset = root.parent / "dual-bbdm/dual_bbdm/dataset.py"
        dataset_names = {"ISICDualBridgeDataset", "_load_rgb", "_load_mask", "_image_to_tensor",
                         "_single_to_tensor", "_horizontal_flip", "_resize_rgb_legacy", "_resize_mask_legacy"}
        if original_dataset.is_file():
            source_definitions = definition_dump(original_dataset, dataset_names)
            release_definitions = definition_dump(root / "dual_bbdm/dataset.py", dataset_names)
            core_checks["bridge_data_loading_unchanged"] = (
                source_definitions.keys() == dataset_names and source_definitions == release_definitions
            )
        else:
            core_checks["bridge_data_loading_unchanged"] = False
        for component in manifest.get("extracted_component_provenance", []):
            original = root.parent / "dual-bbdm" / component["source_file"]
            functions = set(component["preserved_functions"])
            matches = False
            if original.is_file():
                source_definitions = definition_dump(original, functions, bodies_only=True)
                release_definitions = definition_dump(root / component["file"], functions, bodies_only=True)
                matches = source_definitions.keys() == functions and source_definitions == release_definitions
                if component.get("source_sha256"):
                    matches = matches and sha(original) == component["source_sha256"]
            core_checks["kcs_scoring_bodies_unchanged"] = matches
        for name, matches in core_checks.items():
            if not matches:
                errors.append(f"Original implementation equivalence failed: {name}")
    return {
        "status": "failed" if errors else "passed",
        "packaged_files": len(files), "python_files": python_count,
        "copied_source_files": len(manifest["copied_source_files"]),
        "original_source_check_requested": check_source,
        "original_source_files_verified_unchanged": verified,
        "preserved_core_implementation_checks": core_checks,
        "modified_copied_files": changed, "errors": errors,
        "scope": "Static packaging audit, not a fresh installation or paper reproduction.",
        "outstanding": ["manuscript/runtime alignment", "third-party license verification",
                        "exact final split/checkpoint/config provenance", "fresh-environment end-to-end testing"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-source", action="store_true",
                        help="Developer-only: requires the original neighboring source folders")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root.resolve(), args.check_source)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
