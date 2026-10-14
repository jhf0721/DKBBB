from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path


def _is_legacy_root(path: Path) -> bool:
    return (path / "ddbm" / "script_util.py").exists()


def _candidate_legacy_roots() -> list[Path]:
    current = Path(__file__).resolve()
    candidates: list[Path] = []

    env_root = os.environ.get("DIFF_BBDM_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    # This snapshot must work without the author's surrounding workspace.
    candidates.append(current.parents[1] / "vendor" / "diff_bbdm")

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def ensure_legacy_ddbm_on_path() -> Path:
    candidates = _candidate_legacy_roots()
    legacy_root = next((path for path in candidates if _is_legacy_root(path)), None)
    if legacy_root is None:
        checked = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "legacy diff_bbdm root not found. Set DIFF_BBDM_ROOT to the folder "
            "that contains ddbm/script_util.py. The bundled default is "
            "vendor/diff_bbdm inside this repository.\n"
            f"Checked:\n  - {checked}"
        )
    legacy_str = str(legacy_root)
    while legacy_str in sys.path:
        sys.path.remove(legacy_str)
    sys.path.insert(0, legacy_str)
    _drop_cached_ddbm_from_other_roots(legacy_root)
    return legacy_root


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _module_is_from_root(module: object, root: Path) -> bool:
    paths: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(Path(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path:
        paths.extend(Path(path) for path in module_path)
    return bool(paths) and any(_is_relative_to(path, root) for path in paths)


def _drop_cached_ddbm_from_other_roots(root: Path) -> None:
    ddbm_module = sys.modules.get("ddbm")
    if ddbm_module is not None and not _module_is_from_root(ddbm_module, root):
        for name in list(sys.modules):
            if name == "ddbm" or name.startswith("ddbm."):
                del sys.modules[name]
        return

    for name, module in list(sys.modules.items()):
        if name.startswith("ddbm.") and not _module_is_from_root(module, root):
            del sys.modules[name]


def _filter_kwargs(callable_obj, cfg: dict) -> dict:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(cfg)
    return {
        name: cfg[name]
        for name in signature.parameters
        if name != "self" and name in cfg
    }


def _compat_create_model_and_diffusion(script_util):
    create_model = getattr(script_util, "create_model", None)
    if not callable(create_model):
        return None

    karras_denoiser = getattr(script_util, "KarrasDenoiser", None)
    if karras_denoiser is None:
        karras_module = importlib.import_module("ddbm.karras_diffusion")
        karras_denoiser = getattr(karras_module, "KarrasDenoiser", None)
    if karras_denoiser is None:
        return None

    def create_model_and_diffusion(**cfg):
        model = create_model(**_filter_kwargs(create_model, cfg))
        diffusion = karras_denoiser(**_filter_kwargs(karras_denoiser, cfg))
        return model, diffusion

    return create_model_and_diffusion


def load_legacy_factory():
    legacy_root = ensure_legacy_ddbm_on_path()
    script_util = importlib.import_module("ddbm.script_util")
    model_and_diffusion_defaults = getattr(script_util, "model_and_diffusion_defaults", None)
    if not callable(model_and_diffusion_defaults):
        module_path = getattr(script_util, "__file__", "<unknown>")
        raise ImportError(
            "ddbm.script_util does not define model_and_diffusion_defaults. "
            f"Loaded module: {module_path}. DIFF_BBDM_ROOT: {legacy_root}"
        )

    create_model_and_diffusion = getattr(script_util, "create_model_and_diffusion", None)
    if not callable(create_model_and_diffusion):
        create_model_and_diffusion = _compat_create_model_and_diffusion(script_util)
    if not callable(create_model_and_diffusion):
        module_path = getattr(script_util, "__file__", "<unknown>")
        exports = ", ".join(name for name in dir(script_util) if not name.startswith("_"))
        raise ImportError(
            "ddbm.script_util does not define create_model_and_diffusion and cannot be "
            "adapted from create_model + KarrasDenoiser. Set DIFF_BBDM_ROOT to the "
            "diff_bbdm checkout used for bridge training, or copy the compatible "
            f"ddbm/script_util.py. Loaded module: {module_path}. DIFF_BBDM_ROOT: "
            f"{legacy_root}. Available exports: {exports}"
        )

    return create_model_and_diffusion, model_and_diffusion_defaults
