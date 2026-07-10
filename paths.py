"""Project paths and versioned output directories."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_ROOT = PROJECT_ROOT / "_data"

DEFAULT_CKPT = CHECKPOINTS_DIR / "cross_modal_snn.pt"
DEFAULT_CONFIG = CONFIGS_DIR / "v10f.yaml"


def _normalize_version_folder(tag):
    """Map v10a / 10 / outputs_v10a to outputs_v10a."""
    tag = str(tag).strip()
    if tag.startswith("outputs_"):
        return tag
    if re.fullmatch(r"v\d+", tag):
        return f"outputs_{tag}"
    if tag.isdigit():
        return f"outputs_v{tag}"
    return f"outputs_{tag}"


def infer_output_version(cfg):
    """Infer the versioned output directory from config metadata."""
    train = cfg.get("train", {})
    if train.get("output_version"):
        return _normalize_version_folder(train["output_version"])
    path = cfg.get("_config_path", "")
    stem = Path(path).stem
    if re.fullmatch(r"v\d+", stem):
        return f"outputs_{stem}"
    return "outputs_default"


def version_bundle_dir(cfg):
    """Return outputs/outputs_v* for the current config."""
    return OUTPUTS_DIR / infer_output_version(cfg)


def figures_dir(cfg):
    return version_bundle_dir(cfg) / "figures"


def logs_dir(cfg):
    return version_bundle_dir(cfg) / "logs"


def tables_dir(cfg):
    return version_bundle_dir(cfg) / "tables"


def ensure_output_dirs(cfg=None):
    """Create checkpoint and versioned artifact directories."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        return
    for d in (figures_dir(cfg), logs_dir(cfg), tables_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)


def resolve_from_root(path):
    """Resolve a config path relative to the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
