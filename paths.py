"""项目根目录与输出路径（训练权重、日志、图表等）。

目录约定：
  outputs/checkpoints/          各版本 checkpoint（跨版本共用）
  outputs/outputs_v7/figures/   版本专属图表
  outputs/outputs_v7/logs/
  outputs/outputs_v7/tables/
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_ROOT = PROJECT_ROOT / "_data"

DEFAULT_CKPT = CHECKPOINTS_DIR / "cross_modal_snn.pt"
DEFAULT_CONFIG = CONFIGS_DIR / "v7.yaml"


def _normalize_version_folder(tag):
    """'v5' / '5' / 'outputs_v7' -> 'outputs_v7'。"""
    tag = str(tag).strip()
    if tag.startswith("outputs_"):
        return tag
    if re.fullmatch(r"v\d+", tag):
        return f"outputs_{tag}"
    if tag.isdigit():
        return f"outputs_v{tag}"
    return f"outputs_{tag}"


def infer_output_version(cfg):
    """从 train.output_version 或 configs/v7.yaml 推断版本目录名。"""
    train = cfg.get("train", {})
    if train.get("output_version"):
        return _normalize_version_folder(train["output_version"])
    path = cfg.get("_config_path", "")
    stem = Path(path).stem
    if re.fullmatch(r"v\d+", stem):
        return f"outputs_{stem}"
    return "outputs_default"


def version_bundle_dir(cfg):
    """outputs/outputs_v7 等版本根目录。"""
    return OUTPUTS_DIR / infer_output_version(cfg)


def figures_dir(cfg):
    return version_bundle_dir(cfg) / "figures"


def logs_dir(cfg):
    return version_bundle_dir(cfg) / "logs"


def tables_dir(cfg):
    return version_bundle_dir(cfg) / "tables"


def ensure_output_dirs(cfg=None):
    """创建 checkpoints + 当前版本 figures/logs/tables。"""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        return
    for d in (figures_dir(cfg), logs_dir(cfg), tables_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)


def resolve_from_root(path):
    """将配置中的相对路径解析为基于项目根的绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
