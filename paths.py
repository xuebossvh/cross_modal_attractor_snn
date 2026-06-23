"""项目根目录与输出路径（训练权重、日志、图表等）。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_ROOT = PROJECT_ROOT / "_data"

DEFAULT_CKPT = CHECKPOINTS_DIR / "cross_modal_snn.pt"
DEFAULT_CONFIG = CONFIGS_DIR / "v5.yaml"


def ensure_output_dirs():
    """创建 outputs 子目录（若不存在）。"""
    for d in (CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_from_root(path):
    """将配置中的相对路径解析为基于项目根的绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
