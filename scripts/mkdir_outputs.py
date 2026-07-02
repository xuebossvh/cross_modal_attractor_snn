"""创建 outputs/checkpoints 与 outputs/outputs_vN/{figures,logs,tables}。

在 `tee outputs/.../logs/xxx.log` 之前运行，避免目录不存在。
仅依赖标准库，无需 PyYAML。

用法：
    python scripts/mkdir_outputs.py --config configs/v9b.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import CHECKPOINTS_DIR, version_bundle_dir


def _load_output_cfg(config_path):
    """Read only train.output_version using stdlib so this works before torch/yaml."""
    cfg = {"_config_path": config_path, "train": {}}
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path

    in_train = False
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.startswith("train:"):
                in_train = True
                continue
            if line and not line.startswith((" ", "\t")):
                in_train = False
            if in_train and line.lstrip().startswith("output_version:"):
                value = line.split(":", 1)[1].strip().strip("'\"")
                if value:
                    cfg["train"]["output_version"] = value
                break

    if "output_version" not in cfg["train"]:
        cfg["train"]["output_version"] = Path(config_path).stem
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v9b.yaml")
    args = ap.parse_args()

    cfg = _load_output_cfg(args.config)

    bundle = version_bundle_dir(cfg)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("figures", "logs", "tables"):
        (bundle / sub).mkdir(parents=True, exist_ok=True)

    print(f"[mkdir] checkpoints -> {CHECKPOINTS_DIR}", flush=True)
    print(f"[mkdir] version bundle -> {bundle}", flush=True)


if __name__ == "__main__":
    main()
