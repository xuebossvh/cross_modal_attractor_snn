"""创建 outputs/checkpoints 与 outputs/outputs_v5/{figures,logs,tables}。

在 `tee outputs/.../logs/xxx.log` 之前运行，避免目录不存在。
仅依赖标准库，无需 PyYAML。

用法：
    python scripts/mkdir_outputs.py --config configs/v5.yaml
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import CHECKPOINTS_DIR, infer_output_version, version_bundle_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v5.yaml")
    args = ap.parse_args()

    cfg = {"_config_path": args.config}
    stem = Path(args.config).stem
    if re.fullmatch(r"v\d+", stem):
        cfg.setdefault("train", {})["output_version"] = stem

    bundle = version_bundle_dir(cfg)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("figures", "logs", "tables"):
        (bundle / sub).mkdir(parents=True, exist_ok=True)

    print(f"[mkdir] checkpoints -> {CHECKPOINTS_DIR}", flush=True)
    print(f"[mkdir] version bundle -> {bundle}", flush=True)


if __name__ == "__main__":
    main()
