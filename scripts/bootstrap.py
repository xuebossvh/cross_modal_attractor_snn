"""脚本入口：切到项目根目录并加入 sys.path（保证 configs/、common 可导入）。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
os.chdir(ROOT)
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
