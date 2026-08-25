#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
向后兼容入口: 运行模型层对照实验 (Proxy to run_experiments.py)
============================================================
推荐直接使用全新解耦引擎:
    python run_experiments.py --suite configs/models/
============================================================
"""
import sys
from pathlib import Path

STRATEGIES_DIR = Path(__file__).resolve().parent
sys.path.append(str(STRATEGIES_DIR))

from run_experiments import main

if __name__ == "__main__":
    # 如果用户没有指定参数，默认执行 models 专题
    if len(sys.argv) == 1:
        sys.argv.extend(["-s", str(STRATEGIES_DIR / "configs" / "models")])
    main()
