#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
配置加载器 (Strategy Configuration Loader)
========================================================================
功能：
1. 扁平化单文件管理：一个策略对应一个自包含的 `.yaml` 配置文件。
2. 自动补充缺省兜底字段，保证即使配置文件只有精简字段也能稳健运行。
3. 支持按文件路径加载或自动扫描 `configs/*.yaml`。
========================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# 基础兜底默认值 (如果单个 YAML 漏写了某个可选参数，自动补充此兜底值)
DEFAULT_FALLBACKS = {
    "market": "csi500",
    "benchmark": "SH000905",
    "benchmark_ma": 20,
    "label_horizon": 1,
    "train_start": "2008-01-01",
    "train_end": "2024-12-31",
    "valid_start": "2025-01-01",
    "valid_end": "2025-06-30",
    "test_start": "2025-07-01",
    "test_end": "2026-08-24",
    "account_cash": 20000.0,
    "topk_stocks": 1,
    "n_drop_stocks": 1,
    "trade_unit": 100,
    "rebalance_days": 1,
    "risk_degree": 0.95,
    "only_main_board": True,
    "open_cost": 0.0001,
    "close_cost": 0.0001,
    "min_cost": 5.0,
    "deal_price": "close",
    "limit_threshold": 0.095,
    "stock_uptrend_filter": True,
    "drop_rank_threshold": 30,
    "stop_loss_rate": 0.035,
    "trailing_stop_trigger": 0.05,
    "trailing_stop_rate": 0.025,
    "max_holding_days": 10,
    "market_timing_mode": "half_position_timing",
    "account_drawdown_limit": 0.08,
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.0421,
            "max_depth": 8,
            "num_leaves": 210
        }
    }
}


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """加载单个 YAML 文件"""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"配置文件未找到: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_strategy_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载单个策略 YAML 配置文件，并自动补充缺省兜底字段
    """
    p = Path(config_path).resolve()
    cfg = load_yaml(p)

    # 浅拷贝兜底并补充
    merged = dict(DEFAULT_FALLBACKS)
    merged.update(cfg)

    # 规范 key 与 title
    if "key" not in merged or not merged["key"]:
        merged["key"] = p.stem
    if "title" not in merged or not merged["title"]:
        merged["title"] = p.stem
    merged["_config_path"] = str(p)
    return merged


def list_all_strategy_configs(configs_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    扫描 configs/ 目录下的所有策略 YAML 文件
    """
    c_dir = Path(configs_dir).resolve() if configs_dir else CONFIGS_DIR
    if not c_dir.exists():
        return []

    configs = []
    for f in sorted(c_dir.glob("*.yaml")):
        configs.append(load_strategy_config(f))
    return configs
