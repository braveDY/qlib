#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
配置加载与继承合并引擎 (Configuration Loader & Inheritance Engine)
========================================================================
功能：
1. 自动定位并加载 `configs/base_config.yaml` 基础配置。
2. 支持子配置文件对基础配置的深度合并继承（Deep Merge），仅需编写差异化字段。
3. 支持按文件路径、专题目录（Suite）或自动扫描全量配置。
========================================================================
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
BASE_CONFIG_PATH = CONFIGS_DIR / "base_config.yaml"


def deep_merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典 (递归覆盖)"""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """加载单个 YAML 文件"""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"配置文件未找到: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_base_config(base_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """加载全局基础配置字典"""
    p = Path(base_path).resolve() if base_path else BASE_CONFIG_PATH
    return load_yaml(p)


def load_experiment_config(config_path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    加载单个实验配置，并自动与基础配置进行深度合并继承
    """
    p = Path(config_path).resolve()
    exp_cfg = load_yaml(p)
    base_cfg = get_base_config(base_path)

    merged = deep_merge_dicts(base_cfg, exp_cfg)
    # 自动补充 key (默认取文件名不含后缀)
    if "key" not in merged or not merged["key"]:
        merged["key"] = p.stem
    if "title" not in merged or not merged["title"]:
        merged["title"] = p.stem
    merged["_config_path"] = str(p)
    return merged


def load_suite_configs(suite_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """加载某个专题目录下的所有 YAML 实验配置"""
    s_dir = Path(suite_dir).resolve()
    if not s_dir.exists():
        raise FileNotFoundError(f"专题目录未找到: {s_dir}")

    configs = []
    for yaml_file in sorted(s_dir.glob("*.yaml")):
        if yaml_file.name != "base_config.yaml":
            configs.append(load_experiment_config(yaml_file))
    return configs


def discover_all_configs(configs_root: Optional[Union[str, Path]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    自动扫描发现所有专题分类下的配置文件：
    返回格式: { "models": [cfg1, cfg2], "factors": [...], "strategies": [...] }
    """
    root = Path(configs_root).resolve() if configs_root else CONFIGS_DIR
    categories = {}

    for sub_dir in sorted(root.iterdir()):
        if sub_dir.is_dir():
            yaml_files = sorted(sub_dir.glob("*.yaml"))
            if yaml_files:
                categories[sub_dir.name] = [load_experiment_config(f) for f in yaml_files]

    return categories
