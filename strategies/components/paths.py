#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
统一实验输出与路径管理器 (Experiment Outputs & Paths Manager)
========================================================================
功能：
1. 项目根目录与策略目录自适应定位，解决在不同目录下执行脚本导致的相对路径错乱。
2. 支持环境变量 `QLIB_OUTPUT_DIR` 动态重定向输出路径（例如 AutoDL 数据盘 `/root/autodl-tmp/outputs`）。
3. 规范化分层目录结构：
   - cache/       : 跨实验共享的特征缓存 (alpha158_csi500_1d/5d_full.pkl 等)
   - logs/        : 运行日志归档
   - reports/     : 跨实验横向对比报告 (Markdown / CSV)
   - experiments/ : 单个实验隔离产物 (config.json, metrics.json, html看板, 交割单)
4. 自动化结构化指标保存与实验结果归档。
========================================================================
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# 自动定位项目根目录和策略目录 (基于当前文件位置定位)
STRATEGIES_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT_DIR = STRATEGIES_DIR.parent


class CustomJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，支持 numpy / pandas / Path 类型的序列化"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (set, frozenset, tuple)):
            return list(obj)
        return super().default(obj)


class ExperimentPaths:
    """实验路径管理器配置类"""

    def __init__(self, base_output_dir: Optional[str] = None):
        if base_output_dir is not None:
            self._root = Path(base_output_dir).resolve()
        elif "QLIB_OUTPUT_DIR" in os.environ and os.environ["QLIB_OUTPUT_DIR"].strip():
            self._root = Path(os.environ["QLIB_OUTPUT_DIR"].strip()).resolve()
        else:
            self._root = (STRATEGIES_DIR / "outputs").resolve()

    @property
    def root(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    @property
    def cache_dir(self) -> Path:
        """全局共享特征缓存目录"""
        p = self.root / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        """日志归档目录"""
        p = self.root / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_dir(self) -> Path:
        """多实验横向对比报告与综合分析目录"""
        p = self.root / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def experiments_dir(self) -> Path:
        """各独立实验专属目录根路径"""
        p = self.root / "experiments"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_exp_dir(self, exp_key: str) -> Path:
        """获取指定实验的独立产物目录"""
        p = self.experiments_dir / exp_key
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_exp_file(self, exp_key: str, filename: str) -> Path:
        """获取指定实验目录下的文件路径"""
        return self.get_exp_dir(exp_key) / filename


# 全局默认路径实例
paths = ExperimentPaths()


def get_paths(base_output_dir: Optional[str] = None) -> ExperimentPaths:
    """获取或初始化路径管理器"""
    if base_output_dir:
        return ExperimentPaths(base_output_dir)
    return paths


def get_cache_dir() -> Path:
    return paths.cache_dir


def get_logs_dir() -> Path:
    return paths.logs_dir


def get_reports_dir() -> Path:
    return paths.reports_dir


def get_experiments_dir() -> Path:
    return paths.experiments_dir


def get_exp_dir(exp_key: str) -> Path:
    return paths.get_exp_dir(exp_key)


def save_experiment_artifacts(
    exp_key: str,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    exp_paths: Optional[ExperimentPaths] = None,
) -> Path:
    """
    保存实验的结构化快照 (config.json, metrics.json)
    """
    mgr = exp_paths or paths
    exp_dir = mgr.get_exp_dir(exp_key)

    if config is not None:
        cfg_file = exp_dir / "config.json"
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)

    if metrics is not None:
        metrics_file = exp_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)

    return exp_dir


def load_experiment_metrics(exp_key: str, exp_paths: Optional[ExperimentPaths] = None) -> Optional[Dict[str, Any]]:
    """读取指定实验的指标 JSON"""
    mgr = exp_paths or paths
    metrics_file = mgr.get_exp_dir(exp_key) / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def list_all_experiments(
    exp_paths: Optional[ExperimentPaths] = None,
    only_completed: bool = False,
) -> List[Dict[str, Any]]:
    """扫描所有已运行的实验列表及其指标"""
    mgr = exp_paths or paths
    exp_list = []
    if not mgr.experiments_dir.exists():
        return exp_list

    for exp_p in sorted(mgr.experiments_dir.iterdir()):
        if exp_p.is_dir():
            metrics_file = exp_p / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["exp_key"] = exp_p.name
                        if not only_completed or "tot_ret" in data:
                            exp_list.append(data)
                except Exception:
                    if not only_completed:
                        exp_list.append({"exp_key": exp_p.name, "status": "corrupt_metrics"})
            else:
                if not only_completed:
                    exp_list.append({"exp_key": exp_p.name, "status": "no_metrics"})
    return exp_list

