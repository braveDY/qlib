#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
全量特征工程持久化缓存与极速动态切片模块 (Cached Dataset Engine)
支持:
- 158 维 Alpha158 特征持久化缓存 (按 标签周期/股票池 独立隔离)
- 可选追加 个股相对基准的相对强度 (RS) 特征: RS_N = 个股N日收益 / 基准N日收益 - 1
"""

import pickle
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from .paths import get_cache_dir


def compute_rs_features_df(
    market: str = "csi500",
    bench_symbol: str = "SH000905",
    horizons: tuple = (5, 10, 20, 60),
    full_start: str = "2008-01-01",
    full_end: str = "2026-08-24",
) -> pd.DataFrame:
    """
    计算个股相对基准的相对强度 (Relative Strength) 特征矩阵 (MultiIndex: datetime, instrument):
        RS_N(t) = (close_t / close_{t-N}) / (bench_t / bench_{t-N}) - 1
    含义: 个股 N 日收益率相对基准 N 日收益率的超额动量。RS > 0 表示该股跑赢大盘。
    设计动机: 熊市里也有逆势牛股, 绝对动量 (如 MA 多头) 会漏掉它们, 而相对强度
    天然把"大盘跌它不跌/涨得更多"的股票排在前面 —— 这正是 topk=1 单票策略需要的选股信号。
    无未来函数: 只用 t 日及以前的数据。
    """
    from qlib.data import D

    # 1. 个股收盘价 (MultiIndex: instrument, datetime) — 通过 D.instruments 获取股票池配置
    inst_cfg = D.instruments(market)
    close = D.features(inst_cfg, ["$close"], start_time=full_start, end_time=full_end, freq="day")
    close_s = close["$close"]

    # 2. 基准指数收盘价
    bench = D.features([bench_symbol], ["$close"], start_time=full_start, end_time=full_end, freq="day")
    bench_s = bench["$close"]
    if bench_s.index.nlevels > 1:
        bench_s = bench_s.droplevel(0)  # index: datetime

    # 3. 相对强弱比值 (个股 / 基准, 按 datetime 自动广播)
    ratio = close_s / bench_s

    # 4. 各周期相对强度特征
    rs_cols = {}
    for h in horizons:
        lagged = ratio.groupby(level="instrument").shift(h)
        rs_cols[("feature", f"RS{h}")] = (ratio / lagged - 1.0).astype("float32")

    rs_df = pd.DataFrame(rs_cols)
    # 统一 index 顺序为 (datetime, instrument), 与特征矩阵一致
    if rs_df.index.nlevels > 1 and rs_df.index.names != ["datetime", "instrument"]:
        rs_df = rs_df.reorder_levels(["datetime", "instrument"]).sort_index()
    print(f"       📈 RS 相对强度特征计算完成: {list(rs_cols.keys())} ({len(rs_df):,} 行)")
    return rs_df


def get_cached_dataset(
    market: str = "csi500",
    segments: dict = None,
    cache_dir: Optional[Union[str, Path]] = None,
    full_start: str = "2008-01-01",
    full_end: str = "2026-08-24",
    force_recompute: bool = False,
    label_horizon: int = 1,
    rs_horizons: tuple = (),
    bench_symbol: str = "SH000905",
) -> DatasetH:
    """
    全量特征工程缓存与动态切片引擎：
    1. 首次运行：自动计算该股票池 (market) 从 2008 年至今的全量 158 维特征与指定预测周期标签 (label_horizon)，
       并持久化保存至共享 cache_dir；
    2. 后续运行：直接从缓存中 0.3 秒加载全量数据，并根据用户传入的 segments 动态切片，
       即使随意修改训练时间、验证时间、回测时间，也完全不需要重新计算 158 个因子！
    3. 支持 1 日标签 (1d) 与 多日标签 (5d 等) 独立缓存隔离。
    4. 可选 rs_horizons: 非空时在加载后实时追加个股相对基准的相对强度特征 (无需重建 158 维缓存)。
    """
    c_dir = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    c_dir.mkdir(parents=True, exist_ok=True)

    label_tag = f"{label_horizon}d"
    target_cache_path = c_dir / f"alpha158_{market}_{label_tag}_full.pkl"
    legacy_cache_path = c_dir / f"alpha158_{market}_full.pkl"

    cache_path = None
    if not force_recompute:
        if target_cache_path.exists():
            cache_path = target_cache_path
        elif label_horizon == 1 and legacy_cache_path.exists():
            cache_path = legacy_cache_path

    loaded_dict = None


    if cache_path is not None and cache_path.exists():
        print(f"\n[1/5] 🚀 命中 [{market} | {label_tag} 标签] 全量特征持久化缓存 ({cache_path})，0.3 秒极速加载...")
        try:
            with open(cache_path, "rb") as f:
                loaded_dict = pickle.load(f)
            print(f"       ✅ 缓存加载成功！全量数据规模: {loaded_dict['infer'].shape}")
        except Exception as e:
            print(f"       ⚠️ 缓存读取失败 ({e})，将重新全量计算...")
            loaded_dict = None

    if loaded_dict is None:
        save_path = target_cache_path
        print(f"\n[1/5] ⏳ 未检测到有效缓存，首次全量构建 [{market} | {label_tag} 标签] 158 维特征工程 ({full_start} ~ {full_end})...")
        label_expr = f"Ref($close, -{label_horizon + 1}) / Ref($close, -1) - 1"
        label_config = ([label_expr], ["LABEL0"])
        print(f"       🎯 预测标签配置: {label_expr}")

        h = Alpha158(
            instruments=market,
            start_time=full_start,
            end_time=full_end,
            fit_start_time=full_start,
            fit_end_time=full_end,
            label=label_config,
        )
        loaded_dict = {
            "infer": getattr(h, "_infer", None),
            "learn": getattr(h, "_learn", None),
            "data": getattr(h, "_data", None),
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"       💾 正在将全量特征矩阵持久化保存至: {save_path} ...")
        with open(save_path, "wb") as f:
            pickle.dump(loaded_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"       ✅ 全量特征缓存写入完成！下次运行将享受秒级秒开！")

    # 可选: 实时追加个股相对基准的相对强度 (RS) 特征 (不写入缓存, 每次加载后计算)
    if rs_horizons:
        rs_df = compute_rs_features_df(
            market=market,
            bench_symbol=bench_symbol,
            horizons=tuple(rs_horizons),
            full_start=full_start,
            full_end=full_end,
        )
        for key in ("infer", "learn"):
            if loaded_dict.get(key) is not None:
                loaded_dict[key] = loaded_dict[key].join(rs_df)
        print(f"       ✅ RS 特征已并入特征矩阵, 新规模: {loaded_dict['infer'].shape}")

    # 构建动态 Handler 并根据用户配置动态切片
    h_dynamic = Alpha158.__new__(Alpha158)
    h_dynamic._infer = loaded_dict["infer"]
    h_dynamic._learn = loaded_dict["learn"]
    h_dynamic._data = loaded_dict["data"]
    h_dynamic.drop_raw = False
    h_dynamic.fetch_orig = False

    dataset = DatasetH(handler=h_dynamic, segments=segments)
    return dataset
