#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
========================================================================================
模型层多实验对比执行引擎 (Experiment Runner)
========================================================================================
支持的对比实验矩阵：
- 实验 A (exp_a_baseline):         基线 (1日标签 + 2008~2024 17年历史训练 + 风控)
- 实验 B (exp_b_label5d):          5日标签 (5日标签 + 2008~2024 17年历史训练 + 风控)
- 实验 C (exp_c_label5d_roll3y):    5日标签 + 近3年 (5日标签 + 2022~2024 3年窗口训练 + 风控)
- 实验 D (exp_d_label5d_rs):       5日标签 + RS相对强度 (5日标签 + 17年 + RS特征 + 风控)
- 实验 E (exp_e_daily_rotate):     1日标签 + 日频满仓轮动 (1日标签 + 17年 + 无风控轮动)
- 实验 F (exp_f_double_ensemble):  DoubleEnsemble 双重集成模型 (5日标签 + 17年)

特性：
- 集中式共享特征缓存 (outputs/cache)，杜绝重复计算与磁盘冗余
- 自动化指标结构化落盘 (outputs/experiments/<exp_key>/metrics.json, config.json)
- 自动化多实验横向对比报告生成 (outputs/reports/experiment_comparison.md & .csv)
- 命令行支持: --experiments, --output_dir, --list, --report_only
========================================================================================
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
import warnings
import argparse
from pathlib import Path
import copy
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 确保 components 模块可导入
sys.path.append(str(Path(__file__).resolve().parent))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData

from components import (
    AffordableTopkDropoutStrategy,
    DayRotateStrategy,
    generate_daily_delivery_slip,
    generate_all_in_one_dashboard,
    get_cached_dataset,
    get_paths,
    save_experiment_artifacts,
    load_experiment_metrics,
    list_all_experiments,
    ExperimentPaths,
)
from affordable_topk_workflow import CONFIG as BASE_CONFIG


EXPERIMENTS = {
    "exp_a_baseline": {
        "title": "A 组: 基线 (1日标签 + 17年训练)",
        "label_horizon": 1,
        "train_start": "2008-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2026-08-24",
    },
    "exp_b_label5d": {
        "title": "B 组: 5日标签 (5日标签 + 17年训练)",
        "label_horizon": 5,
        "train_start": "2008-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2026-08-24",
    },
    "exp_c_label5d_roll3y": {
        "title": "C 组: 5日标签 + 近3年 (5日标签 + 3年窗口训练)",
        "label_horizon": 5,
        "train_start": "2022-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2026-08-24",
    },
    "exp_d_label5d_rs": {
        "title": "D 组: 5日标签 + RS相对强度 (5日标签 + 17年 + RS特征)",
        "label_horizon": 5,
        "rs_horizons": [5, 10, 20, 60],
        "train_start": "2008-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2026-08-24",
    },
    "exp_e_daily_rotate": {
        "title": "E 组: 1日标签 + 日频满仓轮动 (1日标签 + 17年 + 每天满仓Top1最高分)",
        "label_horizon": 1,
        "strategy_class": "DayRotateStrategy",
        "train_start": "2008-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2026-08-24",
    },
    "exp_f_double_ensemble": {
        "title": "F 组: DoubleEnsemble 双重集成 (5日标签 + 17年 + DEnsembleModel)",
        "label_horizon": 5,
        "model": {
            "class": "DEnsembleModel",
            "module_path": "qlib.contrib.model.double_ensemble",
            "kwargs": {
                "loss": "mse",                  # 目标损失函数
                "base_model": "gbm",            # 基模型 = LightGBM
                "num_models": 6,                # 6 个子模型集成
                "enable_sr": True,              # 样本重加权 (Sample Reweighting)
                "enable_fs": True,              # 特征选择 (Feature Selection)
                "alpha1": 1.0,
                "alpha2": 1.0,
                "decay": 0.5,               # 样本重加权衰减系数 (官方 benchmark 配置)
                "bins_sr": 10,
                "bins_fs": 5,
                "epochs": 100,
                "early_stopping_rounds": 50,
                # 以下直接平铺传给 LightGBM 的超参 (与 B 组一致, 只隔离模型变量)
                "learning_rate": 0.0421,
                "max_depth": 8,
                "num_leaves": 210,
                "subsample": 0.8789,
                "colsample_bytree": 0.8879,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "seed": 42,
                "bagging_seed": 42,
                "feature_fraction_seed": 42,
                "num_threads": 20,
            },
        },
        "train_start": "2008-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2026-08-24",
    },
}


def _to_clean_series(obj) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].dropna()
    elif isinstance(obj, pd.Series):
        return obj.dropna()
    return pd.Series(obj).dropna()


def run_single_experiment(exp_key: str, exp_info: dict, exp_paths: ExperimentPaths) -> dict:
    title = exp_info["title"]
    l_h = exp_info["label_horizon"]
    tr_s = exp_info["train_start"]
    tr_e = exp_info["train_end"]
    va_s = exp_info["valid_start"]
    va_e = exp_info["valid_end"]
    te_s = exp_info["test_start"]
    te_e = exp_info["test_end"]

    exp_dir = exp_paths.get_exp_dir(exp_key)
    dashboard_path = str(exp_dir / "backtest_dashboard.html")
    delivery_slip_path = str(exp_dir / "daily_delivery_slip.csv")

    print("\n" + "=" * 90)
    print(f" 🚀 正在运行实验: 【{title}】 (Key: {exp_key})")
    print(f"    - 标签周期: {l_h} 日收益")
    print(f"    - 训练区间: {tr_s} 至 {tr_e}")
    print(f"    - 验证区间: {va_s} 至 {va_e}")
    print(f"    - 回测区间: {te_s} 至 {te_e}")
    print(f"    - 输出目录: {exp_dir}")
    print("=" * 90)

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(exp_info)

    # 1. 获取共享缓存动态切片数据集
    segments = {
        "train": (tr_s, tr_e),
        "valid": (va_s, va_e),
        "test":  (te_s, te_e),
    }
    dataset = get_cached_dataset(
        market=cfg["market"],
        segments=segments,
        cache_dir=exp_paths.cache_dir,
        force_recompute=cfg.get("force_recompute_cache", False),
        label_horizon=l_h,
        rs_horizons=exp_info.get("rs_horizons", ()),
        bench_symbol=cfg.get("benchmark", "SH000905"),
    )

    # 2. 训练最新模型
    model_cfg = exp_info.get("model", BASE_CONFIG["model"])
    print(f"\n[2/5] 正在训练模型 {model_cfg['class']} ({tr_s} ~ {tr_e})...")
    model = init_instance_by_config(model_cfg)
    with R.start(experiment_name=f"train_{exp_key}"):
        model.fit(dataset)
        R.save_objects(trained_model=model)
    print("       ✅ 模型训练完成！")

    # 3. 配置回测
    strategy_cls = DayRotateStrategy if cfg.get("strategy_class") == "DayRotateStrategy" else AffordableTopkDropoutStrategy
    if cfg.get("strategy_class") == "DayRotateStrategy":
        strategy_kwargs = {
            "model": model,
            "dataset": dataset,
            "topk": cfg["topk_stocks"],
            "risk_degree": cfg["risk_degree"],
            "trade_unit": cfg["trade_unit"],
            "only_main_board": cfg["only_main_board"],
        }
    else:
        strategy_kwargs = {
            "model": model,
            "dataset": dataset,
            "topk": cfg["topk_stocks"],
            "n_drop": cfg["n_drop_stocks"],
            "risk_degree": cfg["risk_degree"],
            "trade_unit": cfg["trade_unit"],
            "rebalance_days": cfg["rebalance_days"],
            "only_main_board": cfg["only_main_board"],
            "stop_loss_rate": cfg["stop_loss_rate"],
            "trailing_stop_trigger": cfg["trailing_stop_trigger"],
            "trailing_stop_rate": cfg["trailing_stop_rate"],
            "max_holding_days": cfg["max_holding_days"],
            "drop_rank_threshold": cfg["drop_rank_threshold"],
            "stock_uptrend_filter": cfg["stock_uptrend_filter"],
            "market_timing_mode": cfg["market_timing_mode"],
            "benchmark_symbol": cfg["benchmark"],
            "benchmark_ma": cfg["benchmark_ma"],
            "account_drawdown_limit": cfg["account_drawdown_limit"],
        }

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": strategy_cls,
            "kwargs": strategy_kwargs,
        },
        "backtest": {
            "start_time": te_s,
            "end_time": te_e,
            "account": cfg["account_cash"],
            "benchmark": cfg["benchmark"],
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": cfg["limit_threshold"],
                "deal_price": cfg["deal_price"],
                "open_cost": cfg["open_cost"],
                "close_cost": cfg["close_cost"],
                "min_cost": cfg["min_cost"],
                "trade_unit": cfg["trade_unit"],
            },
        },
    }

    # 4. 执行回测
    print(f"\n[4/5] 正在执行回测并分析信号...")
    with R.start(experiment_name=f"workflow_{exp_key}"):
        recorder = R.get_recorder()

        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        sar.generate()

        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        ic_df = recorder.load_object("sig_analysis/ic.pkl")
        ric_df = recorder.load_object("sig_analysis/ric.pkl")

        generate_all_in_one_dashboard(
            report_df=report_df,
            positions=positions,
            save_path=dashboard_path,
            benchmark=cfg["benchmark"],
            target_risk_degree=cfg["risk_degree"],
        )
        generate_daily_delivery_slip(positions, save_csv_path=delivery_slip_path)

    # 5. 提取汇总指标
    net_ret = report_df["return"] - report_df["cost"]
    nav = (1.0 + net_ret).cumprod()
    bench_nav = (1.0 + report_df["bench"]).cumprod()

    total_days = len(report_df)
    years = total_days / 252.0 if total_days > 0 else 1.0
    tot_ret = float((nav.iloc[-1] - 1.0) * 100)
    bench_ret = float((bench_nav.iloc[-1] - 1.0) * 100)
    excess_ret = float(tot_ret - bench_ret)

    ann_ret = float((nav.iloc[-1] ** (1.0 / years) - 1.0) * 100) if nav.iloc[-1] > 0 else 0.0
    ann_vol = float(net_ret.std() * np.sqrt(252) * 100)

    rf_daily = 0.02 / 252.0
    sharpe = float(((net_ret - rf_daily).mean() / net_ret.std() * np.sqrt(252))) if net_ret.std() > 0 else 0.0

    mdd_series = (nav / nav.cummax() - 1.0) * 100
    max_mdd = float(abs(mdd_series.min()))
    calmar = float((ann_ret / max_mdd)) if max_mdd > 0 else 0.0

    pos_days = int((net_ret > 0).sum())
    valid_days = int((net_ret != 0).sum())
    win_rate = float(pos_days / valid_days * 100) if valid_days > 0 else 0.0

    ic_series = _to_clean_series(ic_df)
    ric_series = _to_clean_series(ric_df)

    mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
    std_ic = float(ic_series.std()) if len(ic_series) > 0 else 0.0
    icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
    ic_pos_ratio = float((ic_series > 0).mean() * 100) if len(ic_series) > 0 else 0.0

    mean_ric = float(ric_series.mean()) if len(ric_series) > 0 else 0.0
    std_ric = float(ric_series.std()) if len(ric_series) > 0 else 0.0
    ricir = float(mean_ric / std_ric) if std_ric > 0 else 0.0
    ric_pos_ratio = float((ric_series > 0).mean() * 100) if len(ric_series) > 0 else 0.0

    slip_df = pd.read_csv(delivery_slip_path) if Path(delivery_slip_path).exists() else pd.DataFrame()
    n_buys = int((slip_df["操作方向"] == "【买入开仓】").sum()) if not slip_df.empty else 0
    n_sells = int((slip_df["操作方向"] == "【卖出平仓】").sum()) if not slip_df.empty else 0
    total_cost = float(slip_df["交易税费 (¥)"].sum()) if ("交易税费 (¥)" in slip_df.columns) else 0.0

    metrics = {
        "key": exp_key,
        "title": title,
        "label_horizon": f"{l_h} 日",
        "train_range": f"{tr_s[:4]}~{tr_e[:4]}",
        "test_range": f"{te_s}~{te_e}",
        "mean_ic": mean_ic,
        "icir": icir,
        "ic_pos_ratio": ic_pos_ratio,
        "mean_ric": mean_ric,
        "ricir": ricir,
        "ric_pos_ratio": ric_pos_ratio,
        "tot_ret": tot_ret,
        "bench_ret": bench_ret,
        "excess_ret": excess_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "max_mdd": max_mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "total_cost": total_cost,
        "dashboard_path": dashboard_path,
        "delivery_slip_path": delivery_slip_path,
    }

    # 结构化保存实验元数据与指标
    save_experiment_artifacts(exp_key=exp_key, metrics=metrics, config=cfg, exp_paths=exp_paths)

    print(f"       ✅ 实验完成！累计收益: {tot_ret:+.2f}% | 最大回撤: -{max_mdd:.2f}% | IC均值: {mean_ic:.4f} | ICIR: {icir:.2f}")
    return metrics


def generate_comparison_reports(results: list, exp_paths: ExperimentPaths) -> str:
    """生成 Markdown 与 CSV 对比报告"""
    if not results:
        return "⚠️ 没有可供生成的实验结果。"

    report_md = "# 🔬 模型层对照实验多维横向评测报告\n\n"
    report_md += f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_md += "**基准指数**: 中证500 (SH000905) | **初始本金**: ¥20,000 | **持股数**: Top 1\n\n"

    report_md += "## 一、预测模型信号质量 (IC / Rank IC 纯信号层对比)\n\n"
    report_md += "| 实验组 | 预测目标 (标签) | 训练数据区间 | IC 均值 | ICIR | IC>0占比 | Rank IC 均值 | Rank ICIR | Rank IC>0占比 |\n"
    report_md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in results:
        t = r.get("title", r.get("key", "Unknown"))
        lh = r.get("label_horizon", "-")
        tr = r.get("train_range", "-")
        mic = r.get("mean_ic", 0.0)
        icir = r.get("icir", 0.0)
        ic_pos = r.get("ic_pos_ratio", 0.0)
        mric = r.get("mean_ric", 0.0)
        ricir = r.get("ricir", 0.0)
        ric_pos = r.get("ric_pos_ratio", 0.0)
        report_md += f"| **{t}** | {lh} | {tr} | **{mic:.4f}** | {icir:.2f} | {ic_pos:.1f}% | **{mric:.4f}** | {ricir:.2f} | {ric_pos:.1f}% |\n"

    report_md += "\n## 二、投资组合回测绩效 (实盘模拟层对比)\n\n"
    report_md += "| 实验组 | 累计净收益 | 超额收益 (vs基准) | 年化收益 | 最大回撤 | 夏普比率 | 卡玛比率 | 日胜率 | 买卖次数 | 累计交易费 |\n"
    report_md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in results:
        t = r.get("title", r.get("key", "Unknown"))
        tot = r.get("tot_ret", 0.0)
        exc = r.get("excess_ret", 0.0)
        ann = r.get("ann_ret", 0.0)
        mdd = r.get("max_mdd", 0.0)
        sh = r.get("sharpe", 0.0)
        cal = r.get("calmar", 0.0)
        win = r.get("win_rate", 0.0)
        nb = r.get("n_buys", 0)
        ns = r.get("n_sells", 0)
        cost = r.get("total_cost", 0.0)
        report_md += f"| **{t}** | **{tot:+.2f}%** | {exc:+.2f}% | {ann:.2f}% | **-{mdd:.2f}%** | {sh:.2f} | {cal:.2f} | {win:.1f}% | {nb}买/{ns}卖 | ¥{cost:.2f} |\n"

    report_md += "\n## 三、产物文件路径汇总\n\n"
    for r in results:
        k = r.get("key", r.get("exp_key", "unknown"))
        t = r.get("title", k)
        report_md += f"### {t}\n"
        report_md += f"- 🌐 **HTML 大屏看板**: `{r.get('dashboard_path', 'N/A')}`\n"
        report_md += f"- 📑 **交割单 CSV**: `{r.get('delivery_slip_path', 'N/A')}`\n"
        report_md += f"- 📊 **指标 JSON**: `{exp_paths.get_exp_file(k, 'metrics.json')}`\n\n"


    # 保存文件
    reports_dir = exp_paths.reports_dir
    comp_md_file = reports_dir / "experiment_comparison.md"
    comp_csv_file = reports_dir / "experiment_comparison.csv"

    with open(comp_md_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    df_res = pd.DataFrame(results)
    df_res.to_csv(comp_csv_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(" 🏁 全部实验完成！对比汇总报告已保存至:")
    print(f"    - Markdown: {comp_md_file}")
    print(f"    - CSV:      {comp_csv_file}")
    print("=" * 90)
    return report_md


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib 策略多实验并行/批量评测运行引擎")
    parser.add_argument("pos_experiments", nargs="*", help="要运行的实验键名列表 (如 exp_a_baseline exp_b_label5d)")
    parser.add_argument("-e", "--experiments", nargs="+", default=None, help="实验键名列表")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="自定义输出根目录 (默认 strategies/outputs)")
    parser.add_argument("-l", "--list", action="store_true", help="列出所有可用的实验列表与当前运行状态")
    parser.add_argument("--report_only", action="store_true", help="仅从已有 metrics.json 重新生成横向对比报告，不重新训练")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_paths = get_paths(args.output_dir)

    if args.list:
        print("\n📋 可用实验配置矩阵:")
        print("-" * 75)
        for k, v in EXPERIMENTS.items():
            metrics = load_experiment_metrics(k, exp_paths)
            status = f"✅ 已完成 (累计收益 {metrics.get('tot_ret', 0):+.2f}%)" if metrics else "⏳ 未运行"
            print(f"  [{k:22s}] {v['title']:<38s} -> {status}")
        print("-" * 75)
        return

    if args.report_only:
        print("\n📊 正在从已有 metrics.json 汇编多实验对比报告...")
        all_metrics = list_all_experiments(exp_paths, only_completed=True)
        if not all_metrics:
            print("⚠️ 未找到任何已完成的实验 metrics.json。")
            return
        report_md = generate_comparison_reports(all_metrics, exp_paths)
        print("\n" + report_md)
        return


    # 确定要运行的实验列表
    target_keys = []
    if args.experiments:
        target_keys.extend(args.experiments)
    if args.pos_experiments:
        target_keys.extend(args.pos_experiments)
    if not target_keys:
        target_keys = list(EXPERIMENTS.keys())

    # 过滤有效实验
    valid_keys = [k for k in target_keys if k in EXPERIMENTS]
    invalid_keys = [k for k in target_keys if k not in EXPERIMENTS]
    if invalid_keys:
        print(f"⚠️ 忽略未知实验: {invalid_keys}")

    if not valid_keys:
        print(f"❌ 没有有效的待执行实验。可用实验列表: {list(EXPERIMENTS.keys())}")
        return

    print("=" * 90)
    print(f" 🌟 启动模型层对照实验 (共 {len(valid_keys)} 组)")
    print(f"    - 输出根目录: {exp_paths.root}")
    print(f"    - 共享缓存目录: {exp_paths.cache_dir}")
    print("=" * 90)

    # 初始化 Qlib 行情数据环境
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    results = []
    for exp_key in valid_keys:
        res = run_single_experiment(exp_key, EXPERIMENTS[exp_key], exp_paths)
        results.append(res)

    # 生成横向报告
    report_md = generate_comparison_reports(results, exp_paths)
    print("\n" + report_md)


if __name__ == "__main__":
    main()
