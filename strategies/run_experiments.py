#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
========================================================================================
Qlib 通用配置驱动实验执行引擎 (Universal Config-Driven Experiment Runner)
========================================================================
设计理念：
- 【引擎与配置完全解耦】：代码中不硬编码任何具体实验参数，全部由 `configs/` 下的 YAML 文件驱动。
- 【多层级执行模式】：
  1. 单实验运行: `python run_experiments.py -c configs/models/exp_f_double_ensemble.yaml`
  2. 专题批量运行: `python run_experiments.py -s configs/models/`
  3. 全量扫描运行: `python run_experiments.py --all`
  4. 查看配置状态: `python run_experiments.py --list`
  5. 快速汇编报告: `python run_experiments.py --report_only`
- 【自动化生命周期】：特征缓存全局共享，指标结构化落盘 (metrics.json)，自动生成横向评测报告。
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
STRATEGIES_DIR = Path(__file__).resolve().parent
sys.path.append(str(STRATEGIES_DIR))

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
    load_strategy_config,
    list_all_strategy_configs,
)


def _to_clean_series(obj) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].dropna()
    elif isinstance(obj, pd.Series):
        return obj.dropna()
    return pd.Series(obj).dropna()


def run_single_experiment(cfg: dict, exp_paths: ExperimentPaths) -> dict:
    """
    根据给定的完整配置字典执行单次模型训练与实盘回测
    """
    exp_key = cfg.get("key", "unnamed_exp")
    title = cfg.get("title", exp_key)
    l_h = cfg.get("label_horizon", 1)
    tr_s = cfg.get("train_start", "2008-01-01")
    tr_e = cfg.get("train_end", "2024-12-31")
    va_s = cfg.get("valid_start", "2025-01-01")
    va_e = cfg.get("valid_end", "2025-06-30")
    te_s = cfg.get("test_start", "2025-07-01")
    te_e = cfg.get("test_end", "2026-08-24")

    exp_dir = exp_paths.get_exp_dir(exp_key)
    dashboard_path = str(exp_dir / "backtest_dashboard.html")
    delivery_slip_path = str(exp_dir / "daily_delivery_slip.csv")

    print("\n" + "=" * 90)
    print(f" 🚀 正在执行实验: 【{title}】 (Key: {exp_key})")
    print(f"    - 预测目标: {l_h} 日收益标签")
    print(f"    - 训练区间: {tr_s} 至 {tr_e}")
    print(f"    - 验证区间: {va_s} 至 {va_e}")
    print(f"    - 回测区间: {te_s} 至 {te_e}")
    print(f"    - 产物目录: {exp_dir}")
    print("=" * 90)

    # 1. 动态切片并命中共享特征缓存
    segments = {
        "train": (tr_s, tr_e),
        "valid": (va_s, va_e),
        "test":  (te_s, te_e),
    }
    dataset = get_cached_dataset(
        market=cfg.get("market", "csi500"),
        segments=segments,
        cache_dir=exp_paths.cache_dir,
        force_recompute=cfg.get("force_recompute_cache", False),
        label_horizon=l_h,
        rs_horizons=cfg.get("rs_horizons", ()),
        bench_symbol=cfg.get("benchmark", "SH000905"),
    )

    # 2. 初始化并训练模型
    model_cfg = cfg.get("model")
    if not model_cfg:
        raise ValueError("配置中缺少 'model' 字段定义！")

    print(f"\n[2/5] 正在训练模型 {model_cfg.get('class', 'Model')} ({tr_s} ~ {tr_e})...")
    model = init_instance_by_config(model_cfg)
    with R.start(experiment_name=f"train_{exp_key}"):
        model.fit(dataset)
        R.save_objects(trained_model=model)
    print("       ✅ 模型训练完成！")

    # 3. 构建交易策略与回测执行器配置
    strategy_cls_name = cfg.get("strategy_class", "AffordableTopkDropoutStrategy")
    if strategy_cls_name == "DayRotateStrategy":
        strategy_cls = DayRotateStrategy
        strategy_kwargs = {
            "model": model,
            "dataset": dataset,
            "topk": cfg.get("topk_stocks", 1),
            "risk_degree": cfg.get("risk_degree", 0.95),
            "trade_unit": cfg.get("trade_unit", 100),
            "only_main_board": cfg.get("only_main_board", True),
        }
    else:
        strategy_cls = AffordableTopkDropoutStrategy
        strategy_kwargs = {
            "model": model,
            "dataset": dataset,
            "topk": cfg.get("topk_stocks", 1),
            "n_drop": cfg.get("n_drop_stocks", 1),
            "risk_degree": cfg.get("risk_degree", 0.95),
            "trade_unit": cfg.get("trade_unit", 100),
            "rebalance_days": cfg.get("rebalance_days", 1),
            "only_main_board": cfg.get("only_main_board", True),
            "stop_loss_rate": cfg.get("stop_loss_rate", 0.035),
            "trailing_stop_trigger": cfg.get("trailing_stop_trigger", 0.05),
            "trailing_stop_rate": cfg.get("trailing_stop_rate", 0.025),
            "max_holding_days": cfg.get("max_holding_days", 10),
            "drop_rank_threshold": cfg.get("drop_rank_threshold", 30),
            "stock_uptrend_filter": cfg.get("stock_uptrend_filter", True),
            "market_timing_mode": cfg.get("market_timing_mode", "half_position_timing"),
            "benchmark_symbol": cfg.get("benchmark", "SH000905"),
            "benchmark_ma": cfg.get("benchmark_ma", 20),
            "account_drawdown_limit": cfg.get("account_drawdown_limit", 0.08),
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
            "account": cfg.get("account_cash", 20000.0),
            "benchmark": cfg.get("benchmark", "SH000905"),
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": cfg.get("limit_threshold", 0.095),
                "deal_price": cfg.get("deal_price", "close"),
                "open_cost": cfg.get("open_cost", 0.0001),
                "close_cost": cfg.get("close_cost", 0.0001),
                "min_cost": cfg.get("min_cost", 5.0),
                "trade_unit": cfg.get("trade_unit", 100),
            },
        },
    }

    # 4. 执行实盘模拟回测
    print(f"\n[4/5] 正在执行样本外实盘模拟并分析信号...")
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
            benchmark=cfg.get("benchmark", "SH000905"),
            target_risk_degree=cfg.get("risk_degree", 0.95),
        )
        generate_daily_delivery_slip(positions, save_csv_path=delivery_slip_path)

    # 5. 提取并计算全维度复利与风险指标
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

    # 结构化持久化快照
    save_experiment_artifacts(exp_key=exp_key, metrics=metrics, config=cfg, exp_paths=exp_paths)

    print(f"       ✅ 实验完成！累计收益: {tot_ret:+.2f}% | 最大回撤: -{max_mdd:.2f}% | 卡玛比率: {calmar:.2f} | ICIR: {icir:.2f}")
    return metrics


def generate_comparison_reports(results: list, exp_paths: ExperimentPaths, report_suffix: str = "") -> str:
    """生成 Markdown 与 CSV 多维横向对比报告"""
    if not results:
        return "⚠️ 没有可供展示的实验结果。"

    report_md = "# 🔬 量化策略多维横向评测报告\n\n"
    report_md += f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_md += "**测试区间**: 2025-07-01 ~ 2026-08-24 | **对比基准**: 中证500 (SH000905) | **本金**: ¥20,000 | **持仓**: Top 1\n\n"

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

    filename_base = f"experiment_comparison{('_' + report_suffix) if report_suffix else ''}"
    comp_md_file = exp_paths.reports_dir / f"{filename_base}.md"
    comp_csv_file = exp_paths.reports_dir / f"{filename_base}.csv"

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
    parser = argparse.ArgumentParser(description="Qlib 扁平配置驱动策略实验执行引擎")
    parser.add_argument("pos_targets", nargs="*", help="要运行的配置文件路径或策略键名 (例如 double_ensemble)")
    parser.add_argument("-c", "--config", type=str, default=None, help="单个策略 YAML 配置文件路径")
    parser.add_argument("-a", "--all", action="store_true", help="运行 configs/ 下的全部策略配置")
    parser.add_argument("-l", "--list", action="store_true", help="列出所有策略配置文件与完成状态")
    parser.add_argument("--report_only", action="store_true", help="仅从已有 metrics.json 重新汇编多实验对比报告")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="自定义输出根目录")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_paths = get_paths(args.output_dir)

    # 1. 列表模式
    if args.list:
        print("\n📋 发现的策略配置文件清单 (位于 strategies/configs/):")
        print("=" * 80)
        cfgs = list_all_strategy_configs()
        if not cfgs:
            print("⚠️ 未发现任何策略配置文件。")
            return
        for c in cfgs:
            k = c.get("key", "unnamed")
            t = c.get("title", k)
            metrics = load_experiment_metrics(k, exp_paths)
            status = f"✅ 已完成 (净收益 {metrics.get('tot_ret', 0):+.2f}% | 回撤 -{metrics.get('max_mdd', 0):.2f}%)" if metrics else "⏳ 未运行"
            print(f"  • [{k:20s}] {t:<45s} -> {status}")
        print("=" * 80)
        return

    # 2. 仅报告模式
    if args.report_only:
        print("\n📊 正在从已有 metrics.json 汇编多实验对比报告...")
        all_metrics = list_all_experiments(exp_paths, only_completed=True)
        if not all_metrics:
            print("⚠️ 未找到任何已完成的实验 metrics.json。")
            return
        report_md = generate_comparison_reports(all_metrics, exp_paths)
        print("\n" + report_md)
        return

    # 3. 解析待运行的目标配置列表
    configs_to_run = []
    report_suffix = ""
    all_known_cfgs = {c.get("key"): c for c in list_all_strategy_configs()}

    if args.config:
        configs_to_run.append(load_strategy_config(args.config))
        report_suffix = Path(args.config).stem
    elif args.pos_targets:
        for t in args.pos_targets:
            p = Path(t)
            if p.is_file():
                configs_to_run.append(load_strategy_config(p))
            elif t in all_known_cfgs:
                configs_to_run.append(all_known_cfgs[t])
            else:
                # 尝试补上 .yaml 后缀查找
                cand = Path(__file__).resolve().parent / "configs" / f"{t}.yaml"
                if cand.exists():
                    configs_to_run.append(load_strategy_config(cand))
                else:
                    print(f"⚠️ 未找到匹配的策略配置: {t}")
    else:
        # 默认或 --all: 运行全量策略
        configs_to_run.extend(list_all_strategy_configs())
        report_suffix = "all"

    if not configs_to_run:
        print("❌ 未找到任何待执行的有效策略配置。可通过 --list 查看所有可用配置。")
        return

    print("=" * 90)
    print(f" 🌟 启动单文件配置驱动实验引擎 (共 {len(configs_to_run)} 个策略待执行)")
    print(f"    - 输出根目录: {exp_paths.root}")
    print(f"    - 共享缓存目录: {exp_paths.cache_dir}")
    print("=" * 90)

    # 4. 初始化 Qlib 行情数据环境
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 5. 循环执行实验
    results = []
    for cfg in configs_to_run:
        res = run_single_experiment(cfg, exp_paths)
        results.append(res)

    # 6. 生成对比报告
    report_md = generate_comparison_reports(results, exp_paths, report_suffix=report_suffix)
    print("\n" + report_md)



if __name__ == "__main__":
    main()
