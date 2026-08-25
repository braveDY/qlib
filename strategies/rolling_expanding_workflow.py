#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
========================================================================================
Qlib 步进前向滚动再训练回测 (模式一：从 2008 年累积扩张到上月 / Expanding Window)
========================================================================================

【策略与训练模式】：
1. 滚动再训练频率：【按月度滚动 (Monthly Rolling)】
2. 训练窗口模式：【累积扩张窗口 (Expanding Window)】
   - 训练集起始时间固定为 2008-01-01。
   - 每个月回测开始前，把【截至上个月底】产生的所有最新历史数据全部吸收进模型重新训练。
   - 紧贴式切片：
     * 训练集: 2008-01-01 至 上上个月底
     * 验证集: 刚结束的上一个月 (紧挨着预测月，用于早停)
     * 测试集: 目标预测月份 (即当前回测月份)
3. 选股与实盘交易规则：
   - 自适应购买力选股 (Affordable Top-K)：按模型预测打分降序排列，买不起 1 手（100股）的高价股自动顺延跳过，选出买得起 1 手的 Top 3 股票。
   - 账户资金：20,000 元本金。
   - 交易规则：强制 100 股一手，万一免五佣金。
4. 可视化报告：自动输出交互式 HTML 曲线与高清 PNG 净值图表。
========================================================================================
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.report import analysis_position
from qlib.backtest.decision import TradeDecisionWO, Order, OrderDir
from qlib.backtest.position import Position
from qlib.tests.data import GetData
from components.paths import paths


# ======================================================================================
# 1. 自定义自适应购买力 Top-K 选股策略类
# ======================================================================================
class AffordableTopkDropoutStrategyWithPred(TopkDropoutStrategy):
    """自适应购买力顺延选股策略（支持直接传入滚动拼接后的 pred_df/pred_series）"""
    def __init__(self, pred_df, topk=3, n_drop=3, risk_degree=0.95, only_main_board=True, **kwargs):
        super().__init__(topk=topk, n_drop=n_drop, risk_degree=risk_degree, **kwargs)
        self.only_main_board = only_main_board
        super().__init__(topk=topk, n_drop=n_drop, risk_degree=risk_degree, **kwargs)
        if isinstance(pred_df, pd.DataFrame):
            self.pred_series = pred_df.iloc[:, 0]
        else:
            self.pred_series = pred_df

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        try:
            pred_score = self.pred_series.loc[pred_start_time:pred_end_time]
            if isinstance(pred_score.index, pd.MultiIndex):
                pred_score = pred_score.droplevel("datetime")
        except KeyError:
            return TradeDecisionWO([], self)

        if pred_score is None or len(pred_score) == 0:
            return TradeDecisionWO([], self)

        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()

        # 1. 卖出逻辑
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        sell_candidates = list(last[-self.n_drop:]) if len(last) > 0 else []
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL
            ):
                continue
            if code in sell_candidates:
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code, amount=sell_amount,
                    start_time=trade_start_time, end_time=trade_end_time,
                    direction=Order.SELL
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, _ = self.trade_exchange.deal_order(sell_order, position=current_temp)
                    cash += trade_val - trade_cost

        # 2. 买入逻辑：精确按真实股价顺延跳过买不起 1 手 (100股) 的高价股
        holding_stocks = set(current_stock_list) - set(sell_candidates)
        needed_count = self.topk - len(holding_stocks)
        if needed_count > 0 and cash > 0:
            budget_per_stock = (cash * self.risk_degree) / needed_count
            candidate_stocks = pred_score[~pred_score.index.isin(holding_stocks)].sort_values(ascending=False).index
            
            for code in candidate_stocks:
                if getattr(self, 'only_main_board', True) and (code.startswith('SZ30') or code.startswith('SH688') or code.startswith('BJ')):
                    continue
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
                ):
                    continue
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
                )
                if buy_price is None or np.isnan(buy_price) or buy_price <= 0:
                    continue
                
                raw_amount = budget_per_stock / buy_price
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(raw_amount, factor)
                
                if buy_amount > 0:
                    buy_order = Order(
                        stock_id=code, amount=buy_amount,
                        start_time=trade_start_time, end_time=trade_end_time,
                        direction=Order.BUY
                    )
                    buy_order_list.append(buy_order)
                    if len(buy_order_list) >= needed_count:
                        break

        return TradeDecisionWO(sell_order_list + buy_order_list, self)


# ======================================================================================
# 2. 静态曲线绘制辅助函数 (Matplotlib PNG 输出)
# ======================================================================================

def analyze_and_save_holdings(positions: dict, save_csv_path: str = "strategies/outputs/holding_analysis.csv"):
    from collections import defaultdict
    stock_stats = defaultdict(lambda: {
        "total_holding_days": 0, "clear_count": 0, "episodes": [],
        "current_episode_days": 0, "first_buy_date": None, "last_sell_date": None,
    })
    dates = sorted(positions.keys())
    if not dates: return pd.DataFrame()
    prev_held_stocks = set()
    for d in dates:
        pos = positions[d]
        cur_held_stocks = set(pos.get_stock_list() if hasattr(pos, "get_stock_list") else [k for k in pos.keys() if k not in ("cash", "now_account_value")])
        for s in cur_held_stocks:
            stats = stock_stats[s]
            stats["total_holding_days"] += 1
            stats["current_episode_days"] += 1
            if stats["first_buy_date"] is None: stats["first_buy_date"] = d
        cleared_stocks = prev_held_stocks - cur_held_stocks
        for s in cleared_stocks:
            stats = stock_stats[s]
            stats["clear_count"] += 1
            stats["episodes"].append(stats["current_episode_days"])
            stats["current_episode_days"] = 0
            stats["last_sell_date"] = d
        prev_held_stocks = cur_held_stocks
    for s in prev_held_stocks:
        stats = stock_stats[s]
        if stats["current_episode_days"] > 0: stats["episodes"].append(stats["current_episode_days"])
    records = []
    for s, stats in stock_stats.items():
        episodes = stats["episodes"] if stats["episodes"] else [stats["total_holding_days"]]
        avg_days = sum(episodes) / len(episodes) if episodes else 0
        max_days = max(episodes) if episodes else 0
        records.append({
            "股票代码": s, "累计持有天数": stats["total_holding_days"], "清仓次数": stats["clear_count"],
            "平均每次持有天数": round(avg_days, 1), "单次最长持有天数": max_days,
            "首次买入日期": stats["first_buy_date"].strftime("%Y-%m-%d") if stats["first_buy_date"] else "-",
            "最近清仓日期": stats["last_sell_date"].strftime("%Y-%m-%d") if stats["last_sell_date"] else "当前持仓中",
        })
    df_holdings = pd.DataFrame(records).sort_values(by="累计持有天数", ascending=False).reset_index(drop=True)
    Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_holdings.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n" + "=" * 90)
    print(f" 📊 投资组合持仓行为深度统计 (回测期间共持仓过 {len(df_holdings)} 只股票)")
    print("=" * 90)
    print(df_holdings.head(20).to_string(index=False))
    print(f"\n ✅ 完整持仓明细报表已保存至: {save_csv_path}")
    return df_holdings

def plot_and_save_static_report(report_df: pd.DataFrame, save_path: str, title_suffix: str = ""):
    """绘制并保存高清收益率与风险曲线图 (PNG 格式)"""
    df = report_df.copy()
    cum_bench = df["bench"].cumsum()
    cum_return_wo_cost = df["return"].cumsum()
    cum_return_w_cost = (df["return"] - df["cost"]).cumsum()
    cum_ex_return_w_cost = (df["return"] - df["cost"] - df["bench"]).cumsum()
    mdd_w_cost = cum_return_w_cost - cum_return_w_cost.cummax()
    turnover = df["turnover"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.5, 1.5]})
    
    axes[0].plot(df.index, cum_return_w_cost, label="Rolling Strategy (With Cost)", color="#e74c3c", linewidth=2)
    axes[0].plot(df.index, cum_return_wo_cost, label="Rolling Strategy (Without Cost)", color="#f39c12", linestyle="--", alpha=0.8)
    axes[0].plot(df.index, cum_bench, label="Benchmark (SH000300)", color="#3498db", linewidth=1.5)
    axes[0].set_title(f"1. Cumulative Return vs Benchmark ({title_suffix})", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left")

    axes[1].plot(df.index, cum_ex_return_w_cost, label="Excess Return (Alpha with Cost)", color="#2ecc71", linewidth=2)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("2. Cumulative Excess Return (Alpha)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Excess Return")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left")

    axes[2].fill_between(df.index, mdd_w_cost, 0, color="#e74c3c", alpha=0.35, label="Drawdown")
    axes[2].plot(df.index, mdd_w_cost, color="#c0392b", linewidth=1)
    axes[2].set_title("3. Portfolio Max Drawdown", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Drawdown")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="lower left")

    axes[3].bar(df.index, turnover, color="#9b59b6", alpha=0.7, width=1.5, label="Daily Turnover")
    axes[3].set_title("4. Daily Turnover Rate", fontsize=13, fontweight="bold")
    axes[3].set_ylabel("Turnover")
    axes[3].set_xlabel("Trading Date")
    axes[3].grid(True, linestyle=":", alpha=0.6)
    axes[3].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"       📊 高清静态收益曲线已保存至: {save_path}")


# ======================================================================================
# 3. 主执行流程 (Expanding Window)
# ======================================================================================
def main():
    print("=" * 85)
    print(" 🚀 启动 Qlib 按月度滚动再训练策略 (模式一：从 2008 年累积扩张窗口 Expanding Window)")
    print("=" * 85)

    # 1. 初始化 Qlib 行情数据环境
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    # 2. 构建基础特征工程
    print("\n[1/4] 正在加载特征工程与数据集 (Alpha158)...")
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2026-08-24",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-01", "2016-11-30"),
                "valid": ("2016-12-01", "2016-12-31"),
                "test": ("2017-01-01", "2017-01-31"),
            },
        },
    }
    dataset = init_instance_by_config(dataset_config)

    # 3. 生成按月度切片的训练计划 (2017-01 到 2020-08，共 44 个月份)
    month_starts = pd.date_range("2017-01-01", "2026-08-01", freq="MS")
    all_preds = []

    print(f"\n[2/4] 开始执行月度步进再训练 (共 {len(month_starts)} 个月):")
    print(f"       - 训练集: 2008-01-01 起始，累积扩张到预测月前 1 个月")
    print(f"       - 验证集: 预测月前紧挨着的 1 个月最新数据 (用于模型早停 Early Stopping)")
    print(f"       - 测试集: 目标预测月份 (Out-of-Sample 样本外)")
    print("-" * 85)

    for i, m_start in enumerate(month_starts, 1):
        m_end = m_start + pd.offsets.MonthEnd(1)
        test_start_str = m_start.strftime("%Y-%m-%d")
        test_end_str = m_end.strftime("%Y-%m-%d")

        # 紧贴式累积扩张切片设计 (0 间隔利用最新历史数据)
        # 1. 验证集：预测月紧挨着的上一个月
        valid_start = m_start - pd.DateOffset(months=1)
        valid_start_str = valid_start.strftime("%Y-%m-%d")
        valid_end = m_start - pd.Timedelta(days=1)
        valid_end_str = valid_end.strftime("%Y-%m-%d")

        # 2. 训练集：从 2008-01-01 累积至上上个月底
        train_start_str = "2008-01-01"
        train_end = valid_start - pd.Timedelta(days=1)
        train_end_str = train_end.strftime("%Y-%m-%d")

        print(f" [{i:02d}/{len(month_starts)}] 训练: {train_start_str}~{train_end_str} | 验证(上个月): {valid_start_str}~{valid_end_str} | 预测月: {test_start_str}~{test_end_str}")

        # 动态切片并训练模型
        dataset.config(segments={"train": (train_start_str, train_end_str), "valid": (valid_start_str, valid_end_str), "test": (test_start_str, test_end_str)})
        model = init_instance_by_config(GBDT_MODEL_CONFIG)
        model.fit(dataset)
        
        # 预测该月股票得分
        sub_pred = model.predict(dataset, segment="test")
        all_preds.append(sub_pred)

    # 4. 拼接所有月度的样本外预测打分
    final_pred_df = pd.concat(all_preds).sort_index()
    if isinstance(final_pred_df, pd.Series):
        final_pred_df = final_pred_df.to_frame("score")

    print("\n[3/4] 月度滚动预测序列拼接完成！总打分记录数:", len(final_pred_df))

    # 5. 执行自适应购买力策略回测
    account_cash = 20000
    topk_stocks = 3
    n_drop_stocks = 3
    risk_degree = 0.95

    print(f"\n[4/4] 启动自适应购买力策略模拟回测 (2 万元本金，Top 3 选股，强制一手 100 股)...")
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
            "class": AffordableTopkDropoutStrategyWithPred,
            "kwargs": {
                "pred_df": final_pred_df,
                "topk": topk_stocks,
                "n_drop": n_drop_stocks,
                "risk_degree": risk_degree,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2026-08-24",
            "account": account_cash,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0001,
                "close_cost": 0.0001,
                "min_cost": 0,
                "trade_unit": 100,
            },
        },
    }

    with R.start(experiment_name="rolling_expanding_workflow"):
        recorder = R.get_recorder()
        ba_rid = recorder.id
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        # 加载回测产物并生成可视化图表
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        html_path = str(paths.reports_dir / "rolling_expanding_report.html")
        figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
        if figs and len(figs) > 0:
            figs[0].write_html(html_path)
            print(f"       🌐 交互式 HTML 收益曲线已保存至: {html_path}")

        png_path = str(paths.reports_dir / "rolling_expanding_report.png")
        plot_and_save_static_report(report_normal_df, save_path=png_path, title_suffix="Expanding Window")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analyze_and_save_holdings(positions, save_csv_path=str(paths.reports_dir / "rolling_expanding_holdings.csv"))

    print("\n" + "=" * 85)
    print(" 🎉 累积扩张窗口滚动再训练与回测圆满完成！")
    print(f" 实验记录 ID (Recorder ID): {ba_rid}")
    print(f" 交互式 HTML 报告: {html_path}")
    print(f" 静态 PNG 图片:    {png_path}")
    print("=" * 85)



if __name__ == "__main__":
    main()
