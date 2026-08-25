#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
========================================================================================
Qlib 自适应购买力选股实盘回测系统 (全量特征缓存与秒级秒开版本)
========================================================================================

【使用指南】：
所有参数（训练时间、验证时间、回测时间、账户本金、选股只数、交易费率、股票池等）均已集中在
文件顶部的 【CONFIG 参数控制面板】 中。
底层已集成【全量特征持久化缓存引擎】，首次运行生成缓存后，后续随意修改时间区间均可 0.3 秒秒开！
========================================================================================
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

# 添加当前目录以支持 components 模块导入
sys.path.append(str(Path(__file__).resolve().parent))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData

# 导入策略核心组件
from components import (
    AffordableTopkDropoutStrategy,
    generate_daily_delivery_slip,
    generate_all_in_one_dashboard,
    get_cached_dataset,
    get_paths,
    paths,
)



# ======================================================================================
# 🎛️ 策略全局参数控制面板 (用户修改区：只需在此处调整，全局自动联动生效)
# ======================================================================================
CONFIG = {
    # ------------------ 1. 时间周期与标签配置 ------------------
    "label_horizon": 1,                 # 预测目标周期 (1: 预测未来 1 日收益; 5: 预测未来 5 日收益)
    "train_start":   "2008-01-01",      # 训练集起始时间 (2008 年至 2024 年底长周期历史训练)
    "train_end":     "2024-12-31",      # 训练集截止时间
    "valid_start":   "2025-01-01",      # 验证集起始时间 (2025 上半年 6 个月验证防过拟合)
    "valid_end":     "2025-06-30",      # 验证集截止时间
    "test_start":    "2025-07-01",      # 样本外测试/回测起始时间 (约 1 年 2 个月实测样本)
    "test_end":      "2026-08-24",      # 样本外测试/回测截止时间 (最新交易日)

    # ------------------ 2. 股票池与基准 ------------------
    "market":        "csi500",          # 股票池：沪深300 (csi300) 或 中证500 (csi500) 或 中证800 (csi800)
    "benchmark":     "SH000905",        # 对比基准：沪深300指数 (SH000300) 或 中证500指数 (SH000905)

    # ------------------ 3. 账户本金与选股 ------------------
    "account_cash":            20000,     # 账户初始本金 (元)
    "topk_stocks":             1,         # 目标持仓股票只数 (固定持有 Top 1 只)
    "risk_degree":             0.95,      # 仓位资金使用率 (0.95 表示最多用 95% 资金)

    # ------------------ 4. 智能风控与动态退出 (让利润奔跑，截断亏损) ------
    "stop_loss_rate":          0.04,      # 硬止损比例 (0.04 表示单票浮亏 -4% 强制止损；0 表示关闭)
    "trailing_stop_trigger":   0.06,      # 移动止盈触发线 (浮盈达到 +6% 开启移动止盈跟踪)
    "trailing_stop_rate":      0.03,      # 移动止盈回撤阈值 (从最高点回撤 3% 立即锁定利润；0 表示关闭)
    "max_holding_days":        10,        # 最长持股天数 (持股 10 天滞涨则主动换股腾挪资金；0 表示不限)
    "drop_rank_threshold":     30,        # 信号退出容忍排名 (只要模型打分在前 30 名且未触发止损，继续坚定持股，绝不每天乱换)
    "stock_uptrend_filter":    True,      # 个股独立多头通道过滤 (仅买入处于 5日/20日均线之上的主升浪股票，拒绝接飞刀)
    "market_timing_mode":      "strict",  # 大盘择时模式 ("strict": 熊市/弱势期果断 100% 空仓避险; "off": 关闭)
    "benchmark_ma":            20,        # 大盘均线周期 (20日线波段过滤)
    "account_drawdown_limit":  0.08,      # 账户高水位回撤保护 (从最高峰值回撤达到 8% 时，自动收缩仓位并收紧止损至 -2.5%，锁定胜利果实)
    "rebalance_days":          1,         # 轮动检测周期 (1=每日监控止盈止损)
    "n_drop_stocks":           1,         # 备用轮换只数

    # ------------------ 5. 交易费率与撮合规则 --------------
    "open_cost":               0.0001,    # 买入佣金费率 (万一: 0.01%)
    "close_cost":              0.0001,    # 卖出佣金费率 (万一: 0.01%)
    "min_cost":                5,         # 最低佣金 (万一免五填 0，不免五填 5)
    "trade_unit":              100,       # 强制买卖单位 (A股一手 100 股)
    "deal_price":              "close",   # 撮合成交价 (收盘价 "close" 或 开盘价 "open")
    "limit_threshold":         0.095,     # 涨跌停限制 (9.5% 不成交)

    # ------------------ 6. 板块交易权限过滤 ------------------
    "only_main_board":         True,      # 是否仅限主板 (True: 只买 60xxxx 和 00xxxx，自动过滤 30xxxx 创业板和 688xxx 科创板)

    # ------------------ 7. 预测模型与算法超参数配置 ---------
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",                  # 目标损失函数：均方误差 (MSE)
            "learning_rate": 0.0421,        # 学习率
            "max_depth": 8,                 # 决策树最大深度
            "num_leaves": 210,              # 最大叶子节点数
            "subsample": 0.8789,            # 样本行抽样比例
            "colsample_bytree": 0.8879,     # 特征列抽样比例
            "lambda_l1": 205.6999,          # L1 正则化系数 (防止过拟合)
            "lambda_l2": 580.9768,          # L2 正则化系数 (防止过拟合)
            "seed": 42,                     # 随机种子 (固定保证每次训练可复现,实验对比才有意义)
            "bagging_seed": 42,             # bagging 抽样种子 (配合 subsample 固定随机性)
            "feature_fraction_seed": 42,    # 特征列抽样种子 (配合 colsample_bytree 固定随机性)
            "num_threads": 20,              # 并行 CPU 线程数
        },
    },

    # ------------------ 8. 缓存与产物输出路径 (设为 None 则自动自适应) --------------
    "cache_dir":             None,                                          # 全量特征持久化缓存目录 (None 则自动使用共享 outputs/cache)
    "output_dir":            None,                                          # 输出根目录 (None 则自动使用 outputs/ 或 QLIB_OUTPUT_DIR)
    "force_recompute_cache": False,                                         # 是否强制重新计算全量特征 (通常设为 False 享受秒级加载)
    "dashboard_html_path":   None,                                          # 全景大屏 (HTML), None 自动使用 outputs/backtest_dashboard.html
    "delivery_slip_csv_path":None,                                          # 每日流水单 (CSV), None 自动使用 outputs/daily_delivery_slip.csv
}
# ======================================================================================


def main():
    exp_paths = get_paths(CONFIG.get("output_dir"))
    cache_dir = CONFIG.get("cache_dir") or exp_paths.cache_dir
    dashboard_path = CONFIG.get("dashboard_html_path") or str(exp_paths.root / "backtest_dashboard.html")
    delivery_slip_path = CONFIG.get("delivery_slip_csv_path") or str(exp_paths.root / "daily_delivery_slip.csv")

    print("=" * 85)
    print(" 🚀 启动 Qlib 自适应购买力选股实盘回测流程 (多维风控与独立主升浪版)")
    print(f"    - 股票池: {CONFIG['market']} | 对比基准: {CONFIG['benchmark']}")
    print(f"    - 训练区间: {CONFIG['train_start']} 至 {CONFIG['train_end']}")
    print(f"    - 验证区间: {CONFIG['valid_start']} 至 {CONFIG['valid_end']}")
    print(f"    - 回测区间: {CONFIG['test_start']} 至 {CONFIG['test_end']}")
    print(f"    - 账户资金: {CONFIG['account_cash']:,} 元 | 目标持仓: Top {CONFIG['topk_stocks']} 只 | 仅限主板: {CONFIG['only_main_board']}")
    print(f"    - 选股风控: 个股双多头通道过滤 ({'已开启: 只买 Close > MA5/20 主升浪' if CONFIG['stock_uptrend_filter'] else '已关闭'})")
    print(f"    - 择时模式: {CONFIG['market_timing_mode']} (弱势期半仓防御, 兼顾逆势独立行情)")
    print(f"    - 账户锁利: 高水位回撤 -{CONFIG['account_drawdown_limit']*100:.1f}% 自动紧缩防守，锁定浮盈果实")
    print(f"    - 输出根目录: {exp_paths.root}")
    print("=" * 85)

    # 1. 初始化 Qlib 行情数据环境
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 2. 从全量缓存加载并秒级切片数据集 (修改任何时间无需重新计算 158 因子)
    segments = {
        "train": (CONFIG["train_start"], CONFIG["train_end"]),
        "valid": (CONFIG["valid_start"], CONFIG["valid_end"]),
        "test":  (CONFIG["test_start"],  CONFIG["test_end"]),
    }
    dataset = get_cached_dataset(
        market=CONFIG["market"],
        segments=segments,
        cache_dir=cache_dir,
        force_recompute=CONFIG["force_recompute_cache"],
        label_horizon=CONFIG.get("label_horizon", 1),
    )


    # 3. 训练最新模型
    print(f"\n[2/5] 正在训练 LightGBM 模型 (使用 {CONFIG['train_start']} ~ {CONFIG['train_end']} 数据)...")
    model = init_instance_by_config(CONFIG["model"])
    with R.start(experiment_name="train_model_panel"):
        model.fit(dataset)
        R.save_objects(trained_model=model)
    print("       ✅ 模型训练完成！")

    # 4. 配置回测环境 (自动读取顶部 CONFIG)
    print(f"\n[3/5] 正在配置投资组合回测参数...")
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
            "class": AffordableTopkDropoutStrategy,
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": CONFIG["topk_stocks"],
                "n_drop": CONFIG["n_drop_stocks"],
                "risk_degree": CONFIG["risk_degree"],
                "trade_unit": CONFIG["trade_unit"],
                "rebalance_days": CONFIG["rebalance_days"],
                "only_main_board": CONFIG["only_main_board"],
                "stop_loss_rate": CONFIG["stop_loss_rate"],
                "trailing_stop_trigger": CONFIG["trailing_stop_trigger"],
                "trailing_stop_rate": CONFIG["trailing_stop_rate"],
                "max_holding_days": CONFIG["max_holding_days"],
                "drop_rank_threshold": CONFIG["drop_rank_threshold"],
                "stock_uptrend_filter": CONFIG["stock_uptrend_filter"],
                "market_timing_mode": CONFIG["market_timing_mode"],
                "benchmark_symbol": CONFIG["benchmark"],
                "benchmark_ma": CONFIG["benchmark_ma"],
                "account_drawdown_limit": CONFIG["account_drawdown_limit"],
            },
        },
        "backtest": {
            "start_time": CONFIG["test_start"],
            "end_time": CONFIG["test_end"],
            "account": CONFIG["account_cash"],
            "benchmark": CONFIG["benchmark"],
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": CONFIG["limit_threshold"],
                "deal_price": CONFIG["deal_price"],
                "open_cost": CONFIG["open_cost"],
                "close_cost": CONFIG["close_cost"],
                "min_cost": CONFIG["min_cost"],
                "trade_unit": CONFIG["trade_unit"],
            },
        },
    }

    # 5. 执行回测并记录实验
    print("\n[4/5] 正在执行自适应购买力策略模拟回测...")
    with R.start(experiment_name="affordable_topk_workflow"):
        recorder = R.get_recorder()
        ba_rid = recorder.id

        # 5.1 生成预测得分 (pred.pkl)
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 5.2 运行因子信号质量分析 (IC / Rank IC)
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 5.3 执行投资组合回测 (port_analysis_1day.pkl)
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        # 6. 加载回测收益报告并生成全景大屏与每日交割单
        print("\n[5/5] 正在生成全合一全景交互看板与每日交割单...")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        
        # 6.1 生成全合一 HTML 大屏
        generate_all_in_one_dashboard(
            report_df=report_normal_df,
            positions=positions,
            save_path=dashboard_path,
            benchmark=CONFIG["benchmark"],
            target_risk_degree=CONFIG["risk_degree"]
        )

        # 6.2 生成按时间顺序的每日真实交易交割流水单 (CSV)
        generate_daily_delivery_slip(positions, save_csv_path=delivery_slip_path)

    print("\n" + "=" * 85)
    print(" 🎉 全流程回测与每日交割单生成圆满完成！")
    print(f" 实验记录 ID (Recorder ID): {ba_rid}")
    print(f" 🌐 全合一全景交互看板 (HTML): {dashboard_path}")
    print(f" 📑 每日真实交易交割单 (CSV):  {delivery_slip_path}")
    print("=" * 85)



if __name__ == "__main__":
    main()
