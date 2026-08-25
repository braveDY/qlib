# 📈 Qlib 策略投研与回测执行系统 (Strategies Framework)

本模块是一个**“一个策略对应一个独立 YAML 配置文件”**的现代量化策略投研系统。通过将策略参数与执行引擎完全解耦，实现了策略与因子的极简定义、特征缓存跨实验复用、多维度横向对比评测以及机构级全景大屏自动生成。

---

## 目录
- [一、核心架构与目录结构](#一核心架构与目录结构)
- [二、快速上手 (Quick Start)](#二快速上手-quick-start)
- [三、配置文件结构详解 (YAML 规范)](#三配置文件结构详解-yaml-规范)
- [四、如何极简新建一个自己的策略？](#四如何极简新建一个自己的策略)
- [五、核心底层组件 (Components)](#五核心底层组件-components)
- [六、产物与报表输出管理 (Outputs)](#六产物与报表输出管理-outputs)

---

## 一、核心架构与目录结构

所有策略配置文件扁平放置在 `configs/` 目录下，**一个策略即一个自包含的 `.yaml` 文件**：

```text
strategies/
├── README.md                   # 📖 本使用说明文档
├── run_experiments.py          # 🌟【唯一通用执行引擎】(CLI 启动入口)
├── configs/                    # 🎛️【策略配置中心】(一个策略 = 一个独立 YAML)
│   ├── baseline.yaml           # 1. 基线策略 (1日标签 + 17年 LightGBM + 完整风控)
│   ├── double_ensemble.yaml    # 2. 双重集成策略 (5日标签 + DoubleEnsemble + 完整风控)
│   ├── label5d.yaml            # 3. 5日平滑标签策略 (5日标签 + LightGBM + 完整风控)
│   ├── rs_momentum.yaml        # 4. 相对强度动量策略 (5日标签 + RS超额因子 + 完整风控)
│   ├── roll3y.yaml             # 5. 近3年短窗口策略 (5日标签 + 2022~2024短数据 + 完整风控)
│   └── daily_rotate.yaml       # 6. 日频满仓轮动策略 (1日标签 + 每天满仓Top1 + 无风控)
├── components/                 # 🧩【底层组件库】
│   ├── config_loader.py        # 策略 YAML 加载与自动缺省兜底
│   ├── dataset.py              # 共享特征持久化缓存与动态时间切片
│   ├── strategy.py             # 自适应购买力选股与多维风控策略类
│   ├── day_rotate_strategy.py  # 日频满仓轮动策略类
│   ├── visualization.py        # 机构级全景交互 HTML 大屏生成器
│   ├── analysis.py             # 每日交易交割流水单生成器
│   └── paths.py                # 标准化路径与产物生命周期管理
└── outputs/                    # 📊【标准化分层输出中心】
    ├── cache/                  # 全局共享特征缓存 (多策略 0.3 秒秒级加载)
    ├── logs/                   # 执行运行日志
    ├── reports/                # 跨策略横向对比报告 (Markdown / CSV)
    └── experiments/            # 各策略专属产物 (HTML大屏 / 交割单 / metrics.json)
```

---

## 二、快速上手 (Quick Start)

所有操作均通过核心引擎 `run_experiments.py` 驱动：

### 1. 查看所有策略配置与完成状态 (`--list`)
扫描 `configs/` 下的所有策略 YAML 文件，并打印各策略最新的回测绩效：
```bash
python run_experiments.py --list
```

### 2. 运行单组策略
可以直接指定配置文件路径，也可以直接传策略名字：
```bash
# 方式 A: 直接传策略名称
python run_experiments.py double_ensemble

# 方式 B: 指定 YAML 文件路径
python run_experiments.py -c configs/double_ensemble.yaml
```

### 3. 一键运行全量策略并生成横向对比总表
```bash
python run_experiments.py
# 或
python run_experiments.py --all
```

### 4. 快速重新汇编对比报告 (`--report_only`)
无需重新跑模型训练与回测，直接从已有策略的 `metrics.json` 重新计算并生成 Markdown / CSV 横向对比总表：
```bash
python run_experiments.py --report_only
```

### 5. 自定义产物输出根目录 (`-o / --output_dir`)
```bash
python run_experiments.py double_ensemble -o /root/autodl-tmp/outputs
```

---

## 三、配置文件结构详解 (YAML 规范)

每一个策略 `.yaml` 都是**自包含、清晰直观**的。以下以 `configs/double_ensemble.yaml` 为例展示完整参数结构：

```yaml
key: "double_ensemble"
title: "DoubleEnsemble 双重集成策略 (5日标签 + 17年 + 完整风控)"
description: "微软KDD论文模型：在17年历史数据上迭代训练6个LightGBM子模型，动态去噪与特征筛选"

# 1. 股票池与时间切片
market: "csi500"                  # 股票池: 中证500
benchmark: "SH000905"             # 对比基准: 中证500指数
benchmark_ma: 20                  # 基准均线周期 (大盘择时)
label_horizon: 5                  # 预测未来 5 日收益标签
train_start: "2008-01-01"         # 训练集起始
train_end: "2024-12-31"           # 训练集结束
valid_start: "2025-01-01"         # 验证集起始 (用于早停)
valid_end: "2025-06-30"           # 验证集结束
test_start: "2025-07-01"          # 实盘模拟测试起始
test_end: "2026-08-24"            # 实盘模拟测试结束

# 2. 资金与交易规则
account_cash: 20000.0             # 初始本金: 2 万元
topk_stocks: 1                    # 持仓只数: Top 1
n_drop_stocks: 1                  # 每次调仓卖出数量
trade_unit: 100                   # 强制 100 股一手
risk_degree: 0.95                 # 目标仓位: 95%
open_cost: 0.0001                 # 买入佣金: 万一
close_cost: 0.0001                # 卖出佣金: 万一
min_cost: 5.0                     # 最低佣金: 5 元

# 3. 选股与多维风控
stock_uptrend_filter: true        # 只买主升浪 (Close > MA5 & Close > MA20)
drop_rank_threshold: 30           # 跌出前30名强制淘汰卖出
stop_loss_rate: 0.035             # -3.5% 固定硬止损
trailing_stop_trigger: 0.05       # 浮盈达到 +5% 启动跟踪止盈
trailing_stop_rate: 0.025         # 从最高点回撤 2.5% 锁定利润
max_holding_days: 10              # 最多持有 10 天调仓
market_timing_mode: "half_position_timing" # 大盘 < MA20 自动降半仓防御

# 4. 预测模型配置
model:
  class: "DEnsembleModel"
  module_path: "qlib.contrib.model.double_ensemble"
  kwargs:
    loss: "mse"
    base_model: "gbm"
    num_models: 6
    enable_sr: true              # 开启样本重加权 (去噪)
    enable_fs: true              # 开启特征选择 (互补)
    learning_rate: 0.0421
    max_depth: 8
    num_leaves: 210
```

---

## 四、如何极简新建一个自己的策略？

想要测试一个新想法（比如换个止损点、换个模型或测试新特征），**只需在 `configs/` 目录下新建一个 `.yaml` 文件即可**！

### 示例：新建一个 `my_strategy.yaml`
在 `strategies/configs/my_strategy.yaml` 中写入：
```yaml
key: "my_strategy"
title: "我的高频止盈策略"

# 自定义核心参数 (未填写的参数会自动采用系统安全兜底值)
label_horizon: 3
stop_loss_rate: 0.02              # 改为 -2% 严格止损
trailing_stop_trigger: 0.03       # 浮盈 +3% 立即启动止盈
trailing_stop_rate: 0.015         # 回撤 1.5% 锁定利润
```

运行方式：
```bash
python run_experiments.py my_strategy
```

---

## 五、核心底层组件 (Components)

| 组件文件 | 模块职责 | 核心亮点 |
| :--- | :--- | :--- |
| **`config_loader.py`** | 策略配置解析 | 扁平加载 `configs/*.yaml`，自动补充安全兜底参数。 |
| **`dataset.py`** | 特征持久化缓存 | 计算 17 年 Alpha158 因子工程并保存为 `cache/*.pkl`，后续实验 **0.3 秒秒级命中**。 |
| **`strategy.py`** | 实盘级选股策略 | 实现**自适应购买力顺延（Affordable TopK）**、个股双多头通道过滤、动态跟踪止损与大盘均线择时。 |
| **`day_rotate_strategy.py`**| 日频满仓轮动策略 | 每天固定全仓切换到当日预测打分最高的第一名股票（用于对比风控价值）。 |
| **`visualization.py`** | 交互式可视化 | 自动生成包含净值曲线、水下回撤图、持仓甘特图、月度收益热力图的 HTML 单文件大屏。 |
| **`analysis.py`** | 逐笔交易归因 | 解析每日持仓，输出标准**每日交割单 CSV**（包含成交单价、手数、税费、平仓盈亏、日末现金与总资产）。 |
| **`paths.py`** | 统一路径管理 | 隔离各策略专属产物，支持通过环境变量 `QLIB_OUTPUT_DIR` 一键重定向。 |

---

## 六、产物与报表输出管理 (Outputs)

每次回测完成后，产物会自动落盘到标准化分层目录：

```text
outputs/
├── cache/
│   ├── alpha158_csi500_1d_full.pkl     # 1日标签全量特征缓存 (~2.26GB)
│   └── alpha158_csi500_5d_full.pkl     # 5日标签全量特征缓存 (~2.26GB)
├── logs/
│   └── train_all_experiments.log       # 完整运行日志
├── reports/
│   ├── experiment_comparison.md        # 多策略横向对比报告 (Markdown)
│   └── experiment_comparison.csv       # 多策略横向对比表 (CSV)
└── experiments/
    ├── double_ensemble/
    │   ├── config.json                 # 实验超参快照
    │   ├── metrics.json                # 结构化评测指标 (IC/IR/收益率/回撤/夏普/卡玛等)
    │   ├── backtest_dashboard.html     # 机构级全景交互 HTML 大屏
    │   └── daily_delivery_slip.csv     # 每日真实交易交割流水单
    └── ...
```
