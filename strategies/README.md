# 📈 Qlib 配置驱动量化投研框架使用指南 (Strategies Framework)

本模块是一个**“配置驱动（Config-Driven）”**的工业级量化策略投研与回测执行系统。通过将实验参数（YAML）与执行逻辑（Python）完全解耦，实现了策略与因子的极简扩展、特征缓存跨实验复用、多维度横向对比评测以及机构级全景大屏自动生成。

---

## 目录
- [一、核心架构与目录规划](#一核心架构与目录规划)
- [二、快速上手 (Quick Start)](#二快速上手-quick-start)
- [三、配置文件与继承机制 (Config Guide)](#三配置文件与继承机制-config-guide)
- [四、如何极简新增一个实验？](#四如何极简新增一个实验)
- [五、底层核心组件说明 (Components)](#五底层核心组件说明-components)
- [六、产物与报表输出管理 (Outputs)](#六产物与报表输出管理-outputs)
- [七、云端与本地高效工作流](#七云端与本地高效工作流)

---

## 一、核心架构与目录规划

```text
strategies/
├── README.md                   # 📖 本使用说明文档
├── run_experiments.py          # 🌟【唯一通用执行引擎】(CLI 启动入口)
├── run_model_experiments.py    # 兼容入口 (转发至 run_experiments.py)
├── configs/                    # 🎛️【实验配置仓库】(纯 YAML 驱动)
│   ├── base_config.yaml        # 全局基准默认配置 (股票池、成本、本金、风控、基准模型)
│   ├── models/                 # 📂 专题 1: 模型算法对照 (Baseline, Label5D, Roll3Y, DoubleEnsemble)
│   ├── factors/                # 📂 专题 2: 因子特征工程对照 (Alpha158, RS相对强度)
│   └── strategies/             # 📂 专题 3: 交易策略与风控模式对照 (DayRotate, AffordableTopk)
├── components/                 # 🧩【底层组件库】
│   ├── config_loader.py        # YAML 配置加载与深度合并继承器
│   ├── dataset.py              # 共享特征持久化缓存与动态时间切片
│   ├── strategy.py             # 自适应购买力 TopK 选股与多维风控策略类
│   ├── day_rotate_strategy.py  # 日频满仓轮动策略类
│   ├── visualization.py        # 机构级全景交互 HTML 大屏生成器
│   ├── analysis.py             # 每日交易交割流水单生成器
│   └── paths.py                # 标准化路径与产物生命周期管理
└── outputs/                    # 📊【标准化分层输出中心】
    ├── cache/                  # 全局共享特征缓存 (多实验 0.3 秒秒级加载)
    ├── logs/                   # 执行运行日志
    ├── reports/                # 跨实验横向对比报告 (Markdown / CSV)
    └── experiments/            # 独立实验专属产物 (HTML大屏 / 交割单 / metrics.json)
```

---

## 二、快速上手 (Quick Start)

所有操作均通过核心引擎 `run_experiments.py` 驱动：

### 1. 查看所有专题与实验状态 (`--list`)
扫描 `configs/` 下的所有 YAML 配置，并打印各实验的最新完成状态与核心指标：
```bash
python run_experiments.py --list
```

### 2. 运行单组指定实验 (`-c / --config`)
```bash
# 运行 DoubleEnsemble 双重集成模型实验
python run_experiments.py -c configs/models/exp_f_double_ensemble.yaml

# 运行 RS 相对强度因子实验
python run_experiments.py -c configs/factors/exp_d_label5d_rs.yaml
```

### 3. 批量运行某个专题套件 (`-s / --suite`)
批量执行指定目录下的所有实验，并自动生成该专题的专属横向对比报告：
```bash
# 运行 models 专题下的所有 4 组模型对比实验
python run_experiments.py -s configs/models/
```

### 4. 运行全量所有实验 (`-a / --all`)
```bash
python run_experiments.py --all
```

### 5. 快速汇编多实验对比报告 (`--report_only`)
无需重新跑训练，直接从已有实验的 `metrics.json` 重新计算并生成 Markdown / CSV 横向对比总表：
```bash
python run_experiments.py --report_only
```

### 6. 自定义产物输出根目录 (`-o / --output_dir`)
```bash
python run_experiments.py -s configs/models/ -o /root/autodl-tmp/outputs
```

---

## 三、配置文件与继承机制 (Config Guide)

系统内置了 **深度字典合并（Deep Merge）与继承机制**：
- `configs/base_config.yaml` 集中定义了所有默认参数。
- 子配置文件（如 `exp_f_double_ensemble.yaml`）**默认自动继承全部基准参数，且仅需书写需要覆盖的差异字段**。

### `base_config.yaml` 核心参数一览

```yaml
# 1. 股票池与对比基准
market: "csi500"                  # 目标股票池: 中证500成分股
benchmark: "SH000905"             # 基准指数: 中证500
benchmark_ma: 20                  # 基准指数均线周期 (用于大盘牛熊择时)

# 2. 预测周期与时间切片
label_horizon: 1                  # 预测未来 N 日收益率 (1 / 3 / 5 日)
train_start: "2008-01-01"         # 训练集起始时间
train_end: "2024-12-31"           # 训练集结束时间
valid_start: "2025-01-01"         # 验证集起始时间 (用于早停)
valid_end: "2025-06-30"           # 验证集结束时间
test_start: "2025-07-01"          # 样本外实盘回测起始时间
test_end: "2026-08-24"            # 样本外实盘回测结束时间

# 3. 账户资金与交易规则
account_cash: 20000.0             # 初始实盘模拟本金 (¥20,000)
topk_stocks: 1                    # 目标持仓股票数量 (Top 1)
n_drop_stocks: 1                  # 每次调仓卖出数量
trade_unit: 100                   # 强制 A 股 1 手 (100 股) 交易整数倍
risk_degree: 0.95                 # 目标仓位上限 (95% 满仓运作)
open_cost: 0.0001                 # 买入佣金费率: 万一
close_cost: 0.0001                # 卖出佣金费率: 万一
min_cost: 5.0                     # 最低佣金门槛: 5 元 (万一免五设为 0)

# 4. 选股与多维风控
stock_uptrend_filter: true        # 开启个股双多头通道过滤 (Close > MA5 & Close > MA20)
stop_loss_rate: 0.035             # 固定止损线: -3.5% 市价平仓
trailing_stop_trigger: 0.05       # 移动止盈激活线: 浮盈 +5% 启动
trailing_stop_rate: 0.025         # 跟踪止损回撤容忍度: 从最高点回撤 2.5% 锁定利润
max_holding_days: 10              # 最大持仓时间约束: 10 个交易日自动调仓
market_timing_mode: "half_position_timing" # 择时模式: 大盘 < MA20 自动减半仓
```

---

## 四、如何极简新增一个实验？

想要测试一个新算法、新因子或新风控规则，**完全无需修改任何 Python 代码**！

### 示例 1：测试 3 日预测标签的新模型
在 `configs/models/` 目录下新建 `exp_label3d.yaml`：
```yaml
key: "exp_label3d"
title: "3日收益标签对照实验"
description: "测试未来 3 日收益标签的信噪比与回测表现"

# 仅需书写差异字段 (其余自动继承 base_config.yaml)
label_horizon: 3
```

### 示例 2：测试无止损止盈的裸跑策略
在 `configs/strategies/` 目录下新建 `exp_no_stoploss.yaml`：
```yaml
key: "exp_no_stoploss"
title: "无止损对照实验"

# 覆盖风控参数
stop_loss_rate: 1.0               # 禁用止损
trailing_stop_trigger: 1.0        # 禁用跟踪止盈
market_timing_mode: "none"        # 禁用大盘择时
```

运行方式：
```bash
python run_experiments.py -c configs/strategies/exp_no_stoploss.yaml
```

---

## 五、底层核心组件说明 (Components)

| 组件文件 | 模块职责 | 核心亮点 |
| :--- | :--- | :--- |
| **`config_loader.py`** | 配置解析与深度继承 | 支持单文件、专题目录批量加载，提供 `deep_merge_dicts` 递归覆盖。 |
| **`dataset.py`** | 特征持久化缓存 | 计算 17 年 Alpha158 因子工程并保存为 `cache/*.pkl`，后续实验 **0.3 秒秒级命中**。 |
| **`strategy.py`** | 实盘级选股策略 | 实现**自适应购买力顺延（Affordable TopK）**、个股双多头通道过滤、动态跟踪止损与大盘均线择时。 |
| **`day_rotate_strategy.py`**| 日频满仓轮动策略 | 每天固定全仓切换到当日预测打分最高的第一名股票（用于对比风控价值）。 |
| **`visualization.py`** | 交互式可视化 | 自动生成包含净值曲线、水下回撤图、持仓甘特图、月度收益热力图的 HTML 单文件大屏。 |
| **`analysis.py`** | 逐笔交易归因 | 解析每日持仓，输出标准**每日交割单 CSV**（包含成交单价、手数、税费、平仓盈亏、日末现金与总资产）。 |
| **`paths.py`** | 统一路径管理 | 隔离各实验专属产物，支持通过环境变量 `QLIB_OUTPUT_DIR` 一键重定向。 |

---

## 六、产物与报表输出管理 (Outputs)

每次实验完成后，产物会自动落盘到标准化分层目录：

```text
outputs/
├── cache/
│   ├── alpha158_csi500_1d_full.pkl     # 1日标签全量特征缓存 (~2.26GB)
│   └── alpha158_csi500_5d_full.pkl     # 5日标签全量特征缓存 (~2.26GB)
├── logs/
│   └── train_all_experiments.log       # 完整运行日志
├── reports/
│   ├── experiment_comparison.md        # 多实验横向对比报告 (Markdown)
│   └── experiment_comparison.csv       # 多实验横向对比表 (CSV)
└── experiments/
    ├── exp_f_double_ensemble/
    │   ├── config.json                 # 实验超参快照
    │   ├── metrics.json                # 结构化评测指标 (IC/IR/收益率/回撤/夏普/卡玛等)
    │   ├── backtest_dashboard.html     # 机构级全景交互 HTML 大屏
    │   └── daily_delivery_slip.csv     # 每日真实交易交割流水单
    └── ...
```

---

## 七、云端与本地高效工作流

在云端服务器（如 AutoDL）训练完毕后，可仅将轻量的结果文件（HTML 看板、交割单、对比报告）秒级同步回本地电脑：

```bash
# 在本地电脑终端执行 (排除 4GB 特征缓存，仅需几秒钟):
rsync -avzP \
  --exclude 'cache/' \
  --exclude '*.pkl' \
  seeta:/root/autodl-tmp/qlib/strategies/outputs/ \
  /home/brave/open_src/qlib/strategies/outputs/
```
同步完成后，在本地浏览器中直接双击任意 `backtest_dashboard.html` 即可畅快查看交互式回测看板！
