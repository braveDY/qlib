# 🚀 Qlib 量化策略云端服务器部署与训练指南 (0 到 1 实战全流程)

本文档记录了如何在全新的云端 Linux 服务器（如 AutoDL、阿里云、腾讯云等）上，从零搭建 Python 3.10 量化回测环境、同步行情数据、部署策略代码并启动后台实验的完整流程。

---

## 目录
- [一、服务器磁盘规划与目录规范](#一定义磁盘规划与目录规范)
- [二、本地行情数据集打包与同步](#二本地行情数据集打包与同步)
- [三、代码拉取与 GitHub 同步](#三代码拉取与-github-同步)
- [四、Conda 环境搭建与依赖安装](#四conda-环境搭建与依赖安装)
- [五、环境与数据加载快速验证](#五环境与数据加载快速验证)
- [六、后台长期训练与监控 (tmux)](#六后台长期训练与监控-tmux)
- [七、产物输出架构与结果同步回本地](#七产物输出架构与结果同步回本地)
- [八、常见问题排查与技巧 (FAQ)](#八常见问题排查与技巧-faq)

---

## 一、服务器磁盘规划与目录规范

以 **AutoDL** 为例，服务器通常分为：
- **系统盘 (`/`)**：容量较小（通常 30GB），装过多环境与特征缓存极易爆盘。
- **数据盘 (`/root/autodl-tmp/`)**：容量大（50GB+），读写性能高。

### 1. 最佳实践目录映射
| 模块 | 服务器目标路径 | 说明 |
| :--- | :--- | :--- |
| **Qlib 原始数据集** | `/root/autodl-tmp/.qlib/qlib_data/cn_data` | 存放在数据盘，建立软链接 `~/.qlib` |
| **项目代码与策略** | `/root/autodl-tmp/qlib` | 存放在数据盘 |
| **特征缓存与实验输出**| `/root/autodl-tmp/qlib/strategies/outputs` | 随项目存放在数据盘 |

### 2. 初始化软链接命令
登录服务器后执行：
```bash
mkdir -p /root/autodl-tmp/.qlib/qlib_data
ln -sfn /root/autodl-tmp/.qlib /root/.qlib
```

---

## 二、本地行情数据集打包与同步

Qlib 的 A 股日频数据集包含 **6 万多个微小的二进制 `.bin` 文件**。如果直接逐个文件传输，极易触发云服务器 SSH 连接超时中断。

### 正确做法：本地单包打包 + 流式断点续传

在**本地电脑终端**执行：

```bash
# 1. 在本地将行情数据快速归档为单个 tar 文件 (耗时约 1 秒)
tar -C ~/.qlib/qlib_data -cf /tmp/local_cn_data.tar cn_data

# 2. 通过 rsync 断点续传到服务器数据盘 (传输速度可达 10~20 MB/s)
rsync -avzP -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=5" \
  /tmp/local_cn_data.tar seeta:/root/autodl-tmp/.qlib/qlib_data/local_cn_data.tar

# 3. 在服务器端解压并清理临时包
ssh seeta "cd /root/autodl-tmp/.qlib/qlib_data && tar -xf local_cn_data.tar && rm -f local_cn_data.tar"

# 4. 清理本地临时打包文件
rm -f /tmp/local_cn_data.tar
```

---

## 三、代码拉取与 GitHub 同步

登录服务器：
```bash
ssh seeta
```

### 1. 开启学术网络加速 (AutoDL 专属)
```bash
source /etc/network_turbo
```

### 2. 克隆项目代码到数据盘
```bash
cd /root/autodl-tmp
git clone https://github.com/braveDY/qlib.git
cd /root/autodl-tmp/qlib
```

---

## 四、Conda 环境搭建与依赖安装

在服务器终端执行：

### 1. 创建并激活 Python 3.10 环境
```bash
conda create -n qlib python=3.10 -y
conda activate qlib
```

### 2. 配置国内镜像源加速下载
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip
```

### 3. 安装科学计算与机器学习核心依赖
```bash
pip install numpy pandas scipy scikit-learn lightgbm tables pyarrow \
            matplotlib plotly mlflow loguru tqdm fire joblib cvxpy \
            jinja2 markupsafe ruamel.yaml cython
```

### 4. 编译并以开发模式安装 Qlib 框架
```bash
cd /root/autodl-tmp/qlib
pip install --no-build-isolation -e .
```

---

## 五、环境与数据加载快速验证

在服务器终端运行单行测试脚本，验证 Qlib 与行情读取是否正常：

```bash
python -c "
import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)
from qlib.data import D
df = D.features(['SH600519', 'SZ000001'], ['\$close', '\$volume'], start_time='2024-01-01', end_time='2024-01-10')
print('\n🎉 Qlib 行情数据读取验证成功:')
print(df)
"
```
看到成功打印出茅台和深发展的量价数据，即代表环境搭建成功！

---

## 六、后台长期训练与监控 (tmux)

由于模型训练耗时较长（尤其是 `DoubleEnsemble` 或多窗口滚动再训练），**强烈建议使用 `tmux` 后台持久化运行**，防止因本地网络闪断或关闭电脑导致训练中断。

### 1. 新建后台会话
```bash
tmux new -s train
```

### 2. 激活环境并进入策略目录
```bash
conda activate qlib
cd /root/autodl-tmp/qlib/strategies
```

### 3. 执行实验命令

```bash
# 查看所有可用策略及其当前完成状态
python run_experiments.py --list

# 运行全部策略
python run_experiments.py

# 或者仅运行指定策略 (例如运行 double_ensemble 或 baseline)
python run_experiments.py double_ensemble baseline

# 仅从已有 metrics.json 重新汇编对比报告 (无需重新跑训练)
python run_experiments.py --report_only
```

### 4. 退出 / 挂起 tmux 界面
- 键盘按下快捷键：`Ctrl + B`，然后按 `D`（程序会在后台持续运行）。
- 此时可安全关闭本地终端或电脑。

### 5. 随时重新连接查看训练
```bash
# 重新进入训练终端
tmux attach -t train
```

---

## 七、产物输出架构与结果同步回本地

### 1. 标准化分层输出结构 (`strategies/outputs/`)
```text
strategies/outputs/
├── cache/                  # 全局共享特征缓存 (alpha158_csi500_1d/5d_full.pkl)
├── logs/                   # 所有执行日志 (*.log)
├── reports/                # 跨实验对比报告 (Markdown / CSV)
│   ├── experiment_comparison.md
│   ├── experiment_comparison.csv
│   └── experiment_deep_analysis.csv
└── experiments/            # 各独立实验专属产物
    ├── exp_a_baseline/
    │   ├── config.json         # 超参配置快照
    │   ├── metrics.json        # 结构化指标 (IC/IR/收益率/回撤/交易次数等)
    │   ├── backtest_dashboard.html  # 交互大屏看板
    │   └── daily_delivery_slip.csv  # 每日交易交割流水单
    ├── exp_b_label5d/
    └── ...
```

### 2. 将训练产物一键下载到本地电脑
在**本地电脑终端**执行：
```bash
# 将服务器生成的 HTML 看板、交割单与对比报告同步回本地
rsync -avzP seeta:/root/autodl-tmp/qlib/strategies/outputs/ /home/brave/open_src/qlib/strategies/outputs/
```

---

## 八、常见问题排查与技巧 (FAQ)

### Q1: AutoDL 关机后数据会丢失吗？
- **系统盘 (`/`) 与 数据盘 (`/root/autodl-tmp/`)**：只要不手动释放实例，关机数据**均不会丢失**。
- 开机后无需重新安装环境，只需 `conda activate qlib` 即可直接使用。

### Q2: 如何把实验输出重定向到其他路径？
系统支持环境变量 `QLIB_OUTPUT_DIR`，例如：
```bash
export QLIB_OUTPUT_DIR=/root/autodl-tmp/outputs
```
或者在运行脚本时指定参数：
```bash
python run_experiments.py -o /root/autodl-tmp/outputs
```


### Q3: 为什么训练首次很慢，后续很快？
- **首次运行**：会自动计算 2008 年至今 17 年的 Alpha158 因子工程并缓存至 `outputs/cache/`。
- **后续运行**：直接命中本地缓存，0.3 秒即可极速加载 226 万行特征矩阵并进行动态时间切片。
