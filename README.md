# Qlib：面向AI的量化投资平台

Qlib是一个开源、面向AI的量化投资平台，旨在利用AI技术在量化投资中挖掘潜力、赋能研究并创造价值——从探索想法到实现生产。Qlib支持多种机器学习建模范式，包括监督学习、市场动态建模和强化学习。

越来越多的SOTA（当前最优）量化研究工作/论文在不同范式中被整合到Qlib中，以协同解决量化投资中的关键挑战。例如：1）利用监督学习从丰富且异构的金融数据中挖掘市场的复杂非线性模式，2）使用自适应概念漂移技术建模金融市场的动态特性，3）利用强化学习建模连续投资决策并帮助投资者优化交易策略。

它包含完整的机器学习流程：数据处理、模型训练、回测；并覆盖量化投资的完整链条：Alpha挖掘、风险建模、投资组合优化、订单执行。
更多详情，请参考我们的论文 ["Qlib: An AI-oriented Quantitative Investment Platform"](https://arxiv.org/abs/2009.11189)。

**目录**（略，见文档结构）

# 计划
开发中的新功能（按预计发布时间排序）。您对这些功能的反馈非常重要。

# Qlib框架

Qlib的高级框架如上所示（用户可在深入了解时查看Qlib设计的[详细框架](https://qlib.readthedocs.io/en/latest/introduction/introduction.html#framework)）。组件设计为松耦合模块，每个组件均可独立使用。

Qlib提供强大的基础设施以支持量化研究。[数据](https://qlib.readthedocs.io/en/latest/component/data.html)始终是重要组成部分。
强大的学习框架旨在支持不同的学习范式（如[强化学习](https://qlib.readthedocs.io/en/latest/component/rl.html)、[监督学习](https://qlib.readthedocs.io/en/latest/component/workflow.html#model-section)）和不同层次模式（如[市场动态建模](https://qlib.readthedocs.io/en/latest/component/meta.html)）。
通过对市场建模，[交易策略](https://qlib.readthedocs.io/en/latest/component/strategy.html)将生成交易决策并执行。不同层次或粒度的多个交易策略和执行器可以[嵌套在一起进行优化和协同运行](https://qlib.readthedocs.io/en/latest/component/highfreq.html)。
最后，将提供全面的[分析](https://qlib.readthedocs.io/en/latest/component/report.html)，且模型可以低成本[在线服务化](https://qlib.readthedocs.io/en/latest/component/online.html)。

# 快速开始

本快速开始指南旨在演示：
1. 使用_Qlib_构建完整的量化研究工作流并尝试您的想法非常容易。
2. 尽管使用*公开数据*和*简单模型*，机器学习技术在实践量化投资中**表现非常出色**。

这里有一个快速**[演示](https://terminalizer.com/view/3f24561a4470)**，展示如何安装``Qlib``，并使用``qrun``运行LightGBM。**但是**，请确保您已按照[说明](#数据准备)准备好数据。

## 安装

此表格展示`Qlib`支持的Python版本：

|               | 通过pip安装 | 从源码安装 | plot |
| ------------- |:---------------------:|:--------------------:|:------------------:|
| Python 3.8    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.9    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.10   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.11   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.12   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |

**注意**：
1. 建议使用**Conda**管理Python环境。在某些情况下，在`conda`环境外使用Python可能导致缺少头文件，从而造成某些包安装失败。
2. 请注意，在Python 3.6中安装cython时，从源码安装``Qlib``会引发错误。如果用户机器使用Python 3.6，建议*升级*Python至3.8或更高版本，或使用`conda`的Python从源码安装``Qlib``。

### 通过pip安装
用户可以通过以下命令轻松通过pip安装``Qlib``：

```bash
  pip install pyqlib
```

**注意**：pip将安装最新的稳定版qlib。但qlib的主分支处于活跃开发中。如果您想测试主分支中的最新脚本或函数，请按以下方法安装qlib。

### 从源码安装
用户也可以按照以下步骤通过源码安装最新开发版``Qlib``：

* 从源码安装``Qlib``前，用户需安装一些依赖：

  ```bash
  pip install numpy
  pip install --upgrade cython
  ```

* 克隆仓库并按如下方式安装``Qlib``：
    ```bash
    git clone https://github.com/microsoft/qlib.git && cd qlib
    pip install .  # 开发推荐使用 `pip install -e .[dev]`。详情请查看 docs/developer/code_standard_and_dev_guide.rst
    ```

**提示**：如果您在环境中安装`Qlib`或运行示例失败，将您的步骤与[CI工作流](.github/workflows/test_qlib_from_source.yml)对比可能有助于发现问题。

**Mac提示**：如果您使用M1芯片的Mac，在构建LightGBM的wheel时可能遇到问题，这是由于缺少OpenMP的依赖。解决方法：先用``brew install libomp``安装openmp，然后运行``pip install .``成功构建。

## 数据准备
❗ 由于更严格的数据安全政策，官方数据集暂时禁用。您可以尝试社区贡献的[此数据源](https://github.com/chenditc/investment_data/releases)。
以下是下载最新数据的示例：
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

官方数据集将在短期内恢复。

----

通过运行以下代码加载和准备数据：

### 通过模块获取
  ```bash
  # 获取日线数据
  python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取1分钟数据
  python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
  ```

### 从源码获取
  ```bash
  # 获取日线数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取1分钟数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
  ```

该数据集由[crawler脚本](scripts/data_collector/)收集的公开数据创建，这些脚本已发布在同一仓库中。
用户可以使用相同方法创建相同数据集。[数据集描述](https://github.com/microsoft/qlib/tree/main/scripts/data_collector#description-of-dataset)

*请注意，数据收集自[Yahoo Finance](https://finance.yahoo.com/lookup)，数据可能不完美。
如果用户有高质量数据集，建议准备自己的数据。更多信息请参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*。

### 日频数据自动更新（来自雅虎财经）
  > 如果用户仅想在历史数据上尝试模型和策略，此步骤*可选*。
  >
  > 建议用户先手动更新一次数据（--trading_date 2021-05-25），然后设置为自动更新。
  >
  > **注意**：用户不能基于Qlib提供的离线数据进行增量更新（部分字段已被移除以减小数据大小）。用户应使用[yahoo collector](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)从头下载Yahoo数据，然后进行增量更新。
  >
  > 更多信息请参考：[yahoo collector](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)

  * 每个交易日自动更新数据至"qlib"目录（Linux）
      * 使用*crontab*：`crontab -e`
      * 设置定时任务：
        ```
        * * * * 1-5 python <脚本路径> update_data_to_bin --qlib_data_1d_dir <用户数据目录>
        ```
        * **脚本路径**：*scripts/data_collector/yahoo/collector.py*

  * 手动更新数据
      ```
      python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <用户数据目录> --trading_date <开始日期> --end_date <结束日期>
      ```
      * *trading_date*：交易日开始日期
      * *end_date*：交易结束日期（不包含）

### 检查数据健康状况
  * 我们提供一个脚本检查数据健康状况，您可运行以下命令检查数据是否健康：
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data
    ```
  * 当然，您也可以添加一些参数调整测试结果，例如：
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data --missing_data_num 30055 --large_step_threshold_volume 94485 --large_step_threshold_price 20
    ```
  * 如需更多关于`check_data_health`的信息，请参考[文档](https://qlib.readthedocs.io/en/latest/component/data.html#checking-the-health-of-the-data)。

## Docker镜像
1. 从Docker Hub仓库拉取镜像
    ```bash
    docker pull pyqlib/qlib_image_stable:stable
    ```
2. 启动新的Docker容器
    ```bash
    docker run -it --name <容器名称> -v <挂载的本地目录>:/app pyqlib/qlib_image_stable:stable
    ```
3. 此时您已进入docker环境，可运行qlib脚本。示例：
    ```bash
    >>> python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
    >>> python qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
4. 退出容器
    ```bash
    >>> exit
    ```
5. 重启容器
    ```bash
    docker start -i -a <容器名称>
    ```
6. 停止容器
    ```bash
    docker stop <容器名称>
    ```
7. 删除容器
    ```bash
    docker rm <容器名称>
    ```
8. 如需更多信息，请参考[文档](https://qlib.readthedocs.io/en/latest/developer/how_to_build_image.html)。

## 自动量化研究工作流
Qlib提供了一个名为`qrun`的工具，可自动运行整个工作流（包括构建数据集、训练模型、回测和评估）。您可以通过以下步骤启动自动量化研究工作流并获得图形化报告分析：

1. 量化研究工作流：使用LightGBM工作流配置（[workflow_config_lightgbm_Alpha158.yaml](examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)）运行`qrun`，如下所示：
    ```bash
      cd examples  # 避免在包含`qlib`的目录下运行
      qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    如果用户想在调试模式下使用`qrun`，请使用以下命令：
    ```bash
    python -m pdb qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    `qrun`的结果如下，更多结果解释请参考[docs](https://qlib.readthedocs.io/en/latest/component/strategy.html#result)：

    ```bash
    'The following are analysis results of the excess return without cost.'
                           risk
    mean               0.000708
    std                0.005626
    annualized_return  0.178316
    information_ratio  1.996555
    max_drawdown      -0.081806
    'The following are analysis results of the excess return with cost.'
                           risk
    mean               0.000512
    std                0.005626
    annualized_return  0.128982
    information_ratio  1.444287
    max_drawdown      -0.091078
    ```
    以下是`qrun`和[workflow](https://qlib.readthedocs.io/en/latest/component/workflow.html)的详细文档。

2. 图形化报告分析：首先运行`python -m pip install .[analysis]`安装所需依赖。然后使用`jupyter notebook`运行`examples/workflow_by_code.ipynb`以获取图形化报告。
    - 预测信号（模型预测）分析
      - 分组累计收益
      ![Cumulative Return](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_cumulative_return.png)
      - 收益分布
      ![long_short](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_long_short.png)
      - 信息系数（IC）
      ![Information Coefficient](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_IC.png)
      ![Monthly IC](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_monthly_IC.png)
      ![IC](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_NDQ.png)
      - 预测信号（模型预测）的自相关性
      ![Auto Correlation](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_auto_correlation.png)

    - 投资组合分析
      - 回测收益
      ![Report](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/report.png)

   - 上述结果的[解释](https://qlib.readthedocs.io/en/latest/component/report.html)

## 通过代码构建自定义量化研究工作流
自动化工作流可能不适合所有量化研究人员的研究工作流。为支持灵活的量化研究工作流，Qlib还提供模块化接口，允许研究人员通过代码构建自己的工作流。这里有[一个自定义量化研究工作流的演示](examples/workflow_by_code.ipynb)。

# 量化研究中的主要挑战与解决方案
量化投资是一个非常特殊的场景，有许多关键挑战需要解决。
目前，Qlib为其中一些挑战提供了解决方案。

## 预测：发现有价值的信号/模式
准确预测股票价格趋势是构建盈利投资组合非常重要的一环。
然而，金融市场中大量不同格式的数据使得构建预测模型具有挑战性。

越来越多的SOTA量化研究工作/论文，专注于构建预测模型以在复杂金融数据中挖掘有价值的信号/模式，已在`Qlib`中发布。

### [量化模型（论文）动物园](examples/benchmarks)

以下是在`Qlib`上构建的模型列表：
- [基于XGBoost的GBDT (Tianqi Chen, et al. KDD 2016)](examples/benchmarks/XGBoost/)
- [基于LightGBM的GBDT (Guolin Ke, et al. NIPS 2017)](examples/benchmarks/LightGBM/)
- [基于Catboost的GBDT (Liudmila Prokhorenkova, et al. NIPS 2018)](examples/benchmarks/CatBoost/)
- [基于pytorch的MLP](examples/benchmarks/MLP/)
- [基于pytorch的LSTM (Sepp Hochreiter, et al. Neural computation 1997)](examples/benchmarks/LSTM/)
- [基于pytorch的GRU (Kyunghyun Cho, et al. 2014)](examples/benchmarks/GRU/)
- [基于pytorch的ALSTM (Yao Qin, et al. IJCAI 2017)](examples/benchmarks/ALSTM)
- [基于pytorch的GATs (Petar Velickovic, et al. 2017)](examples/benchmarks/GATs/)
- [基于pytorch的SFM (Liheng Zhang, et al. KDD 2017)](examples/benchmarks/SFM/)
- [基于tensorflow的TFT (Bryan Lim, et al. International Journal of Forecasting 2019)](examples/benchmarks/TFT/)
- [基于pytorch的TabNet (Sercan O. Arik, et al. AAAI 2019)](examples/benchmarks/TabNet/)
- [基于LightGBM的DoubleEnsemble (Chuheng Zhang, et al. ICDM 2020)](examples/benchmarks/DoubleEnsemble/)
- [基于pytorch的TCTS (Xueqing Wu, et al. ICML 2021)](examples/benchmarks/TCTS/)
- [基于pytorch的Transformer (Ashish Vaswani, et al. NeurIPS 2017)](examples/benchmarks/Transformer/)
- [基于pytorch的Localformer (Juyong Jiang, et al.)](examples/benchmarks/Localformer/)
- [基于pytorch的TRA (Hengxu, Dong, et al. KDD 2021)](examples/benchmarks/TRA/)
- [基于pytorch的TCN (Shaojie Bai, et al. 2018)](examples/benchmarks/TCN/)
- [基于pytorch的ADARNN (YunTao Du, et al. 2021)](examples/benchmarks/ADARNN/)
- [基于pytorch的ADD (Hongshun Tang, et al.2020)](examples/benchmarks/ADD/)
- [基于pytorch的IGMTF (Wentao Xu, et al.2021)](examples/benchmarks/IGMTF/)
- [基于pytorch的HIST (Wentao Xu, et al.2021)](examples/benchmarks/HIST/)
- [基于pytorch的KRNN](examples/benchmarks/KRNN/)
- [基于pytorch的Sandwich](examples/benchmarks/Sandwich/)

热烈欢迎您提交新的量化模型PR。

每个模型在`Alpha158`和`Alpha360`数据集上的表现可在[这里](examples/benchmarks/README.md)查看。

### 运行单个模型
上述所有模型均可通过``Qlib``运行。用户可通过[benchmarks](examples/benchmarks)文件夹找到我们提供的配置文件及模型相关细节。更多信息可在上述列出的模型文件中获取。

`Qlib`提供三种不同方式运行单个模型，用户可根据情况选择最适合的方式：
- 用户可使用上述工具`qrun`，基于配置文件运行模型工作流。
- 用户可基于`examples`文件夹中的[示例](examples/workflow_by_code.py)创建`workflow_by_code` Python脚本。

- 用户可使用`examples`文件夹中的脚本[`run_all_model.py`](examples/run_all_model.py)运行模型。以下是具体shell命令示例：`python run_all_model.py run --models=lightgbm`，其中`--models`参数可接受上述任意数量的模型（可用模型可在[benchmarks](examples/benchmarks/)中找到）。更多用例请参考该文件的[文档字符串](examples/run_all_model.py)。
    - **注意**：每个基线有不同的环境依赖，请确保您的Python版本符合要求（例如，由于`tensorflow==1.15.0`的限制，TFT仅支持Python 3.6~3.7）。

### 运行多个模型
`Qlib`还提供脚本[`run_all_model.py`](examples/run_all_model.py)，可多次迭代运行多个模型。（**注意**：目前该脚本仅支持*Linux*。未来将支持其他操作系统。此外，目前也不支持并行多次运行同一模型，这个问题也将在未来开发中修复。）

该脚本将为每个模型创建独立的虚拟环境，并在训练完成后删除环境。因此，只会生成和存储如`IC`和回测结果等实验结果。

以下是运行所有模型10次迭代的示例：
```python
python run_all_model.py run 10
```

它还提供了同时运行特定模型的API。更多用例请参考该文件的[文档字符串](examples/run_all_model.py)。

### 重大变更
在`pandas`中，`group_key`是`groupby`方法的参数之一。从`pandas`版本1.5到2.0，`group_key`的默认值从`无默认`变为`True`，这会导致qlib在运行时报错。因此我们设置了`group_key=False`，但不能保证所有程序都能正确运行，包括：
* qlib\examples\rl_order_execution\scripts\gen_training_orders.py
* qlib\examples\benchmarks\TRA\src\dataset.MTSDatasetH.py
* qlib\examples\benchmarks\TFT\tft.py

## [适应市场动态](examples/benchmarks_dynamic)

由于金融市场环境的非平稳性，数据分布可能在不同时期发生变化，这导致基于训练数据构建的模型在未来测试数据上的表现下降。
因此，使预测模型/策略适应市场动态对模型/策略的表现非常重要。

以下是在`Qlib`上构建的解决方案列表：
- [滚动再训练](examples/benchmarks_dynamic/baseline/)
- [基于pytorch的DDG-DA (Wendi, et al. AAAI 2022)](examples/benchmarks_dynamic/DDG-DA/)

## 强化学习：建模连续决策
Qlib现已支持强化学习，该功能旨在建模连续投资决策。此功能通过让投资者与环境交互以最大化某种累积奖励，帮助投资者优化其交易策略。

以下是在`Qlib`上按场景分类构建的解决方案列表。

### [订单执行的强化学习](examples/rl_order_execution)
[这里](https://qlib.readthedocs.io/en/latest/component/rl/overall.html#order-execution)是该场景的介绍。以下所有方法均[在此处](examples/rl_order_execution)进行比较。
- [TWAP](examples/rl_order_execution/exp_configs/backtest_twap.yml)
- [PPO: "基于近端策略优化的端到端最优交易执行框架", IJCAL 2020](examples/rl_order_execution/exp_configs/backtest_ppo.yml)
- [OPDS: "基于Oracle策略蒸馏的通用交易执行", AAAI 2021](examples/rl_order_execution/exp_configs/backtest_opds.yml)

# 量化数据集动物园
数据集在量化中扮演着非常重要的角色。以下是在`Qlib`上构建的数据集列表：

| 数据集                                    | 美国市场 | 中国市场 |
| --                                         | --        | --           |
| [Alpha360](./qlib/contrib/data/handler.py) |  √        |  √           |
| [Alpha158](./qlib/contrib/data/handler.py) |  √        |  √           |

[这里](https://qlib.readthedocs.io/en/latest/advanced/alpha.html)是使用`Qlib`构建数据集的教程。
热烈欢迎您提交构建新量化数据集的PR。

# 学习框架
Qlib具有高度可定制性，其许多组件是可学习的。
可学习组件是`预测模型`和`交易代理`的实例。它们基于`学习框架`层学习，然后应用于`工作流`层的多种场景。
学习框架也利用`工作流`层（例如，共享`信息提取器`，基于`执行环境`创建环境）。

基于学习范式，它们可分为强化学习和监督学习。
- 监督学习，详细文档可在[这里](https://qlib.readthedocs.io/en/latest/component/model.html)找到。
- 强化学习，详细文档可在[这里](https://qlib.readthedocs.io/en/latest/component/rl.html)找到。Qlib的强化学习框架利用`工作流`层中的`执行环境`创建环境。值得注意的是，`NestedExecutor`也得到支持。这使用户能够共同优化不同层次的策略/模型/代理（例如，针对特定投资组合管理策略优化订单执行策略）。

# 更多关于Qlib
如果您想快速浏览Qlib最常用的组件，可以尝试[这里的notebooks](examples/tutorial/)。

详细文档组织在[docs](docs/)中。
构建HTML格式文档需要[Sphinx](http://www.sphinx-doc.org)和readthedocs主题。
```bash
cd docs/
conda install sphinx sphinx_rtd_theme -y
# 或者，您可以通过pip安装它们
# pip install sphinx sphinx_rtd_theme
make html
```
您也可以直接在线查看[最新文档](http://qlib.readthedocs.io/)。

Qlib处于活跃和持续开发中。我们的计划在路线图中，以[github项目](https://github.com/microsoft/qlib/projects/1)形式管理。

# 离线模式与在线模式
Qlib的数据服务器可部署为`离线`模式或`在线`模式。默认模式为离线模式。

在`离线`模式下，数据将部署在本地。

在`在线`模式下，数据将作为共享数据服务部署。数据和缓存将由所有客户端共享。由于缓存命中率提高，数据检索性能预计将得到提升。它也会消耗更少的磁盘空间。在线模式的文档可在[Qlib-Server](https://qlib-server.readthedocs.io/)中找到。在线模式可使用[基于Azure CLI的脚本](https://qlib-server.readthedocs.io/en/latest/build.html#one-click-deployment-in-azure)自动部署。在线数据服务器的源代码可在[Qlib-Server仓库](https://github.com/microsoft/qlib-server)中找到。

## Qlib数据服务器的性能
数据处理性能对AI技术等数据驱动方法非常重要。作为面向AI的平台，Qlib为数据存储和数据处理提供了解决方案。为展示Qlib数据服务器的性能，我们将其与其他几种数据存储解决方案进行了比较。

我们通过完成相同任务来评估几种存储解决方案的性能，该任务从股票市场（2007年至2020年每天800只股票）的基础OHLCV日数据中创建数据集（14个特征/因子）。任务涉及数据查询和处理。

|                         | HDF5      | MySQL     | MongoDB   | InfluxDB  | Qlib -E -D  | Qlib +E -D   | Qlib +E +D  |
| --                      | ------    | ------    | --------  | --------- | ----------- | ------------ | ----------- |
| 总计（1CPU）（秒）  | 184.4±3.7 | 365.3±7.5 | 253.6±6.7 | 368.2±3.6 | 147.0±8.8   | 47.6±1.0     | **7.4±0.3** |
| 总计（64CPU）（秒） |           |           |           |           | 8.8±0.6     | **4.2±0.2**  |             |
* `+(-)E` 表示有（无）`ExpressionCache`
* `+(-)D` 表示有（无）`DatasetCache`

大多数通用数据库加载数据耗时过长。研究底层实现后，我们发现数据在通用数据库解决方案中经过了太多层接口和不必要的格式转换。
这些开销大大减慢了数据加载过程。
Qlib数据以紧凑格式存储，便于组合成数组进行科学计算。

# 相关报告
- [Guide To Qlib: Microsoft's AI Investment Platform](https://analyticsindiamag.com/qlib/)
- [微软也搞AI量化平台？还是开源的！](https://mp.weixin.qq.com/s/47bP5YwxfTp2uTHjUBzJQQ)
- [微矿Qlib：业内首个AI量化投资开源平台](https://mp.weixin.qq.com/s/vsJv7lsgjEi-ALYUz4CvtQ)


