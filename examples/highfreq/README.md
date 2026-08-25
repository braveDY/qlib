# 引言
本文件夹包含2个示例
- 高频数据集示例
- 基于高频数据预测价格趋势的示例

## 高频数据集

该数据集为强化学习高频交易的示例数据集。

### 获取高频数据

执行以下命令获取高频数据：
```bash
python workflow.py get_data
```

### 数据集的转储、重载与重新初始化

高频数据集在`workflow.py`中以`qlib.data.dataset.DatasetH`实现。`DatasetH`是[`qlib.utils.serial.Serializable`](https://qlib.readthedocs.io/en/latest/advanced/serial.html)的子类，可通过`pickle`格式将自身状态保存到磁盘，也可从磁盘加载状态。

### 关于重新初始化

从磁盘重载数据集后，Qlib还支持对数据集重新初始化。即用户可以重置`Dataset`或`DataHandler`的部分状态，例如标的、开始时间、结束时间、数据分段等，并依据更新后的状态生成新数据。

`workflow.py`中提供了对应示例，用户可按如下方式运行代码。

### 运行代码

执行下述命令运行示例：
```bash
python workflow.py dump_and_load_dataset
```

## 基准模型性能（高频数据价格趋势预测）

以下为高频数据价格趋势预测的模型实验结果。后续我们会持续更新基准模型。

|模型名称|数据集|IC|ICIR|Rank IC|Rank ICIR|多头精度|空头精度|多空平均收益率|多空平均夏普比率|
|---|---|---|---|---|---|---|---|---|---|
|LightGBM|Alpha158|0.0349±0.00|0.3805±0.00|0.0435±0.00|0.4724±0.00|0.5111±0.00|0.5428±0.00|0.000074±0.00|0.2677±0.00|