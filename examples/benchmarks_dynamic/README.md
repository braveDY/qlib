# 引言
金融市场环境具有非平稳特性，不同时期的数据分布可能发生变化，这会导致基于训练数据构建的模型在后续测试数据上性能下降。
因此，让预测模型/策略适配市场动态，对模型/策略的表现至关重要。

下表展示了不同方案在各类预测模型上的性能表现。

## Alpha158数据集
以下是qlib数据集的众包版本：[data_collector/crowd_source/README.md](https://github.com/chenditc/investment_data/releases)
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
rm -f qlib_bin.tar.gz
```

|模型名称|数据集|IC|ICIR|Rank IC|Rank ICIR|年化收益率|信息比率|最大回撤|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|RR[线性模型]|Alpha158|0.0945|0.5989|0.1069|0.6495|0.0857|1.3682|-0.0986|
|DDG‑DA[线性模型]|Alpha158|0.0983|0.6157|0.1108|0.6646|0.0764|1.1904|-0.0769|
|RR[LightGBM]|Alpha158|0.0816|0.5887|0.0912|0.6263|0.0771|1.3196|-0.0909|
|DDG‑DA[LightGBM]|Alpha158|0.0878|0.6185|0.0975|0.6524|0.1261|2.0096|-0.0744|

- `Alpha158`数据集的标签预测周期设为20。
- 滚动时间间隔设置为20个交易日。
- 测试滚动周期为2017年1月至2020年8月。
- 实验结果基于众包版本数据集。雅虎版本的qlib数据集不包含成交量加权平均价（VWAP），因此所有相关因子缺失并被填充为0，会造成矩阵秩亏（矩阵不具备满秩特性），导致DDG‑DA下层优化问题无法求解。