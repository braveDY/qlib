#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
日频满仓 Top1 因子轮动策略 (DayRotateStrategy)
==================================================================
最纯粹的信徒策略: 完全信任模型打分, 不做任何择时 / 止盈止损 / 均线过滤。

每天收盘后:
1. 取模型预测打分最高的股票 (仅过滤: 主板 + 可交易);
2. 若当前持仓就是目标股票 → 不做任何操作 (避免无谓换手白交手续费);
3. 否则 → 卖出全部旧持仓, 以可用资金的 risk_degree 比例满仓买入目标股票。

设计动机:
- 1 日标签模型预测的是"明日收益", 每日轮动恰好与信号有效期匹配;
- 去掉大盘择时 (熊市也有牛股, 让模型自己选);
- 去掉 uptrend filter (追高/均线过滤是主观干预, 此处完全交给因子排序);
- topk=1 满仓单票, 风险完全由选股质量承担 —— 用最干净的形态检验"模型 alpha 到底有多少"。
"""

import copy
import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import TradeDecisionWO, Order, OrderDir


class DayRotateStrategy(TopkDropoutStrategy):
    """日频满仓 Top1 因子轮动策略 (无任何主观风控, 每日持有打分最高的一只)"""

    def __init__(
        self,
        topk: int = 1,
        risk_degree: float = 0.95,
        trade_unit: int = 100,
        only_main_board: bool = True,
        **kwargs,
    ):
        super().__init__(topk=topk, n_drop=1, risk_degree=risk_degree, **kwargs)
        self.trade_unit = trade_unit
        self.only_main_board = only_main_board

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)

        # T-1 日收盘生成的预测打分 (无未来函数)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None or len(pred_score) == 0:
            return TradeDecisionWO([], self)

        current_temp = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()

        # 1. 选出当天打分最高的可交易目标 (主板过滤 + 涨跌停过滤)
        target_code = None
        for code in pred_score.dropna().sort_values(ascending=False).index:
            if self.only_main_board and (
                code.startswith("SZ30") or code.startswith("SH688") or code.startswith("BJ")
            ):
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
            target_code = code
            break

        # 2. 卖出所有非目标持仓
        for code in current_stock_list:
            if code == target_code:
                continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL
            ):
                continue
            sell_amount = current_temp.get_stock_amount(code=code)
            if sell_amount <= 0:
                continue
            sell_order = Order(
                stock_id=code, amount=sell_amount,
                start_time=trade_start_time, end_time=trade_end_time, direction=Order.SELL
            )
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                trade_val, trade_cost, _ = self.trade_exchange.deal_order(sell_order, position=current_temp)
                cash += trade_val - trade_cost

        # 3. 若目标未持有, 满仓买入
        if target_code is not None and target_code not in current_stock_list:
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=target_code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            if buy_price is not None and not np.isnan(buy_price) and buy_price > 0:
                raw_amount = (cash * self.risk_degree) / buy_price
                factor = self.trade_exchange.get_factor(
                    stock_id=target_code, start_time=trade_start_time, end_time=trade_end_time
                )
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(raw_amount, factor)
                if buy_amount > 0:
                    buy_order = Order(
                        stock_id=target_code, amount=buy_amount,
                        start_time=trade_start_time, end_time=trade_end_time, direction=Order.BUY
                    )
                    if self.trade_exchange.check_order(buy_order):
                        buy_order_list.append(buy_order)
                        self.trade_exchange.deal_order(buy_order, position=current_temp)

        return TradeDecisionWO(sell_order_list + buy_order_list, self)
