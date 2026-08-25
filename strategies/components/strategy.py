#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
自适应购买力选股策略模块 (AffordableTopkDropoutStrategy)
"""

import copy
import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import TradeDecisionWO, Order, OrderDir
from qlib.backtest.position import Position


class AffordableTopkDropoutStrategy(TopkDropoutStrategy):
    """
    自适应购买力与多维动态风控选股策略 (小资金机构级实战版)：
    1. 资金自适应：按手 (100股) 精确计算，买不起 1 手自动顺延至下一高分股；
    2. 个股独立多头趋势过滤 (Stock Uptrend Filter)：即便大盘弱势，只要个股自身处于 5日/20日双多头主升浪，允许半仓出击逆势牛股！拒绝买入破位阴跌股；
    3. 动态连续仓位管理 (Dynamic Position Sizing)：大盘多头全仓进攻 (95%)，大盘弱势半仓防守 (50%)，告别一刀切滞后；
    4. 账户高水位回撤硬锁利 (High-Watermark Drawdown Protection)：当账户从历史最高峰值回撤达到阈值 (如 8%)，自动收紧止损并收缩仓位，死守盈利果实；
    5. 多维动态退出：硬止损 (-4%)、移动止盈锁定、滞涨时间止损与非对称持股。
    """
    
    def __init__(
        self,
        topk: int = 1,
        n_drop: int = 1,
        risk_degree: float = 0.95,
        trade_unit: int = 100,
        rebalance_days: int = 1,
        only_main_board: bool = True,
        stop_loss_rate: float = 0.04,          # 硬止损比例 (0.04 表示 -4% 止损)
        trailing_stop_trigger: float = 0.06,   # 移动止盈触发门槛 (0.06 表示盈利 6% 开启移动止盈)
        trailing_stop_rate: float = 0.03,      # 移动止盈回撤阈值 (0.03 表示自最高点回撤 3% 止盈)
        max_holding_days: int = 10,            # 最长持股天数 (10 天滞涨则调仓换股；0 为不限)
        drop_rank_threshold: int = 30,         # 退出容忍排名 (打分在前 30 名坚定持有；0 为使用 n_drop)
        stock_uptrend_filter: bool = True,     # 个股独立多头通道过滤 (仅买入 close > MA5 且 close > MA20 的主升浪股票)
        market_timing_mode: str = "dynamic",   # 大盘择时模式 ("dynamic": 弱势半仓防御; "strict": 弱势全空仓; "off": 纯个股)
        benchmark_symbol: str = "SH000905",    # 大盘基准代码
        benchmark_ma: int = 20,                # 大盘均线周期
        account_drawdown_limit: float = 0.08,  # 账户高水位回撤保护阈值 (从最高峰回撤超 8% 强制防守锁利)
        **kwargs
    ):
        super().__init__(topk=topk, n_drop=n_drop, risk_degree=risk_degree, **kwargs)
        self.trade_unit = trade_unit
        self.rebalance_days = rebalance_days
        self.only_main_board = only_main_board
        self.stop_loss_rate = stop_loss_rate
        self.trailing_stop_trigger = trailing_stop_trigger
        self.trailing_stop_rate = trailing_stop_rate
        self.max_holding_days = max_holding_days
        self.drop_rank_threshold = drop_rank_threshold
        self.stock_uptrend_filter = stock_uptrend_filter
        self.market_timing_mode = market_timing_mode
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_ma = benchmark_ma
        self.account_drawdown_limit = account_drawdown_limit

        self.day_counter = 0
        self.holding_meta = {}  # {code: {"buy_price": float, "peak_price": float, "holding_days": int}}
        self._bench_df = None
        self._infer_df = None
        self.max_account_val = 0.0

    def _get_market_state(self, date_str: str) -> bool:
        """检查指定日期大盘是否处于多头均线之上"""
        if self.market_timing_mode == "off":
            return True
        if self._bench_df is None:
            try:
                from qlib.data import D
                b_df = D.features([self.benchmark_symbol], ["$close"])
                if isinstance(b_df.index, pd.MultiIndex):
                    b_df = b_df.droplevel(0)
                b_df.columns = ["close"]
                b_df["ma"] = b_df["close"].rolling(self.benchmark_ma).mean()
                b_df["bull"] = b_df["close"] >= b_df["ma"]
                self._bench_df = b_df
            except Exception:
                return True

        if date_str in self._bench_df.index:
            return bool(self._bench_df.loc[date_str, "bull"])
        
        matched = self._bench_df[self._bench_df.index <= date_str]
        if len(matched) > 0:
            return bool(matched.iloc[-1]["bull"])
        return True

    def _is_stock_in_uptrend(self, code: str, date_str: str) -> bool:
        """检查个股自身是否处于独立多头通道 (MA5 < 1.0 且 MA20 < 1.0 即 Close > MA5 > MA20)"""
        if not self.stock_uptrend_filter:
            return True
        if self._infer_df is None:
            if hasattr(self, "dataset") and self.dataset is not None:
                handler = getattr(self.dataset, "handler", None)
                if handler is not None:
                    self._infer_df = getattr(handler, "_infer", None)

        if self._infer_df is not None:
            try:
                # 提取 (feature, MA5) 和 (feature, MA20)
                ma5_val = self._infer_df.loc[(date_str, code), ("feature", "MA5")]
                ma20_val = self._infer_df.loc[(date_str, code), ("feature", "MA20")]
                # MA5 < 1.0 表示当前收盘价高于5日均线；MA20 < 1.0 表示当前收盘价高于20日均线
                return (ma5_val < 1.0) and (ma20_val < 1.05)
            except Exception:
                return True
        return True

    def generate_trade_decision(self, execute_result=None):
        self.day_counter += 1
        
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()

        # 清理已不在持仓中的元数据
        for code in list(self.holding_meta.keys()):
            if code not in current_stock_list:
                del self.holding_meta[code]

        # 1. 账户级高水位回撤风控监控 (High-Watermark Profit Lock)
        cur_total_val = current_temp.calculate_value()
        self.max_account_val = max(self.max_account_val, cur_total_val)
        account_drawdown = (self.max_account_val - cur_total_val) / self.max_account_val if self.max_account_val > 0 else 0.0

        # 若账户从最高峰回撤超过阈值，启动紧缩防守：收窄止损并收缩仓位
        is_account_defense = self.account_drawdown_limit > 0 and (account_drawdown >= self.account_drawdown_limit)
        effective_stop_loss = 0.025 if is_account_defense else self.stop_loss_rate
        
        # 2. 检查 T-1 日大盘多空状态 (无未来函数)
        date_str = str(pred_start_time).split(" ")[0]
        is_bull_market = self._get_market_state(date_str)

        # ---------------- 3. 智能风控与动态退出判断 ----------------
        sell_candidates = []
        sorted_rank_series = pred_score.sort_values(ascending=False)
        rank_dict = {code: rank + 1 for rank, code in enumerate(sorted_rank_series.index)}

        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL
            ):
                continue
            
            current_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL
            )
            if current_price is None or np.isnan(current_price) or current_price <= 0:
                continue

            if code not in self.holding_meta:
                self.holding_meta[code] = {
                    "buy_price": current_price,
                    "peak_price": current_price,
                    "holding_days": 0,
                }
            
            meta = self.holding_meta[code]
            meta["holding_days"] += 1
            if current_price > meta["peak_price"]:
                meta["peak_price"] = current_price

            buy_price = meta["buy_price"]
            peak_price = meta["peak_price"]
            pnl_ratio = (current_price - buy_price) / buy_price
            pullback_from_peak = (peak_price - current_price) / peak_price
            code_rank = rank_dict.get(code, 9999)

            should_sell = False

            # (1) 硬止损 / 账户回撤紧缩止损
            if effective_stop_loss > 0 and pnl_ratio <= -effective_stop_loss:
                should_sell = True

            # (2) 灵敏移动止盈：浮盈达到门槛且自最高点回撤超过阈值
            elif (
                self.trailing_stop_trigger > 0
                and (peak_price - buy_price) / buy_price >= self.trailing_stop_trigger
                and pullback_from_peak >= self.trailing_stop_rate
            ):
                should_sell = True

            # (3) 滞涨时间止损
            elif self.max_holding_days > 0 and meta["holding_days"] >= self.max_holding_days and pnl_ratio <= 0.01:
                should_sell = True

            # (4) 信号排名严重下滑 (跌破容忍区间)
            elif self.drop_rank_threshold > 0 and code_rank > self.drop_rank_threshold:
                should_sell = True

            # (5) 传统 n_drop 轮动
            elif self.drop_rank_threshold <= 0 and (self.day_counter % self.rebalance_days == 0):
                last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
                if code in list(last[-self.n_drop:]):
                    should_sell = True

            if should_sell:
                sell_candidates.append(code)

        # 执行卖单
        for code in sell_candidates:
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
                if code in self.holding_meta:
                    del self.holding_meta[code]

        # ---------------- 4. 买入逻辑 (支持个股逆势独立主升浪 + 连续动态仓位管理) ----------------
        holding_stocks = set(current_stock_list) - set(sell_candidates)
        needed_count = self.topk - len(holding_stocks)

        # 动态计算当前可用资金使用率 (Risk Degree)
        if self.market_timing_mode == "strict" and not is_bull_market:
            effective_risk = 0.0
        elif self.market_timing_mode == "dynamic" and not is_bull_market:
            effective_risk = self.risk_degree * 0.50   # 熊市防守期：允许以 50% 半仓精选逆势独立强股
        elif is_account_defense:
            effective_risk = self.risk_degree * 0.70   # 账户高水位回撤防守期：降至 70% 仓位
        else:
            effective_risk = self.risk_degree          # 正常多头期：95% 全仓进攻

        if needed_count > 0 and cash > 0 and effective_risk > 0:
            budget_per_stock = (cash * effective_risk) / needed_count
            candidate_stocks = pred_score[~pred_score.index.isin(holding_stocks)].sort_values(ascending=False).index
            
            for code in candidate_stocks:
                # 1. 主板权限过滤
                if self.only_main_board and (code.startswith("SZ30") or code.startswith("SH688") or code.startswith("BJ")):
                    continue
                
                # 2. 个股独立多头通道过滤：必须处于多头通道 (Close > MA5 & Close > MA20)，拒绝接飞刀
                if not self._is_stock_in_uptrend(code, date_str):
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
                
                # 3. 购买力自适应计算
                raw_amount = budget_per_stock / buy_price
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(raw_amount, factor)
                
                if buy_amount > 0:
                    buy_order = Order(
                        stock_id=code, amount=buy_amount,
                        start_time=trade_start_time, end_time=trade_end_time,
                        direction=Order.BUY
                    )
                    if self.trade_exchange.check_order(buy_order):
                        buy_order_list.append(buy_order)
                        trade_val, trade_cost, _ = self.trade_exchange.deal_order(buy_order, position=current_temp)
                        cash -= trade_val + trade_cost
                        
                        self.holding_meta[code] = {
                            "buy_price": buy_price,
                            "peak_price": buy_price,
                            "holding_days": 0,
                        }
                        if len(buy_order_list) >= needed_count:
                            break

        return TradeDecisionWO(sell_order_list + buy_order_list, self)
