#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
回测真实每日交割单生成模块 (Daily Delivery & Settlement Slip Generator)
"""

from typing import Optional, Union
from pathlib import Path
import pandas as pd
from qlib.data import D


def generate_daily_delivery_slip(
    positions: dict,
    save_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    生成按日期时间顺序的【每日真实交易交割流水单 (Delivery Slip)】：
    包含：交易日期、证券代码、买卖方向 (买入开仓/卖出平仓/持仓观望)、
          成交单价(真实未复权)、成交股数、成交手数、成交金额、
          交易税费、平仓盈亏金额与比例、日末持仓股数、日末现金与总资产、仓位占比。
    """
    dates = sorted(positions.keys())
    if not dates:
        return pd.DataFrame()

    all_stocks = list(set([s for pos in positions.values() for s in (pos.get_stock_list() if hasattr(pos, "get_stock_list") else [])]))
    
    factor_df = None
    if all_stocks:
        try:
            factor_df = D.features(all_stocks, ["$factor", "$close"], start_time=dates[0], end_time=dates[-1])
        except Exception:
            factor_df = None

    trade_history = {}  # 记录买入成本，用于精准核算平仓盈亏
    slips = []
    prev_pos = None

    for d in dates:
        pos = positions[d]
        cur_stocks = set(pos.get_stock_list() if hasattr(pos, "get_stock_list") else [])
        prev_stocks = set(prev_pos.get_stock_list() if hasattr(prev_pos, "get_stock_list") else []) if prev_pos else set()
        
        total_val = pos.calculate_value() if hasattr(pos, "calculate_value") else 0
        cash = pos.get_cash() if hasattr(pos, "get_cash") else 0
        stock_val = pos.calculate_stock_value() if hasattr(pos, "calculate_stock_value") else 0
        
        bought = cur_stocks - prev_stocks
        sold = prev_stocks - cur_stocks

        # 1. 处理卖出 (平仓结算)
        for s in sold:
            p_info = prev_pos.position[s]
            adj_amt = p_info["amount"]
            factor = factor_df.loc[(s, d), "$factor"] if (factor_df is not None and (s, d) in factor_df.index) else 1.0
            close_adj = factor_df.loc[(s, d), "$close"] if (factor_df is not None and (s, d) in factor_df.index) else p_info["price"]
            
            real_amt = int(round(adj_amt * factor))
            real_price = close_adj / factor if factor else close_adj
            deal_val = real_amt * real_price
            cost = max(5.0, deal_val * 0.0001)
            
            buy_info = trade_history.get(s, {"buy_val": deal_val, "buy_price": real_price})
            pnl = deal_val - buy_info["buy_val"] - cost
            pnl_pct = (pnl / buy_info["buy_val"]) * 100 if buy_info["buy_val"] > 0 else 0
            
            slips.append({
                "交易日期": d.strftime("%Y-%m-%d"),
                "证券代码": s,
                "操作方向": "【卖出平仓】",
                "成交单价 (¥)": round(real_price, 2),
                "成交股数": real_amt,
                "成交手数": real_amt // 100,
                "成交金额 (¥)": round(deal_val, 2),
                "交易税费 (¥)": round(cost, 2),
                "平仓盈亏 (¥)": round(pnl, 2),
                "盈亏比例 (%)": f"{pnl_pct:+.2f}%",
                "日末持仓状态": "0股 (已清仓)",
                "日末可用现金 (¥)": round(cash, 2),
                "日末账户总资产 (¥)": round(total_val, 2),
                "股票总仓位 (%)": round(stock_val / total_val * 100, 1) if total_val > 0 else 0,
            })

        # 2. 处理买入 (开仓交割)
        for s in bought:
            p_info = pos.position[s]
            adj_amt = p_info["amount"]
            factor = factor_df.loc[(s, d), "$factor"] if (factor_df is not None and (s, d) in factor_df.index) else 1.0
            close_adj = factor_df.loc[(s, d), "$close"] if (factor_df is not None and (s, d) in factor_df.index) else p_info["price"]
            
            real_amt = int(round(adj_amt * factor))
            real_price = close_adj / factor if factor else close_adj
            deal_val = real_amt * real_price
            cost = max(5.0, deal_val * 0.0001)
            
            trade_history[s] = {"buy_val": deal_val, "buy_price": real_price, "buy_date": d}
            
            slips.append({
                "交易日期": d.strftime("%Y-%m-%d"),
                "证券代码": s,
                "操作方向": "【买入开仓】",
                "成交单价 (¥)": round(real_price, 2),
                "成交股数": real_amt,
                "成交手数": real_amt // 100,
                "成交金额 (¥)": round(deal_val, 2),
                "交易税费 (¥)": round(cost, 2),
                "平仓盈亏 (¥)": "-",
                "盈亏比例 (%)": "-",
                "日末持仓状态": f"{s} ({real_amt}股/{real_amt//100}手)",
                "日末可用现金 (¥)": round(cash, 2),
                "日末账户总资产 (¥)": round(total_val, 2),
                "股票总仓位 (%)": round(stock_val / total_val * 100, 1) if total_val > 0 else 0,
            })

        # 3. 处理持仓观望或空仓日
        if not bought and not sold:
            held_desc = []
            for s in cur_stocks:
                p_info = pos.position[s]
                factor = factor_df.loc[(s, d), "$factor"] if (factor_df is not None and (s, d) in factor_df.index) else 1.0
                real_amt = int(round(p_info["amount"] * factor))
                held_desc.append(f"{s}({real_amt}股)")
                
            slips.append({
                "交易日期": d.strftime("%Y-%m-%d"),
                "证券代码": ", ".join(cur_stocks) if cur_stocks else "无",
                "操作方向": "【持仓观望】" if cur_stocks else "【空仓观望】",
                "成交单价 (¥)": 0.0,
                "成交股数": 0,
                "成交手数": 0,
                "成交金额 (¥)": 0.0,
                "交易税费 (¥)": 0.0,
                "平仓盈亏 (¥)": "-",
                "盈亏比例 (%)": "-",
                "日末持仓状态": ", ".join(held_desc) if held_desc else "空仓 (无持股)",
                "日末可用现金 (¥)": round(cash, 2),
                "日末账户总资产 (¥)": round(total_val, 2),
                "股票总仓位 (%)": round(stock_val / total_val * 100, 1) if total_val > 0 else 0,
            })
            
        prev_pos = pos

    df_slips = pd.DataFrame(slips)
    if save_csv_path:
        Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_slips.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n" + "=" * 90)
        print(f" 📑 每日真实交易交割流水单 (前 15 笔明细):")
        print("=" * 90)
        print(df_slips.head(15).to_string(index=False))
        print(f"\n ✅ 完整每日交割单已成功保存至: {save_csv_path}")

    return df_slips
