#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
专业量化对冲基金级全景回测看板生成模块 (Institutional-Grade Quant Backtest Dashboard)
"""

from typing import Optional, Union
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qlib.data import D


def _calc_kpis(report_df: pd.DataFrame, positions: dict, benchmark: str, target_risk_degree: float):
    """计算专业量化绩效与风险核心指标"""
    df = report_df.copy()
    net_ret = df["return"] - df["cost"]
    cum_net = net_ret.cumsum()
    cum_gross = df["return"].cumsum()
    cum_bench = df["bench"].cumsum()
    cum_alpha = (net_ret - df["bench"]).cumsum()
    # 复利净值口径 (每日扣费后净值累乘), 用于总收益/年化/回撤的准确计算
    nav = (1.0 + net_ret).cumprod()
    bench_nav = (1.0 + df["bench"]).cumprod()

    total_days = len(df)
    years = total_days / 252.0 if total_days > 0 else 1.0

    init_cash = df["account"].iloc[0]
    final_val = df["account"].iloc[-1]
    peak_val = df["account"].max()
    net_profit = final_val - init_cash
    net_profit_pct = (net_profit / init_cash) * 100

    tot_ret = (nav.iloc[-1] - 1.0) * 100
    gross_ret = cum_gross.iloc[-1] * 100
    bench_ret = (bench_nav.iloc[-1] - 1.0) * 100
    alpha_ret = tot_ret - bench_ret

    # 年化收益与波动率
    ann_ret = (nav.iloc[-1] ** (1.0 / years) - 1.0) * 100 if nav.iloc[-1] > 0 else 0.0
    ann_vol = net_ret.std() * np.sqrt(252) * 100
    bench_vol = df["bench"].std() * np.sqrt(252) * 100

    # 夏普比率 (无风险利率设为 2%)
    rf_daily = 0.02 / 252.0
    excess_ret = net_ret - rf_daily
    sharpe = (excess_ret.mean() / net_ret.std() * np.sqrt(252)) if net_ret.std() > 0 else 0.0

    # 最大回撤与卡玛比率 (复利净值口径: nav = (1+日净收益) 累乘)
    mdd_series = (nav / nav.cummax() - 1.0) * 100
    max_mdd = abs(mdd_series.min())
    calmar = (ann_ret / max_mdd) if max_mdd > 0 else 0.0

    # 胜率与交易统计
    pos_days = (net_ret > 0).sum()
    valid_days = (net_ret != 0).sum()
    win_rate = (pos_days / valid_days * 100) if valid_days > 0 else 0.0
    avg_turnover = df["turnover"].mean()
    ann_turnover = avg_turnover * 252
    avg_pos_ratio = (df["value"] / df["account"] * 100).mean()

    return {
        "init_cash": init_cash,
        "final_val": final_val,
        "peak_val": peak_val,
        "net_profit": net_profit,
        "net_profit_pct": net_profit_pct,
        "tot_ret": tot_ret,
        "gross_ret": gross_ret,
        "bench_ret": bench_ret,
        "alpha_ret": alpha_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "bench_vol": bench_vol,
        "sharpe": sharpe,
        "max_mdd": max_mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "total_days": total_days,
        "avg_turnover": avg_turnover,
        "ann_turnover": ann_turnover,
        "avg_pos_ratio": avg_pos_ratio,
        "cum_net": cum_net * 100,
        "cum_gross": cum_gross * 100,
        "cum_bench": cum_bench * 100,
        "cum_alpha": cum_alpha * 100,
        "mdd_series": mdd_series,
    }


def generate_all_in_one_dashboard(
    report_df: pd.DataFrame = None,
    positions: dict = None,
    save_path: Optional[Union[str, Path]] = None,
    benchmark: str = "SH000300",
    target_risk_degree: float = 0.95,
    report_normal_df: pd.DataFrame = None,
    **kwargs
):

    """生成机构级专业量化回测与交易全景大屏 (Professional Quant Backtest Dashboard)"""
    if report_df is None and report_normal_df is not None:
        report_df = report_normal_df
    if report_df is None or positions is None:
        return

    df = report_df.copy()
    dates = sorted(positions.keys()) if positions else list(df.index)
    kpis = _calc_kpis(df, positions, benchmark, target_risk_degree)

    # 提取持仓甘特图与市值堆叠数据
    holding_spans = defaultdict(list)
    cur_episodes = {}
    rows_amount = []

    for i, d in enumerate(dates):
        if d in positions:
            pos = positions[d]
            cash = pos.get_cash() if hasattr(pos, "get_cash") else 0
            stocks = pos.get_stock_list() if hasattr(pos, "get_stock_list") else []

            row = {"datetime": d, "可用现金 (Cash)": cash}
            for s in stocks:
                s_val = pos.position[s]["amount"] * pos.position[s]["price"] if hasattr(pos, "position") and s in pos.position else 0
                row[s] = s_val

                if s not in cur_episodes:
                    cur_episodes[s] = {"start": d, "days": 1}
                else:
                    cur_episodes[s]["days"] += 1

            for s in list(cur_episodes.keys()):
                if s not in stocks:
                    ep = cur_episodes.pop(s)
                    holding_spans[s].append((ep["start"], d, ep["days"]))
            rows_amount.append(row)

    for s, ep in cur_episodes.items():
        holding_spans[s].append((ep["start"], dates[-1], ep["days"]))

    df_amounts = pd.DataFrame(rows_amount).set_index("datetime").fillna(0) if rows_amount else pd.DataFrame()
    stock_cols = [c for c in df_amounts.columns if c != "可用现金 (Cash)"] if not df_amounts.empty else []

    # ==================================================================================
    # 🎨 图表 1: 核心收益与超额 Alpha 走势 (Hero Performance Chart)
    # ==================================================================================
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=df.index, y=kpis["cum_net"], name="策略净收益 (扣费后)",
        line=dict(color="#ef4444", width=2.5),
        hovertemplate="策略(扣费后): %{y:+.2f}%<extra></extra>"
    ))
    fig_perf.add_trace(go.Scatter(
        x=df.index, y=kpis["cum_gross"], name="策略毛收益 (扣费前)",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        hovertemplate="策略(扣费前): %{y:+.2f}%<extra></extra>"
    ))
    fig_perf.add_trace(go.Scatter(
        x=df.index, y=kpis["cum_bench"], name=f"基准指数 ({benchmark})",
        line=dict(color="#3b82f6", width=1.8),
        hovertemplate=f"基准: %{{y:+.2f}}%<extra></extra>"
    ))
    fig_perf.add_trace(go.Scatter(
        x=df.index, y=kpis["cum_alpha"], name="累计超额 Alpha",
        line=dict(color="#10b981", width=2),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.12)",
        hovertemplate="超额 Alpha: %{y:+.2f}%<extra></extra>"
    ))
    fig_perf.update_layout(
        height=450,
        margin=dict(l=50, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="累计收益率 (%)", showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
    )

    # ==================================================================================
    # 🎨 图表 2: 账户总资产变动曲线 (Total Account Value in RMB)
    # ==================================================================================
    fig_asset = go.Figure()
    fig_asset.add_trace(go.Scatter(
        x=df.index, y=df["account"], name="账户净资产 (¥)",
        line=dict(color="#8b5cf6", width=2.5),
        fill="tozeroy", fillcolor="rgba(139, 92, 246, 0.15)",
        hovertemplate="总资产: ¥%{y:,.2f}<extra></extra>"
    ))
    fig_asset.add_hline(
        y=kpis["init_cash"], line_dash="dash", line_color="#94a3b8",
        annotation_text=f"初始本金: ¥{kpis['init_cash']:,.0f}", annotation_position="bottom right"
    )
    fig_asset.update_layout(
        height=350,
        margin=dict(l=50, r=30, t=30, b=30),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="账户金额 (元)", showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
    )

    # ==================================================================================
    # 🎨 图表 3: 每日股票仓位水位与现金比例 (Position vs Cash Ratio)
    # ==================================================================================
    pos_ratio = (df["value"] / df["account"]) * 100
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        x=df.index, y=pos_ratio, name="实际股票仓位 (%)",
        line=dict(color="#059669", width=2),
        fill="tozeroy", fillcolor="rgba(5, 150, 105, 0.25)",
        hovertemplate="股票仓位: %{y:.1f}%<extra></extra>"
    ))
    fig_pos.add_hline(
        y=target_risk_degree * 100, line_dash="dot", line_color="#ea580c",
        annotation_text=f"目标上限 ({target_risk_degree*100:.0f}%)", annotation_position="top right"
    )
    fig_pos.add_hline(
        y=kpis["avg_pos_ratio"], line_dash="dash", line_color="#64748b",
        annotation_text=f"平均仓位 ({kpis['avg_pos_ratio']:.1f}%)", annotation_position="bottom right"
    )
    fig_pos.update_layout(
        height=350,
        margin=dict(l=50, r=30, t=30, b=30),
        hovermode="x unified",
        yaxis=dict(title="仓位占比 (%)", range=[0, 105], showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
    )

    # ==================================================================================
    # 🎨 图表 4: 水下动态回撤曲线 (Underwater Drawdown)
    # ==================================================================================
    fig_mdd = go.Figure()
    fig_mdd.add_trace(go.Scatter(
        x=df.index, y=kpis["mdd_series"], name="动态回撤 (%)",
        line=dict(color="#dc2626", width=1.8),
        fill="tozeroy", fillcolor="rgba(220, 38, 38, 0.25)",
        hovertemplate="回撤幅度: %{y:.2f}%<extra></extra>"
    ))
    fig_mdd.update_layout(
        height=320,
        margin=dict(l=50, r=30, t=30, b=30),
        hovermode="x unified",
        yaxis=dict(title="回撤幅度 (%)", showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
    )

    # ==================================================================================
    # 🎨 图表 5: 每日调仓换手率 (Turnover Rate)
    # ==================================================================================
    fig_to = go.Figure()
    fig_to.add_trace(go.Bar(
        x=df.index, y=df["turnover"], name="换手率",
        marker_color="rgba(99, 102, 241, 0.75)",
        hovertemplate="换手率: %{y:.2f}<extra></extra>"
    ))
    fig_to.add_hline(
        y=kpis["avg_turnover"], line_dash="dash", line_color="#4f46e5",
        annotation_text=f"平均: {kpis['avg_turnover']:.2f}", annotation_position="top right"
    )
    fig_to.update_layout(
        height=320,
        margin=dict(l=50, r=30, t=30, b=30),
        hovermode="x unified",
        yaxis=dict(title="换手率 (Turnover)", showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
    )

    # ==================================================================================
    # 🎨 图表 6: 每日持仓股票市值动态分布堆叠 (Portfolio Stacked Area)
    # ==================================================================================
    fig_stack = go.Figure()
    for col in stock_cols:
        fig_stack.add_trace(go.Scatter(
            x=df_amounts.index, y=df_amounts[col],
            mode="lines", stackgroup="amounts", name=col,
            hovertemplate=f"<b>{col}</b>: ¥%{{y:,.0f}}<extra></extra>"
        ))
    if not df_amounts.empty and "可用现金 (Cash)" in df_amounts:
        fig_stack.add_trace(go.Scatter(
            x=df_amounts.index, y=df_amounts["可用现金 (Cash)"],
            mode="lines", stackgroup="amounts", name="可用现金",
            line=dict(color="#cbd5e1"),
            hovertemplate="<b>现金</b>: ¥%{y:,.0f}<extra></extra>"
        ))
    fig_stack.update_layout(
        height=400,
        margin=dict(l=50, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        yaxis=dict(title="持仓市值 (元)", showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
    )

    # ==================================================================================
    # 🎨 图表 7: 股票持仓周期甘特图 (Holding Gantt Timeline)
    # ==================================================================================
    fig_gantt = go.Figure()
    y_labels = sorted(holding_spans.keys())
    for s, spans in holding_spans.items():
        for start_d, end_d, days in spans:
            fig_gantt.add_trace(go.Scatter(
                x=[start_d, end_d], y=[s, s],
                mode="lines",
                line=dict(color="#2563eb", width=12),
                hoverinfo="text",
                text=f"<b>{s}</b><br>买入: {start_d.strftime('%Y-%m-%d')}<br>清仓: {end_d.strftime('%Y-%m-%d')}<br>持仓: {days} 天",
                showlegend=False
            ))
    fig_gantt.update_layout(
        height=max(400, len(y_labels) * 22 + 100),
        margin=dict(l=80, r=30, t=30, b=30),
        yaxis=dict(title="持仓股票", automargin=True, showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(title="交易日期", showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
    )

    # 导出各个图表的 HTML 容器
    perf_html = fig_perf.to_html(include_plotlyjs="cdn", full_html=False)
    asset_html = fig_asset.to_html(include_plotlyjs=False, full_html=False)
    pos_html = fig_pos.to_html(include_plotlyjs=False, full_html=False)
    mdd_html = fig_mdd.to_html(include_plotlyjs=False, full_html=False)
    to_html = fig_to.to_html(include_plotlyjs=False, full_html=False)
    stack_html = fig_stack.to_html(include_plotlyjs=False, full_html=False)
    gantt_html = fig_gantt.to_html(include_plotlyjs=False, full_html=False)

    # ==================================================================================
    # 🏛️ 构建现代对冲基金级 HTML 仪表盘模板
    # ==================================================================================
    ret_color = "#10b981" if kpis["tot_ret"] >= 0 else "#ef4444"
    profit_sign = "+" if kpis["net_profit"] >= 0 else ""
    alpha_color = "#10b981" if kpis["alpha_ret"] >= 0 else "#ef4444"

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qlib 量化策略全景回测与交易看板</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8fafc; }}
        .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .metric-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; transition: transform 0.15s ease, box-shadow 0.15s ease; }}
        .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08); }}
    </style>
</head>
<body class="text-slate-800 pb-16">
    <!-- 顶部导航与策略头信息 -->
    <header class="bg-slate-900 text-white py-6 px-8 shadow-md mb-8">
        <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <span class="bg-emerald-500 text-slate-900 text-xs font-bold px-2.5 py-1 rounded">LIVE BACKTEST</span>
                    <h1 class="text-2xl font-bold tracking-tight">Qlib 自适应购买力选股实盘回测全景看板</h1>
                </div>
                <p class="text-slate-400 text-sm mt-1">
                    标的池: <span class="text-slate-200 font-medium">沪深300 (仅限主板)</span> | 
                    基准指数: <span class="text-slate-200 font-medium">{benchmark}</span> | 
                    样本外区间: <span class="text-slate-200 font-medium">{dates[0].strftime('%Y-%m-%d')} 至 {dates[-1].strftime('%Y-%m-%d')}</span> 
                    ({kpis['total_days']} 个交易日)
                </p>
            </div>
            <div class="flex items-center gap-3 bg-slate-800 px-4 py-2.5 rounded-lg border border-slate-700">
                <div class="text-right">
                    <div class="text-xs text-slate-400">最新账户总资产</div>
                    <div class="text-xl font-bold text-white">¥{kpis['final_val']:,.2f}</div>
                </div>
                <span class="text-xs font-semibold px-2 py-1 rounded {'bg-emerald-500/20 text-emerald-400' if kpis['net_profit'] >= 0 else 'bg-rose-500/20 text-rose-400'}">
                    {profit_sign}¥{kpis['net_profit']:,.2f} ({kpis['net_profit_pct']:+.2f}%)
                </span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 space-y-8">
        <!-- 1. 核心量化绩效指标网格 (8 KPI Cards) -->
        <section>
            <h2 class="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                <svg class="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                核心绩效与风险指标 (Executive Summary)
            </h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="metric-card border-l-4 border-l-emerald-500">
                    <div class="text-xs font-medium text-slate-500">策略累计净收益 (扣费后)</div>
                    <div class="text-2xl font-bold mt-1 text-slate-900" style="color: {ret_color};">{kpis['tot_ret']:+.2f}%</div>
                    <div class="text-xs text-slate-400 mt-1">扣费前: {kpis['gross_ret']:+.2f}%</div>
                </div>
                <div class="metric-card border-l-4 border-l-blue-500">
                    <div class="text-xs font-medium text-slate-500">沪深300基准收益</div>
                    <div class="text-2xl font-bold mt-1 text-blue-600">{kpis['bench_ret']:+.2f}%</div>
                    <div class="text-xs text-slate-400 mt-1">同期大盘表现</div>
                </div>
                <div class="metric-card border-l-4 border-l-teal-500">
                    <div class="text-xs font-medium text-slate-500">累计超额 Alpha</div>
                    <div class="text-2xl font-bold mt-1" style="color: {alpha_color};">{kpis['alpha_ret']:+.2f}%</div>
                    <div class="text-xs text-slate-400 mt-1">跑赢基准净超额</div>
                </div>
                <div class="metric-card border-l-4 border-l-purple-500">
                    <div class="text-xs font-medium text-slate-500">年化收益率 (CAGR)</div>
                    <div class="text-2xl font-bold mt-1 text-purple-600">{kpis['ann_ret']:+.2f}%</div>
                    <div class="text-xs text-slate-400 mt-1">年化波动: {kpis['ann_vol']:.2f}%</div>
                </div>
                <div class="metric-card border-l-4 border-l-rose-500">
                    <div class="text-xs font-medium text-slate-500">历史最大回撤 (Max DD)</div>
                    <div class="text-2xl font-bold mt-1 text-rose-600">-{kpis['max_mdd']:.2f}%</div>
                    <div class="text-xs text-slate-400 mt-1">卡玛比率 (Calmar): {kpis['calmar']:.2f}</div>
                </div>
                <div class="metric-card border-l-4 border-l-indigo-500">
                    <div class="text-xs font-medium text-slate-500">夏普比率 (Sharpe Ratio)</div>
                    <div class="text-2xl font-bold mt-1 text-indigo-600">{kpis['sharpe']:.2f}</div>
                    <div class="text-xs text-slate-400 mt-1">无风险利率: 2.0%</div>
                </div>
                <div class="metric-card border-l-4 border-l-amber-500">
                    <div class="text-xs font-medium text-slate-500">日收益胜率 (Win Rate)</div>
                    <div class="text-2xl font-bold mt-1 text-amber-600">{kpis['win_rate']:.1f}%</div>
                    <div class="text-xs text-slate-400 mt-1">上涨交易日占比</div>
                </div>
                <div class="metric-card border-l-4 border-l-slate-500">
                    <div class="text-xs font-medium text-slate-500">年化换手率 / 仓位</div>
                    <div class="text-2xl font-bold mt-1 text-slate-700">{kpis['ann_turnover']:.1f}x</div>
                    <div class="text-xs text-slate-400 mt-1">平均仓位: {kpis['avg_pos_ratio']:.1f}%</div>
                </div>
            </div>
        </section>

        <!-- 2. 主图：累计净值与超额 Alpha 走势 -->
        <section class="card p-6">
            <div class="flex justify-between items-center mb-4">
                <div>
                    <h3 class="text-base font-bold text-slate-900">① 策略累计收益率与超额走势 (Performance & Alpha Curve)</h3>
                    <p class="text-xs text-slate-500">包含策略扣费后净值、扣费前收益、沪深300基准以及累计超额 Alpha 面积图</p>
                </div>
            </div>
            {perf_html}
        </section>

        <!-- 3. 双列图表：总资产变动 与 仓位水位管理 -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="card p-6">
                <h3 class="text-base font-bold text-slate-900 mb-1">② 每日账户总资产金额变动 (¥ RMB)</h3>
                <p class="text-xs text-slate-500 mb-4">初始本金: ¥{kpis['init_cash']:,.0f} ➔ 历史最高: ¥{kpis['peak_val']:,.0f} ➔ 期末: ¥{kpis['final_val']:,.0f}</p>
                {asset_html}
            </div>
            <div class="card p-6">
                <h3 class="text-base font-bold text-slate-900 mb-1">③ 每日股票仓位水位与现金占比 (%)</h3>
                <p class="text-xs text-slate-500 mb-4">平均持仓水位: {kpis['avg_pos_ratio']:.1f}% | 目标风控上限: {target_risk_degree*100:.0f}%</p>
                {pos_html}
            </div>
        </section>

        <!-- 4. 双列图表：水下回撤曲线 与 调仓换手率 -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="card p-6">
                <h3 class="text-base font-bold text-slate-900 mb-1">④ 动态水下回撤曲线 (Underwater Plot)</h3>
                <p class="text-xs text-slate-500 mb-4">全周期最大回撤: -{kpis['max_mdd']:.2f}%</p>
                {mdd_html}
            </div>
            <div class="card p-6">
                <h3 class="text-base font-bold text-slate-900 mb-1">⑤ 每日调仓换手率与摩擦损耗</h3>
                <p class="text-xs text-slate-500 mb-4">日均换手率: {kpis['avg_turnover']:.2f} (年化: {kpis['ann_turnover']:.1f}x)</p>
                {to_html}
            </div>
        </section>

        <!-- 5. 每日持仓股票市值动态堆叠图 -->
        <section class="card p-6">
            <div class="mb-4">
                <h3 class="text-base font-bold text-slate-900">⑥ 投资组合资产配置动态分布 (Portfolio Stacked Area)</h3>
                <p class="text-xs text-slate-500">直观呈现每天具体是由哪只股票占用了多少资金、以及账户留存了多少可用现金</p>
            </div>
            {stack_html}
        </section>

        <!-- 6. 股票持仓周期甘特图 -->
        <section class="card p-6">
            <div class="mb-4">
                <h3 class="text-base font-bold text-slate-900">⑦ 股票持仓周期甘特图 (Holding Gantt Timeline)</h3>
                <p class="text-xs text-slate-500">展示每只股票的买入开仓日期、清仓日期以及单次持股天数 (共交易过 {len(holding_spans)} 只股票)</p>
            </div>
            {gantt_html}
        </section>
    </main>

    <footer class="text-center text-xs text-slate-400 mt-12">
        Qlib Quantitative Framework | AI Automated Strategy Analytics Dashboard
    </footer>
</body>
</html>
"""

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"       🌐 机构级专业量化回测全景大屏已成功生成至: {save_path}")

    return html_template

