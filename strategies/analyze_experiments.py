#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
多实验深度分析与归因对比脚本
========================================================================
在 run_model_experiments.py 跑完后运行，基于各实验的 mlruns 信号记录 + 交割单，
输出信号层、组合层、交易行为层全维度对比，并保存至 outputs/reports/ 目录。
========================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保 components 模块可导入
sys.path.append(str(Path(__file__).resolve().parent))
from components import paths, get_paths, ExperimentPaths

EXPERIMENTS = {
    "exp_a_baseline": {
        "title": "A组 基线(1日标签+17年, 带风控)",
    },
    "exp_b_label5d": {
        "title": "B组 5日标签+17年, 带风控",
    },
    "exp_c_label5d_roll3y": {
        "title": "C组 5日标签+3年, 带风控",
    },
    "exp_d_label5d_rs": {
        "title": "D组 5日标签+17年+RS, 带风控",
    },
    "exp_e_daily_rotate": {
        "title": "E组 1日标签+17年, 日频满仓轮动(无风控)",
    },
    "exp_f_double_ensemble": {
        "title": "F组 5日标签+17年+DoubleEnsemble, 带风控",
    },
}


def find_workflow_dir(exp_key: str) -> Path:
    """在 mlruns 中按 experiment 名匹配 workflow 实验记录, 返回包含 artifacts 的 run 目录"""
    # 优先在 repo 根目录与当前目录下查找 mlruns
    for candidate_root in [Path(__file__).resolve().parent.parent / "mlruns", Path("mlruns")]:
        if candidate_root.exists():
            for exp_dir in candidate_root.iterdir():
                meta = exp_dir / "meta.yaml"
                if not meta.exists():
                    continue
                try:
                    exp_name = meta.read_text(encoding="utf-8", errors="ignore").split("name: ")[1].split("\n")[0].strip()
                except Exception:
                    continue
                if exp_name != f"workflow_{exp_key}":
                    continue
                best = None
                for run_dir in exp_dir.iterdir():
                    if run_dir.is_dir() and (run_dir / "artifacts").exists():
                        if best is None or run_dir.stat().st_mtime > best.stat().st_mtime:
                            best = run_dir
                if best is not None:
                    return best
    return None


def load_ic(run_dir: Path):
    ic = pickle.load(open(run_dir / "artifacts/sig_analysis/ic.pkl", "rb"))
    ric = pickle.load(open(run_dir / "artifacts/sig_analysis/ric.pkl", "rb"))
    return ic.dropna(), ric.dropna()


def analyze_slip(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty or "交易日期" not in df.columns:
        return None
    df["交易日期"] = pd.to_datetime(df["交易日期"])
    assets = (
        df[["交易日期", "日末账户总资产 (¥)"]]
        .drop_duplicates("交易日期")
        .set_index("交易日期")["日末账户总资产 (¥)"]
    )
    nav = assets / assets.iloc[0]
    mdd = (nav / nav.cummax() - 1.0) * 100
    buys = df[df["操作方向"].str.contains("买入开仓")]
    sells = df[df["操作方向"].str.contains("卖出平仓")]
    cash_days = df[df["操作方向"].str.contains("空仓")]
    # 持仓周期
    hold_days = []
    for _, s in sells.iterrows():
        prev = buys[(buys["证券代码"] == s["证券代码"]) & (buys["交易日期"] <= s["交易日期"])]
        if len(prev) > 0:
            hold_days.append((s["交易日期"] - prev.iloc[-1]["交易日期"]).days)
    hd = pd.Series(hold_days) if hold_days else pd.Series(dtype=float)
    monthly = nav.resample("ME").last().pct_change().dropna() * 100
    return {
        "tot_ret": (nav.iloc[-1] - 1) * 100,
        "mdd": mdd.min(),
        "n_buys": len(buys),
        "n_sells": len(sells),
        "cash_days": len(cash_days),
        "hold_mean": hd.mean() if len(hd) else float("nan"),
        "hold_median": hd.median() if len(hd) else float("nan"),
        "monthly": monthly,
    }


def main():
    print("=" * 100)
    print(" 🔬 模型层对照实验深度分析 (Deep Attribution & Signals Analysis)")
    print("=" * 100)

    exp_paths = get_paths()
    rows = []
    for key, info in EXPERIMENTS.items():
        print(f"\n### {info['title']} ({key})")
        wdir = find_workflow_dir(key)
        if wdir is None or not wdir.exists():
            print("  ⚠️ 未找到 workflow 记录, 跳过信号层分析")
            ic = ric = None
        else:
            try:
                ic, ric = load_ic(wdir)
                print(f"  IC={ic.mean():.4f} ICIR={ic.mean()/ic.std():.3f} IC>0={(ic>0).mean()*100:.0f}% | RankIC={ric.mean():.4f}")
                mic = ic.groupby(ic.index.to_period("M")).mean()
                print("  月度IC:", " ".join(f"{d}:{v:.3f}" for d, v in mic.items()))
            except Exception as e:
                print(f"  ⚠️ 信号读取失败: {e}")
                ic = ric = None

        # 查找交割单路径 (优先 experiments/ 目录, 兼容旧 outputs/ 目录)
        slip_p = exp_paths.get_exp_file(key, "daily_delivery_slip.csv")
        if not slip_p.exists():
            slip_p = exp_paths.root / key / "daily_delivery_slip.csv"

        s = analyze_slip(slip_p)
        if s is not None:
            print(f"  收益 {s['tot_ret']:+.2f}% | 回撤 {s['mdd']:.2f}% | 买{s['n_buys']}卖{s['n_sells']} | 空仓{s['cash_days']}天 | 持仓均值{s['hold_mean']:.1f}天")
            print("  月度收益:", " ".join(f"{d.strftime('%y%m')}:{v:+.1f}%" for d, v in s["monthly"].items()))
            rows.append({**info, "key": key, "ic": ic, "ric": ric, **s})
        else:
            print(f"  ⚠️ 未找到交割单: {slip_p}")

    if not rows:
        print("\n⚠️ 未收集到任何实验分析数据。")
        return

    # 汇总表
    print("\n" + "=" * 100)
    print(" 📊 汇总对比")
    print("=" * 100)
    summ = pd.DataFrame(
        [
            {
                "实验": r["title"],
                "总收益%": round(r["tot_ret"], 2),
                "最大回撤%": round(r["mdd"], 2),
                "IC": round(r["ic"].mean(), 4) if r["ic"] is not None else None,
                "RankIC": round(r["ric"].mean(), 4) if r["ric"] is not None else None,
                "ICIR": round(r["ic"].mean() / r["ic"].std(), 3) if r["ic"] is not None else None,
                "买卖次数": f"{r['n_buys']}/{r['n_sells']}",
                "空仓天数": r["cash_days"],
                "持仓均值(天)": round(r["hold_mean"], 1),
            }
            for r in rows
        ]
    )
    print(summ.to_string(index=False))

    out = exp_paths.reports_dir / "experiment_deep_analysis.csv"
    summ.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n ✅ 深度分析表已保存: {out}")


if __name__ == "__main__":
    main()
