#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
========================================================================================
Qlib 投资组合全合一看板与交割单生成工具 (All-in-One Dashboard & Delivery Slip CLI)
========================================================================================
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
import pickle
import argparse
from pathlib import Path

# 添加当前目录以支持 components 模块导入
sys.path.append(str(Path(__file__).resolve().parent))

import qlib
from qlib.constant import REG_CN
from qlib.workflow import R

from components import (
    generate_daily_delivery_slip,
    generate_all_in_one_dashboard,
    paths,
)


def main():
    parser = argparse.ArgumentParser(description="生成全合一回测全景交互看板与交割单")
    parser.add_argument("--exp_name", type=str, default="affordable_topk_workflow", help="实验名称")
    parser.add_argument("--pos_path", type=str, default=None, help="直接指定 positions_normal_1day.pkl 路径")
    parser.add_argument("--report_path", type=str, default=None, help="直接指定 report_normal_1day.pkl 路径")
    parser.add_argument("--output_html", type=str, default=None, help="看板 HTML 输出路径")
    parser.add_argument("--output_csv", type=str, default=None, help="交割单 CSV 输出路径")
    args = parser.parse_args()

    html_out = args.output_html or str(paths.root / "backtest_dashboard.html")
    csv_out = args.output_csv or str(paths.root / "daily_delivery_slip.csv")

    positions = None
    report_df = None

    if args.pos_path and Path(args.pos_path).exists():
        with open(args.pos_path, "rb") as f:
            positions = pickle.load(f)
    if args.report_path and Path(args.report_path).exists():
        with open(args.report_path, "rb") as f:
            report_df = pickle.load(f)

    if positions is None or report_df is None:
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
        try:
            exp = R.get_exp(experiment_name=args.exp_name)
            recorders = exp.list_recorders(rtype=exp.RT_L)
            if recorders:
                latest_recorder = recorders[0]
                if positions is None:
                    positions = latest_recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
                if report_df is None:
                    report_df = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
                print(f"成功从实验 [{args.exp_name}] (Recorder ID: {latest_recorder.id}) 加载回测与持仓记录！")
        except Exception as e:
            print(f"未能自动从实验加载: {e}")

    if positions is None or report_df is None:
        print("未找到有效回测记录！")
        return

    # 1. 生成全合一 HTML 看板
    generate_all_in_one_dashboard(report_df=report_df, positions=positions, save_path=html_out)

    # 2. 导出每日真实交易交割流水单 (CSV)
    generate_daily_delivery_slip(positions, save_csv_path=csv_out)


if __name__ == "__main__":
    main()
