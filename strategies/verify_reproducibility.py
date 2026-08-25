# 验证 seed 修复后实验可复现性: 用完全相同配置重跑 A 组, 对比两次结果
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import copy

sys.path.append(str(Path(__file__).resolve().parent))

import qlib
from qlib.constant import REG_CN
from qlib.tests.data import GetData
from components import paths

# 初始化 qlib 环境 (与 run_model_experiments.main 一致)
provider_uri = "~/.qlib/qlib_data/cn_data"
GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
qlib.init(provider_uri=provider_uri, region=REG_CN)

from run_model_experiments import run_single_experiment, EXPERIMENTS

info = copy.deepcopy(EXPERIMENTS["exp_a_baseline"])
r = run_single_experiment("exp_a_baseline_rerun", info, paths)

print("\n" + "=" * 70)
print(" 🧪 seed 复现性验证结果")
print("=" * 70)
print(f"  本次重跑 A 组:  IC={r['mean_ic']:.4f} | 总收益={r['tot_ret']:+.2f}% | 回撤=-{r['max_mdd']:.2f}%")
print(f"  首次运行 A 组:  IC=0.0268 | 总收益=+26.07% | 回撤=-34.47%")
ic_match = abs(r["mean_ic"] - 0.0268) < 1e-4
ret_match = abs(r["tot_ret"] - 26.07) < 0.5
mdd_match = abs(r["max_mdd"] - 34.47) < 0.5
print(f"\n  IC 一致: {ic_match} | 收益一致: {ret_match} | 回撤一致: {mdd_match}")
print("  ✅ seed 修复生效, 实验可复现" if (ic_match and ret_match and mdd_match) else "  ❌ 仍存在随机性, 需进一步排查")
