"""示例 4：简单回测 + 四项验证演示

演示：分层回测 / 纯样本外 / 多子区间 / 压力测试 / 蒙特卡洛。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.portfolio import portfolio_stats, subperiod_stability  # noqa: E402
from src.validation import monte_carlo_dd, stress_correlation, train_test_split  # noqa: E402


def main():
    rng = np.random.default_rng(42)
    n = 140
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")

    # 模拟三条线的月频收益
    stock = pd.Series(rng.normal(0.006, 0.03, n), index=idx)
    comm = pd.Series(rng.normal(0.004, 0.02, n), index=idx)
    bond = pd.Series(rng.normal(0.002, 0.008, n), index=idx)
    R = pd.DataFrame({"股票": stock, "商品": comm, "国债": bond})

    # 风险平价
    vol = R.std()
    w = (1 / vol) / (1 / vol).sum()
    port = R.to_numpy() @ w.to_numpy()
    port = pd.Series(port, index=idx)

    print("=" * 90)
    print("回测与验证演示（模拟数据）")
    print("=" * 90)
    print("风险平价权重：%s" % {k: round(float(v), 3) for k, v in w.items()})
    print()
    print("=== ① 全样本统计 ===")
    for k, v in portfolio_stats(port).items():
        print("  %-8s %s" % (k, v))

    print("\n=== ② 纯样本外（训练 ≤2020-12-31）===")
    tr, te = train_test_split(port, "2020-12-31")
    for nm, s in (("训练", tr), ("测试", te)):
        st = portfolio_stats(s)
        print("  %-4s 期数 %3d ｜ 夏普 %+.3f ｜ 年化 %+.2f%% ｜ 回撤 %.1f%% ｜ 月胜 %.1f%%"
              % (nm, st["期数"], st["夏普"], st["年化"], st["最大回撤"], st["月胜率"]))

    print("\n=== ③ 多子区间稳定性（5 段）===")
    ss = subperiod_stability(port, n_seg=5)
    print("  各区间夏普 %s" % ss["子区间"])
    print("  正比例 %.0f%% ｜ 最小区间 %+.3f ｜ 中位 %+.3f"
          % (100 * ss["正比例"], ss["最小区间"], ss["中位"]))

    print("\n=== ④ 压力测试 ===")
    s = stress_correlation(R, rho=0.95)
    print("  相关性骤升 0.95 → 模拟夏普 %+.3f" % s["模拟夏普"])
    mc = monte_carlo_dd(port, horizon=12)
    print("  蒙特卡洛 12 个月 → 中位回撤 %.1f%% ｜ 99%%分位 %.1f%%" % (mc["中位"], mc["99分位"]))

    print("\n✅ 四项验证演示完成（真实项目应同时通过：24 项一致性 + 41 项端到端）")


if __name__ == "__main__":
    main()
