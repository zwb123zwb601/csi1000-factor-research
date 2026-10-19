"""示例 3：B35 四步筛选法演示

演示：多子区间正比例 / 最小区间 / 与已有因子相关性 / 组合增量。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.screening import (b35_screen, step3_correlation,  # noqa: E402
                           step4_combo_increment)


def main():
    rng = np.random.default_rng(7)
    n = 140
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")

    # 构造三个假因子的 IC 序列
    # A：稳定有效（全期正）
    icA = pd.Series(rng.normal(0.03, 0.06, n), index=idx)
    # B：全期夏普更高，但只在部分区间有效
    icB = pd.Series(np.concatenate([
        rng.normal(0.09, 0.06, n // 2),
        rng.normal(-0.02, 0.06, n - n // 2),
    ]), index=idx)
    # C：几乎无效
    icC = pd.Series(rng.normal(0.005, 0.07, n), index=idx)

    # 注：待检验因子不放入 zoo，避免与自身比较得到 1.000
    zoo = {"A": icA}
    print("=" * 90)
    print("B35 四步筛选演示")
    print("=" * 90)
    print("因子 A：全期稳定 ｜ 因子 B：前半强后半弱 ｜ 因子 C：无效\n")
    for nm, ic in (("A", icA), ("B", icB), ("C", icC)):
        r = b35_screen(ic)
        print("  %-3s 全样本 ICIR %+6.3f ｜ 正比例 %3.0f%% ｜ 最小区间 %+7.3f ｜ 中位 %+7.3f ｜ 通过 %s"
              % (nm, r["ICIR"], 100 * r["正比例"], r["最小区间"], r["中位"],
                 "✅" if r["通过"] else "❌"))
    print("\n  ⭐ 注意：因子 B 的全样本 ICIR 可能最高，但「最小区间」为负 → 拒绝")
    print("     → 这就是 B35 的核心作用\n")

    # 第 ③ 步：相关性
    print("=== 第 ③ 步：与已有因子的相关性 ===")
    for nm, ic in (("B", icB), ("C", icC)):
        mc, which, ok = step3_correlation(ic, zoo)
        print("  %-3s 最大相关 %+.3f（%s）→ %s" % (nm, mc, which, "✅ 通过" if ok else "❌ 冗余"))

    # 第 ④ 步：组合增量
    print("\n=== 第 ④ 步：组合增量（每个子区间都需为正）===")
    base = pd.Series(rng.normal(0.005, 0.02, n), index=idx)
    for nm, ic in (("A", icA), ("B", icB)):
        new = base + ic * 0.3
        r = step4_combo_increment(base, new)
        print("  %-3s 增量均值 %+.5f ｜ 子区间增量 %s ｜ 全正 %s"
              % (nm, r.get("增量均值", np.nan),
                 [round(s, 5) for s in r.get("子区间增量", [])], r.get("全正")))


if __name__ == "__main__":
    main()
