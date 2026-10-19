"""示例 2：从日线数据构建因子并做中性化

演示：动量 / 反转 / 波动 / 换手 / Amihud，以及 去极值→标准化→中性化。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factors import (amihud, mad_winsorize, momentum, neutralize,  # noqa: E402
                         rank_normalize, volatility)


def main():
    # 构造一个假的多股票面板（真实场景从 CSV/parquet 读）
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    codes = ["A%03d" % i for i in range(50)]
    ret = pd.DataFrame(rng.normal(0.0005, 0.02, (len(dates), len(codes))),
                       index=dates, columns=codes)
    px = (1 + ret).cumprod() * 10
    amount = pd.DataFrame(rng.lognormal(18, 0.5, (len(dates), len(codes))),
                          index=dates, columns=codes)
    mv = pd.DataFrame(rng.lognormal(22, 0.8, (len(dates), len(codes))),
                      index=dates, columns=codes)
    industry = pd.Series(rng.choice(["银行", "医药", "电子", "化工", "食品"], len(codes)),
                         index=codes)

    # 构造因子
    fac_raw = {
        "MOM_20": momentum(ret, 20),
        "VOL_21": volatility(ret, 21),
        "AMIHUD_20": amihud(ret, amount, 20),
        "TURN_20": amount.rolling(20).mean() / mv,
    }
    print("原始因子 %d 个" % len(fac_raw))

    # 取最后一个截面做处理演示
    dt = dates[-1]
    print("\n截面日期：%s" % dt.date())
    for nm, f in fac_raw.items():
        x = f.loc[dt].dropna()
        if len(x) < 20:
            continue
        x1 = mad_winsorize(x)
        x2 = rank_normalize(x1)
        x3 = neutralize(x2, np.log(mv.loc[dt]), industry)
        print("  %-10s 原始 std %.4f → 去极值 %.4f → 标准化 %.4f → 中性化 %.4f"
              % (nm, x.std(), x1.std(), x2.std(), x3.std()))
    print("\n✅ 处理流程演示完成（去极值 → 标准化 → 中性化）")


if __name__ == "__main__":
    main()
