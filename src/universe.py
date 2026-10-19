"""股票池构建 —— 中证1000 动态成分

【核心纪律】股票池必须与策略池一致。用全市场算因子、用中证1000选股，
回测结果是假的（本项目踩过此坑）。
"""
import numpy as np
import pandas as pd


def build_universe(rebal_dates, all_codes, mask, float_mv,
                   exclude_top=800, n=1000):
    """构造中证1000 动态股票池。

    Parameters
    ----------
    rebal_dates : list of Timestamp  调仓日
    all_codes : list                 全部股票代码
    mask : DataFrame (date × code)   可交易掩码（未停牌/未ST/上市满N日）
    float_mv : DataFrame             流通市值
    exclude_top : int                剔除市值最大的前 N 只
    n : int                          取接下来的 n 只

    Returns
    -------
    uni : ndarray (T, N) bool   True = 在池内
    """
    # 剔除前 exclude_top 大市值，取接下来的 n 只（近似中证1000）
    uni = np.zeros((len(rebal_dates), len(all_codes)), dtype=bool)
    for i, dt in enumerate(rebal_dates):
        ok = mask.loc[dt].reindex(all_codes).fillna(False).to_numpy()
        mv = float_mv.loc[dt].reindex(all_codes).to_numpy(dtype=float)
        rank = pd.Series(np.where(ok, mv, np.nan)).rank(ascending=False).to_numpy()
        uni[i] = (rank > exclude_top) & (rank <= exclude_top + n)
    return uni
