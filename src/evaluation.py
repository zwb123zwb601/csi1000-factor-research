"""因子评估：IC / ICIR / 分层 / 相关性

【核心纪律】
  · ICIR 比 IC 重要
  · 多子区间比全样本重要（B35）
  · 新因子必须与已有因子相关 <0.7（B15）
"""
import numpy as np
import pandas as pd


def ic_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
              valid: pd.DataFrame, min_n: int = 30) -> pd.Series:
    """逐期截面 RankIC"""
    out = {}
    for dt in factor.index:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        v = valid.loc[dt]
        m = f.notna() & r.notna() & v.fillna(False)
        if m.sum() < min_n:
            continue
        out[dt] = f[m].corr(r[m], method="spearman")
    return pd.Series(out, dtype=float)


def ic_stats(ic: pd.Series) -> dict:
    """IC 统计"""
    ic = ic.dropna()
    if len(ic) < 12 or ic.std() == 0:
        return dict(IC=np.nan, ICIR=np.nan, 期数=len(ic))
    return dict(IC=ic.mean(), ICIR=ic.mean() / ic.std(), 期数=len(ic))


def layered_returns(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                    valid: pd.DataFrame, n_layer: int = 10) -> pd.DataFrame:
    """分层回测：每期按因子值分 n_layer 层，返回各层平均收益"""
    rows = {}
    for dt in factor.index:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        v = valid.loc[dt]
        m = f.notna() & r.notna() & v.fillna(False)
        if m.sum() < n_layer * 5:
            continue
        layer = pd.qcut(f[m].rank(method="first"), n_layer, labels=False)
        rows[dt] = r[m].groupby(layer).mean()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


def monotonicity(layered: pd.DataFrame) -> float:
    """分层单调性：各层平均收益与层序号的 Spearman 相关"""
    if layered.empty:
        return np.nan
    avg = layered.mean(axis=0)
    return float(pd.Series(avg.values).corr(pd.Series(avg.index.astype(float)), method="spearman"))


def factor_corr_streams(ic_a: pd.Series, ic_b: pd.Series) -> float:
    """两个因子「收益流」的相关（用 IC 序列近似）"""
    j = ic_a.dropna().index.intersection(ic_b.dropna().index)
    if len(j) < 24:
        return np.nan
    return float(np.corrcoef(ic_a.reindex(j), ic_b.reindex(j))[0, 1])
