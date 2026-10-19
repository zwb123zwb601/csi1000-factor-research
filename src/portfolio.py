"""组合构建：加权、优化、风险平价

【核心纪律】
  · 因子数量要与加权方法匹配（B36）
  · 风险平价最稳健；权重优化的过拟合风险高（B37）
"""
import numpy as np
import pandas as pd


def icir_weight(ic_dict: dict) -> pd.Series:
    """ICIR 加权（线性方法，必须精选因子）"""
    w = {}
    for nm, ic in ic_dict.items():
        ic = ic.dropna()
        if len(ic) < 12 or ic.std() == 0:
            continue
        w[nm] = ic.mean() / ic.std()
    s = pd.Series(w)
    return s / s.abs().sum()


def risk_parity(returns: pd.DataFrame) -> pd.Series:
    """风险平价（按波动倒数）"""
    vol = returns.std()
    w = 1.0 / vol.replace(0, np.nan)
    w = w.fillna(0)
    return w / w.sum()


def equal_weight(n: int) -> pd.Series:
    return pd.Series(np.ones(n) / n, index=range(n))


def portfolio_stats(returns: pd.Series, ppy: int = 12) -> dict:
    """组合统计"""
    r = returns.dropna()
    if len(r) < 6:
        return {}
    ann = (1 + r.mean()) ** ppy - 1
    vol = r.std() * np.sqrt(ppy)
    nav = (1 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    return dict(期数=len(r), 年化=round(100 * ann, 2), 波动=round(100 * vol, 2),
                夏普=round(ann / vol, 3) if vol > 0 else np.nan,
                最大回撤=round(100 * mdd, 1),
                月胜率=round(100 * (r > 0).mean(), 1))


def subperiod_stability(returns: pd.Series, n_seg: int = 5, ppy: int = 12) -> dict:
    """多子区间稳定性（B35 精神）"""
    r = returns.dropna()
    if len(r) < n_seg * 4:
        return {}
    n = len(r)
    b = [int(round(i * n / n_seg)) for i in range(n_seg + 1)]
    ss = []
    for i in range(n_seg):
        s = r.iloc[b[i]:b[i + 1]]
        ss.append(float(s.mean() / s.std() * np.sqrt(ppy)) if s.std() > 0 else np.nan)
    ss = np.array(ss)
    return dict(子区间=ss.round(3).tolist(),
                正比例=round(float(np.mean([v > 0 for v in ss if not np.isnan(v)])), 2),
                最小区间=round(float(np.nanmin(ss)), 3),
                中位=round(float(np.nanmedian(ss)), 3))
