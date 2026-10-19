"""因子构建与预处理

【核心纪律】
  · 因子必须做市值+行业中性化，否则挖到的是「小市值因子」
  · 处理顺序不能错：去极值 → 标准化 → 中性化
"""
import numpy as np
import pandas as pd


def mad_winsorize(x: pd.Series, k: float = 3.0) -> pd.Series:
    """MAD 去极值（比 3σ 稳健）"""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or not np.isfinite(mad):
        return x
    return x.clip(med - k * 1.4826 * mad, med + k * 1.4826 * mad)


def rank_normalize(x: pd.Series) -> pd.Series:
    """rank 百分位标准化（比 z-score 稳健）"""
    return x.rank(pct=True)


def neutralize(x: pd.Series, log_mv: pd.Series, industry: pd.Series) -> pd.Series:
    """市值+行业中性化：对 [log(市值), 行业哑变量] 回归取残差

    ★ 必须做！否则因子可能只是「小市值」或「行业」的代理变量
    """
    df = pd.DataFrame({"x": x, "lmv": log_mv, "ind": industry}).dropna()
    if len(df) < 30:
        return x
    X = pd.concat([df["lmv"],
                   pd.get_dummies(df["ind"], drop_first=True, dtype=float)], axis=1)
    X = np.column_stack([np.ones(len(X)), X.to_numpy()])
    y = df["x"].to_numpy()
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
    except np.linalg.LinAlgError:
        return x
    return pd.Series(resid, index=df.index).reindex(x.index)


def process_factor(x, log_mv, industry):
    """完整处理：去极值 → 标准化 → 中性化"""
    return neutralize(rank_normalize(mad_winsorize(x)), log_mv, industry)


# ---------- 常用因子示例 ----------

def momentum(ret: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """动量（取负 = 反转）"""
    return -ret.rolling(window).sum()


def volatility(ret: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """波动率"""
    return ret.rolling(window).std()


def amihud(ret: pd.DataFrame, amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Amihud 流动性（价格冲击）"""
    return (ret.abs() / amount.replace(0, np.nan)).rolling(window).mean()


def overnight_intraday(open_: pd.DataFrame, close: pd.DataFrame):
    """隔夜/日内拆分（B13）"""
    overnight = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    return overnight, intraday
