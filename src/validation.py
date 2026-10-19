"""验证：样本外 / 多子区间 / 压力测试

【核心纪律】每个结论都要独立复算（B2）；验证脚本本身也要被验证。
"""
import numpy as np
import pandas as pd


def train_test_split(returns: pd.Series, split_date: str):
    """训练/测试切分"""
    d = pd.Timestamp(split_date)
    return returns[returns.index <= d], returns[returns.index > d]


def stress_correlation(returns_df: pd.DataFrame, rho: float = 0.95,
                       n_sim: int = 2000, seed: int = 42) -> dict:
    """相关性骤升压力测试"""
    rng = np.random.default_rng(seed)
    mu = returns_df.mean().to_numpy()
    sd = returns_df.std().to_numpy()
    k = len(mu)
    C = np.full((k, k), rho)
    np.fill_diagonal(C, 1.0)
    L = np.linalg.cholesky(C)
    x = rng.standard_normal((n_sim, k)) @ L.T
    # ★ 保留一阶矩（B19）：用 mu + sd*x 而非仅 sd*x
    sim = mu + x * sd
    w = (1 / sd) / (1 / sd).sum()
    port = sim @ w
    sharpe = port.mean() / port.std() * np.sqrt(12) if port.std() > 0 else np.nan
    return dict(模拟夏普=round(float(sharpe), 3))


def monte_carlo_dd(returns: pd.Series, n_sim: int = 2000, horizon: int = 12,
                   seed: int = 42) -> dict:
    """蒙特卡洛最大回撤分布"""
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    dds = []
    for _ in range(n_sim):
        s = rng.choice(r, size=horizon, replace=True)
        nav = np.cumprod(1 + s)
        dds.append((nav / np.maximum.accumulate(nav) - 1).min())
    dds = np.array(dds)
    return dict(中位=round(100 * float(np.median(dds)), 1),
                **{"99分位": round(100 * float(np.percentile(dds, 1)), 1)})


def check_expectation(name: str, actual, expected, tol=None) -> dict:
    """一致性检查单条"""
    if expected is None:
        ok = actual is not None
    elif tol is None:
        ok = str(actual) == str(expected)
    else:
        try:
            ok = abs(float(actual) - float(expected)) <= tol
        except (TypeError, ValueError):
            ok = False
    return dict(项=name, 实测=actual, 期望=expected, 判定="✅" if ok else "❌")
