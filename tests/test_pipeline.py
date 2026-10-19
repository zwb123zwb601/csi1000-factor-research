"""基础测试：确保核心函数可用"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import ic_series, ic_stats  # noqa: E402
from src.factors import (mad_winsorize, neutralize,  # noqa: E402
                         rank_normalize)
from src.screening import b35_screen  # noqa: E402


def test_mad_winsorize_clips_outliers():
    x = pd.Series([1, 2, 3, 4, 5, 1000])
    out = mad_winsorize(x)
    assert out.max() < 1000


def test_rank_normalize_range():
    x = pd.Series([10, 20, 30, 40, 50])
    out = rank_normalize(x)
    assert out.min() == pytest.approx(0.2)
    assert out.max() == pytest.approx(1.0)


def test_neutralize_removes_size_exposure():
    rng = np.random.default_rng(0)
    n = 200
    lmv = pd.Series(rng.normal(22, 1, n))
    ind = pd.Series(rng.choice(["a", "b", "c"], n))
    x = lmv * 0.5 + pd.Series(rng.normal(0, 0.1, n))     # 强市值暴露
    out = neutralize(x, lmv, ind)
    assert abs(np.corrcoef(out.dropna(), lmv[out.notna()])[0, 1]) < 0.1


def test_b35_rejects_one_sided_factor():
    """前半强、后半弱的因子应被拒绝"""
    rng = np.random.default_rng(1)
    n = 140
    ic = pd.Series(np.concatenate([rng.normal(0.09, 0.05, n // 2),
                                   rng.normal(-0.03, 0.05, n - n // 2)]))
    r = b35_screen(ic)
    assert r["最小区间"] < 0
    assert not r["通过"]


def test_b35_accepts_stable_factor():
    """全期稳定的因子应通过"""
    rng = np.random.default_rng(2)
    ic = pd.Series(rng.normal(0.05, 0.05, 140))
    r = b35_screen(ic)
    assert r["正比例"] >= 0.8
    assert r["通过"]
