"""B35 四步筛选法（本项目核心方法论）

① 全样本夏普/ICIR          → 粗筛
② 多子区间正比例 ≥80%      → ★ 核心
   且最小区间 > 0
③ 与已有因子相关 <0.7      → 去冗余
④ 组合增量 >0（每子区间）  → 最终确认

【关键实证】全样本夏普最高的因子，往往最不可用。
"""
import numpy as np
import pandas as pd


def b35_screen(ic: pd.Series, n_seg: int = 5, min_pos: float = 0.8,
               min_abs_icir: float = 0.05) -> dict:
    """单因子 B35 检验（第 ①② 步）"""
    ic = ic.dropna()
    if len(ic) < 40:
        return dict(通过=False, 原因="期数不足")
    n = len(ic)
    b = [int(round(i * n / n_seg)) for i in range(n_seg + 1)]
    seg_icir = []
    for i in range(n_seg):
        s = ic.iloc[b[i]:b[i + 1]]
        seg_icir.append(float(s.mean() / s.std()) if s.std() > 0 else np.nan)
    seg_icir = np.array(seg_icir)
    pos_rate = float(np.mean([v > 0 for v in seg_icir if not np.isnan(v)]))
    worst = float(np.nanmin(seg_icir))
    icir = float(ic.mean() / ic.std())
    ok = (abs(icir) >= min_abs_icir) and (pos_rate >= min_pos) and (worst > 0)
    return dict(ICIR=icir, 正比例=pos_rate, 最小区间=worst,
                中位=float(np.nanmedian(seg_icir)), 通过=ok)


def step3_correlation(ic_new: pd.Series, ic_zoo: dict, max_corr: float = 0.7):
    """第 ③ 步：与已有因子的最大相关"""
    best = 0.0
    best_f = ""
    for nm, ic in ic_zoo.items():
        j = ic_new.dropna().index.intersection(ic.dropna().index)
        if len(j) < 24:
            continue
        c = abs(float(np.corrcoef(ic_new.reindex(j), ic.reindex(j))[0, 1]))
        if c > best:
            best, best_f = c, nm
    return best, best_f, best < max_corr


def step4_combo_increment(base_stream: pd.Series, new_stream: pd.Series,
                          n_seg: int = 4) -> dict:
    """第 ④ 步：组合增量（每个子区间都为正才算通过）"""
    a = base_stream.dropna()
    b = new_stream.dropna()
    j = a.index.intersection(b.index)
    if len(j) < 24:
        return dict(通过=False, 原因="共同期数不足")
    a, b = a.reindex(j), b.reindex(j)
    inc = b - a                      # 增量流
    n = len(inc)
    bb = [int(round(i * n / n_seg)) for i in range(n_seg + 1)]
    segs = [float(inc.iloc[bb[i]:bb[i + 1]].mean()) for i in range(n_seg)]
    return dict(增量均值=float(inc.mean()), 子区间增量=segs,
                全正=all(s > 0 for s in segs), 通过=all(s > 0 for s in segs))
