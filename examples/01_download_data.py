"""示例 1：下载数据（akshare）

注意：akshare 接口可能变动，请以官方文档为准。
"""
import os

import pandas as pd

OUT = "data"
os.makedirs(OUT, exist_ok=True)


def main():
    import akshare as ak

    # 1. 中证1000 历史成分（示例：当前成分，实盘需历史快照）
    print("下载中证1000 成分 …")
    try:
        df = ak.index_stock_cons_csindex(symbol="000852")
        df.to_csv(f"{OUT}/zz1000_cons.csv", index=False, encoding="utf-8-sig")
        print("  成分股 %d 只 → %s/zz1000_cons.csv" % (len(df), OUT))
    except Exception as e:                                       # noqa: BLE001
        print("  失败：%s" % str(e)[:80])

    # 2. 个股日线（示例：取 3 只）
    codes = ["000001", "600000", "000002"]
    for c in codes:
        print("下载 %s 日线 …" % c)
        try:
            d = ak.stock_zh_a_hist(symbol=c, period="daily",
                                   start_date="20150101", end_date="20261231",
                                   adjust="hfq")
            d.to_csv(f"{OUT}/{c}.csv", index=False, encoding="utf-8-sig")
            print("  %d 行 → %s/%s.csv" % (len(d), OUT, c))
        except Exception as e:                                   # noqa: BLE001
            print("  失败：%s" % str(e)[:60])

    # 3. 指数日线
    print("下载中证1000 指数 …")
    try:
        idx = ak.index_zh_a_hist(symbol="000852", period="daily",
                                 start_date="20150101", end_date="20261231")
        idx.to_csv(f"{OUT}/zz1000_index.csv", index=False, encoding="utf-8-sig")
        print("  %d 行 → %s/zz1000_index.csv" % (len(idx), OUT))
    except Exception as e:                                       # noqa: BLE001
        print("  失败：%s" % str(e)[:80])


if __name__ == "__main__":
    main()
