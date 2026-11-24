from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

ROOT = Path(__file__).resolve().parents[1]

screen = pd.read_csv(ROOT / "outputs/cointegration_screen.csv")
candidates = screen.loc[screen["passes_bh_5pct"]]

prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date",
    parse_dates=True,
).sort_index()

rows = []
for pair in candidates.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b

    for period, start, end in [
        ("Validation", "2022", "2023"),
        ("Test", "2024", "2024"),
    ]:
        window = prices.loc[start:end, [a, b]]
        valid = window.gt(0) & np.isfinite(window)
        complete = not window.empty and valid.all().all()

        p_value = np.nan
        if complete:
            logs = np.log(window)
            _, p_value, _ = coint(
                logs[a], logs[b],
                trend="c", maxlag=0, autolag=None,
            )

        rows.append({
            "ticker_a": a,
            "ticker_b": b,
            "period": period,
            "observations": len(window),
            "complete_data": complete,
            "p_value": p_value,
        })

results = pd.DataFrame(rows)
results.to_csv(ROOT / "outputs/pair_stability.csv", index=False)

for period, group in results.groupby("period", sort=False):
    tested = group["p_value"].notna().sum()
    passed = group["p_value"].lt(0.05).sum()
    print(f"{period}: {tested} tested, {passed} below raw p = 0.05")