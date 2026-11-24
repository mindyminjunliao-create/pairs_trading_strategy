"""Screen all formation-period pairs with a fixed Engle-Granger specification."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

ROOT = Path(__file__).resolve().parents[1]


def screen_pair(log_prices, ticker_a, ticker_b):
    """Return the test statistic and p-value with ticker_a as the dependent variable."""
    statistic, p_value, _ = coint(
        log_prices[ticker_a], log_prices[ticker_b],
        trend="c", maxlag=0, autolag=None,
    )
    return ticker_a, ticker_b, statistic, p_value


def main():
    prices = pd.read_csv(
        ROOT / "data/processed/formation_prices.csv",
        index_col="Date", parse_dates=True,
    )
    log_prices = np.log(prices.sort_index(axis=1))
    rows = []
    for number, (a, b) in enumerate(combinations(log_prices.columns, 2), start=1):
        rows.append(screen_pair(log_prices, a, b))
        if number % 20000 == 0:
            print(f"Completed {number:,} pairs", flush=True)

    results = pd.DataFrame(rows, columns=["ticker_a", "ticker_b", "statistic", "p_value"])
    if not np.isfinite(results["p_value"]).all():
        raise ValueError("Non-finite p-values require inspection before correction.")
    rejected, adjusted, _, _ = multipletests(
        results["p_value"], alpha=0.05, method="fdr_bh",
    )
    results["bh_adjusted_p"] = adjusted
    results["passes_raw_5pct"] = results["p_value"] < 0.05
    results["passes_bh_5pct"] = rejected
    results = results.sort_values(["p_value", "ticker_a", "ticker_b"])
    assert len(results) == len(prices.columns) * (len(prices.columns) - 1) // 2
    results.to_csv(ROOT / "outputs/cointegration_screen.csv", index=False)
    print(f"Pairs tested: {len(results):,}")
    print(f"Raw 5% candidates: {results.passes_raw_5pct.sum():,}")
    print(f"BH 5% candidates: {results.passes_bh_5pct.sum():,}")
    print(results.head(5).round(8).to_string(index=False))
    print(results.query("ticker_a == 'COP' and ticker_b == 'CVX'").to_string(index=False))


if __name__ == "__main__":
    main()
