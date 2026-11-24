from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine import backtest_pair, CAPITAL
from pair_model import fit_pair, make_signals

ROOT = Path(__file__).resolve().parents[1]

prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date",
    parse_dates=True,
).sort_index()

screen = pd.read_csv(ROOT / "outputs/cointegration_screen.csv")
pool = screen.loc[screen["bh_adjusted_p"] < 0.10].reset_index(drop=True)

selected_indices = {
    cutoff: pool.index[pool["bh_adjusted_p"] < cutoff].tolist()
    for cutoff in [0.01, 0.05, 0.10]
}

# Prepare each pair model once per estimation window.
prepared = {}

for pair_id, pair in pool.iterrows():
    a, b = pair["ticker_a"], pair["ticker_b"]
    sample = prices.loc["2018":"2023", [a, b]]

    if (
        sample.isna().any().any()
        or not np.isfinite(sample).all().all()
        or (sample <= 0).any().any()
    ):
        raise ValueError(f"Inspect price coverage for {a} and {b}.")

    for lookback in [126, 252, 504, 1008]:
        model = fit_pair(sample.loc["2018":"2021"], a, b, lookback)
        signals = make_signals(sample, model).loc["2022":"2023"]
        prepared[pair_id, lookback] = (
            sample.loc[signals.index],
            signals,
            model["beta"],
        )


@lru_cache(maxsize=256)
def run_pair(pair_id, lookback, days, entry, exit_band, stop):
    pair_prices, signals, beta = prepared[pair_id, lookback]
    nav, trades = backtest_pair(
        pair_prices,
        signals,
        beta,
        max_days=days,
        entry_z=entry,
        exit_z=exit_band,
        stop_loss=stop,
        fee_rate=0.0005,
    )
    return nav.to_numpy() / CAPITAL, trades


rows = []

# Keep cutoff innermost so overlapping portfolios reuse pair backtests.
settings = product(
    [126, 252, 504, 1008],
    [10, 20, 60],
    [1.5, 2.0, 2.5],
    [0.0, 0.25, 0.5],
    [None, 0.03, 0.06],
    [0.01, 0.05, 0.10],
)

for number, (lookback, days, entry, exit_band, stop, cutoff) in enumerate(
    settings, start=1
):
    runs = [
        run_pair(pair_id, lookback, days, entry, exit_band, stop)
        for pair_id in selected_indices[cutoff]
    ]

    equity = np.mean([curve for curve, _ in runs], axis=0)
    values = np.r_[1.0, equity]
    if (values <= 0).any():
        raise ValueError("Nonpositive portfolio equity requires review.")

    returns = np.diff(values) / values[:-1]
    daily_sd = returns.std(ddof=1)

    rows.append({
        "bh_cutoff": cutoff,
        "lookback": lookback,
        "max_days": days,
        "entry_z": entry,
        "exit_z": exit_band,
        "stop_loss": "None" if stop is None else str(stop),
        "pairs": len(runs),
        "trades": sum(count for _, count in runs),
        "total_return": equity[-1] - 1,
        "max_drawdown": (
            values / np.maximum.accumulate(values) - 1
        ).min(),
        "sharpe": (
            returns.mean() / daily_sd * np.sqrt(252)
            if daily_sd > 0 else np.nan
        ),
    })

    if number % 108 == 0:
        print(f"Completed {number}/972 configurations", flush=True)

results = pd.DataFrame(rows)
assert len(results) == 972

# Confirm that the joint search reproduces a previous experiment.
reference = results.loc[
    results["bh_cutoff"].eq(0.05)
    & results["lookback"].eq(1008)
    & results["max_days"].eq(60)
    & results["entry_z"].eq(2.0)
    & results["exit_z"].eq(0.25)
    & results["stop_loss"].eq("0.03")
].iloc[0]

assert np.isclose(reference["total_return"], 0.03976147, atol=0.000001)

results.to_csv(ROOT / "outputs/joint_validation_grid.csv", index=False)

ranked = results.loc[
    results["trades"].ge(5) & np.isfinite(results["sharpe"])
].sort_values("sharpe", ascending=False, kind="stable")

ranked.head(1).to_csv(
    ROOT / "outputs/joint_validation_winner.csv", index=False
)

print("\nTop 10 validation configurations:")
print(ranked.head(10).round(4).to_string(index=False))