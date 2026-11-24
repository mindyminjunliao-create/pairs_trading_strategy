from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine import backtest_pair, CAPITAL
from pair_model import fit_pair, make_signals

ROOT = Path(__file__).resolve().parents[1]

winner = pd.read_csv(
    ROOT / "outputs/joint_validation_winner.csv",
    keep_default_na=False,
).iloc[0]

stop = (
    None if winner["stop_loss"] == "None"
    else float(winner["stop_loss"])
)

screen = pd.read_csv(ROOT / "outputs/cointegration_screen.csv")
selected = screen.loc[
    screen["bh_adjusted_p"] < float(winner["bh_cutoff"])
]

prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date", parse_dates=True,
).sort_index()

curves = []
contributions = []

for pair in selected.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b
    sample = prices.loc["2018":"2024", [a, b]]

    if (
        sample.isna().any().any()
        or not np.isfinite(sample).all().all()
        or (sample <= 0).any().any()
    ):
        raise ValueError(f"Inspect prices for {a} and {b}.")

    model = fit_pair(
        sample.loc["2018":"2021"],
        a, b, int(winner["lookback"]),
    )
    signals = make_signals(sample, model).loc["2024"]

    nav, count = backtest_pair(
        sample.loc[signals.index],
        signals,
        model["beta"],
        max_days=int(winner["max_days"]),
        entry_z=float(winner["entry_z"]),
        exit_z=float(winner["exit_z"]),
        stop_loss=stop,
        fee_rate=0.0005,
    )

    curves.append(nav / CAPITAL)
    contributions.append({
        "pair": f"{a}-{b}",
        "trades": count,
        "sleeve_return": nav.iloc[-1] / CAPITAL - 1,
    })

unit_equity = pd.concat(curves, axis=1).mean(axis=1)
returns = unit_equity / unit_equity.shift(1).fillna(1.0) - 1
peak = unit_equity.cummax().clip(lower=1.0)
sd = returns.std(ddof=1)

summary = pd.DataFrame([{
    "pairs": len(selected),
    "trades": sum(row["trades"] for row in contributions),
    "total_return": unit_equity.iloc[-1] - 1,
    "max_drawdown": (unit_equity / peak - 1).min(),
    "sharpe": returns.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
}])

summary.to_csv(ROOT / "outputs/joint_winner_test_metrics.csv", index=False)
unit_equity.to_csv(
    ROOT / "outputs/joint_winner_test_equity.csv",
    header=["unit_equity"],
)

print(pd.DataFrame(contributions).round(4).to_string(index=False))
print(summary.round(4).to_string(index=False))