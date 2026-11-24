from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]

screen = pd.read_csv(ROOT / "outputs/cointegration_screen.csv")
candidates = screen.loc[screen["passes_bh_5pct"]].sort_values("p_value")
a, b = candidates.iloc[0][["ticker_a", "ticker_b"]]

prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date",
    parse_dates=True,
).sort_index()

pair = prices.loc["2018":"2024", [a, b]]
if pair.isna().any().any() or not np.isfinite(pair).all().all():
    raise ValueError("Inspect missing or non-finite prices before proceeding.")
if (pair <= 0).any().any():
    raise ValueError("Log prices require positive prices.")

logs = np.log(pair)
formation = logs.loc["2018":"2021"]

model = sm.OLS(
    formation[a],
    sm.add_constant(formation[b]),
).fit()

spread = logs[a] - model.params["const"] - model.params[b] * logs[b]
formation_spread = spread.loc["2018":"2021"]
spread_mean = formation_spread.mean()
spread_sd = formation_spread.std()

if spread_sd <= 0:
    raise ValueError("Spread standard deviation must be positive.")

z = (spread - spread_mean) / spread_sd

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

spread.plot(ax=axes[0], color="teal")
axes[0].axhline(spread_mean, color="gray", linestyle="--")
axes[0].set_title(f"{a}–{b}: spread with frozen formation coefficients")
axes[0].set_ylabel("Log-price spread")

z.plot(ax=axes[1], color="navy")
for level in [-2, 2]:
    axes[1].axhline(level, color="red", linestyle="--")
for level in [-0.5, 0.5]:
    axes[1].axhline(level, color="gray", linestyle=":")
axes[1].set_ylabel("Z-score")
axes[1].set_xlabel("Date")

for ax in axes:
    ax.axvline(pd.Timestamp("2022-01-01"), color="black", linestyle="--")
    ax.axvline(pd.Timestamp("2024-01-01"), color="black", linestyle=":")

fig.tight_layout()
fig.savefig(ROOT / "outputs/selected_pair_spread.png", dpi=160)
plt.close(fig)

print("Selected pair:", a, b)
print("Hedge ratio:", round(model.params[b], 4))
print("Formation spread SD:", round(spread_sd, 4))
print("Saved: outputs/selected_pair_spread.png")

from statsmodels.tsa.stattools import coint

rows = []
periods = {
    "Formation": ("2018", "2021"),
    "Validation": ("2022", "2023"),
    "Test": ("2024", "2024"),
}

for name, (start, end) in periods.items():
    window = logs.loc[start:end]
    statistic, p_value, _ = coint(
        window[a], window[b],
        trend="c", maxlag=0, autolag=None,
    )
    rows.append({
        "period": name,
        "observations": len(window),
        "p_value": p_value,
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(ROOT / "outputs/pair_period_tests.csv", index=False)
print(comparison.round(6).to_string(index=False))
