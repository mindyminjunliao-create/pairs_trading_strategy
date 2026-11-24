"""Inspect one formation-period regression and its cointegration test."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

ROOT = Path(__file__).resolve().parents[1]
prices = pd.read_csv(
    ROOT / "data/processed/formation_prices.csv",
    index_col="Date",
    parse_dates=True,
)
logs = np.log(prices[["CVX", "COP"]])
y = logs["CVX"]
x = logs["COP"]
model = sm.OLS(y, sm.add_constant(x)).fit()
statistic, p_value, critical_values = coint(
    y, x, trend="c", maxlag=0, autolag=None
)
spread = model.resid
assert np.allclose(spread, y - model.params["const"] - model.params["COP"] * x)

result = pd.DataFrame([{
    "dependent": "CVX",
    "explanatory": "COP",
    "observations": len(logs),
    "intercept": model.params["const"],
    "hedge_ratio": model.params["COP"],
    "test_statistic": statistic,
    "critical_5pct": critical_values[1],
    "p_value": p_value,
}])
result.to_csv(ROOT / "outputs/single_pair_cointegration.csv", index=False)
print(result.round(6).to_string(index=False))
