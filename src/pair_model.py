import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_pair(formation_prices, a, b, lookback):
    """Estimate a pair model using only formation-period prices."""
    logs = np.log(formation_prices[[a, b]].tail(lookback))
    regression = sm.OLS(logs[a], sm.add_constant(logs[b])).fit()

    spread = regression.resid
    spread_sd = spread.std()
    if not np.isfinite(spread_sd) or spread_sd <= 0:
        raise ValueError("The formation spread must have positive variation.")

    return {
        "a": a,
        "b": b,
        "intercept": regression.params["const"],
        "beta": regression.params[b],
        "spread_mean": spread.mean(),
        "spread_sd": spread_sd,
    }


def make_signals(prices, pair_model):
    """Apply a fitted model without estimating any new parameters."""
    logs = np.log(prices[[pair_model["a"], pair_model["b"]]])

    spread = (
        logs[pair_model["a"]]
        - pair_model["intercept"]
        - pair_model["beta"] * logs[pair_model["b"]]
    )
    z = (spread - pair_model["spread_mean"]) / pair_model["spread_sd"]

    return pd.DataFrame({
        "z": z,
        "previous_z": z.shift(1),
    })