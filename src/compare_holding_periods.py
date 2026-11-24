from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 10_000.0
FEE_RATE = 0.0005

screen = pd.read_csv(ROOT / "outputs/cointegration_screen.csv")
pair = screen.loc[screen["passes_bh_5pct"]].sort_values("p_value").iloc[0]
a, b = pair["ticker_a"], pair["ticker_b"]

prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date", parse_dates=True,
).sort_index()

signals = pd.read_csv(
    ROOT / "outputs/validation_signals.csv",
    index_col="Date", parse_dates=True,
)

formation = np.log(prices.loc["2018":"2021", [a, b]])
model = sm.OLS(
    formation[a], sm.add_constant(formation[b])
).fit()
beta = model.params[b]


def backtest(max_days, entry_z=2.0, exit_z=0.5, stop_loss=None, fee_rate=FEE_RATE):
    """Execute closing signals one trading day later."""
    quotes = prices.loc[signals.index, [a, b]].to_numpy()
    equity = CAPITAL
    state = 0
    pending = None
    shares = np.zeros(2)
    entry_day = 0
    trade_start_equity = CAPITAL
    trade_profits = []
    trade_records = []
    values = []

    for i, (_, signal) in enumerate(signals.iterrows()):
        # Existing shares earn today's close-to-close price change.
        if i > 0:
            equity += shares @ (quotes[i] - quotes[i - 1])

        time_exit = (
            state != 0
            and max_days is not None
            and i - entry_day >= max_days
        )
        final_day = i == len(signals) - 1

        if state != 0 and (pending == 0 or time_exit or final_day):
            equity -= fee_rate * (np.abs(shares) @ quotes[i])
            trade_profits.append(equity - trade_start_equity)
            if final_day:
                reason = "period_end"
            elif pending == 0:
                reason = "mean_reversion"
            else:
                reason = "holding_limit"

            trade_records.append({
                "entry_date": signals.index[entry_day],
                "exit_date": signals.index[i],
                "direction": state,
                "holding_days": i - entry_day,
                "exit_reason": reason,
                "net_pnl": equity - trade_start_equity,
            })
            shares = np.zeros(2)
            state = 0

        elif state == 0 and pending in (-1, 1) and not final_day:
            state = pending
            entry_day = i
            trade_start_equity = equity
            dollars = (
                state * CAPITAL * np.array([1.0, -beta])
                / (1 + abs(beta))
            )
            shares = dollars / quotes[i]
            equity -= fee_rate * np.abs(dollars).sum()

        pending = None

        # Today's signal prepares an order for the next close.
        previous_z = signal["previous_z"]
        current_z = signal["z"]

        stop_hit = (
            state != 0
            and stop_loss is not None
            and equity - trade_start_equity <= -CAPITAL * stop_loss
        )

        if state == 0:
            if previous_z < entry_z <= current_z:
                pending = -1
            elif previous_z > -entry_z >= current_z:
                pending = 1
        elif stop_hit:
            pending = 0
        elif exit_z == 0:
            # Exit after the spread crosses its formation mean.
            if state * current_z >= 0:
                pending = 0
        elif abs(current_z) <= exit_z:
            pending = 0

        values.append(equity)

    nav = pd.Series(values, index=signals.index)
    previous = nav.shift(1).fillna(CAPITAL)
    returns = nav / previous - 1
    peak = nav.cummax().clip(lower=CAPITAL)
    sd = returns.std(ddof=1)

    assert np.isclose(sum(trade_profits), nav.iloc[-1] - CAPITAL)

    if signals.index[0].year == 2024:
        ledger = pd.DataFrame(trade_records)
        ledger.to_csv(ROOT / "outputs/frozen_test_trades.csv", index=False)
        print("\nFrozen test trade ledger:")
        print(
            ledger.round({"net_pnl": 2}).to_string(index=False)
        )

    return {
        "max_days": "Unlimited" if max_days is None else str(max_days),
        "trades": len(trade_profits),
        "net_pnl": nav.iloc[-1] - CAPITAL,
        "total_return": nav.iloc[-1] / CAPITAL - 1,
        "max_drawdown": (nav / peak - 1).min(),
        "sharpe": returns.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
    }, nav


rows = []
curves = {}
for limit in [None, 10, 20, 60]:
    result, nav = backtest(limit)
    rows.append(result)
    curves[result["max_days"]] = nav

summary = pd.DataFrame(rows)
summary.to_csv(ROOT / "outputs/holding_period_comparison.csv", index=False)
pd.DataFrame(curves).to_csv(ROOT / "outputs/holding_period_equity.csv")

print("Pair:", a, b)
print(summary.round(4).to_string(index=False))

threshold_rows = []

for entry_z in [1.5, 2.0, 2.5]:
    for exit_z in [0.0, 0.25, 0.5]:
        result, _ = backtest(
            max_days=20,
            entry_z=entry_z,
            exit_z=exit_z,
        )
        result["entry_z"] = entry_z
        result["exit_z"] = exit_z
        threshold_rows.append(result)

thresholds = pd.DataFrame(threshold_rows)
thresholds.to_csv(
    ROOT / "outputs/threshold_comparison.csv",
    index=False,
)

print("\nValidation Sharpe by entry and exit threshold:")
print(
    thresholds.pivot(
        index="entry_z",
        columns="exit_z",
        values="sharpe",
    ).round(3)
)

grid_rows = []

for max_days in [None, 10, 20, 60]:
    for entry_z in [1.5, 2.0, 2.5]:
        for exit_z in [0.0, 0.25, 0.5]:
            result, _ = backtest(
                max_days=max_days,
                entry_z=entry_z,
                exit_z=exit_z,
            )
            result["entry_z"] = entry_z
            result["exit_z"] = exit_z
            grid_rows.append(result)

grid = pd.DataFrame(grid_rows)
grid.to_csv(ROOT / "outputs/validation_parameter_grid.csv", index=False)

eligible = grid.loc[
    grid["trades"].ge(5) & np.isfinite(grid["sharpe"])
]
ranked = eligible.sort_values("sharpe", ascending=False, kind="stable")

print("\nTop validation configurations with at least five trades:")
print(
    ranked[
        ["max_days", "entry_z", "exit_z", "trades",
         "total_return", "max_drawdown", "sharpe"]
    ].head(5).round(4).to_string(index=False)
)

# Freeze the winner selected exclusively from validation results.
winner = ranked.iloc[0]
holding_limit = (
    None if winner["max_days"] == "Unlimited"
    else int(winner["max_days"])
)
entry_threshold = float(winner["entry_z"])
exit_threshold = float(winner["exit_z"])

# Reuse the formation-period regression and normalization.
all_logs = np.log(prices.loc["2018":"2024", [a, b]])
spread = (
    all_logs[a]
    - model.params["const"]
    - beta * all_logs[b]
)
reference = spread.loc["2018":"2021"]
z = (spread - reference.mean()) / reference.std()

all_signals = pd.DataFrame({
    "z": z,
    "previous_z": z.shift(1),
})

# backtest() reads this signals table; start the test account flat.
signals = all_signals.loc["2024"].copy()
if signals.empty or signals.isna().any().any():
    raise ValueError("Inspect missing test-period signals.")

test_result, test_equity = backtest(
    max_days=holding_limit,
    entry_z=entry_threshold,
    exit_z=exit_threshold,
)

pd.DataFrame([test_result]).to_csv(
    ROOT / "outputs/frozen_test_metrics.csv", index=False
)
test_equity.to_csv(
    ROOT / "outputs/frozen_test_equity.csv", header=["equity"]
)

print("\nFrozen parameters:", holding_limit, entry_threshold, exit_threshold)
print(pd.DataFrame([test_result]).round(4).to_string(index=False))

# Compare stops on validation data using the frozen entry and exit settings.
signals = all_signals.loc["2022":"2023"].copy()

stop_results = []
for stop in [None, 0.03, 0.06]:
    result, _ = backtest(
        max_days=10,
        entry_z=1.5,
        exit_z=0.25,
        stop_loss=stop,
    )
    result["stop_loss"] = "None" if stop is None else str(stop)
    stop_results.append(result)

stop_table = pd.DataFrame(stop_results)
stop_table.to_csv(ROOT / "outputs/validation_stop_comparison.csv", index=False)

print("\nValidation stop-loss comparison:")
print(
    stop_table[
        ["stop_loss", "trades", "total_return", "max_drawdown", "sharpe"]
    ].round(4).to_string(index=False)
)

signals = all_signals.loc["2022":"2023"].copy()

cost_results = []
for fee_bps in [0, 5, 10, 20]:
    result, _ = backtest(
        max_days=10,
        entry_z=1.5,
        exit_z=0.25,
        stop_loss=0.03,
        fee_rate=fee_bps / 10_000,
    )
    result["fee_bps_per_fill"] = fee_bps
    cost_results.append(result)

cost_table = pd.DataFrame(cost_results)
cost_table.to_csv(
    ROOT / "outputs/validation_cost_comparison.csv", index=False
)

print("\nValidation transaction-cost sensitivity:")
print(
    cost_table[
        ["fee_bps_per_fill", "trades", "total_return", "sharpe"]
    ].round(4).to_string(index=False)
)

lookback_results = []

for lookback in [126, 252, 504, len(formation)]:
    estimation = formation.tail(lookback)

    window_model = sm.OLS(
        estimation[a],
        sm.add_constant(estimation[b]),
    ).fit()

    # backtest() uses beta to calculate the two position sizes.
    beta = window_model.params[b]

    window_spread = (
        all_logs[a]
        - window_model.params["const"]
        - beta * all_logs[b]
    )
    reference_spread = window_spread.loc[estimation.index]
    window_z = (
        window_spread - reference_spread.mean()
    ) / reference_spread.std()

    window_signals = pd.DataFrame({
        "z": window_z,
        "previous_z": window_z.shift(1),
    })
    signals = window_signals.loc["2022":"2023"].copy()

    result, _ = backtest(
        max_days=10,
        entry_z=1.5,
        exit_z=0.25,
        stop_loss=0.03,
        fee_rate=0.0005,
    )
    result["lookback"] = lookback
    result["hedge_ratio"] = beta
    result["intercept"] = window_model.params["const"]
    result["spread_mean"] = reference_spread.mean()
    result["spread_sd"] = reference_spread.std()
    lookback_results.append(result)

lookback_table = pd.DataFrame(lookback_results)
lookback_table.to_csv(
    ROOT / "outputs/validation_lookback_comparison.csv", index=False
)

print("\nValidation estimation-window sensitivity:")
print(
    lookback_table[
        ["lookback", "hedge_ratio", "intercept",
         "spread_mean", "spread_sd", "trades", "sharpe"]
    ].round(6).to_string(index=False)
)

from pair_model import fit_pair, make_signals

pair_model = fit_pair(prices.loc["2018":"2021"], a, b, 1008)
rebuilt = make_signals(prices.loc["2018":"2024"], pair_model)

assert np.isclose(pair_model["beta"], 0.781011, atol=0.000001)
assert np.isclose(pair_model["spread_sd"], 0.057696, atol=0.000001)
assert np.allclose(rebuilt["z"], window_z)

print("Reusable pair model matches the existing calculation.")
