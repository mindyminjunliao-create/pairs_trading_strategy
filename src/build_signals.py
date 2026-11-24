from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]

screen = pd.read_csv(ROOT / "outputs/cointegration_screen.csv")
candidate = screen.loc[screen["passes_bh_5pct"]].sort_values("p_value").iloc[0]
a, b = candidate["ticker_a"], candidate["ticker_b"]

prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date",
    parse_dates=True,
).sort_index()

logs = np.log(prices.loc["2018":"2023", [a, b]])
formation = logs.loc["2018":"2021"]

model = sm.OLS(
    formation[a], sm.add_constant(formation[b])
).fit()

spread = logs[a] - model.params["const"] - model.params[b] * logs[b]
reference = spread.loc["2018":"2021"]
z = (spread - reference.mean()) / reference.std()

signals = pd.DataFrame({"z": z})
signals["previous_z"] = signals["z"].shift(1)

signals["short_spread_entry"] = (
    signals["previous_z"].lt(2) & signals["z"].ge(2)
)
signals["long_spread_entry"] = (
    signals["previous_z"].gt(-2) & signals["z"].le(-2)
)
signals["exit_condition"] = signals["z"].abs().le(0.5)

validation = signals.loc["2022":"2023"]
validation.to_csv(ROOT / "outputs/validation_signals.csv")

print("Pair:", a, b)
print("Hedge ratio:", round(model.params[b], 4))
print(validation[
    ["short_spread_entry", "long_spread_entry", "exit_condition"]
].sum().to_string())

state = 0
pending = None
actions = []

for date, row in validation.iterrows():
    # Execute yesterday's order at today's close.
    if pending is not None:
        old_state = state
        state = pending
        actions.append({
            "date": date,
            "action": "entry" if state != 0 else "exit",
            "position_before": old_state,
            "position_after": state,
            "execution_z": row["z"],
        })
        pending = None

    # Use today's closing signal to prepare tomorrow's order.
    if state == 0:
        if row["short_spread_entry"]:
            pending = -1
        elif row["long_spread_entry"]:
            pending = 1
    elif row["exit_condition"]:
        pending = 0

action_table = pd.DataFrame(
    actions,
    columns=[
        "date", "action", "position_before",
        "position_after", "execution_z",
    ],
)
action_table.to_csv(ROOT / "outputs/validation_actions.csv", index=False)

print(action_table.to_string(index=False))
print("Position at period end:", state)
print("Unexecuted order at period end:", pending)

entry = action_table.loc[action_table["action"] == "entry"].iloc[0]
exit_trade = action_table.loc[action_table["action"] == "exit"].iloc[0]

entry_prices = prices.loc[entry["date"], [a, b]]
exit_prices = prices.loc[exit_trade["date"], [a, b]]

beta = model.params[b]
direction = entry["position_after"]
gross_exposure = 10_000
fee_rate = 0.0005

# The spread is log(A) - intercept - beta * log(B).
dollar_a = direction * gross_exposure / (1 + abs(beta))
dollar_b = -direction * beta * gross_exposure / (1 + abs(beta))

shares_a = dollar_a / entry_prices[a]
shares_b = dollar_b / entry_prices[b]

pnl_a = shares_a * (exit_prices[a] - entry_prices[a])
pnl_b = shares_b * (exit_prices[b] - entry_prices[b])

fees = fee_rate * (
    abs(shares_a) * (entry_prices[a] + exit_prices[a])
    + abs(shares_b) * (entry_prices[b] + exit_prices[b])
)

print("Trade:", entry["date"].date(), "to", exit_trade["date"].date())
print(f"{a} signed entry value: {dollar_a:.2f}")
print(f"{b} signed entry value: {dollar_b:.2f}")
print(f"{a} gross P&L: {pnl_a:.2f}")
print(f"{b} gross P&L: {pnl_b:.2f}")
print(f"Total fees: {fees:.2f}")
print(f"Net P&L: {pnl_a + pnl_b - fees:.2f}")

# Pair each entry with its exit, including liquidation at period end.
round_trips = []
open_entry = None

for action in action_table.itertuples(index=False):
    if action.action == "entry":
        open_entry = action
    else:
        round_trips.append((open_entry, action.date, "signal"))
        open_entry = None

if open_entry is not None:
    round_trips.append((open_entry, validation.index[-1], "period_end"))

trades = []
for entry, exit_date, exit_reason in round_trips:
    entry_prices = prices.loc[entry.date, [a, b]]
    exit_prices = prices.loc[exit_date, [a, b]]
    direction = entry.position_after

    dollar_a = direction * gross_exposure / (1 + abs(beta))
    dollar_b = -direction * beta * gross_exposure / (1 + abs(beta))
    shares_a = dollar_a / entry_prices[a]
    shares_b = dollar_b / entry_prices[b]

    gross_pnl = (
        shares_a * (exit_prices[a] - entry_prices[a])
        + shares_b * (exit_prices[b] - entry_prices[b])
    )
    fees = fee_rate * (
        abs(shares_a) * (entry_prices[a] + exit_prices[a])
        + abs(shares_b) * (entry_prices[b] + exit_prices[b])
    )

    trades.append({
        "entry_date": entry.date,
        "exit_date": exit_date,
        "direction": direction,
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": gross_pnl - fees,
    })

trade_log = pd.DataFrame(trades)
trade_log.to_csv(ROOT / "outputs/validation_trade_log.csv", index=False)

print(
    trade_log.round({"gross_pnl": 2, "fees": 2, "net_pnl": 2})
    .to_string(index=False)
)
print("Total net P&L:", round(trade_log["net_pnl"].sum(), 2))
print("Win rate:", round((trade_log["net_pnl"] > 0).mean(), 3))

initial_capital = 10_000.0
daily_pnl = pd.Series(0.0, index=validation.index)

for trade in trade_log.itertuples(index=False):
    holding_prices = prices.loc[trade.entry_date:trade.exit_date, [a, b]]
    entry_prices = holding_prices.iloc[0]

    dollar_a = trade.direction * gross_exposure / (1 + abs(beta))
    dollar_b = -trade.direction * beta * gross_exposure / (1 + abs(beta))
    shares_a = dollar_a / entry_prices[a]
    shares_b = dollar_b / entry_prices[b]

    # Entry occurs at the close, so the first mark-to-market change is zero.
    changes = holding_prices.diff().fillna(0.0)
    holding_pnl = shares_a * changes[a] + shares_b * changes[b]

    entry_fee = fee_rate * (
        abs(shares_a) * entry_prices[a]
        + abs(shares_b) * entry_prices[b]
    )
    exit_fee = fee_rate * (
        abs(shares_a) * holding_prices[a].iloc[-1]
        + abs(shares_b) * holding_prices[b].iloc[-1]
    )

    holding_pnl.iloc[0] -= entry_fee
    holding_pnl.iloc[-1] -= exit_fee
    daily_pnl.loc[holding_pnl.index] += holding_pnl

equity = initial_capital + daily_pnl.cumsum()
running_peak = equity.cummax().clip(lower=initial_capital)
drawdown = equity / running_peak - 1

assert np.isclose(daily_pnl.sum(), trade_log["net_pnl"].sum())

pd.DataFrame({
    "daily_pnl": daily_pnl,
    "equity": equity,
    "drawdown": drawdown,
}).to_csv(ROOT / "outputs/validation_equity.csv")

print("Final equity:", round(equity.iloc[-1], 2))
print(f"Total return: {equity.iloc[-1] / initial_capital - 1:.2%}")
print(f"Maximum drawdown: {drawdown.min():.2%}")

# Divide today's P&L by the account value at the previous close.
previous_equity = equity.shift(1).fillna(initial_capital)
daily_returns = daily_pnl / previous_equity

trading_days_per_year = 252
years = len(daily_returns) / trading_days_per_year

annual_return = (
    equity.iloc[-1] / initial_capital
) ** (1 / years) - 1

annual_volatility = daily_returns.std(ddof=1) * np.sqrt(
    trading_days_per_year
)

# Assume a zero risk-free rate for this baseline calculation.
daily_sd = daily_returns.std(ddof=1)
sharpe = (
    daily_returns.mean() / daily_sd * np.sqrt(trading_days_per_year)
    if daily_sd > 0 else np.nan
)

print(f"Annualized return: {annual_return:.2%}")
print(f"Annualized volatility: {annual_volatility:.2%}")
print(f"Annualized Sharpe: {sharpe:.3f}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

equity.plot(ax=axes[0], color="navy")
axes[0].axhline(initial_capital, color="gray", linestyle="--")
axes[0].set_title(f"{a}–{b}: validation performance after fees")
axes[0].set_ylabel("Account value ($)")

drawdown_pct = 100 * drawdown
axes[1].fill_between(
    drawdown_pct.index,
    drawdown_pct.to_numpy(),
    0,
    color="firebrick",
    alpha=0.35,
)
axes[1].set_ylabel("Drawdown (%)")
axes[1].set_xlabel("Date")

fig.tight_layout()
fig.savefig(ROOT / "outputs/validation_performance.png", dpi=160)
plt.close(fig)

worst_date = drawdown.idxmin()
print("Deepest drawdown date:", worst_date.date())
print("Equity on that date:", round(equity.loc[worst_date], 2))