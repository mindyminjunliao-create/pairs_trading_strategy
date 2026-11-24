from pathlib import Path

import numpy as np
import pandas as pd

from pair_model import fit_pair, make_signals

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 10_000.0
FEE = 0.0005


def backtest_pair(
        prices, signals, beta, 
        max_days=10, entry_z=1.5, exit_z=0.25,
        trade_records=None,
        fee_rate=FEE,
        stop_loss=0.03,
):
    """Trade one fixed-capital sleeve using next-close execution."""
    equity = CAPITAL
    shares = np.zeros(2)
    state = 0
    pending = None
    entry_day = 0
    entry_equity = CAPITAL
    values = []
    trades = []

    quotes = prices.to_numpy()

    for i, signal in enumerate(signals.itertuples()):
        if i > 0:
            equity += shares @ (quotes[i] - quotes[i - 1])

        final_day = i == len(prices) - 1
        time_exit = state != 0 and i - entry_day >= max_days

        if state != 0 and (pending == 0 or time_exit or final_day):
            equity -= fee_rate * (np.abs(shares) @ quotes[i])
            trades.append(equity - entry_equity)
            if trade_records is not None:
                entry_notional = np.abs(shares) @ quotes[entry_day]
                exit_notional = np.abs(shares) @ quotes[i]
                fees = fee_rate * (entry_notional + exit_notional)
                net_pnl = equity - entry_equity

                trade_records.append({
                    "entry_date": prices.index[entry_day],
                    "exit_date": prices.index[i],
                    "direction": state,
                    "holding_days": i - entry_day,
                    "entry_notional": entry_notional,
                    "exit_notional": exit_notional,
                    "fees": fees,
                    "gross_pnl": net_pnl + fees,
                    "net_pnl": net_pnl,
                    "trade_return": net_pnl / entry_notional,
                })
            shares = np.zeros(2)
            state = 0

        elif state == 0 and pending in (-1, 1) and not final_day:
            state = pending
            entry_day = i
            entry_equity = equity
            dollars = state * CAPITAL * np.array([1, -beta]) / (1 + abs(beta))
            shares = dollars / quotes[i]
            equity -= fee_rate * np.abs(dollars).sum()

        pending = None

        if state == 0:
            if signal.previous_z < entry_z <= signal.z:
                pending = -1
            elif signal.previous_z > -entry_z >= signal.z:
                pending = 1
        else:
            stop_hit = (
                stop_loss is not None
                and equity - entry_equity <= -stop_loss * CAPITAL
            )

            if exit_z == 0:
                exit_hit = state * signal.z >= 0
            else:
                exit_hit = abs(signal.z) <= exit_z

            if stop_hit or exit_hit:
                pending = 0

        values.append(equity)

    nav = pd.Series(values, index=prices.index)
    assert np.isclose(nav.iloc[-1] - CAPITAL, sum(trades))
    return nav, len(trades)


prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date", parse_dates=True,
).sort_index()

screen = pd.read_csv(ROOT / "outputs/cointegration_screen.csv")
candidates = screen.loc[screen["passes_bh_5pct"]]

curves = {}
rows = []

for pair in candidates.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b
    sample = prices.loc["2018":"2023", [a, b]]
    if sample.isna().any().any() or not np.isfinite(sample).all().all():
        raise ValueError(f"Inspect price coverage for {a} and {b}.")
    if (sample <= 0).any().any():
        raise ValueError(f"Nonpositive prices for {a} and {b}.")

    model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
    signals = make_signals(sample, model).loc["2022":"2023"]
    nav, count = backtest_pair(
        sample.loc[signals.index], signals, model["beta"]
    )

    name = f"{a}-{b}"
    curves[name] = nav
    rows.append({
        "pair": name,
        "trades": count,
        "net_pnl": nav.iloc[-1] - CAPITAL,
    })

sleeves = pd.DataFrame(curves)
portfolio = sleeves.sum(axis=1)
initial = CAPITAL * len(candidates)

returns = portfolio / portfolio.shift(1).fillna(initial) - 1
drawdown = portfolio / portfolio.cummax().clip(lower=initial) - 1
sharpe = (
    returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    if returns.std(ddof=1) > 0 else np.nan
)

contributions = pd.DataFrame(rows).sort_values("net_pnl", ascending=False)
sleeves.to_csv(ROOT / "outputs/validation_pair_equities.csv")
contributions.to_csv(ROOT / "outputs/validation_pair_contributions.csv")
portfolio.to_csv(ROOT / "outputs/validation_portfolio_equity.csv", header=["equity"])

print(contributions.round(2).to_string(index=False))
print(f"Portfolio return: {portfolio.iloc[-1] / initial - 1:.2%}")
print(f"Maximum drawdown: {drawdown.min():.2%}")
print(f"Annualized Sharpe: {sharpe:.3f}")

# Estimate each model once; reuse the same signals across holding limits.
prepared = []

for pair in candidates.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b
    sample = prices.loc["2018":"2023", [a, b]]
    model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
    signals = make_signals(sample, model).loc["2022":"2023"]

    prepared.append((
        sample.loc[signals.index],
        signals,
        model["beta"],
    ))

from itertools import product

comparison = []

for limit, entry_z, exit_z in product(
    [10, 20, 60],
    [1.5, 2.0, 2.5],
    [0.0, 0.25, 0.5],
):
    pair_curves = []
    total_trades = 0

    for pair_prices, pair_signals, beta in prepared:
        nav, count = backtest_pair(
            pair_prices,
            pair_signals,
            beta,
            max_days=limit,
            entry_z=entry_z,
            exit_z=exit_z,
        )
        pair_curves.append(nav)
        total_trades += count

    combined = pd.concat(pair_curves, axis=1).sum(axis=1)
    daily_returns = combined / combined.shift(1).fillna(initial) - 1
    peak = combined.cummax().clip(lower=initial)
    daily_sd = daily_returns.std(ddof=1)

    comparison.append({
        "max_days": limit,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "trades": total_trades,
        "total_return": combined.iloc[-1] / initial - 1,
        "max_drawdown": (combined / peak - 1).min(),
        "sharpe": (
            daily_returns.mean() / daily_sd * np.sqrt(252)
            if daily_sd > 0 else np.nan
        ),
    })

comparison = pd.DataFrame(comparison)
comparison.to_csv(
    ROOT / "outputs/portfolio_threshold_grid.csv", index=False
)

ranked = comparison.loc[
    comparison["trades"].ge(5) & np.isfinite(comparison["sharpe"])
].sort_values("sharpe", ascending=False, kind="stable")

print("\nTop portfolio validation configurations:")
print(ranked.head(5).round(4).to_string(index=False))

winner = ranked.iloc[0]
best_days = int(winner["max_days"])
best_entry = float(winner["entry_z"])
best_exit = float(winner["exit_z"])

test_curves = {}
test_rows = []

for pair in candidates.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b
    sample = prices.loc["2018":"2024", [a, b]]

    if sample.isna().any().any() or not np.isfinite(sample).all().all():
        raise ValueError(f"Inspect test price coverage for {a} and {b}.")
    if (sample <= 0).any().any():
        raise ValueError(f"Nonpositive prices for {a} and {b}.")

    model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
    test_signals = make_signals(sample, model).loc["2024"]

    nav, count = backtest_pair(
        sample.loc[test_signals.index],
        test_signals,
        model["beta"],
        max_days=best_days,
        entry_z=best_entry,
        exit_z=best_exit,
    )

    name = f"{a}-{b}"
    test_curves[name] = nav
    test_rows.append({
        "pair": name,
        "trades": count,
        "net_pnl": nav.iloc[-1] - CAPITAL,
    })

test_sleeves = pd.DataFrame(test_curves)
test_portfolio = test_sleeves.sum(axis=1)
test_initial = CAPITAL * len(test_curves)
test_returns = (
    test_portfolio / test_portfolio.shift(1).fillna(test_initial) - 1
)
test_peak = test_portfolio.cummax().clip(lower=test_initial)
test_sd = test_returns.std(ddof=1)

test_summary = pd.DataFrame([{
    "max_days": best_days,
    "entry_z": best_entry,
    "exit_z": best_exit,
    "trades": sum(row["trades"] for row in test_rows),
    "total_return": test_portfolio.iloc[-1] / test_initial - 1,
    "max_drawdown": (test_portfolio / test_peak - 1).min(),
    "sharpe": (
        test_returns.mean() / test_sd * np.sqrt(252)
        if test_sd > 0 else np.nan
    ),
}])

test_summary.to_csv(ROOT / "outputs/portfolio_test_metrics.csv", index=False)
test_sleeves.to_csv(ROOT / "outputs/portfolio_test_pair_equities.csv")
pd.DataFrame(test_rows).to_csv(
    ROOT / "outputs/portfolio_test_contributions.csv", index=False
)

print("\nFrozen portfolio test:")
print(test_summary.round(4).to_string(index=False))

contributions = pd.DataFrame(test_rows)
contributions["return_contribution"] = (
    contributions["net_pnl"] / test_initial
)

print("\nTest pair contributions:")
print(
    contributions.sort_values("net_pnl", ascending=False)
    .round({"net_pnl": 2, "return_contribution": 6})
    .to_string(index=False)
)

print("Pairs with trades:", contributions["trades"].gt(0).sum())
print("Pairs without trades:", contributions["trades"].eq(0).sum())
print(f"Total net P&L: {contributions['net_pnl'].sum():.2f}")
print(
    f"Portfolio return: "
    f"{test_portfolio.iloc[-1] / test_initial - 1:.6%}"
)

assert np.isclose(
    contributions["net_pnl"].sum(),
    test_portfolio.iloc[-1] - test_initial,
)

diagnostics = []

for pair in candidates.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b
    sample = prices.loc["2018":"2024", [a, b]]
    model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
    signals_2024 = make_signals(sample, model).loc["2024"]

    z = signals_2024["z"]
    previous_z = signals_2024["previous_z"]

    crossings = (
        ((previous_z < best_entry) & (z >= best_entry))
        | ((previous_z > -best_entry) & (z <= -best_entry))
    )

    diagnostics.append({
        "pair": f"{a}-{b}",
        "first_z": z.iloc[0],
        "last_z": z.iloc[-1],
        "days_outside_entry": (z.abs() >= best_entry).sum(),
        "entry_crossings": crossings.sum(),
    })

diagnostics = pd.DataFrame(diagnostics).merge(
    contributions[["pair", "trades"]],
    on="pair",
    validate="one_to_one",
)

diagnostics.to_csv(
    ROOT / "outputs/test_signal_diagnostics.csv", index=False
)

print("\nPairs with no executed trades:")
print(
    diagnostics.loc[diagnostics["trades"].eq(0)]
    .round({"first_z": 2, "last_z": 2})
    .to_string(index=False)
)

minimum_equity = test_sleeves.min()
ending_equity = test_sleeves.iloc[-1]

account_check = pd.DataFrame({
    "minimum_equity": minimum_equity,
    "ending_equity": ending_equity,
    "fixed_trade_notional": CAPITAL,
})

print("\nPair account diagnostics:")
print(account_check.round(2).to_string())

print(
    "Any nonpositive account equity:",
    bool((test_sleeves <= 0).any().any()),
)

portfolio_records = []

for pair in candidates.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b
    sample = prices.loc["2018":"2024", [a, b]]
    model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
    pair_signals = make_signals(sample, model).loc["2024"]

    records = []
    nav, _ = backtest_pair(
        sample.loc[pair_signals.index],
        pair_signals,
        model["beta"],
        max_days=best_days,
        entry_z=best_entry,
        exit_z=best_exit,
        trade_records=records,
    )

    for record in records:
        record["ticker_a"] = a
        record["ticker_b"] = b
        portfolio_records.append(record)

ledger = pd.DataFrame(portfolio_records)
ledger.to_csv(ROOT / "outputs/portfolio_test_trades.csv", index=False)

assert np.isclose(
    ledger["net_pnl"].sum(),
    test_portfolio.iloc[-1] - test_initial,
)

years = len(test_portfolio) / 252
traded_notional = (
    ledger["entry_notional"] + ledger["exit_notional"]
).sum()

trade_metrics = pd.DataFrame([{
    "trades": len(ledger),
    "win_rate": ledger["net_pnl"].gt(0).mean(),
    "average_trade_return": ledger["trade_return"].mean(),
    "total_fees": ledger["fees"].sum(),
    "annualized_turnover": traded_notional / test_initial / years,
}])

trade_metrics.to_csv(
    ROOT / "outputs/portfolio_test_trade_metrics.csv", index=False
)

print("\nPortfolio test trade metrics:")
print(trade_metrics.round(6).to_string(index=False))

validation_curves = {}

for pair in candidates.itertuples(index=False):
    a, b = pair.ticker_a, pair.ticker_b
    sample = prices.loc["2018":"2023", [a, b]]
    model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
    pair_signals = make_signals(sample, model).loc["2022":"2023"]

    nav, _ = backtest_pair(
        sample.loc[pair_signals.index],
        pair_signals,
        model["beta"],
        max_days=best_days,
        entry_z=best_entry,
        exit_z=best_exit,
    )
    validation_curves[f"{a}-{b}"] = nav

pd.DataFrame(validation_curves).to_csv(
    ROOT / "outputs/portfolio_selected_validation_equities.csv"
)

cost_results = []
cost_trade_records = []

for fee_bps in [0, 5, 10, 20]:
    ending_values = []
    scenario_records = []

    for pair in candidates.itertuples(index=False):
        a, b = pair.ticker_a, pair.ticker_b
        sample = prices.loc["2018":"2024", [a, b]]
        model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
        signals = make_signals(sample, model).loc["2024"]

        records = []
        nav, count = backtest_pair(
            sample.loc[signals.index],
            signals,
            model["beta"],
            max_days=best_days,
            entry_z=best_entry,
            exit_z=best_exit,
            fee_rate=fee_bps / 10_000,
            trade_records=records,
        )
        ending_values.append(nav.iloc[-1])

        for record in records:
            record.update({
                "fee_bps": fee_bps,
                "pair": f"{a}-{b}",
            })
            scenario_records.append(record)

    ledger = pd.DataFrame(scenario_records)
    initial_value = CAPITAL * len(ending_values)
    net_pnl = sum(ending_values) - initial_value

    assert np.isclose(ledger["net_pnl"].sum(), net_pnl)

    cost_results.append({
        "fee_bps": fee_bps,
        "trades": len(ledger),
        "gross_pnl": ledger["gross_pnl"].sum(),
        "fees": ledger["fees"].sum(),
        "net_pnl": net_pnl,
    })
    cost_trade_records.extend(scenario_records)

cost_comparison = pd.DataFrame(cost_results)
cost_ledger = pd.DataFrame(cost_trade_records)

cost_comparison.to_csv(
    ROOT / "outputs/portfolio_test_cost_sensitivity.csv", index=False
)
cost_ledger.to_csv(
    ROOT / "outputs/portfolio_test_cost_trades.csv", index=False
)

print(cost_comparison.round(2).to_string(index=False))
print("\nNet P&L by pair and fee scenario:")
print(
    cost_ledger.pivot_table(
        index="pair",
        columns="fee_bps",
        values="net_pnl",
        aggfunc="sum",
        fill_value=0,
    ).round(2).to_string()
)

base_trades = cost_ledger.loc[cost_ledger["fee_bps"].eq(5)].copy()

fixed_gross = base_trades["gross_pnl"].sum()
fixed_notional = (
    base_trades["entry_notional"] + base_trades["exit_notional"]
).sum()

fixed_rows = []
for fee_bps in [0, 5, 10, 20]:
    fees = fixed_notional * fee_bps / 10_000
    fixed_rows.append({
        "fee_bps": fee_bps,
        "gross_pnl": fixed_gross,
        "fees": fees,
        "net_pnl": fixed_gross - fees,
    })

fixed_costs = pd.DataFrame(fixed_rows)
fixed_costs.to_csv(
    ROOT / "outputs/portfolio_fixed_trade_costs.csv", index=False
)

print("\nCost sensitivity with identical executions:")
print(fixed_costs.round(2).to_string(index=False))

print("\nExecution dates for affected pairs:")
affected = cost_ledger.loc[
    cost_ledger["fee_bps"].isin([5, 20])
    & cost_ledger["pair"].isin(["BA-NCLH", "CFG-MGM", "PEG-YUM"])
]
print(
    affected[
        ["pair", "fee_bps", "entry_date", "exit_date", "net_pnl"]
    ].sort_values(["pair", "fee_bps", "entry_date"])
    .round({"net_pnl": 2})
    .to_string(index=False)
)

cutoff_results = []
portfolio_capital = 160_000.0

for cutoff in [0.01, 0.05, 0.10]:
    selected = screen.loc[screen["bh_adjusted_p"] < cutoff]
    normalized_curves = []
    total_trades = 0

    for pair in selected.itertuples(index=False):
        a, b = pair.ticker_a, pair.ticker_b
        sample = prices.loc["2018":"2023", [a, b]]

        if (
            sample.isna().any().any()
            or not np.isfinite(sample).all().all()
            or (sample <= 0).any().any()
        ):
            raise ValueError(f"Inspect prices for {a} and {b}.")

        model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
        pair_signals = make_signals(sample, model).loc["2022":"2023"]

        nav, count = backtest_pair(
            sample.loc[pair_signals.index],
            pair_signals,
            model["beta"],
            max_days=60,
            entry_z=2.0,
            exit_z=0.25,
            fee_rate=0.0005,
        )

        normalized_curves.append(nav / CAPITAL)
        total_trades += count

    # Equal initial allocations, with the same total capital for every cutoff.
    unit_equity = pd.concat(normalized_curves, axis=1).mean(axis=1)
    equity = portfolio_capital * unit_equity

    returns = equity / equity.shift(1).fillna(portfolio_capital) - 1
    peak = equity.cummax().clip(lower=portfolio_capital)
    sd = returns.std(ddof=1)

    cutoff_results.append({
        "bh_cutoff": cutoff,
        "pairs": len(selected),
        "capital_per_pair": portfolio_capital / len(selected),
        "trades": total_trades,
        "total_return": equity.iloc[-1] / portfolio_capital - 1,
        "max_drawdown": (equity / peak - 1).min(),
        "sharpe": returns.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
    })

cutoff_performance = pd.DataFrame(cutoff_results)
cutoff_performance.to_csv(
    ROOT / "outputs/validation_cutoff_performance.csv", index=False
)

print("\nValidation cointegration-cutoff comparison:")
print(cutoff_performance.round(4).to_string(index=False))

selected = screen.loc[screen["bh_adjusted_p"] < 0.05]
lookback_results = []

for lookback in [126, 252, 504, 1008]:
    unit_curves = []
    total_trades = 0

    for pair in selected.itertuples(index=False):
        a, b = pair.ticker_a, pair.ticker_b
        sample = prices.loc["2018":"2023", [a, b]]

        model = fit_pair(
            sample.loc["2018":"2021"], a, b, lookback
        )
        pair_signals = make_signals(sample, model).loc["2022":"2023"]

        nav, count = backtest_pair(
            sample.loc[pair_signals.index],
            pair_signals,
            model["beta"],
            max_days=60,
            entry_z=2.0,
            exit_z=0.25,
            fee_rate=0.0005,
        )
        unit_curves.append(nav / CAPITAL)
        total_trades += count

    unit_equity = pd.concat(unit_curves, axis=1).mean(axis=1)
    returns = unit_equity / unit_equity.shift(1).fillna(1.0) - 1
    peak = unit_equity.cummax().clip(lower=1.0)
    sd = returns.std(ddof=1)

    lookback_results.append({
        "lookback": lookback,
        "pairs": len(selected),
        "trades": total_trades,
        "total_return": unit_equity.iloc[-1] - 1,
        "max_drawdown": (unit_equity / peak - 1).min(),
        "sharpe": returns.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
    })

lookback_performance = pd.DataFrame(lookback_results)
lookback_performance.to_csv(
    ROOT / "outputs/portfolio_lookback_comparison.csv", index=False
)

print("\nPortfolio validation estimation-window comparison:")
print(lookback_performance.round(4).to_string(index=False))

stop_results = []
selected = screen.loc[screen["bh_adjusted_p"] < 0.05]

for stop in [None, 0.03, 0.06]:
    unit_curves = []
    total_trades = 0

    for pair in selected.itertuples(index=False):
        a, b = pair.ticker_a, pair.ticker_b
        sample = prices.loc["2018":"2023", [a, b]]
        model = fit_pair(sample.loc["2018":"2021"], a, b, 1008)
        pair_signals = make_signals(sample, model).loc["2022":"2023"]

        nav, count = backtest_pair(
            sample.loc[pair_signals.index],
            pair_signals,
            model["beta"],
            max_days=60,
            entry_z=2.0,
            exit_z=0.25,
            fee_rate=0.0005,
            stop_loss=stop,
        )
        unit_curves.append(nav / CAPITAL)
        total_trades += count

    equity = pd.concat(unit_curves, axis=1).mean(axis=1)
    returns = equity / equity.shift(1).fillna(1.0) - 1
    peak = equity.cummax().clip(lower=1.0)
    sd = returns.std(ddof=1)

    stop_results.append({
        "stop_loss": "None" if stop is None else f"{stop:.0%}",
        "trades": total_trades,
        "total_return": equity.iloc[-1] - 1,
        "max_drawdown": (equity / peak - 1).min(),
        "sharpe": returns.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
    })

stop_comparison = pd.DataFrame(stop_results)
stop_comparison.to_csv(
    ROOT / "outputs/portfolio_stop_comparison.csv", index=False
)

print("\nPortfolio validation stop-loss comparison:")
print(stop_comparison.round(4).to_string(index=False))