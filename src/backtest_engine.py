import numpy as np
import pandas as pd

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