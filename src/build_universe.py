from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START_DATE = pd.Timestamp("2018-01-01")

snapshots = pd.read_csv(
    ROOT / "data/raw/sp500_membership.csv",
    parse_dates=["date"],
).sort_values("date")

eligible = snapshots.loc[snapshots["date"] < START_DATE]
snapshot = eligible.iloc[-1]

tickers = sorted({
    symbol.strip().replace(".", "-")
    for symbol in snapshot["tickers"].split(",")
})

universe = pd.DataFrame({
    "ticker": tickers,
    "membership_snapshot": snapshot["date"].date(),
})
universe.to_csv(ROOT / "data/processed/sp500_universe.csv", index=False)

print("Membership snapshot:", snapshot["date"].date())
print("Number of securities:", len(universe))
print(universe.head().to_string(index=False))