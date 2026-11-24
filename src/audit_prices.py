from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"

universe = pd.read_csv(PROCESSED / "sp500_universe.csv")
prices = pd.read_csv(
    ROOT / "data/raw/sp500_prices.csv",
    index_col="Date",
    parse_dates=True,
)

if prices.index.has_duplicates:
    raise ValueError("Duplicate trading dates found in the price file.")

prices = prices.sort_index()
formation = prices.loc["2018":"2021"].reindex(columns=universe["ticker"])

valid = formation.gt(0) & np.isfinite(formation)
audit = pd.DataFrame(index=formation.columns)
audit["expected_days"] = len(formation)
audit["valid_days"] = valid.sum()
audit["missing_days"] = formation.isna().sum()
audit["invalid_days"] = (formation.notna() & ~valid).sum()
audit["included"] = audit["valid_days"].eq(len(formation))

audit["status"] = "included"
audit.loc[~audit["included"], "status"] = "incomplete_history"
audit.loc[audit["valid_days"].eq(0), "status"] = "no_valid_prices"

audit.to_csv(PROCESSED / "formation_price_audit.csv")
clean_prices = formation.loc[:, audit["included"]]
clean_prices.to_csv(PROCESSED / "formation_prices.csv")

print("Formation dates:", formation.index.min().date(), "to", formation.index.max().date())
print("Trading dates in file:", len(formation))
print("Securities audited:", len(audit))
print(audit["status"].value_counts().to_string())
print("Clean price matrix:", clean_prices.shape)