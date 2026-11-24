# S&P 500 Pairs Trading with Cointegration

## Objective

Evaluate whether pairs identified from historical S&P 500 prices
produce profitable mean-reversion trades after transaction costs.

The study examines cointegration stability, parameter sensitivity,
portfolio performance, and the concentration of gains and losses.

## Data

The initial universe contains 505 securities from the historical
membership snapshot dated December 29, 2017.

Adjusted daily prices come from a saved Yahoo Finance snapshot.
Of the initial securities, 407 have complete positive, finite prices
over the 1,008 formation-period trading dates.

The historical membership data are community-maintained.
Unavailable prices and corporate ticker changes limit coverage.

## Experimental design

| Period | Dates | Purpose |
| --- | --- | --- |
| Formation | 2018–2021 | Screen pairs and estimate price relationships |
| Validation | 2022–2023 | Compare trading parameters |
| Test | 2024 | Evaluate the selected portfolio rules |

The test period has been inspected during project development.
The reported evaluation is retrospective.

## Method

1. Audit price coverage using formation-period data.
2. Test all 82,621 pairs using Engle–Granger cointegration.
3. Apply Benjamini–Hochberg correction across the screening tests.
4. Estimate hedge ratios and standardized spreads.
5. Execute closing signals at the following trading day's close.
6. Compare validation-period parameter configurations.
7. Evaluate the selected portfolio and inspect pair-level contributions.

## Current findings

Sixteen pairs passed the formation-period BH screen at 5%.
None passed an unadjusted 5% cointegration test in validation,
and one passed in the test period.

The selected portfolio configuration used an entry threshold of 2.0,
an exit threshold of 0.25, a 60-trading-day holding limit, and a
3% stop relative to each trade's initial gross exposure.

It returned 3.98% in validation and approximately zero in 2024.
Eight pairs generated no test-period entries because their spreads
remained outside the entry bands without crossing them.

## Execution assumptions

Each pair starts with $10,000 and uses fixed initial gross exposure
of $10,000 per trade. Positions allow fractional shares.

Each fill incurs 5 basis points. Cash earns zero.
Borrow fees, financing costs, and market impact are omitted.
Adjusted closing prices approximate execution.
All remaining positions are closed at the period end.

## Project structure

- data/raw/ — saved source data
- data/processed/ — universe and cleaned formation prices
- src/ — analysis and backtest scripts
- notebooks/ — research walkthrough
- outputs/ — tables and charts
- reports/ — final written report or presentation