def leg_pnl(quantity, entry_price, exit_price):
    """Calculate gross profit or loss for one trading leg."""
    return quantity * (exit_price - entry_price)


short_b = leg_pnl(-1, 220, 216)
long_a = leg_pnl(2, 108, 108)

print("Short B:", short_b)
print("Long A:", long_a)
print("Pair gross P&L:", short_b + long_a)

def leg_fees(quantity, entry_price, exit_price, fee_rate):
    """Calculate the total fees for opening and closing one leg."""
    fees = abs(quantity) * (entry_price + exit_price) * fee_rate
    return fees


fee_rate = 0.0005
total_fees = leg_fees(-1, 220, 216, fee_rate) + leg_fees(2, 108, 108, fee_rate)
net_pnl = short_b + long_a - total_fees

print("Total fees:", round(total_fees, 4))
print("Pair net P&L:", round(net_pnl, 4))

hedge_ratio = 1.0
price_a = 108
price_b = 220
shares_b = -1

target_value_a = -shares_b * price_b * hedge_ratio
shares_a = target_value_a / price_a
print("Target value of A:", target_value_a)
print("Shares of A:", round(shares_a, 4))

exit_a = price_a * 1.02
exit_b = price_b * 1.01

pnl_a = leg_pnl(shares_a, price_a, exit_a)
pnl_b = leg_pnl(shares_b, price_b, exit_b)

print("Long A gross P&L:", round(pnl_a, 2))
print("Short B gross P&L:", round(pnl_b, 2))
print("Pair gross P&L:", round(pnl_a + pnl_b, 2))

fees_a = leg_fees(shares_a, price_a, exit_a, fee_rate)
fees_b = leg_fees(shares_b, price_b, exit_b, fee_rate)
scenario_net_pnl = (pnl_a + pnl_b) - (fees_a + fees_b)

print("Scenario fees:", round(fees_a + fees_b, 4))
print("Scenario net P&L:", round(scenario_net_pnl, 4))