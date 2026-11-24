import math
from statistics import correlation

def log_return(previous_price, current_price):
    """Calculate the natural log of the price ratio."""
    return math.log(current_price / previous_price) 

def simple_return(previous_price, current_price):
    """Calculate the simple return between two prices."""
    return (current_price - previous_price) / previous_price


print(simple_return(100, 110))
print(simple_return(110, 100))

up = log_return(100, 110)
down = log_return(110, 100)

print("Day 1 log return:", round(up, 6))
print("Day 2 log return:", round(down, 6))
print("Two-day log return:", round(up + down, 6))

prices_a = [100, 110, 100]
prices_b = [100, 90, 100]

for day in range(1, len(prices_a)):
    return_a = log_return(prices_a[day - 1], prices_a[day])
    return_b = log_return(prices_b[day-1], prices_b[day])
    print(f"Day {day}: A = {return_a:.2%}, B = {return_b:.2%}")

returns_a = [0.01, -0.02, 0.03, 0.00]
returns_b = [0.02, -0.04, -0.06, 0.00]

print("Correlation:", round(correlation(returns_a, returns_b), 3))

from statistics import linear_regression

fit = linear_regression(returns_a, returns_b)
print("Slope:", round(fit.slope,3))

fitted_b = fit.intercept + fit.slope * returns_a[3]
residual = returns_b[3] - fitted_b

print(f"Intercept: {fit.intercept:.2%}")
print(f"Day 4 fitted B: {fitted_b:.2%}")
print(f"Day 4 residual: {residual:.2%}")

residuals = []

residuals = []

for actual_a, actual_b in zip(returns_a, returns_b):
    fitted_b = fit.intercept + fit.slope * actual_a
    residual = actual_b - fitted_b  
    residuals.append(residual)

print("Residuals:", [round(value, 4) for value in residuals])
print("Residual sum:", round(sum(residuals), 10))