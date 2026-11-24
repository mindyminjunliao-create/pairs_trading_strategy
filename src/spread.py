import math
from statistics import linear_regression

prices_a = [100, 105, 98, 110, 104]
prices_b = [200, 210, 196, 220, 208]

log_a = [math.log(price) for price in prices_a]
log_b = [math.log(price) for price in prices_b]

fit = linear_regression(log_a, log_b)

print("Hedge ratio:", round(fit.slope, 4))
print("Intercept:", round(fit.intercept, 4))

new_price_a = 108
new_price_b = 220

fitted_log_b = fit.intercept + fit.slope * math.log(new_price_a)
spread = math.log(new_price_b) - fitted_log_b

print("Fitted price of B:", round(math.exp(fitted_log_b), 2))
print("Log-price spread:", round(spread, 4))

from statistics import mean, stdev

noisy_prices_b = [201, 209, 197, 218, 209]
noisy_log_b = [math.log(price) for price in noisy_prices_b]
noisy_fit = linear_regression(log_a, noisy_log_b)

formation_spreads = []
for x, y in zip(log_a, noisy_log_b):
    fitted_y = noisy_fit.intercept + noisy_fit.slope * x
    formation_spreads.append(y - fitted_y)

new_fitted_log_b = noisy_fit.intercept + noisy_fit.slope * math.log(new_price_a)
new_spread = math.log(new_price_b) - new_fitted_log_b

spread_mean = mean(formation_spreads)
spread_sd = stdev(formation_spreads)
z_score = (math.log(new_price_b) - new_fitted_log_b - spread_mean) / spread_sd

print("New spread:", round(new_spread, 4))
print("Formation spread SD:", round(spread_sd, 4))
print("Z-score:", round(z_score, 2))