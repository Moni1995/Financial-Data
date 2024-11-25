import math
import numpy as np
import yfinance as yf
from scipy.stats import norm

# Step 1: Gather Historical Data for Macy's Inc. (M)
data = yf.download('M', start='2023-11-25', end='2024-11-25')

# Step 2: Calculate Daily Returns and Annualized Volatility
data['Daily Return'] = data['Adj Close'].pct_change()
sigma = 0.6444  # Annualized volatility

# Step 3: Set Parameters for Black-Scholes Model
S = 15.8  # Current stock price (latest closing price)
K = 16  # Strike price
T = 18 / 365  # Time to expiration in years (18 days from today)
r = 0.0435  # Risk-free interest rate (4.35%)

# Step 4: Define Black-Scholes Call and Put Option Pricing Functions
def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Step 5: Calculate Theoretical Call and Put Option Prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Step 6: Display Results
print(f"Theoretical Call Option Price for Macy's Inc. (M): ${call_price:.2f}")
print(f"Theoretical Put Option Price for Macy's Inc. (M): ${put_price:.2f}")
