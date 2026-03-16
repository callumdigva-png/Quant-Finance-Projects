import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma):
    
    d1 = [np.log(S/K) + (r + (sigma ** 2)/2) * T] / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(call),float(put)

stock_prices = np.linspace(50,250,200)
K = 160      # strike price £160
T = 0.5      # 6 months (0.5 years)
r = 0.05     # 5% risk free rate
sigma = 0.2  # 20% volatility

call_prices = []
put_prices = []

for S in stock_prices:
    call, put = black_scholes(S, K, T, r, sigma)
    call_prices.append(call)
    put_prices.append(put)

print(f"Call option price: £{call:.2f}")
print(f"Put option price: £{put:.2f}")

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(stock_prices, call_prices, label="Call Price", color="green")
ax.plot(stock_prices, put_prices, label="Put Price", color="red")
ax.axvline(x=K, color='black', linestyle='--', label=f'Strike Price £{K}')

ax.set_title("Black-Scholes Option Prices vs Stock Price", fontsize=13, fontweight="bold")
ax.set_xlabel("Stock Price")
ax.set_ylabel("Option Price")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("black_scholes.png", dpi=150)
#plt.show()

def greeks(S, K, T, r, sigma):
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    
    d1 = (np.log(S/K) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_call = norm.cdf(d1)
    delta_put  = norm.cdf(d1) - 1
    gamma      = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega       = S * norm.pdf(d1) * np.sqrt(T)
    
    return delta_call, delta_put, gamma, vega

delta_call, delta_put, gamma, vega = greeks(150, 160, 0.5, 0.05, 0.2)
print(f"Call Delta : {delta_call:.4f}")
print(f"Put Delta  : {delta_put:.4f}")
print(f"Gamma      : {gamma:.4f}")
print(f"Vega       : {vega:.4f}")

delta_calls = []
delta_puts = []

for S in stock_prices:
    dc, dp, g, v = greeks(S, K, T, r, sigma)
    delta_calls.append(dc)
    delta_puts.append(dp)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(stock_prices, delta_calls, label="Call Delta", color="green")
ax.plot(stock_prices, delta_puts, label="Put Delta", color="red")
ax.axvline(x=K, color='black', linestyle='--', label=f'Strike Price £{K}')

ax.set_title("Black-Scholes Delta vs Stock Price", fontsize=13, fontweight="bold")
ax.set_xlabel("Stock Price")
ax.set_ylabel("Delta")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("black_scholes_delta.png", dpi=150)
plt.show()