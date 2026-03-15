import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#1. DOWNLOAD STOCK DATA
# yfinance downloads historical price data for any stock ticker, this is currently running for Palintir, but can be changed e.g AAPL
df = yf.download("PLTR", start="2020-10-01", end="2024-01-01")
print(df.head())
print(df.shape)

# 2. CALCULATE MOVING AVERAGES
# MA20 = average closing price over last 20 days (short term trend)
# MA50 = average closing price over last 50 days (long term trend)
# rolling().mean() slides a window across the data calculating the average
df['MA20'] = df['Close']['PLTR'].rolling(window=20).mean()
df['MA50'] = df['Close']['PLTR'].rolling(window=50).mean()

# 3. PLOT PRICE AND MOVING AVERAGES 
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df.index, df['Close']['PLTR'], label="Close Price", color="#1f77b4")
ax.plot(df.index, df['MA20'], label="20 Day MA", color="orange")
ax.plot(df.index, df['MA50'], label="50 Day MA", color="green")
ax.set_title("Palantir Stock Close Price - Moving Averages",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha="right")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("PLTR_Stock_Price.png", dpi=150)
print("\n[Chart saved: PLTR_Stock_Price.png]")

# 4. GENERATE TRADING SIGNALS 
# Signal = 1 when MA20 is above MA50 (bullish/upward momentum)
# Signal = -1 when MA20 is below MA50 (bearish/downward momentum)
df['Signal'] = 0
df.loc[df['MA20'] > df['MA50'], 'Signal'] = 1
df.loc[df['MA20'] < df['MA50'], 'Signal'] = -1

# diff() finds where the signal CHANGES between days
# Change from -1 to 1 = diff of 2 = BUY signal (MA20 crossed above MA50)
# Change from 1 to -1 = diff of -2 = SELL signal (MA20 crossed below MA50)
df['Position'] = df['Signal'].diff()

# Filter dataframe to only the rows where a crossover happened
buy_signals  = df[df['Position'] == 2]
sell_signals = df[df['Position'] == -2]

#5. PLOT BUY AND SELL SIGNALS ON CHART 
# Green upward triangles = buy signals
# Red downward triangles = sell signals
# zorder=5 ensures markers appear on top of the price lines
ax.scatter(buy_signals.index, buy_signals['Close']['PLTR'],
           marker='^', color='green', s=200, label='Buy', zorder=5)
ax.scatter(sell_signals.index, sell_signals['Close']['PLTR'],
           marker='v', color='red', s=200, label='Sell', zorder=5)

#6. VALIDATE SIGNAL ALIGNMENT 
# Check signal counts and whether we start with a buy or sell
# We need to start with a buy — can't sell something we haven't bought
print(f"Buy signals: {len(buy_signals)}")
print(f"Sell signals: {len(sell_signals)}")
print(f"First real signal: {df[df['Position'].isin([2, -2])]['Position'].iloc[0]}")

# First signal is a sell (-2) meaning MA20 started below MA50
# Drop the first sell since we haven't bought anything yet
sell_signals = sell_signals.iloc[1:]

# 7. CALCULATE TRADE RETURNS
# Extract the closing prices at each buy and sell signal
buy_prices  = buy_signals['Close']['PLTR'].values
sell_prices = sell_signals['Close']['PLTR'].values

# Profit per trade = sell price minus buy price
profits = sell_prices - buy_prices

# Total profit and percentage return per trade
total_profit = profits.sum()
total_return = (profits / buy_prices * 100)

# 8. PRINT TRADE SUMMARY
print("Individual trades:")
for i, (b, s, p, r) in enumerate(zip(buy_prices, sell_prices, profits, total_return)):
    print(f"  Trade {i+1}: Buy £{b:.2f} → Sell £{s:.2f} | Profit: £{p:.2f} | Return: {r:.1f}%")

# Compare strategy returns vs simply holding the stock the whole time
print(f"\nTotal profit per share: £{total_profit:.2f}")
print(f"Buy and hold profit: £{(df['Close']['PLTR'].iloc[-1] - df['Close']['PLTR'].iloc[0]):.2f}")

plt.show()