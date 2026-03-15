"""
Natural Gas Price Estimator
============================
This script:
  1. Loads monthly natural gas price data from a CSV file
  2. Visualises the data to reveal seasonal and trend patterns
  3. Builds a model that can estimate the price on ANY date
     (historical, within-data, or up to one year beyond the data)
  4. Lets you type in a date and get back a price estimate


REQUIRED FILE
-------------
  Nat_Gas.csv  — must be in the same folder as this script
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
# These are standard Python libraries we need for maths, dates, and graphs.

import pandas as pd          # reading CSV files and working with tables
import numpy as np           # maths and arrays
import matplotlib.pyplot as plt  # drawing graphs
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline  # smooth curve through data points
from scipy.stats import linregress          # simple trend line

# ── 1. LOAD THE DATA ──────────────────────────────────────────────────────────

CSV_FILE = "Nat_Gas.csv"   # <-- change this path if your CSV is in a different folder

df = pd.read_csv(CSV_FILE)
df["Dates"]  = pd.to_datetime(df["Dates"], dayfirst=False)
df["Prices"] = df["Prices"].astype(float)
df = df.sort_values("Dates").reset_index(drop=True)

# Convert dates to a plain number (days since 1970-01-01) so we can do maths on them
df["DateNum"] = df["Dates"].apply(lambda d: d.toordinal())

print("=" * 55)
print("  Natural Gas Price Data – Summary")
print("=" * 55)
print(f"  Records loaded : {len(df)}")
print(f"  Date range     : {df['Dates'].iloc[0].date()} → {df['Dates'].iloc[-1].date()}")
print(f"  Price range    : £{df['Prices'].min():.2f}  –  £{df['Prices'].max():.2f}")
print("=" * 55)

# ── 2. VISUALISE THE DATA ─────────────────────────────────────────────────────

# ── 2a. Raw time series ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df["Dates"], df["Prices"], marker="o", linewidth=2,
        color="#1f77b4", markersize=5, label="Actual monthly price")
ax.set_title("Natural Gas Prices – Monthly Snapshot (Oct 2020 – Sep 2024)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (£/therm or per unit)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha="right")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("plot_1_raw_prices.png", dpi=150)
print("\n[Chart saved: plot_1_raw_prices.png]")
plt.show()

# ── 2b. Seasonal pattern – average price by month of year ────────────────────
df["Month"] = df["Dates"].dt.month
monthly_avg = df.groupby("Month")["Prices"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(month_names, monthly_avg.values, color="#ff7f0e", edgecolor="black", alpha=0.85)
ax.set_title("Average Price by Month of Year\n(Seasonal Pattern Across All Years)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Average Price")
ax.grid(axis="y", alpha=0.3)
# Label each bar with its value
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plot_2_seasonal_pattern.png", dpi=150)
print("[Chart saved: plot_2_seasonal_pattern.png]")
plt.show()

# ── 2c. Year-on-year comparison ───────────────────────────────────────────────
df["Year"] = df["Dates"].dt.year
fig, ax = plt.subplots(figsize=(11, 5))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for i, (year, group) in enumerate(df.groupby("Year")):
    ax.plot(group["Month"], group["Prices"],
            marker="o", label=str(year), color=colors[i % len(colors)], linewidth=2)
ax.set_title("Price by Month – Year-on-Year Comparison", fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Price")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.legend(title="Year")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_3_year_on_year.png", dpi=150)
print("[Chart saved: plot_3_year_on_year.png]")
plt.show()

print("""
PATTERNS OBSERVED
-----------------
1. SEASONAL TREND : Prices tend to be HIGHER in winter (Nov–Jan) and
   LOWER in late spring / summer (Apr–Jun). This reflects heating demand.
2. LONG-TERM TREND: There is a gradual upward drift over the four years,
   suggesting general price inflation in the energy market.
3. The model below captures both of these effects.
""")

# ── 3. BUILD THE PRICE MODEL ──────────────────────────────────────────────────
#
# Strategy:
#   - Use a CubicSpline to interpolate smoothly BETWEEN known data points
#     (this gives exact-ish estimates for dates already in our range).
#   - For dates OUTSIDE our range we use:
#       (a) a linear trend fitted to all the data  (captures the drift)
#       (b) a seasonal correction based on the average monthly premium/discount
#           relative to the yearly average  (captures winter/summer pattern)
#
# This is a simple but transparent approach suitable for indicative pricing.

# 3a. Cubic spline (for interpolation within the data range)
spline = CubicSpline(df["DateNum"].values, df["Prices"].values)

# 3b. Linear trend across all data
slope, intercept, r_value, p_value, std_err = linregress(
    df["DateNum"].values, df["Prices"].values)

# 3c. Seasonal adjustment: deviation of each month's average from the overall mean
overall_mean = df["Prices"].mean()
seasonal_adj = {}
for m in range(1, 13):
    avg_m = df.loc[df["Month"] == m, "Prices"].mean()
    # If a month has no data, default to 0 adjustment
    seasonal_adj[m] = (avg_m - overall_mean) if not np.isnan(avg_m) else 0.0

# 3d. Extrapolation end – one year beyond the last data point
last_date   = df["Dates"].iloc[-1]
extrap_end  = last_date + pd.DateOffset(years=1)

def estimate_price(date_input):
    """
    Returns an estimated natural gas price for any given date.

    Parameters
    ----------
    date_input : str or datetime-like
        Any date string, e.g. "2024-03-15" or "15/03/2024"

    Returns
    -------
    float
        Estimated price (same units as the source data)
    """
    # Parse the date flexibly
    try:
        target_date = pd.to_datetime(date_input, dayfirst=False)
    except Exception:
        try:
            target_date = pd.to_datetime(date_input, dayfirst=True)
        except Exception:
            raise ValueError(f"Could not parse date: {date_input!r}. "
                             "Try a format like 'YYYY-MM-DD' or 'DD/MM/YYYY'.")

    first_date = df["Dates"].iloc[0]

    if target_date < first_date:
        # ── Before data: use trend + seasonal adjustment ──────────────────
        trend_price = slope * target_date.toordinal() + intercept
        season      = seasonal_adj.get(target_date.month, 0.0)
        price       = trend_price + season
        method      = "trend+seasonal (before data)"

    elif target_date <= last_date:
        # ── Within data range: cubic spline interpolation ─────────────────
        price  = float(spline(target_date.toordinal()))
        method = "cubic spline (interpolation)"

    elif target_date <= extrap_end:
        # ── Up to 1 year beyond data: trend + seasonal ────────────────────
        trend_price = slope * target_date.toordinal() + intercept
        season      = seasonal_adj.get(target_date.month, 0.0)
        price       = trend_price + season
        method      = "trend+seasonal (extrapolation)"

    else:
        # ── Beyond the allowed extrapolation window ────────────────────────
        raise ValueError(
            f"Date {target_date.date()} is more than one year beyond the last "
            f"data point ({last_date.date()}). Extrapolation that far is unreliable."
        )

    return price, method, target_date

# ── 4. PLOT: FULL PICTURE WITH EXTRAPOLATION ──────────────────────────────────

# Build a smooth curve over the entire range + 1-year extrapolation
all_dates  = pd.date_range(start=df["Dates"].iloc[0], end=extrap_end, freq="D")
all_prices = []

for d in all_dates:
    p, _, _ = estimate_price(d)
    all_prices.append(p)

fig, ax = plt.subplots(figsize=(14, 6))

# Shade the extrapolation region
ax.axvspan(last_date, extrap_end, alpha=0.08, color="orange",
           label="Extrapolation zone (1 year)")
ax.axvline(last_date, color="orange", linestyle="--", linewidth=1.2)

# Smooth model curve
ax.plot(all_dates, all_prices, color="green", linewidth=2,
        label="Model estimate (interpolation + extrapolation)")

# Actual data points on top
ax.scatter(df["Dates"], df["Prices"], color="#1f77b4", s=50, zorder=5,
           label="Actual monthly data", edgecolors="white", linewidths=0.5)

ax.set_title("Natural Gas Price Model – Historical + 1-Year Extrapolation",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_4_model_and_extrapolation.png", dpi=150)
print("[Chart saved: plot_4_model_and_extrapolation.png]")
plt.show()

# ── 5. INTERACTIVE DATE LOOKUP ────────────────────────────────────────────────

print("=" * 55)
print("  PRICE ESTIMATOR  –  type 'quit' to exit")
print("=" * 55)
print(f"  Data range  : {df['Dates'].iloc[0].date()} → {last_date.date()}")
print(f"  Extrapolates: up to {extrap_end.date()}")
print("  Date format : YYYY-MM-DD  or  DD/MM/YYYY")
print("=" * 55)

while True:
    user_input = input("\nEnter a date (or 'quit'): ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not user_input:
        continue

    try:
        price, method, parsed_date = estimate_price(user_input)
        print(f"\n  Date   : {parsed_date.strftime('%d %B %Y')}")
        print(f"  Price  : £{price:.4f}")
        print(f"  Method : {method}")
    except ValueError as e:
        print(f"  ⚠  {e}")
