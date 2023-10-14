import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the CSV data
csv_filename = "usdt_price_inr.csv"
df = pd.read_csv(csv_filename)

# Extract the price column and convert it to numpy array
prices = df["Price"].values

# Define a function to check stationarity using the Augmented Dickey-Fuller test
def check_stationarity(data):
    result = adfuller(data)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("Stationary (Reject the Null Hypothesis)")
    else:
        print("Non-Stationary (Fail to Reject the Null Hypothesis)")

# Check stationarity of the data
check_stationarity(prices)

# Differencing to make the data stationary
differenced_prices = np.diff(prices)
check_stationarity(differenced_prices)

# Plot ACF and PACF to determine ARIMA parameters
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(differenced_prices, ax=ax1, lags=40)
plot_pacf(differenced_prices, ax=ax2, lags=40)
plt.show()

# Define ARIMA parameters
p = 1  # AR (AutoRegressive) order
d = 1  # Differencing order
q = 1  # MA (Moving Average) order

# Fit ARIMA model
model = ARIMA(prices, order=(p, d, q))
results = model.fit(disp=-1)

# Plot predicted vs. actual prices
plt.figure(figsize=(12, 6))
plt.plot(df.index, prices, label="Actual Prices", color="green")
plt.plot(df.index[1:], results.fittedvalues, label="Fitted Prices", color="blue")
plt.title("USDT Price Prediction with ARIMA")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.show()

# Make predictions for future days
future_days = 30  # Number of days to predict into the future
forecast, stderr, conf_int = results.forecast(steps=future_days)

# Create date range for future predictions
last_date = pd.to_datetime(df["Date"].iloc[-1])
date_range = pd.date_range(start=last_date, periods=future_days + 1, closed='right')

# Plot the predicted prices for future days
plt.figure(figsize=(12, 6))
plt.plot(df.index, prices, label="Actual Prices", color="green")
plt.plot(date_range[1:], forecast, label="Predicted Prices", color="blue")
plt.title("USDT Price Prediction for Future Days with ARIMA")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.show()
