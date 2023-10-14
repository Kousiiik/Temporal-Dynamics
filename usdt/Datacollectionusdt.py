import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def fetch_crypto_data(coin_id, currency="inr"):
    base_url = "https://api.coingecko.com/api/v3"
    endpoint = f"/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": currency,
        "days": "365"  # Fetch data for the last 365 days (1 year)
    }
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code)
        return None

def perform_time_series_analysis(data):
    timestamps = data["prices"]
    prices = [price[1] for price in timestamps]  # Extract closing prices (INR)

    # Convert timestamps to a datetime index
    df = pd.DataFrame(prices, columns=["Price"])
    df.index = pd.to_datetime([timestamp[0] for timestamp in timestamps], unit="ms")

    # Save the DataFrame to a CSV file
    csv_filename = "usdt_price_inr.csv"
    df.to_csv(csv_filename)
    print(f"Data saved to {csv_filename}")

    # Perform time series analysis (e.g., visualize, analyze trends)
    # (You can remove the plotting and ADF test if not needed)

def main():
    coin_id = "tether"
    currency = "usd"
    
    data = fetch_crypto_data(coin_id, currency)
    if data:
        perform_time_series_analysis(data)

if __name__ == "__main__":
    main()
