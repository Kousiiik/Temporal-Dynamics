import pandas as pd
import numpy as np
from scipy import stats

# Load the CSV data
csv_filename = "bitcoin_price_inr.csv"
df = pd.read_csv(csv_filename)

# Extract the price column
prices = df["Price"].values

# Define the hypothesized mean price (you can change this value)
hypothesized_mean = 5000000  # Replace with your own value

# Perform a one-sample t-test
t_statistic, p_value = stats.ttest_1samp(prices, hypothesized_mean)

# Define significance level (alpha)
alpha = 0.05  # Commonly used significance level, you can change this

# Check if the p-value is less than alpha to determine significance
if p_value < alpha:
    print(f"The p-value ({p_value}) is less than alpha ({alpha}). Reject the null hypothesis.")
    print(f"The mean Bitcoin price is significantly different from {hypothesized_mean}.")
else:
    print(f"The p-value ({p_value}) is greater than alpha ({alpha}). Fail to reject the null hypothesis.")
    print(f"There is no significant difference in the mean Bitcoin price from {hypothesized_mean}.")
