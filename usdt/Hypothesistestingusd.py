import pandas as pd
from scipy import stats

def load_data(csv_filename):
    # Load the CSV data
    df = pd.read_csv(csv_filename)
    return df

def perform_t_test(before_date_prices, after_date_prices):
    # Perform a t-test
    t_statistic, p_value = stats.ttest_ind(before_date_prices, after_date_prices)
    return t_statistic, p_value

def main():
    csv_filename = "usdt_price_inr.csv"
    df = load_data(csv_filename)

    # Specify the date to divide the data into two groups
    split_date = "2022-01-01"

    # Divide the data into two groups based on timestamps
    before_date_prices = df[df["Timestamp"] < split_date]["Price"]
    after_date_prices = df[df["Timestamp"] >= split_date]["Price"]

    # Perform a t-test
    t_statistic, p_value = perform_t_test(before_date_prices, after_date_prices)

    # Set the significance level
    alpha = 0.05

    # Check if the p-value is less than alpha
    if p_value < alpha:
        print("Reject the null hypothesis (H0)")
        print("Mean Tether prices have changed significantly over time.")
    else:
        print("Fail to reject the null hypothesis (H0)")
        print("Mean Tether prices have not changed significantly over time.")

if __name__ == "__main__":
    main()
