import warnings

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from matplotlib.patches import Rectangle

from src.gpcpd import gpcpd

warnings.filterwarnings("ignore")


def get_data_yfinance(ticker, start_date, end_date):
    """Helper function to load data using yfinance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    # Select desired columns (replace as needed)
    return data[["Close"]]


def plot_cpd_region(cpd_df: pd.DataFrame, threshold_val: float = 0.99):
    """Plot the output of the GPCPD module along price data"""
    # Your existing code
    ax = cpd_df.dropna(subset=["cp_score"])["Close"].plot(kind="line", figsize=(15, 7))
    ax1 = ax.twinx()

    # Get the data where cp_score > threshold_val
    filtered_data = cpd_df[cpd_df["cp_score"] > threshold_val].dropna(
        subset=["cp_score"]
    )

    # Plot colored rectangles
    for index, _ in filtered_data.iterrows():
        rect = Rectangle(
            (index, min(cpd_df["Close"])),
            dt.timedelta(days=1),
            max(cpd_df["Close"]) - min(cpd_df["Close"]),
            color="orange",
            alpha=0.4,
        )
        ax.add_patch(rect)

    ax1.plot(cpd_df["cp_score"].dropna(), color="red", linestyle="--")

    plt.title("Price data Vs.the marked high probability regions for change points")
    plt.show()
    return


def run_module(lbw: int = 10, plot_regions: bool = True) -> pd.DataFrame:
    """Run the module to detect changepoints.

    Args:
        lbw:
            LookBack window period.

    Returns:
        A DataFrame containing the ["date", "t", "cp_location", "cp_location_norm", "cp_score"]
        as columns.
    """
    price_data = get_data_yfinance("GOOG", "2023-07-01", "2024-03-15")
    price_data.loc[:, "daily_returns"] = price_data["Close"].pct_change()

    change_points_df = gpcpd.run_module(
        price_data, lookback_window_length=lbw, output_csv_file_path=f"output_{lbw}.csv"
    )

    if plot_regions:
        plot_cpd_region(cpd_df=change_points_df, threshold_val=0.99)

    return change_points_df
