import logging

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt

from src.adaga import inducing_points

# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel(logging.ERROR)


def get_data(use_internet: bool = True, ticker: str = None):
    tse_file_path = "/content/drive/MyDrive/Colab Notebooks/TSE/price_panel.parquet"
    if use_internet:
        # ticker = ticker  # Replace with your desired stock ticker symbol
        start_date = "2022-09-01"
        end_date = "2024-03-16"  # Update with today's date
        data = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
        data = data.reset_index(drop=True)
        data_pct = data.pct_change().dropna()
    else:
        data = pd.read_parquet(tse_file_path)
        data = data[ticker].dropna().reset_index(drop=True)
        data = data[-200:].reset_index(drop=True)
        data_pct = data.pct_change().dropna()

    return data, data_pct


def plot_data(plot_data: pd.DataFrame, change_points: list, ticker_name: str):
    # Plot stock price data with change points
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(plot_data.index, plot_data.values)
    ax.scatter(
        change_points,
        plot_data[change_points],
        marker="o",
        color="red",
        label="Change Points",
    )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price")
    plt.yscale("log")
    plt.title(f"Stock Price for {ticker_name} with Detected Change Points")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("Output.png")

    plt.show()
    return


def run_module():
    ticker, use_net = "GOOG", True

    func_data = get_data(use_internet=use_net, ticker=ticker)
    input_data = func_data[0]

    # Set hyperparameters (adjust as needed)
    regionalization_delta = 0.9
    regionalization_min_w_size = 3
    regionalization_n_ind_pts = 4
    regionalization_kern = (
        "Matern12"  # Choose a kernel function (e.g., "RBF", "Matern52", etc.)
    )
    regionalization_batch_size = 1

    # Create AdaptiveRegionalization object
    regionalization = inducing_points.AdaptiveRegionalization(
        domain_data=input_data.index.values.reshape(-1, 1),
        system_data=input_data.values.reshape(-1, 1),
        delta=regionalization_delta,
        min_w_size=regionalization_min_w_size,
        n_ind_pts=regionalization_n_ind_pts,
        seed=1234,
        batch_size=regionalization_batch_size,
        kern=regionalization_kern,
    )

    try:
        # Perform regionalization
        regionalization.regionalize()
    except ValueError as e:
        print(e)
        pass

    # Extract detected change points
    change_points = [
        window["window_start"] for window in regionalization.closed_windows
    ]

    plot_data(func_data[0], change_points, ticker)
