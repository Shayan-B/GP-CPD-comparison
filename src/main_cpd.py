from src.flow_cpd import cpd

import warnings

import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")


def get_data_yfinance(ticker, start_date, end_date):
    """Helper function to load data using yfinance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    # Select desired columns (replace as needed)
    return data[["Close"]]


def run_module(lbw: int = 10) -> pd.DataFrame:
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

    change_points_df = cpd.run_module(
        price_data, lookback_window_length=lbw, output_csv_file_path=f"output_{lbw}.csv"
    )

    return change_points_df
