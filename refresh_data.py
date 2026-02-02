"""
Daily Data Refresh Script

This script is designed to be run automatically (e.g., via a cron job or GitHub Actions)
to refresh the historical market data stored in the `data_parquet` directory.

It iterates through all existing `.parquet` files, identifies the stock symbols,
and uses the MarketDataLoader to efficiently download only the latest incremental data.

Note: Uses Indian Standard Time (IST) for date calculations since this is for
NSE/BSE Indian stock market data. NSE trading hours are 9:15 AM - 3:30 PM IST.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from utils import MarketDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Configuration ---
DATA_DIRECTORY = "data_parquet"

# Indian Standard Time offset (UTC+5:30)
IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def get_ist_dates() -> tuple[datetime, datetime]:
    """
    Calculate start and end dates using Indian Standard Time (IST).

    Returns dates in IST timezone for proper alignment with NSE/BSE trading calendar.
    The end date is set to yesterday (IST) to ensure we request data for completed
    trading days only, avoiding issues with requesting data for days that haven't
    completed trading yet.

    Returns:
        Tuple of (start_date, end_date) as timezone-naive datetime objects.
    """
    # Get current time in IST
    now_ist = datetime.now(IST_OFFSET)

    # Use yesterday as end date to ensure we get completed trading day data
    # This avoids issues when the script runs before market close
    end_date_ist = now_ist - timedelta(days=1)

    # Start date is 5 years before end date
    start_date_ist = end_date_ist - timedelta(days=5 * 365)

    # Return as timezone-naive for compatibility with pandas/yfinance
    # (yfinance handles timezone conversion internally for Indian stocks)
    return (
        start_date_ist.replace(tzinfo=None),
        end_date_ist.replace(tzinfo=None),
    )


# Load data for the last 5 years to ensure history is complete.
# The loader is smart and will only fetch missing data.
START_DATE, END_DATE = get_ist_dates()


def get_symbols_from_directory(directory: str) -> list[str]:
    """Extracts stock symbols from the filenames in the data directory."""
    p = Path(directory)
    if not p.exists():
        logging.warning(f"Data directory '{directory}' not found.")
        return []

    symbols = set()
    for f in p.glob("*.parquet"):
        # Assumes filename format like 'SYMBOL.NS_1d.parquet'
        symbol = f.name.split("_")[0]
        symbols.add(symbol)

    logging.info(f"Found {len(symbols)} unique symbols in '{directory}'.")
    return sorted(list(symbols))


def refresh_market_data(symbols: list[str]):
    """
    Refreshes market data for the given list of symbols.

    Args:
        symbols: A list of stock symbols to update.
    """
    if not symbols:
        logging.info("No symbols to refresh. Exiting.")
        return

    logging.info(f"Starting data refresh for {len(symbols)} symbols...")

    loader = MarketDataLoader(cache_dir=DATA_DIRECTORY, verbose=True)

    # The loader will intelligently handle caching and only download new data.
    loader.load_market_data(
        symbols=symbols,
        start_date=START_DATE,
        end_date=END_DATE,
        interval="1d",
        force_refresh=False,  # Important: Set to False to use incremental updates
        use_parallel=True,
        max_workers=8,  # Use more workers for faster network I/O
    )

    logging.info("Data refresh process completed successfully.")


if __name__ == "__main__":
    logging.info("--- Starting Daily Market Data Refresh ---")
    all_symbols = get_symbols_from_directory(DATA_DIRECTORY)
    refresh_market_data(all_symbols)
    logging.info("--- Data Refresh Finished ---")
