"""
Daily Data Refresh Script

This script is designed to be run automatically (e.g., via a cron job or GitHub Actions)
to refresh the historical market data stored in the `data_parquet` directory.

It iterates through all existing `.parquet` files, identifies the stock symbols,
and uses the MarketDataLoader to efficiently download only the latest incremental data.
"""

import logging
from datetime import datetime, timedelta
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
# Load data for the last 5 years to ensure history is complete.
# The loader is smart and will only fetch missing data.
START_DATE = datetime.now() - timedelta(days=5 * 365)
END_DATE = datetime.now()


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
