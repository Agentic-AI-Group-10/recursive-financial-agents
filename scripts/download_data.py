#!/usr/bin/env python3
# ==============================================================================
# Data Downloader Script
# Location: /home/ow9800/recursive-financial-agents/scripts/download_data.py
# ==============================================================================
# This script downloads:
# 1. $SPY historical price data using yfinance (free, public API).
# 2. Daily Financial News for 6000+ Stocks using the Kaggle API (if configured).
#
# Usage:
#   python3 scripts/download_data.py --price-only
#   python3 scripts/download_data.py --interval 1h --start 2024-01-01 --end 2024-12-31

import os
import sys
import argparse

def install_package(package_name):
    """Installs a python package using pip."""
    import subprocess
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def check_and_install_dependencies():
    """Ensures yfinance and pandas are installed."""
    try:
        import pandas
    except ImportError:
        install_package("pandas")
    
    try:
        import yfinance
    except ImportError:
        install_package("yfinance")

def download_price_data(ticker, start_date, end_date, interval, output_path):
    """Downloads price data from Yahoo Finance."""
    print(f"\n--- Downloading Price Data for {ticker} ---")
    import yfinance as yf
    
    # Validation of intervals vs constraints
    # yfinance limits: 
    # - 1m data: max 7 days
    # - 5m/15m data: max 60 days
    # - 1h data: max 730 days (2 years)
    print(f"Requesting ticker: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if df.empty:
            print(f"Error: No data downloaded for {ticker}. Check dates and interval limits.")
            print("Note: yfinance limits 5m data to the last 60 days, and 1h data to the last 730 days.")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path)
        print(f"Success! Saved price data ({len(df)} rows) to: {output_path}")
        return True
    except Exception as e:
        print(f"An error occurred during price download: {e}")
        return False

def download_news_data(output_dir):
    """Downloads the financial news dataset using kagglehub."""
    print("\n--- Downloading Financial News from Kaggle ---")
    dataset_handle = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    print(f"Dataset: {dataset_handle}")
    
    try:
        import kagglehub
    except ImportError:
        install_package("kagglehub")
        import kagglehub
        
    try:
        print("Using kagglehub to download the latest version of the dataset...")
        # kagglehub automatically reads ~/.kaggle/kaggle.json for credentials
        downloaded_path = kagglehub.dataset_download(dataset_handle)
        print(f"Kagglehub downloaded files to: {downloaded_path}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Sync the downloaded files to our workspace directory to keep things self-contained
        import shutil
        print(f"Syncing files from cache to local workspace: {output_dir}")
        for item in os.listdir(downloaded_path):
            s = os.path.join(downloaded_path, item)
            d = os.path.join(output_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
                
        print(f"Success! Saved news data to local folder: {output_dir}")
        return True
    except Exception as e:
        print(f"An error occurred during kagglehub download: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SPY Price Data and Kaggle Financial News")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date YYYY-MM-DD (default: 2024-12-31)")
    parser.add_argument("--interval", type=str, default="1d", choices=["1m", "5m", "15m", "1h", "1d"], help="Data interval (default: 1d). Note limits (e.g. 1h max 2 years)")
    parser.add_argument("--price-out", type=str, default="data/spy_prices.csv", help="Path to save price CSV (default: data/spy_prices.csv)")
    parser.add_argument("--news-dir", type=str, default="data/news", help="Directory to save news files (default: data/news)")
    parser.add_argument("--price-only", action="store_true", help="Download only price data")
    parser.add_argument("--news-only", action="store_true", help="Download only news data")
    
    args = parser.parse_args()
    
    # 1. Install/Verify Dependencies (only for price download)
    if not args.news_only:
        check_and_install_dependencies()
        
    # 2. Run downloads
    if args.news_only:
        download_news_data(args.news_dir)
    elif args.price_only:
        download_price_data(args.ticker, args.start, args.end, args.interval, args.price_out)
    else:
        # Download both
        price_success = download_price_data(args.ticker, args.start, args.end, args.interval, args.price_out)
        news_success = download_news_data(args.news_dir)
