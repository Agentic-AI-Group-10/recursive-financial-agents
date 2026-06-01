#!/usr/bin/env python3
# ==============================================================================
# Simulation Pack Joiner Script
# Location: /home/ow9800/recursive-financial-agents/scripts/build_simulation_pack.py
# ==============================================================================
# This script joins $SPY historical price data with processed news headlines.
# It implements key professional optimizations:
# 1. Look-Ahead Safety: Strictly ensures news_timestamp < price_timestamp.
# 2. Timezone Normalization: Standardizes all dates to UTC.
# 3. Macro & Mega-Cap Filtering: Excludes corporate noise and keeps news 
#    high-signal (macro keywords and S&P 500 mega-caps).
# 4. Date Range Pruning: Discards news outside the price timeframe to run in seconds.

import os
import sys
import pandas as pd
import numpy as np

# List of mega-cap stocks heavily weighting the S&P 500 (and SPY itself)
TARGET_TICKERS = {'SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'GOOGL', 'META', 'TSLA'}

# Macroeconomic keywords that dictate general market sentiment
MACRO_KEYWORDS = [
    'fed', 'federal reserve', 'inflation', 'interest rate', 'rate hike', 
    'rate cut', 'cpi', 'gdp', 'tariff', 'yield', 'unemployment', 'treasury',
    'jobs report', 'recession', 'monetary policy', 'hawkish', 'dovish'
]

def load_price_data(price_path):
    """Loads and standardizes yfinance multi-index price data."""
    print(f"Loading price data from {price_path}...")
    
    # Read CSV, skipping secondary headers if multi-index
    # We inspect the header structure of yfinance:
    # Row 0: Price, Close, High, Low, Open, Volume
    # Row 1: Ticker, SPY, SPY, SPY, SPY, SPY
    # Row 2: Date,,,,,
    df = pd.read_csv(price_path, header=[0, 1], index_col=0)
    
    # Flatten the multi-index columns (e.g. ('Close', 'SPY') -> 'Close')
    df.columns = [col[0] for col in df.columns]
    
    # Clean index name
    df.index.name = 'Timestamp'
    
    # Parse index as datetime, timezone-naive first, then localize or convert to UTC
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    print(f"Loaded price data successfully. Range: {df.index.min()} to {df.index.max()} ({len(df)} rows)")
    return df

def load_news_data(news_paths, start_time, end_time, filter_signal=True):
    """Loads, filters, and standardizes news data within the active date range."""
    news_dfs = []
    
    for path in news_paths:
        if not os.path.exists(path):
            print(f"Warning: News file not found: {path}. Skipping...")
            continue
            
        print(f"Loading news from {path}...")
        
        # Determine column names based on file header
        # analyst_ratings_processed: title, date, stock
        # raw_partner_headlines: headline, date, stock
        header_df = pd.read_csv(path, nrows=5)
        cols_to_use = []
        rename_map = {}
        
        # Check for date column
        if 'date' in header_df.columns:
            cols_to_use.append('date')
            rename_map['date'] = 'date'
            
        # Check for stock/ticker column
        if 'stock' in header_df.columns:
            cols_to_use.append('stock')
            rename_map['stock'] = 'stock'
            
        # Check for title/headline column
        if 'title' in header_df.columns:
            cols_to_use.append('title')
            rename_map['title'] = 'headline'
        elif 'headline' in header_df.columns:
            cols_to_use.append('headline')
            rename_map['headline'] = 'headline'
            
        # Load optimized dataframe
        df = pd.read_csv(path, usecols=cols_to_use)
        df = df.rename(columns=rename_map)
        
        # Drop rows with missing values
        df = df.dropna(subset=['headline', 'date'])
        
        # Normalize news datetime column to UTC
        # News dates can be messy; handle format errors gracefully
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df = df.dropna(subset=['date'])
        
        # Step 1: Date Range Pruning (Pre-filter news within the trading window + 1 day buffer)
        buffer_start = start_time - pd.Timedelta(days=2)
        df = df[(df['date'] >= buffer_start) & (df['date'] <= end_time)]
        
        print(f"  Rows within date range: {len(df)}")
        news_dfs.append(df)
        
    if not news_dfs:
        print("Error: No news files loaded.")
        sys.exit(1)
        
    combined_news = pd.concat(news_dfs, ignore_index=True)
    combined_news = combined_news.sort_values(by='date')
    
    if not filter_signal:
        print(f"Skipping signal filtering (Unfiltered mode active). Kept {len(combined_news)} raw news rows.")
        return combined_news
        
    # Step 2: Signal Filtering (Macro keywords or Mega-Cap tickers)
    print("Filtering news for macro signal and mega-cap S&P 500 components...")
    
    # Pre-compile keyword checks and ticker sets
    stock_mask = combined_news['stock'].isin(TARGET_TICKERS)
    
    # Vectorized string search for macro keywords (case-insensitive)
    combined_news['headline_lower'] = combined_news['headline'].str.lower()
    macro_pattern = '|'.join(MACRO_KEYWORDS)
    macro_mask = combined_news['headline_lower'].str.contains(macro_pattern, regex=True, na=False)
    
    filtered_news = combined_news[stock_mask | macro_mask].copy()
    filtered_news = filtered_news.drop(columns=['headline_lower'])
    
    print(f"Filtered news rows from {len(combined_news)} down to {len(filtered_news)} high-signal rows.")
    return filtered_news

def build_simulation_pack(price_df, news_df, lookback_hours=24, max_headlines=-1):
    """Joins price timeline with news context using strict point-in-time constraints."""
    print(f"\n--- Joining Price and News (Look-back window: {lookback_hours} hours) ---")
    
    news_contexts = []
    lookback_delta = pd.Timedelta(hours=lookback_hours)
    
    # Convert news lists to numpy or fast structures for speed
    news_times = news_df['date'].values
    news_headlines = news_df['headline'].values
    news_tickers = news_df['stock'].values
    
    total_bars = len(price_df)
    print(f"Processing {total_bars} timeline bars...")
    
    for i, timestamp in enumerate(price_df.index):
        # Strict temporal boundaries: news_timestamp must be before the bar timestamp
        # AND within the look-back window (e.g. last 24h)
        # Convert to numpy datetime64 for seamless and fast comparison inside np.searchsorted
        cutoff_start = (timestamp - lookback_delta).to_datetime64()
        cutoff_end = timestamp.to_datetime64()
        
        # Fast search using numpy searchsorted (since news_df is sorted chronologically)
        start_idx = np.searchsorted(news_times, cutoff_start, side='left')
        end_idx = np.searchsorted(news_times, cutoff_end, side='left')  # side='left' ensures strict < cutoff_end
        
        # Extract headlines in the active range
        if start_idx < end_idx:
            batch_headlines = news_headlines[start_idx:end_idx]
            batch_tickers = news_tickers[start_idx:end_idx]
            
            # Limit number of headlines to prevent exceeding LLM context windows in unfiltered mode
            if max_headlines > 0 and len(batch_headlines) > max_headlines:
                batch_headlines = batch_headlines[-max_headlines:]
                batch_tickers = batch_tickers[-max_headlines:]
                
            # Format as: "TICKER: Headline 1 | TICKER: Headline 2"
            formatted_list = [f"[{t}]: {h}" for t, h in zip(batch_tickers, batch_headlines)]
            joined_context = " | ".join(formatted_list)
        else:
            joined_context = "No major news"
            
        news_contexts.append(joined_context)
        
        # Display progress every 10%
        if (i + 1) % max(1, total_bars // 10) == 0 or i == total_bars - 1:
            print(f"  Processed {i + 1}/{total_bars} bars ({(i + 1)/total_bars*100:.1f}%)")
            
    # Assign news column to price dataframe
    price_df['news_context'] = news_contexts
    return price_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create Simulation Pack (Unified Price + News timeline)")
    parser.add_argument("--price-file", type=str, default="data/spy_prices.csv", help="SPY price file path")
    parser.add_argument("--lookback", type=int, default=24, help="News look-back window in hours (default: 24)")
    parser.add_argument("--no-filter", action="store_true", help="Include all raw news without applying macro/mega-cap filters")
    parser.add_argument("--max-headlines", type=int, default=100, help="Maximum headlines per trading bar in unfiltered mode (default: 100, use -1 for unlimited)")
    parser.add_argument("--output", type=str, default=None, help="Path to save simulation pack (default: data/simulation_pack_[filtered/unfiltered].csv)")
    
    args = parser.parse_args()
    
    # Determine default output file name based on filtering
    if args.output is None:
        if args.no_filter:
            args.output = "data/simulation_pack_unfiltered.csv"
        else:
            args.output = "data/simulation_pack_filtered.csv"
            
    # Paths to the downloaded news files
    news_files = [
        "data/news/analyst_ratings_processed.csv",
        "data/news/raw_partner_headlines.csv",
        "data/news/raw_analyst_ratings.csv"
    ]
    
    # 1. Verify existence of required files
    if not os.path.exists(args.price_file):
        print(f"Error: Price file {args.price_file} not found. Please run the download script first!")
        sys.exit(1)
        
    # 2. Load and standardize prices
    price_df = load_price_data(args.price_file)
    price_start = price_df.index.min()
    price_end = price_df.index.max()
    
    # 3. Load and filter news
    news_df = load_news_data(news_files, price_start, price_end, filter_signal=not args.no_filter)
    
    # 4. Perform the temporal join
    simulation_pack = build_simulation_pack(
        price_df, news_df, 
        lookback_hours=args.lookback, 
        max_headlines=args.max_headlines if args.no_filter else -1
    )
    
    # 5. Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    simulation_pack.to_csv(args.output)
    print(f"\n🚀 Simulation pack created successfully! Saved {len(simulation_pack)} rows to: {args.output}")
