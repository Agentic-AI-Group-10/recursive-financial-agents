import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
        ema_values[0] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(1, len(ema_values)):
            ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing method."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    avg_gain = seed_gains / period
    avg_loss = seed_losses / period
    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta >= 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the middle, upper, and lower Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    sma = calculate_sma(prices, period)
    std_dev = np.std(prices[-period:])
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return sma, upper_band, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a composite "Market Health Score" and an ATR-based
    trailing stop for more dynamic, volatility-aware decision-making.
    1.  Market Health Score: A 0-100 score combining trend (price vs SMAs),
        momentum (RSI, MACD), and volatility (ATR ratio) to create a nuanced
        market regime filter, replacing the previous binary crisis mode.
    2.  ATR-Based Trailing Stop: The stop-loss is now dynamic, calculated as
        (Recent High - N * ATR), making it tighter in calm markets and looser
        in volatile ones for superior risk management.
    3.  Bollinger Band Breakouts: Entries are refined to look for breakouts
        confirmed by price action relative to Bollinger Bands, improving signal quality.
    4.  Sentiment-Modulated Thresholds: Extreme fear/greed news now dynamically
        adjusts RSI overbought/oversold levels, preventing chasing euphoria and
        enabling high-conviction contrarian entries.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "soft landing": 2.5, "cooling inflation": 2.5,
        "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "de-escalation": 2.0, "short squeeze": 3.5, "capitulation": 3.0,
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "black swan": -4.0, "systemic risk": -4.0,
        "credit crunch": -3.5, "rate hike": -2.5, "bankruptcy": -2.5, "hawkish": -2.0,
        "sell-off": -2.0, "bubble": -2.0, "uncertainty": -1.5,
        # Extreme Psychology (for RSI modulation)
        "extreme fear": 2.0, "panic selling": 2.5,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    rsi_sentiment_modifier = 0.0
    for keyword, weight in sentiment_keywords.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', context_lower):
            net_sentiment_score += weight
            if keyword in ["extreme greed", "euphoria", "mania", "irrational exuberance"]:
                rsi_sentiment_modifier = -5.0 # Be quicker to sell
            elif keyword in ["extreme fear", "panic selling", "capitulation"]:
                rsi_sentiment_modifier = 5.0 # Be slower to sell / more willing to buy

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_LONG = 100
    SMA_MEDIUM = 50
    SMA_SHORT = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    STOP_LOSS_LOOKBACK = 25
    ATR_STOP_MULTIPLIER = 2.5

    required_history_length = max(SMA_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_LONG)
    sma_50 = calculate_sma(all_prices, SMA_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(all_prices, SMA_SHORT)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, bb_mid, donchian_high]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Market Health Score (0-100) ---
    # Trend Component (40 points)
    trend_score = 0
    if current_price > sma_50: trend_score += 20
    if sma_50 > sma_100: trend_score += 20
    # Momentum Component (40 points)
    momentum_score = 0
    if rsi > 50: momentum_score += 20
    if macd_histogram > 0: momentum_score += 20
    # Volatility Component (20 points) - Lower volatility is good in uptrends
    volatility_score = 20 if atr < (np.mean(all_prices[-SMA_LONG:]) * 0.03) else 10
    
    market_health_score = trend_score + momentum_score + volatility_score

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY OVERRIDE)
    is_deeply_oversold = rsi < 25
    is_momentum_reversing_up = macd_histogram > prev_macd_histogram
    if is_deeply_oversold and is_momentum_reversing_up and net_sentiment_score < -3.0:
        return "BUY"

    # REGIME 2: RISK MANAGEMENT (SELL TRIGGERS)
    # Priority 1: Dynamic ATR-Based Trailing Stop
    atr_stop_level = donchian_high - (ATR_STOP_MULTIPLIER * atr)
    if current_price < atr_stop_level:
        return "SELL"

    # Priority 2: Extreme Market Deterioration (Low Health Score)
    if market_health_score < 30:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with fading momentum
    is_overbought = rsi > (80 + rsi_sentiment_modifier)
    is_momentum_fading = macd_histogram < prev_macd_histogram
    if is_overbought and is_momentum_fading:
        return "SELL"

    # REGIME 3: NORMAL MARKET CONDITIONS (BUY TRIGGERS)
    is_healthy_market = market_health_score > 70
    is_momentum_turning_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overextended = rsi < (75 + rsi_sentiment_modifier)
    is_above_short_term_trend = current_price > bb_mid # Price is above 20-day SMA
    is_sentiment_permissive = net_sentiment_score > -2.0

    if is_healthy_market and is_momentum_turning_up and is_not_overextended and is_above_short_term_trend and is_sentiment_permissive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"