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
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        ema_values = np.zeros_like(data_arr, dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
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
    prices_arr = np.array(prices, dtype=float)
    short_ema_full = calculate_ema_series(prices_arr, short_period)
    long_ema_full = calculate_ema_series(prices_arr, long_period)
    macd_line = short_ema_full[long_period-1:] - long_ema_full[long_period-1:]
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line_full = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[signal_period-1:] - signal_line_full[signal_period-1:]
    return macd_line, signal_line_full, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    prices_slice = prices[-period:]
    sma = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version refactors the decision-making process into a flexible composite
    scoring system, moving away from rigid binary rules. It enhances risk management
    with a volatility-adaptive ATR-based trailing stop and improves signal quality
    by incorporating Bollinger Bands for mean-reversion signals. Sentiment analysis
    is now frequency-weighted to better capture the magnitude of news themes. The
    proven crisis and capitulation regimes are retained as high-priority overrides
    to ensure robust performance during market stress.
    """
    # --- 1. Sentiment Analysis (Frequency-Weighted) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        matches = list(re.finditer(pattern, context_lower))
        if not matches:
            continue
        
        count = 0
        for match in matches:
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            count += -1 if is_negated else 1
        net_sentiment_score += count * weight

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]
    
    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    STOP_LOSS_LOOKBACK = 25

    required_history_length = max(SMA_TREND_LONG + 1, ATR_PERIOD + 1, BB_PERIOD + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, bb_upper]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection (Retained from Parent) ---
    is_long_term_downtrend = current_price < sma_100
    is_crisis_regime = is_long_term_downtrend and (atr > np.std(all_prices[-100:]) * 1.5)
    is_deeply_oversold = rsi < 25
    is_capitulation_candidate = is_crisis_regime and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: VOLATILITY-ADAPTIVE TRAILING STOP (RISK MANAGEMENT)
    atr_stop_price = donchian_high - (2.5 * atr)
    if current_price < atr_stop_price:
        return "SELL"

    # REGIME 3: CRISIS AVERSION
    if is_crisis_regime:
        return "SELL" # In a crisis, default to cash unless a capitulation buy is triggered.

    # REGIME 4: NORMAL MARKET - COMPOSITE SCORING SYSTEM
    bullish_score = 0.0
    bearish_score = 0.0

    # Trend Analysis
    if current_price > sma_50: bullish_score += 1.0
    else: bearish_score += 1.0
    if sma_50 > sma_100: bullish_score += 1.0
    else: bearish_score += 1.0

    # Momentum Analysis
    if macd_histogram > 0: bullish_score += 1.0
    else: bearish_score += 1.0
    if macd_hist_delta > 0: bullish_score += 1.5 # Accelerating momentum is a strong signal
    else: bearish_score += 1.5

    # Mean Reversion / Overbought-Oversold
    if rsi > 75: bearish_score += 2.0
    if rsi < 30: bullish_score += 2.0
    if current_price > bb_upper: bearish_score += 1.5 # Stretched price
    if current_price < bb_lower: bullish_score += 1.5 # Stretched price

    # Sentiment Overlay
    if net_sentiment_score > 2.0: bullish_score += 1.0
    if net_sentiment_score < -2.0: bearish_score += 1.0

    # Final Decision based on scores
    # Require a clear edge to place a trade, reducing whipsaws.
    if bullish_score >= 4.0 and bearish_score < 2.5:
        return "BUY"
    
    if bearish_score >= 4.0 and bullish_score < 2.5:
        return "SELL"

    # Default action is to hold the current position.
    return "HOLD"