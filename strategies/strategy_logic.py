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
        return ema_values[period-1:]

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
    
    # Align series by taking the tail of the shorter one
    macd_line = short_ema_full[len(short_ema_full)-len(long_ema_full):] - long_ema_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align again for histogram
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

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None, None, None
    prices_arr = np.array(prices[-period:], dtype=float)
    sma = np.mean(prices_arr)
    std_dev = np.std(prices_arr)
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    
    # Calculate bandwidth for squeeze detection
    bandwidth = (upper_band - lower_band) / sma if sma != 0 else 0
    return upper_band, sma, lower_band, bandwidth

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces Bollinger Bands to create a more adaptive,
    volatility-aware system, building on the successful regime-based parent.
    1.  Volatility Breakout Signal: A new Bollinger Band Squeeze detector
        identifies periods of low volatility. Buy/Sell signals are now
        prioritized when breaking out from these consolidations, aiming to
        catch nascent trends and filter out choppy market noise.
    2.  Dynamic Overbought/Oversold Levels: Static RSI thresholds for
        profit-taking are replaced by the upper Bollinger Band. A sell is
        triggered if the price exceeds the upper band and momentum fades,
        making the exit more responsive to current volatility.
    3.  Enhanced Sentiment Density Scoring: The sentiment engine now applies a
        multiplier if multiple strong keywords are detected, giving more weight
        to news cycles with a clear, unified narrative (e.g., multiple positive
        inflation reports) for higher conviction trades.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "ai revolution": 2.5, "stimulus": 2.0, "dovish": 2.0,
        "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0, "beat estimates": 1.5,
        "short squeeze": 3.5, "capitulation": 3.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "supply chain disruption": -2.5,
        "geopolitical tensions": -2.5, "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0,
        "bubble": -2.0, "uncertainty": -1.5, "euphoria": -2.5, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    positive_hits = 0
    negative_hits = 0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            score = -weight if is_negated else weight
            net_sentiment_score += score
            if score > 1.5: positive_hits += 1
            if score < -1.5: negative_hits += 1

    # Density multiplier
    if positive_hits >= 2: net_sentiment_score *= 1.2
    if negative_hits >= 2: net_sentiment_score *= 1.2

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20
    BBAND_PERIOD = 20

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])
    upper_bband, middle_bband, lower_bband, bband_width = calculate_bollinger_bands(all_prices, BBAND_PERIOD)
    
    # Calculate historical bandwidth to detect a squeeze
    _, _, _, prev_bband_width = calculate_bollinger_bands(all_prices[:-1], BBAND_PERIOD)
    is_bb_squeeze = bband_width < 0.06 and (prev_bband_width is not None and prev_bband_width > 0.06)

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_20, upper_bband]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD"

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic Stop-Loss.
    if current_price < (donchian_high_20 * 0.93):
        return "SELL"

    # Priority 2: Dynamic Profit-Taking on Over-extension.
    is_overextended = current_price > upper_bband
    is_momentum_fading = macd_hist_delta < 0
    if is_overextended and is_momentum_fading:
        return "SELL"

    # Priority 3: Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_primary_downtrend and is_momentum_confirming_down:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50 and current_price > middle_bband
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.0

    # Standard Buy Signal
    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"
        
    # Volatility Breakout Buy Signal (Higher Conviction)
    is_breakout_up = current_price > middle_bband and all_prices[-2] <= middle_bband
    if is_bb_squeeze and is_breakout_up and is_sentiment_permissive_for_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"