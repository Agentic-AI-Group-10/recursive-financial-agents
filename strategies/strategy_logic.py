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
        ema_values = np.zeros(len(data_arr), dtype=float)
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
    macd_line = short_ema_full[len(short_ema_full)-len(long_ema_full):] - long_ema_full
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

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands."""
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
    This version evolves V2 with a focus on adaptability and robustness.
    1.  Multi-Factor Scoring System: Replaces rigid boolean logic in normal markets
        with a scoring system that aggregates evidence for BUY/SELL signals from
        trend, momentum, and sentiment, reducing sensitivity to single thresholds.
    2.  Dynamic ATR-Based Stop-Loss: The fixed-percentage stop-loss is replaced
        with a dynamic stop based on a multiple of the Average True Range (ATR),
        allowing risk management to adapt to current market volatility.
    3.  Bollinger Band Integration: Incorporates Bollinger Bands to better identify
        mean-reversion entry points ("buy the dip") and exhaustion-based exits,
        adding a layer of volatility context to trading decisions.
    4.  Preservation of Crisis Logic: Retains the successful hierarchical regime
        detection from V2, ensuring that the high-conviction capitulation and
        crisis aversion logic overrides the normal scoring system during turmoil.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0,
        "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0, "beat estimates": 1.5,
        "short squeeze": 3.5, "capitulation": 3.0,
        "recession": -3.0, "crisis": -3.5, "stagflation": -3.5, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.5, "systemic risk": -4.5, "contagion": -4.0, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "supply chain disruption": -2.5,
        "uncertainty": -1.5, "euphoria": -2.5, "mania": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]
    
    # Indicator Periods
    EMA_TREND_LONG = 100
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20
    BBAND_PERIOD = 20
    
    required_history_length = max(EMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100_series = calculate_ema_series(all_prices, EMA_TREND_LONG)
    ema_50_series = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)
    ema_100 = ema_100_series[-1] if len(ema_100_series) > 0 else None
    ema_50 = ema_50_series[-1] if len(ema_50_series) > 0 else None
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])
    bband_upper, bband_mid, bband_lower = calculate_bollinger_bands(all_prices, BBAND_PERIOD)

    # Null check for all indicators
    if any(v is None for v in [ema_100, ema_50, rsi, atr, roc_20, bband_upper]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < ema_100
    is_high_volatility = atr > (np.mean([calculate_atr(all_prices[-100:-1], ATR_PERIOD) or atr]) * 1.8)
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
        if macd_histogram < 0 or current_price < ema_50:
            return "SELL"
        return "HOLD"

    # REGIME 3: NORMAL MARKET CONDITIONS (Scoring System)

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR-Based Stop-Loss.
    ATR_STOP_MULTIPLIER = 3.0
    atr_stop_level = donchian_high_20 - (ATR_STOP_MULTIPLIER * atr)
    if current_price < atr_stop_level:
        return "SELL"

    # --- Scoring System ---
    buy_score = 0.0
    sell_score = 0.0

    # Factor 1: Trend (Weight: 3)
    if current_price > ema_50 and ema_50 > ema_100:
        buy_score += 3.0 # Strong uptrend
    elif current_price < ema_50 and ema_50 < ema_100:
        sell_score += 3.0 # Strong downtrend
    elif current_price < ema_50:
        sell_score += 1.5 # Weakening trend

    # Factor 2: Momentum (Weight: 4)
    if macd_histogram > 0:
        buy_score += 2.0
        if macd_hist_delta > 0:
            buy_score += 2.0 # Positive and accelerating
    elif macd_histogram < 0:
        sell_score += 2.0
        if macd_hist_delta < 0:
            sell_score += 2.0 # Negative and accelerating

    # Factor 3: Overbought/Oversold (Weight: 3)
    if rsi > 80:
        sell_score += 3.0
    elif rsi > 70:
        sell_score += 1.5
    elif rsi < 30:
        buy_score += 2.0
    
    # Factor 4: Volatility (Bollinger Bands) (Weight: 2)
    if current_price < bband_lower:
        buy_score += 2.0 # Potential reversal buy
    elif current_price > bband_upper:
        sell_score += 2.0 # Potential exhaustion sell

    # Factor 5: Sentiment (Dynamic Weight)
    if net_sentiment_score > 1.0:
        buy_score += net_sentiment_score
    elif net_sentiment_score < -1.0:
        sell_score += abs(net_sentiment_score)

    # --- Final Decision from Scores ---
    BUY_THRESHOLD = 5.5
    SELL_THRESHOLD = 5.5

    if buy_score >= BUY_THRESHOLD and sell_score < (BUY_THRESHOLD / 2):
        return "BUY"
    
    if sell_score >= SELL_THRESHOLD and buy_score < (SELL_THRESHOLD / 2):
        return "SELL"

    # Default action is to hold the current position.
    return "HOLD"