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
        ema_values = np.zeros_like(data_arr)
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
    if period == 0: return None # Avoid division by zero
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
    ema_long = calculate_ema_series(prices, long_period)
    ema_short = calculate_ema_series(prices, short_period)
    macd_line = ema_short[long_period-1:] - ema_long[long_period-1:]
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[signal_period-1:] - signal_line[signal_period-1:]
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Use Wilder's smoothing for ATR, consistent with RSI
    atr_values = np.zeros_like(price_ranges)
    atr_values[period-1] = np.mean(price_ranges[:period])
    for i in range(period, len(price_ranges)):
        atr_values[i] = (atr_values[i-1] * (period - 1) + price_ranges[i]) / period
    return atr_values[-1] if len(atr_values) > 0 else None

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands and Bandwidth."""
    if len(prices) < period:
        return None, None, None, None
    sma = np.mean(prices[-period:])
    std_dev = np.std(prices[-period:])
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    bandwidth = ((upper_band - lower_band) / sma) if sma != 0 else 0
    return upper_band, sma, lower_band, bandwidth

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY v2:
    This version evolves the prior successful model by:
    1.  Robust Scoring System: Replaces rigid if/else logic with a flexible scoring
        system, making decisions less brittle and more nuanced.
    2.  Dynamic ATR Trailing Stop: Upgrades the fixed-percentage stop-loss to a
        volatility-adaptive ATR trailing stop, improving risk management.
    3.  Volatility Squeeze Detection: Incorporates Bollinger Band Width to identify
        periods of consolidation, enhancing entry timing before major moves.
    4.  Refined Sentiment & Crisis Logic: Expands the sentiment dictionary and tunes
        crisis detection thresholds for greater accuracy in extreme conditions.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "disinflation": 2.5, "peak inflation": 2.0, "cpi miss": 2.5,
        "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "record high": 1.5, "bullish": 2.0,
        "strong earnings": 2.0, "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "de-escalation": 2.0, "goldilocks": 2.5,
        # Contrarian Positive (Fear)
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        # Negative
        "recession": -3.0, "crisis": -3.5, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -5.0, "systemic risk": -5.0, "contagion": -4.0, "credit crunch": -4.0,
        "liquidity crisis": -4.0, "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5,
        "geopolitical risk": -2.5, "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0,
        "bearish": -2.0, "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0,
        "bubble": -2.0, "uncertainty": -1.5,
        # Contrarian Negative (Greed)
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
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
    required_history_length = 101 # For 100-day SMA
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Indicator Periods
    SMA_LONG = 100
    SMA_MEDIUM = 50
    SMA_SHORT = 20
    RSI_PERIOD = 14
    ATR_VOL_SHORT = 10
    ATR_VOL_LONG = 50
    ATR_STOP_PERIOD = 14
    ROC_PERIOD = 20
    STOP_LOOKBACK = 20
    BB_PERIOD = 20

    # Calculate indicators
    sma_100 = calculate_sma(all_prices, SMA_LONG)
    sma_50 = calculate_sma(all_prices, SMA_MEDIUM)
    sma_20 = calculate_sma(all_prices, SMA_SHORT)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr_vol = calculate_atr(all_prices, ATR_VOL_SHORT)
    long_atr_vol = calculate_atr(all_prices, ATR_VOL_LONG)
    atr_for_stop = calculate_atr(all_prices, ATR_STOP_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOOKBACK:])
    _, _, _, bbw = calculate_bollinger_bands(all_prices, BB_PERIOD)

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, sma_20, rsi, short_atr_vol, long_atr_vol, atr_for_stop, roc_20, bbw]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_hist = macd_hist_series[-1]
    prev_macd_hist = macd_hist_series[-2]

    # --- 3. Regime Detection & Risk Management ---
    is_high_volatility = short_atr_vol > (long_atr_vol * 1.9)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (current_price < sma_100 and is_high_volatility) or is_crash_velocity

    # Dynamic ATR Trailing Stop
    atr_stop_multiplier = 3.0
    atr_trailing_stop = donchian_high_20 - (atr_for_stop * atr_stop_multiplier)

    # --- 4. Signal Generation (Scoring System) ---
    bullish_score = 0.0
    bearish_score = 0.0

    # Trend Signals
    if current_price > sma_50 and sma_50 > sma_100: bullish_score += 2.0 # Confirmed uptrend
    elif current_price > sma_50: bullish_score += 1.0
    if current_price < sma_50 and sma_50 < sma_100: bearish_score += 2.0 # Confirmed downtrend
    elif current_price < sma_50: bearish_score += 1.0

    # Momentum Signals
    if macd_hist > 0 and prev_macd_hist <= 0: bullish_score += 2.0 # Bullish Crossover
    if macd_hist < 0 and prev_macd_hist >= 0: bearish_score += 2.0 # Bearish Crossover
    if macd_hist > 0 and macd_hist > prev_macd_hist: bullish_score += 0.5 # Increasing bullish momentum
    if macd_hist < 0 and macd_hist < prev_macd_hist: bearish_score += 0.5 # Increasing bearish momentum

    # Overbought/Oversold Signals
    if rsi < 35: bullish_score += 1.0 # Oversold condition
    if rsi > 78: bearish_score += 1.5 # Overbought condition

    # Volatility Squeeze Signal
    if bbw < 0.06: # Bandwidth is tight, indicating a squeeze
        if current_price > sma_20: bullish_score += 1.5 # Primed for upside breakout
        if current_price < sma_20: bearish_score += 1.5 # Primed for downside breakout

    # Sentiment Overlay
    bullish_score += max(0, net_sentiment_score / 2.0)
    bearish_score -= min(0, net_sentiment_score / 2.0)

    # --- 5. Final Decision Logic ---

    # Priority 1: Crisis Aversion - Get out immediately
    if is_crisis_regime:
        return "SELL"

    # Priority 2: Dynamic Stop-Loss - Protect capital and profits
    if current_price < atr_trailing_stop:
        return "SELL"

    # Priority 3: Score-Based Trading Decisions
    BUY_THRESHOLD = 4.0
    SELL_THRESHOLD = 4.0

    if bullish_score >= BUY_THRESHOLD and bullish_score > bearish_score:
        return "BUY"

    if bearish_score >= SELL_THRESHOLD and bearish_score > bullish_score:
        return "SELL"

    # Default action is to hold the current position
    return "HOLD"