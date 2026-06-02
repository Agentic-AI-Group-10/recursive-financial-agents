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
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    prices_slice = prices[-period:]
    sma = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces dynamic, volatility-adaptive logic for enhanced robustness.
    1.  Dynamic Thresholds via Bollinger Bands: Replaces static RSI levels for overbought/
        oversold conditions with Bollinger Bands, making profit-taking and entry signals
        more adaptive to the current volatility regime.
    2.  ATR-Based Trailing Stop-Loss: The static percentage stop-loss is upgraded to a
        dynamic ATR-based trailing stop, which tightens in low volatility and widens
        in high volatility, improving risk management.
    3.  Dual-Confirmation Entry Signal: The standard MACD crossover buy signal is now
        confirmed by requiring the price to be above a short-term EMA (20-day),
        reducing whipsaws and false entries in choppy markets.
    4.  Sentiment Override: A high-priority override is added for catastrophic news
        (e.g., "black swan"), forcing a risk-off "SELL" to protect capital during
        extreme events.
    """
    # --- 0. Strategy Parameters ---
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    EMA_CONFIRMATION = 20
    RSI_PERIOD = 14
    ATR_STOP_PERIOD = 14
    ATR_STOP_MULTIPLIER = 2.5
    BB_PERIOD = 20
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "soft landing": 2.5, "cooling inflation": 2.5,
        "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0,
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "rate hike": -2.5, "hawkish": -2.0,
        "sell-off": -2.0, "bubble": -2.0, "uncertainty": -1.5,
        # High-Impact Negative (Override Triggers)
        "black swan": -10.0, "systemic risk": -10.0, "contagion": -9.0, "credit crunch": -9.0,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    sentiment_override_sell = net_sentiment_score <= -8.0

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]
    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    ema_20_series = calculate_ema_series(np.array(all_prices, dtype=float), EMA_CONFIRMATION)
    ema_20 = ema_20_series[-1] if len(ema_20_series) > 0 else None
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr_14 = calculate_atr(all_prices, ATR_STOP_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None
    upper_bb, _, lower_bb = calculate_bollinger_bands(all_prices, BB_PERIOD)

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, ema_20, rsi, atr_14, roc_20, donchian_high_20, upper_bb]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # PRIORITY 1: SENTIMENT OVERRIDE
    if sentiment_override_sell:
        return "SELL"

    # PRIORITY 2: CONTRARIAN CAPITULATION
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # PRIORITY 3: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD" # Stay in cash during crisis unless a clear capitulation buy signal appears.

    # PRIORITY 4: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # 4a: Dynamic ATR Trailing Stop-Loss
    trailing_stop_price = donchian_high_20 - (ATR_STOP_MULTIPLIER * atr_14)
    if current_price < trailing_stop_price:
        return "SELL"

    # 4b: Trend breakdown signal
    is_trend_down = current_price < sma_50
    is_momentum_crossing_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_down and is_momentum_crossing_down:
        return "SELL"

    # 4c: Profit-taking on overbought conditions with FADING momentum
    is_overbought = current_price > upper_bb
    is_momentum_fading = macd_hist_delta < 0
    if is_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_trend_up = current_price > sma_50
    is_momentum_crossing_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_confirmed_by_price_action = current_price > ema_20 # DUAL CONFIRMATION
    is_not_chasing_peak = current_price < upper_bb # Avoid buying into extreme extensions

    if is_trend_up and is_momentum_crossing_up and is_confirmed_by_price_action and is_not_chasing_peak:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"