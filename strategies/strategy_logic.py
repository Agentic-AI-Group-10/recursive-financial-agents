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
        ema_values = []
        sma = sum(data_arr[0:period]) / period
        ema_values.append(sma)
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema = (data_arr[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        return np.array(ema_values)

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
    """Calculates Bollinger Bands and Band Width."""
    if len(prices) < period:
        return None, None, None, None, None
    prices_arr = np.array(prices, dtype=float)
    
    # Calculate band width series for squeeze detection
    rolling_mean = np.array([np.mean(prices_arr[i-period:i]) for i in range(period, len(prices_arr) + 1)])
    rolling_std = np.array([np.std(prices_arr[i-period:i]) for i in range(period, len(prices_arr) + 1)])
    
    upper_bands = rolling_mean + (rolling_std * num_std_dev)
    lower_bands = rolling_mean - (rolling_std * num_std_dev)
    
    # Avoid division by zero
    safe_rolling_mean = np.where(rolling_mean == 0, 1e-9, rolling_mean)
    band_widths = (upper_bands - lower_bands) / safe_rolling_mean
    
    return upper_bands[-1], rolling_mean[-1], lower_bands[-1], band_widths[-1], band_widths

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version integrates Bollinger Bands to create a more adaptive, volatility-aware
    system. Key improvements include:
    1.  Dynamic Entry/Exit Signals: Instead of fixed RSI levels, it uses Bollinger
        Bands to identify dynamic overbought/oversold conditions. This includes two
        new buy triggers: buying dips to the lower band in an uptrend and buying
        volatility breakouts from a "Bollinger Squeeze."
    2.  Enhanced Confirmation: Sell signals for profit-taking now require confirmation
        from both RSI and a price breach above the upper Bollinger Band, reducing
        premature exits in strong trends.
    3.  Refined Sentiment Context: The sentiment keyword dictionary is expanded with
        more nuanced economic terms like "GDP growth" and "consumer confidence" to
        better capture the macroeconomic environment.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "gdp growth": 2.0,
        "consumer confidence": 2.0, "de-escalation": 2.0, "short squeeze": 3.5,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        # Ambiguous
        "strong jobs report": 0.5,
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "liquidity crisis": -3.5,
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

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_SQUEEZE_LOOKBACK = 60 # Lookback for detecting lowest band width
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(SMA_TREND_LONG + 1, BB_PERIOD + BB_SQUEEZE_LOOKBACK, 65)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])
    
    # Calculate Bollinger Bands and Squeeze indicator
    bb_upper, bb_middle, bb_lower, bb_width, bb_width_series = calculate_bollinger_bands(all_prices, BB_PERIOD)
    is_bb_squeeze = bb_width < np.min(bb_width_series[-BB_SQUEEZE_LOOKBACK:-1]) if bb_width_series is not None and len(bb_width_series) >= BB_SQUEEZE_LOOKBACK else False

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, roc_20, bb_upper]) or macd_hist_series is None or len(macd_hist_series) < 2:
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
    # Priority 1: Dynamic Stop-Loss (7% drop from 20-day high).
    if current_price < (donchian_high_20 * 0.93):
        return "SELL"

    # Priority 2: Trend Breakdown Signal.
    is_trend_breakdown = current_price < sma_50 and all_prices[-2] >= sma_50
    is_momentum_confirming_down = macd_histogram < 0
    if is_trend_breakdown and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with fading momentum.
    is_overextended = current_price > bb_upper and rsi > 78
    is_momentum_fading = macd_hist_delta < 0
    if is_overextended and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50
    is_not_overbought = rsi < 75
    is_sentiment_permissive = net_sentiment_score > -3.0

    if is_primary_uptrend and is_not_overbought and is_sentiment_permissive:
        # Signal 1: Buy the dip to the lower Bollinger Band in an uptrend.
        buy_the_dip_signal = current_price <= bb_lower

        # Signal 2: Volatility breakout from a Bollinger Squeeze.
        breakout_signal = is_bb_squeeze and current_price > bb_upper

        # Signal 3: Classic momentum confirmation (MACD histogram crossover).
        momentum_crossover_signal = macd_histogram > 0 and prev_macd_histogram <= 0

        if buy_the_dip_signal or breakout_signal or momentum_crossover_signal:
            return "BUY"

    # Default action is to hold the current position.
    return "HOLD"