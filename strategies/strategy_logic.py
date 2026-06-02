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
    short_ema_full_series = calculate_ema_series(prices, short_period)
    long_ema_full_series = calculate_ema_series(prices, long_period)
    
    # Ensure series are aligned from the end
    min_len = min(len(short_ema_full_series), len(long_ema_full_series))
    macd_line = short_ema_full_series[-min_len:] - long_ema_full_series[-min_len:]

    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    signal_line_full_series = calculate_ema_series(macd_line, signal_period)
    min_len_signal = min(len(macd_line), len(signal_line_full_series))
    
    histogram = macd_line[-min_len_signal:] - signal_line_full_series[-min_len_signal:]
    return macd_line, signal_line_full_series, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    prices_arr = np.array(prices[-period:], dtype=float)
    sma = np.mean(prices_arr)
    std_dev = np.std(prices_arr)
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a more robust, adaptive framework based on lessons from V2.
    1.  Dynamic Regime Filtering: Introduces Bollinger Bands to detect and filter out
        low-volatility, "choppy" market regimes, significantly reducing whipsaw trades
        that erode profits in sideways markets.
    2.  Enhanced Trend Confirmation: Replaces the single SMA-50 trend filter with a
        more responsive dual EMA (20/50) crossover system. This provides earlier and
        more reliable trend confirmation, improving entry and exit timing.
    3.  Adaptive Risk Management: The static percentage-based stop-loss is replaced
        with a dynamic, ATR-based trailing stop. This automatically adjusts the risk
        threshold based on current market volatility, protecting capital more
        effectively during flash crashes while giving trades room to breathe in
        calmer periods.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "dovish surprise": 3.0, "soft landing": 2.5,
        "cooling inflation": 2.5, "disinflation": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5,
        "strong jobs report": 0.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "hawkish surprise": -3.0, "bankruptcy": -2.5, "hard landing": -2.5,
        "geopolitical tensions": -2.5, "supply chain disruption": -2.5, "vix spike": -2.5,
        "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0,
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
    EMA_TREND_FAST = 20
    EMA_TREND_SLOW = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(EMA_TREND_SLOW + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_20_series = calculate_ema_series(all_prices, EMA_TREND_FAST)
    ema_50_series = calculate_ema_series(all_prices, EMA_TREND_SLOW)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [rsi, atr, bb_upper, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2 or len(ema_20_series) < 1 or len(ema_50_series) < 1:
        return "HOLD"

    ema_20 = ema_20_series[-1]
    ema_50 = ema_50_series[-1]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    # Crisis Regime: High-volatility, fear-driven market.
    is_long_term_downtrend = current_price < ema_50
    is_crisis_regime = is_long_term_downtrend and (atr / current_price) > 0.03 # 3% daily ATR

    # Choppy Market Regime: Low volatility, sideways movement.
    bollinger_bandwidth = (bb_upper - bb_lower) / bb_middle
    is_choppy_market = bollinger_bandwidth < 0.04 # Bands are tight (less than 4% width)

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CRISIS AVERSION (HIGHEST PRIORITY)
    # In a crisis, the primary goal is capital preservation. Exit all positions.
    if is_crisis_regime:
        return "SELL"

    # REGIME 2: CHOP ZONE AVOIDANCE
    # In a sideways, low-volatility market, avoid taking new positions to prevent whipsaws.
    if is_choppy_market:
        return "HOLD"

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR Stop-Loss. Sell if price drops 3x ATR from the 20-day high.
    if current_price < (donchian_high_20 - 3 * atr):
        return "SELL"

    # Priority 2: Trend Breakdown Signal. EMA fast crosses below slow, confirmed by momentum.
    is_trend_reversing_down = ema_20 < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_reversing_down and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with fading momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_overbought = rsi > 78
    is_price_extended = current_price > bb_upper
    if is_overbought and is_momentum_fading and is_price_extended:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = ema_20 > ema_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 75
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.0 # Stricter negative filter

    # Buy on a confirmed uptrend with positive momentum, but not if the market is already overheated.
    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"