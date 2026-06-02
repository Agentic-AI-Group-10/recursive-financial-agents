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
    # Ensure full series are calculated for alignment
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    
    # Align the series before subtraction
    macd_line = full_short_ema[long_period-short_period:] - full_long_ema
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    # Calculate signal line on the aligned MACD line
    signal_line_full = calculate_ema_series(macd_line, signal_period)
    
    # Align histogram calculation
    histogram = macd_line[signal_period-1:] - signal_line_full
    
    return macd_line[signal_period-1:], signal_line_full, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility for simplicity."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Using SMA for ATR calculation for simplicity and robustness without pandas
    if len(price_ranges) < period:
        return None
    return np.mean(price_ranges[-period:])

def calculate_adx(prices, period=14):
    """Calculates the Average Directional Index (ADX)."""
    if len(prices) < 2 * period:
        return None
    
    prices_arr = np.array(prices, dtype=float)
    up_moves = np.diff(prices_arr)
    down_moves = -up_moves

    plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0)
    minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0)
    
    tr = np.abs(np.diff(prices_arr)) # Simplified TR for close-only data

    # Using EMA-like smoothing (Wilder's smoothing)
    atr = np.zeros(len(tr))
    plus_di = np.zeros(len(tr))
    minus_di = np.zeros(len(tr))
    
    atr[period-1] = np.mean(tr[:period])
    smooth_plus_dm = np.mean(plus_dm[:period])
    smooth_minus_dm = np.mean(minus_dm[:period])

    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        smooth_plus_dm = (smooth_plus_dm * (period - 1) + plus_dm[i]) / period
        smooth_minus_dm = (smooth_minus_dm * (period - 1) + minus_dm[i]) / period
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr
        plus_di[atr == 0] = 0
        minus_di[atr == 0] = 0

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx)] = 0
    
    if len(dx) < period:
        return None
        
    adx = np.mean(dx[-period:]) # SMA of DX for final ADX value
    return adx

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces three major enhancements for improved adaptiveness and risk control:
    1.  Trend Strength Confirmation (ADX): A new buy condition requires the Average Directional
        Index (ADX) to be above 20. This filters out weak or sideways markets, reducing
        whipsaw trades and ensuring capital is deployed only in strong, established trends.
    2.  Dynamic Volatility-Based Stop-Loss: The fixed percentage stop-loss is replaced
        with a dynamic stop based on a multiple of the Average True Range (ATR). This allows
        the stop to widen during high volatility and tighten in calm markets, providing
        more intelligent risk management.
    3.  Parabolic Extension Profit-Taking: A new sell trigger is added to take profits
        when the price extends too far (e.g., >10%) above its 20-day EMA, capturing gains
        from unsustainable "blow-off top" scenarios before a sharp reversal.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0,
        "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0, "short squeeze": 3.5,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "bubble": -2.0, "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight
    
    # Cap sentiment score to prevent it from dominating technicals
    net_sentiment_score = max(min(net_sentiment_score, 7.0), -7.0)

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_MEDIUM = 50
    EMA_SHORT = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    STOP_LOSS_LOOKBACK = 25

    required_history_length = max(SMA_TREND_MEDIUM + 1, ADX_PERIOD * 2, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    ema_20_series = calculate_ema_series(all_prices, EMA_SHORT)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_50, rsi, atr, adx, donchian_high]) or macd_hist_series is None or len(macd_hist_series) < 2 or ema_20_series is None or len(ema_20_series) < 1:
        return "HOLD"

    ema_20 = ema_20_series[-1]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    
    # --- 3. Decision Logic (Hierarchical) ---

    # --- SELL LOGIC (Risk Management First) ---
    
    # Priority 1: Dynamic Volatility Stop-Loss. Exit if price drops more than 3 * ATR from recent high.
    stop_loss_price = donchian_high - (3.0 * atr)
    if current_price < stop_loss_price:
        return "SELL"

    # Priority 2: Parabolic Extension Profit-Taking. Sell if price is >10% above 20-day EMA.
    if current_price > (ema_20 * 1.10) and rsi > 75:
        return "SELL"

    # Priority 3: Standard trend breakdown signal.
    is_trend_down = current_price < sma_50
    is_momentum_crossing_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_down and is_momentum_crossing_down:
        return "SELL"

    # Priority 4: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_histogram < prev_macd_histogram
    is_overbought = rsi > 80
    if is_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_trend_up = current_price > sma_50
    is_momentum_crossing_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive = net_sentiment_score > -4.0
    
    # NEW: Trend Strength Confirmation using ADX
    is_strong_trend = adx > 20

    if is_trend_up and is_momentum_crossing_up and is_not_overbought and is_sentiment_permissive and is_strong_trend:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"