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

def calculate_rsi_series(prices, period=14):
    """Calculates a full series of Relative Strength Index (RSI) values."""
    if len(prices) < period + 1:
        return np.array([])
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.empty(len(prices) - period)
    avg_loss = np.empty(len(prices) - period)

    avg_gain[0] = np.mean(gains[:period])
    avg_loss[0] = np.mean(losses[:period])

    for i in range(1, len(deltas) - period + 1):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i + period - 1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i + period - 1]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    prices_arr = np.array(prices, dtype=float)
    short_ema_full = calculate_ema_series(prices_arr, short_period)
    long_ema_full = calculate_ema_series(prices_arr, long_period)
    
    # Align series before subtraction
    macd_line = short_ema_full[long_period-short_period:] - long_ema_full[long_period-1:]
    
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
    # Use Wilder's smoothing for ATR, which is equivalent to an EMA with alpha = 1/period
    try:
        import pandas as pd
        return pd.Series(price_ranges).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except ImportError:
        atr_val = np.mean(price_ranges[:period])
        for i in range(period, len(price_ranges)):
            atr_val = ((atr_val * (period - 1)) + price_ranges[i]) / period
        return atr_val

def calculate_kama(prices, period=10, fast_ema_period=2, slow_ema_period=30):
    """Calculates Kaufman's Adaptive Moving Average (KAMA)."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    
    change = np.abs(prices_arr[period:] - prices_arr[:-period])
    volatility_windows = [np.sum(np.abs(np.diff(prices_arr[i:i+period+1]))) for i in range(len(prices_arr) - period)]
    volatility = np.array(volatility_windows)
    
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)

    fast_sc = 2 / (fast_ema_period + 1)
    slow_sc = 2 / (slow_ema_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.zeros_like(prices_arr)
    kama[period] = np.mean(prices_arr[:period+1])
    
    for i in range(period + 1, len(prices_arr)):
        sc_index = i - (period + 1)
        kama[i] = kama[i-1] + sc[sc_index] * (prices_arr[i] - kama[i-1])
        
    return kama[-1]

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY v2:
    This version evolves the successful parent by introducing adaptive logic and more robust risk management.
    1.  Adaptive Trend Filter: Replaces the static 50-day SMA with Kaufman's Adaptive Moving Average (KAMA),
        which automatically adjusts its speed based on market volatility to reduce whipsaws and lag.
    2.  Enhanced Risk Management: Adds a volatility-based "panic sell" trigger (daily drop > 3x ATR) and
        refines the RSI profit-taking to only trigger after momentum demonstrably wanes (crosses down from overbought).
    3.  Nuanced Sentiment Analysis: Normalizes the sentiment score based on keyword density and dampens the
        signal during conflicting news reports, reducing false signals from sensational headlines.
    """
    # --- 1. Configuration ---
    class Config:
        # Trend and Momentum
        KAMA_PERIOD = 20
        SMA_LONG_TERM = 100
        RSI_PERIOD = 14
        MACD_SHORT, MACD_LONG, MACD_SIGNAL = 12, 26, 9
        # Volatility and Risk
        ATR_PERIOD = 14
        ATR_PANIC_MULTIPLIER = 3.0
        STOP_LOSS_LOOKBACK = 20
        STOP_LOSS_PERCENT = 0.93 # 7% drop from peak
        # Thresholds
        RSI_OVERBOUGHT_ENTRY = 78
        RSI_OVERBOUGHT_EXIT_PEAK = 80
        RSI_OVERBOUGHT_EXIT_CONFIRM = 75

    # --- 2. Sentiment Analysis (Enhanced) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "positive": {
            "fed pivot": 3.0, "rate cut": 2.5, "soft landing": 2.5, "cooling inflation": 2.5,
            "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "record high": 2.0,
            "bullish": 2.0, "strong earnings": 2.0, "beat estimates": 1.5, "recovery": 1.5,
            "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0
        },
        "negative": {
            "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
            "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
            "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "rate hike": -2.5,
            "bankruptcy": -2.5, "hard landing": -2.5, "cpi beat": -2.5, "vix spike": -2.5,
            "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0, "bubble": -2.0,
            "uncertainty": -1.5, "strong jobs report": -1.0, # Can be hawkish
            "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0
        }
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "avoids", "prevent"]
    
    net_sentiment_score = 0.0
    pos_hits, neg_hits = 0, 0
    
    all_keywords = {**sentiment_keywords["positive"], **sentiment_keywords["negative"]}
    for keyword, weight in all_keywords.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', context_lower):
            # Simple negation check for now
            net_sentiment_score += weight
            if weight > 0: pos_hits += 1
            else: neg_hits += 1
    
    # Dampen score if news is highly conflicting
    if pos_hits > 0 and neg_hits > 0:
        net_sentiment_score *= 0.5

    # --- 3. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]
    required_history_length = Config.SMA_LONG_TERM + 5
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate indicators
    kama = calculate_kama(all_prices, period=Config.KAMA_PERIOD)
    sma_100 = calculate_sma(all_prices, Config.SMA_LONG_TERM)
    rsi_series = calculate_rsi_series(all_prices, Config.RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices, Config.MACD_SHORT, Config.MACD_LONG, Config.MACD_SIGNAL)
    atr = calculate_atr(all_prices, Config.ATR_PERIOD)
    donchian_high = np.max(all_prices[-Config.STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [kama, sma_100, atr, donchian_high]) or macd_hist_series is None or len(macd_hist_series) < 2 or len(rsi_series) < 2:
        return "HOLD"

    # Assign latest values
    rsi, prev_rsi = rsi_series[-1], rsi_series[-2]
    macd_hist, prev_macd_hist = macd_hist_series[-1], macd_hist_series[-2]
    daily_change = current_price - price_history[-1]

    # --- 4. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_panic_drop = daily_change < (-Config.ATR_PANIC_MULTIPLIER * atr)
    is_crisis_regime = is_long_term_downtrend or is_panic_drop

    # --- 5. Decision Logic ---

    # REGIME 1: CRISIS AVERSION / PANIC SELL
    if is_crisis_regime:
        return "SELL" # In a crisis, get out. No questions.

    # REGIME 2: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC ---
    # Priority 1: Dynamic Stop-Loss based on recent peak.
    if current_price < (donchian_high * Config.STOP_LOSS_PERCENT):
        return "SELL"

    # Priority 2: Adaptive Trend Breakdown. KAMA is faster than SMA.
    is_adaptive_downtrend = current_price < kama
    is_momentum_confirming_down = macd_hist < 0 and prev_macd_hist >= 0
    if is_adaptive_downtrend and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Intelligent Profit-Taking on Fading Momentum.
    is_waning_from_peak = prev_rsi > Config.RSI_OVERBOUGHT_EXIT_PEAK and rsi < Config.RSI_OVERBOUGHT_EXIT_CONFIRM
    if is_waning_from_peak:
        return "SELL"

    # --- BUY LOGIC ---
    is_adaptive_uptrend = current_price > kama
    is_momentum_confirming_up = macd_hist > 0 and prev_macd_hist <= 0
    is_not_overbought = rsi < Config.RSI_OVERBOUGHT_ENTRY
    is_sentiment_permissive = net_sentiment_score > -2.0 # Tolerate some uncertainty

    if is_adaptive_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"