import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    # Using pandas is preferred for speed and accuracy if available.
    try:
        import pandas as pd
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        # Fallback to a manual calculation if pandas is not installed.
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

def calculate_rsi_series(prices, period=14):
    """Calculates the full series of Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return np.array([])
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    gains = np.where(deltas >= 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros_like(prices_arr)
    avg_loss = np.zeros_like(prices_arr)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices_arr)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0 # Handle case where avg_loss is zero
    return rsi[period:]

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    # Align series before subtraction
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align series before subtraction
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr_series(prices, period=14):
    """Calculates a series of Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return np.array([])
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    return calculate_ema_series(price_ranges, period)

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY v2:
    This version evolves the successful parent by introducing more adaptive mechanisms:
    1.  Volatility-Adjusted Momentum: MACD signals are now normalized by ATR, making
        them more comparable across different volatility regimes.
    2.  Adaptive Trailing Stop-Loss: The fixed 8% stop-loss is replaced with an
        ATR-based trailing stop, which widens in volatile markets and tightens in
        calm ones to reduce whipsaws and protect capital more effectively.
    3.  Capitulation-Based Re-entry: A new signal is added to the crisis regime to
        detect potential V-shaped bottoms, allowing for strategic re-entry after
        panic selling subsides, preventing prolonged cash holdings.
    """
    # --- 1. Sentiment Analysis (with Intensity) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "strong jobs report": -1.0,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
    }
    high_impact_keywords = {"black swan", "systemic risk", "crisis", "contagion", "war"}
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    
    net_sentiment_score = 0.0
    high_impact_count = 0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight
            if keyword in high_impact_keywords:
                high_impact_count += 1

    # Amplify score if multiple high-impact events are mentioned
    if high_impact_count > 1:
        net_sentiment_score *= 1.5

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ROC_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 3.5

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi_series = calculate_rsi_series(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr_series = calculate_atr_series(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2 or len(rsi_series) < 3 or len(atr_series) < 2:
        return "HOLD"

    # Extract latest values
    rsi = rsi_series[-1]
    prev_rsi = rsi_series[-2]
    atr = atr_series[-1]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    
    # Volatility-Adjusted Momentum
    vol_adj_macd_hist = macd_histogram / atr if atr > 0 else 0
    prev_vol_adj_macd_hist = prev_macd_histogram / atr_series[-2] if len(atr_series) > 1 and atr_series[-2] > 0 else 0

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    # --- 4. Decision Logic ---

    # REGIME 1: CRISIS AVERSION & CAPITULATION
    if is_crisis_regime:
        # CAPITULATION BUY SIGNAL: Look for V-shaped bottom opportunities
        is_deeply_oversold_recovering = rsi > prev_rsi and prev_rsi < 25
        is_momentum_reversing_sharply = vol_adj_macd_hist > prev_vol_adj_macd_hist and prev_vol_adj_macd_hist < -0.5
        is_sentiment_not_catastrophic = net_sentiment_score > -5.0
        
        if is_deeply_oversold_recovering and is_momentum_reversing_sharply and is_sentiment_not_catastrophic:
            return "BUY"

        # Standard Crisis SELL: Exit on any sign of weakness
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        
        return "HOLD" # Otherwise, hold cash and wait for clarity

    # REGIME 2: NORMAL / TREND-FOLLOWING

    # --- SELL LOGIC (Enhanced with Adaptive Stop-Loss) ---
    # Priority 1: Adaptive Trailing Stop-Loss based on ATR.
    stop_loss_price = donchian_high_20 - (atr * ATR_STOP_MULTIPLIER)
    if current_price < stop_loss_price:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = vol_adj_macd_hist < 0 and prev_vol_adj_macd_hist >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 1.0 # More sensitive sell trigger
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with fading momentum.
    is_momentum_fading = vol_adj_macd_hist < prev_vol_adj_macd_hist
    is_overbought = rsi > 78
    if is_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC (Enhanced with Volatility-Adjusted Momentum) ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = vol_adj_macd_hist > 0 and prev_vol_adj_macd_hist <= 0
    is_not_overbought = rsi < 75
    is_sentiment_supportive = net_sentiment_score >= 0.0 # Require neutral or positive news
    
    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_supportive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"