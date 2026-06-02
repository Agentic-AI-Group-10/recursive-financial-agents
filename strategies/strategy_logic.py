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
        # Using pandas is preferred for accuracy and standard implementation
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback pure-python EMA calculation
        ema_values = np.zeros(len(data_arr), dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[period-1:]

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
    short_ema_full = calculate_ema_series(prices, short_period)
    long_ema_full = calculate_ema_series(prices, long_period)
    
    # Align series before subtraction
    macd_line = short_ema_full[len(short_ema_full)-len(long_ema_full):] - long_ema_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align macd_line and signal_line for histogram calculation
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

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances the successful V2 strategy by increasing responsiveness and adaptability.
    1.  EMA-Based Trend: Replaces SMAs with more reactive EMAs (50, 100) for faster
        and more accurate trend identification.
    2.  Dynamic ATR Stop-Loss: Implements a volatility-adjusted trailing stop-loss
        (3x ATR below 20-day high), providing superior risk management that adapts
        to market conditions, tightening in calm markets and loosening in volatile ones.
    3.  Refined Regime Detection: The crisis/bear market regime is now primarily
        defined by a more stable EMA(50) vs EMA(100) cross, making the risk-off
        posture more robust against short-term noise.
    4.  Complacency Filter: A new filter prevents buying into over-extended, low-
        volatility market tops (high RSI + low ATR), preserving capital by avoiding
        entries with poor risk/reward profiles.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "disinflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5,
        "generative ai": 2.5, "stimulus": 2.0, "dovish": 2.0, "record high": 2.0,
        "bullish": 2.0, "strong earnings": 2.0, "beat estimates": 1.5, "recovery": 1.5,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5,
        "strong jobs report": 0.5, # Ambiguous
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "supply chain disruption": -2.0,
        "uncertainty": -1.5, "bubble": -2.0, "euphoria": -2.5, "mania": -3.0,
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

    # Indicator Periods & Parameters
    EMA_TREND_LONG = 100
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ATR_STOP_PERIOD = 20
    ATR_STOP_MULTIPLIER = 3.0
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(EMA_TREND_LONG + 1, ATR_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100_series = calculate_ema_series(all_prices, EMA_TREND_LONG)
    ema_50_series = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)
    ema_100 = ema_100_series[-1] if len(ema_100_series) > 0 else None
    ema_50 = ema_50_series[-1] if len(ema_50_series) > 0 else None
    
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    atr_20 = calculate_atr(all_prices, ATR_STOP_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_100, ema_50, rsi, short_atr, long_atr, atr_20, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    # Bear Market / Crisis Regime: High-risk environment defined by trend, volatility, or velocity.
    is_bear_market_structure = ema_50 < ema_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_bear_market_structure or is_high_volatility or is_crash_velocity

    # Capitulation Regime: An extreme subset of crisis, signaling a potential bottom.
    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold
    
    # Complacency Regime: Overbought with low volatility, a risky time to buy.
    is_complacent = rsi > 70 and short_atr < (long_atr * 0.9)

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    # Buy when there is blood in the streets, but only if momentum shows signs of turning.
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    # If in a general crisis (but not a specific capitulation buy signal), be defensive.
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < ema_50:
            return "SELL"
        return "HOLD" # Hold cash and wait for the storm to pass.

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic Volatility Stop-Loss (ATR-based).
    stop_price = donchian_high_20 - (ATR_STOP_MULTIPLIER * atr_20)
    if current_price < stop_price:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_trend_breakdown = current_price < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 3.0
    if is_trend_breakdown and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 82
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > ema_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.0
    is_sufficient_volatility = short_atr > (long_atr * 0.6) # Avoids entering dead, sideways markets.

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility and not is_complacent:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"