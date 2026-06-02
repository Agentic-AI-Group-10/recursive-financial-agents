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
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
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
    
    # Align series by taking the tail of the shorter period's EMA
    macd_line_full = short_ema_full[long_period-short_period:] - long_ema_full
    
    if len(macd_line_full) < signal_period:
        return macd_line_full, None, None
        
    signal_line_full = calculate_ema_series(macd_line_full, signal_period)
    
    # Align again for histogram
    histogram_full = macd_line_full[signal_period-1:] - signal_line_full
    
    return macd_line_full, signal_line_full, histogram_full

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Use Wilder's smoothing (equivalent to an EMA with alpha = 1/period)
    try:
        import pandas as pd
        return pd.Series(price_ranges).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except ImportError:
        atr_val = np.mean(price_ranges[:period])
        for i in range(period, len(price_ranges)):
            atr_val = (atr_val * (period - 1) + price_ranges[i]) / period
        return atr_val

def calculate_adx(prices, period=14):
    """Calculates the Average Directional Index (ADX) using only close prices."""
    if len(prices) < 2 * period:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    plus_dm = np.where((deltas > 0), deltas, 0)
    minus_dm = np.where((deltas < 0), -deltas, 0)
    
    # Approximate True Range with close-to-close volatility
    tr = np.abs(deltas)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.zeros(len(tr) - period + 1)
    smooth_plus_dm = np.zeros_like(atr)
    smooth_minus_dm = np.zeros_like(atr)
    
    atr[0] = np.mean(tr[:period])
    smooth_plus_dm[0] = np.mean(plus_dm[:period])
    smooth_minus_dm[0] = np.mean(minus_dm[:period])
    
    for i in range(1, len(atr)):
        idx = i + period - 1
        atr[i] = (atr[i-1] * (period - 1) + tr[idx]) / period
        smooth_plus_dm[i] = (smooth_plus_dm[i-1] * (period - 1) + plus_dm[idx]) / period
        smooth_minus_dm[i] = (smooth_minus_dm[i-1] * (period - 1) + minus_dm[idx]) / period
        
    # Avoid division by zero
    atr[atr == 0] = 1e-10
    
    plus_di = 100 * (smooth_plus_dm / atr)
    minus_di = 100 * (smooth_minus_dm / atr)
    
    di_sum = plus_di + minus_di
    di_sum[di_sum == 0] = 1e-10 # Avoid division by zero
    dx = 100 * (np.abs(plus_di - minus_di) / di_sum)
    
    # Smooth DX to get ADX
    adx = np.zeros(len(dx) - period + 1)
    adx[0] = np.mean(dx[:period])
    for i in range(1, len(adx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i + period - 1]) / period
        
    return adx[-1] if len(adx) > 0 else None

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a robust trend-strength filter and adaptive risk management.
    1.  ADX Trend Filter: The Average Directional Index (ADX) is now used to
        differentiate between trending and choppy markets. A new "Choppy Regime"
        (ADX < 20) is defined to prevent taking new positions during directionless
        periods, reducing whipsaw losses. Core trend signals require ADX > 22.
    2.  ATR-Based Trailing Stop: The static percentage stop-loss is replaced with
        a dynamic stop based on Average True Range (ATR), making it more responsive
        to changes in market volatility for superior capital preservation.
    3.  Capped Sentiment Score: The influence of the news sentiment score is now
        capped to prevent a single news cycle from overriding strong technical signals,
        balancing fundamental inputs with a robust quantitative framework.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5,
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
    
    # Cap sentiment score to prevent extreme influence
    net_sentiment_score = max(-5.0, min(5.0, net_sentiment_score))

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_VOL_PERIOD = 14
    ADX_PERIOD = 14
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(SMA_TREND_LONG + 1, 2 * ADX_PERIOD, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_VOL_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)
    roc_20 = ((all_prices[-1] - all_prices[-1 - ROC_CRASH_PERIOD]) / all_prices[-1 - ROC_CRASH_PERIOD]) * 100
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, adx, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_capitulation_candidate = roc_20 < -18.0 and rsi < 25
    is_crisis_regime = (current_price < sma_100 and atr > calculate_atr(all_prices, 100) * 1.75) or roc_20 < -15.0
    is_choppy_regime = adx < 20
    is_trending_regime = adx > 22

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD"

    # REGIME 3: CHOPPY MARKET
    if is_choppy_regime:
        # In choppy markets, avoid new entries. Only consider selling if extremely overbought.
        if rsi > 80:
            return "SELL"
        return "HOLD"

    # REGIME 4: TRENDING MARKET (or undefined transition state)
    
    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR-based Trailing Stop-Loss.
    if current_price < (donchian_high_20 - 2.5 * atr):
        return "SELL"

    # Priority 2: Confirmed trend breakdown signal.
    is_downtrend_confirmed = current_price < sma_50 and is_trending_regime
    is_momentum_crossing_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_downtrend_confirmed and is_momentum_crossing_down:
        return "SELL"

    # Priority 3: Profit-taking on exhaustion.
    is_exhausted = rsi > 75 and macd_hist_delta < 0
    if is_exhausted:
        return "SELL"

    # --- BUY LOGIC ---
    is_uptrend_confirmed = current_price > sma_50 and is_trending_regime
    is_momentum_crossing_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 72
    is_sentiment_supportive = net_sentiment_score > -1.5

    if is_uptrend_confirmed and is_momentum_crossing_up and is_not_overbought and is_sentiment_supportive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"