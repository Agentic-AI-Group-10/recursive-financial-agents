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
    gains = np.where(deltas >= 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi_values = np.zeros(len(prices) - period)
    
    if avg_loss == 0:
        rs = 100.0
    else:
        rs = avg_gain / avg_loss
    rsi_values[0] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rs = 100.0
        else:
            rs = avg_gain / avg_loss
        rsi_values[i - period + 1] = 100.0 - (100.0 / (1.0 + rs))
        
    return rsi_values

def calculate_stoch_rsi(prices, rsi_period=14, stoch_period=14, k_period=3):
    """Calculates the Stochastic RSI %K value."""
    rsi_series = calculate_rsi_series(prices, rsi_period)
    if len(rsi_series) < stoch_period:
        return None
    
    stoch_rsi_values = np.zeros(len(rsi_series) - stoch_period + 1)
    for i in range(stoch_period - 1, len(rsi_series)):
        rsi_window = rsi_series[i - stoch_period + 1 : i + 1]
        min_rsi = np.min(rsi_window)
        max_rsi = np.max(rsi_window)
        if max_rsi == min_rsi:
            stoch_rsi_values[i - stoch_period + 1] = 100.0
        else:
            stoch_rsi_values[i - stoch_period + 1] = (rsi_series[i] - min_rsi) / (max_rsi - min_rsi) * 100.0
            
    if len(stoch_rsi_values) < k_period:
        return None
        
    # Calculate %K as a simple moving average of the StochRSI values
    k_value = np.mean(stoch_rsi_values[-k_period:])
    return k_value / 100.0 # Normalize to 0-1 range

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    short_ema_full = calculate_ema_series(prices, short_period)
    long_ema_full = calculate_ema_series(prices, long_period)
    
    # Align series by taking the tail of the shorter period EMA
    macd_line = short_ema_full[-len(long_ema_full):] - long_ema_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[-len(signal_line):] - signal_line
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
    This version enhances the successful V2 strategy with three key upgrades:
    1.  Dynamic ATR Stop-Loss: Replaces the static 7% stop-loss with a dynamic,
        volatility-adjusted stop based on the Average True Range (ATR). This
        adapts risk management to current market conditions, tightening during
        calm periods and loosening during volatility.
    2.  Stochastic RSI for Precision: Integrates the Stochastic RSI oscillator
        to provide more sensitive and timely overbought/oversold signals,
        improving entry/exit precision compared to using RSI alone.
    3.  Sentiment Circuit Breaker: Introduces a high-conviction sentiment
        override. Extremely negative news can now trigger a defensive SELL,
        acting as a circuit breaker against unforeseen "black swan" events.
    """
    # --- 1. Strategy Parameters ---
    # Trend
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    # Momentum
    RSI_PERIOD = 14
    STOCH_RSI_PERIOD = 14
    STOCH_K_PERIOD = 3
    ROC_CRASH_PERIOD = 20
    # Volatility & Risk
    ATR_SHORT = 10
    ATR_LONG = 50
    STOP_LOSS_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 3.0
    # Thresholds
    RSI_CAPITULATION = 25
    ROC_CRASH_VELOCITY = -18.0
    STOCH_RSI_OVERBOUGHT = 0.90
    STOCH_RSI_OVERSOLD = 0.15
    SENTIMENT_SELL_OVERRIDE = -8.0

    # --- 2. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "disinflation": 2.5, "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5,
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0,
        "strong earnings": 2.0, "beat estimates": 1.5, "recovery": 1.5,
        "de-escalation": 2.0, "short squeeze": 3.5, "capitulation": 3.0,
        "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical tensions": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0,
        "supply chain disruption": -2.5, "uncertainty": -1.5,
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

    # --- 3. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]
    required_history_length = max(SMA_TREND_LONG + 1, ATR_LONG + 1, RSI_PERIOD + STOCH_RSI_PERIOD + STOCH_K_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi_series(all_prices, RSI_PERIOD)[-1] if len(calculate_rsi_series(all_prices, RSI_PERIOD)) > 0 else 50.0
    stoch_rsi_k = calculate_stoch_rsi(all_prices, RSI_PERIOD, STOCH_RSI_PERIOD, STOCH_K_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, stoch_rsi_k, short_atr, long_atr, roc_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 4. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity
    is_capitulation_candidate = (roc_20 < ROC_CRASH_VELOCITY) and (rsi < RSI_CAPITULATION)

    # --- 5. Decision Logic (Hierarchical) ---

    # PRIORITY 0: SENTIMENT CIRCUIT BREAKER
    if net_sentiment_score < SENTIMENT_SELL_OVERRIDE:
        return "SELL"

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
    # Priority 1: Dynamic ATR Stop-Loss.
    stop_price = donchian_high_20 - (ATR_STOP_MULTIPLIER * short_atr)
    if current_price < stop_price:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_primary_downtrend and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_stoch_rsi_overbought = stoch_rsi_k > STOCH_RSI_OVERBOUGHT
    if is_stoch_rsi_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_stoch_rsi_not_overbought = stoch_rsi_k < STOCH_RSI_OVERBOUGHT
    is_pullback_in_uptrend = stoch_rsi_k < STOCH_RSI_OVERSOLD # Optional: buy dips

    # Primary Buy Signal: Trend and momentum alignment, not overbought.
    if is_primary_uptrend and is_momentum_confirming_up and is_stoch_rsi_not_overbought:
        return "BUY"
    
    # Secondary Buy Signal: Buy a small dip (oversold StochRSI) within a strong primary uptrend.
    if is_primary_uptrend and macd_histogram > 0 and is_pullback_in_uptrend:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"