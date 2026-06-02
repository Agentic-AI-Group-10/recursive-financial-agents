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
        # Ensure data is a Series for ewm to work correctly
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        # Fallback pure-python EMA calculation
        if len(data_arr) < period:
            return np.array([])
        ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
        # Initial SMA for the first EMA point
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

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing method."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    # Handle cases where deltas might be too short for initial period
    if len(deltas) < period:
        return None

    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    
    avg_gain = seed_gains / period
    avg_loss = seed_losses / period
    
    # If there's no history beyond the seed period, calculate RSI based on seed
    if len(deltas) == period:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

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
    if len(prices) < long_period + signal_period - 1: # Ensure enough data for all EMAs
        return None, None, None
    
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)

    # Align the series for subtraction
    if len(short_ema_series) < len(long_ema_series):
        return None, None, None # Should not happen if long_period > short_period
    
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    if len(signal_line) == 0: # Handle case where signal_line could not be calculated
        return macd_line, None, None

    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(prices)):
        high_low = abs(prices[i] - prices[i-1]) # Using close-to-close as proxy for high-low
        high_prev_close = abs(prices[i] - prices[i-1]) # Using close-to-close as proxy for high-prev_close
        low_prev_close = abs(prices[i-1] - prices[i]) # Using close-to-close as proxy for low-prev_close
        true_ranges.append(max(high_low, high_prev_close, low_prev_close)) # Max of these three
    
    # In a real scenario, we'd need actual High, Low, Close.
    # Given only 'prices' (closing prices), we approximate True Range as abs(current_close - previous_close).
    # This is a simplification, but consistent with the input data.
    price_ranges = np.abs(np.diff(np.array(prices, dtype=float)))
    
    if len(price_ranges) < period:
        return None

    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    # Ensure the denominator is not zero to avoid division by zero error
    if prices[-1 - period] == 0:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version refines the parent strategy by focusing on robust risk management
    and enhanced regime adaptation, particularly during stress periods.
    1.  **ATR-based Trailing Stop:** Replaces the fixed percentage stop-loss with a
        dynamic stop based on Average True Range (ATR). This allows the stop to
        adapt to current market volatility, tightening in calm markets and
        widening in volatile ones, improving capital preservation.
    2.  **Reinforced Crisis Aversion:** Strengthens the defensive posture during
        general crisis regimes, prioritizing being in cash unless a high-conviction
        contrarian capitulation signal is present.
    3.  **Refined Sentiment Keywords:** Further updates to sentiment keywords to
        capture more nuanced market conditions and potential turning points.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5, # Ambiguous: good for economy, bad for inflation/rates
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "slowdown": -1.5, "inflationary pressure": -2.0,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "decline in", "ease"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            # Check a broader context for negation
            pre_context = context_lower[max(0, match.start() - 50):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14 # Using one ATR period for consistency
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20 # For Donchian High reference

    # ATR Stop Loss Multiplier - tuned for volatility adaptation
    ATR_STOP_MULTIPLIER = 2.75 # A common value is 2-3x ATR

    # Ensure enough history for all indicators
    required_history_length = max(SMA_TREND_LONG, RSI_PERIOD + 1, ATR_PERIOD + 1, ROC_CRASH_PERIOD + 1, STOP_LOSS_LOOKBACK, 26 + 9 -1) # MACD needs 26+9-1 days
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    
    # Donchian High for trailing stop reference
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all critical indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    # Crisis Regime: General high-risk environment
    is_long_term_downtrend = current_price < sma_100
    # High volatility check: ATR relative to a longer-term average (if available) or absolute threshold
    # For simplicity with one ATR, we'll use ROC as a proxy for 'crash velocity'
    is_high_volatility_proxy = atr > np.mean(all_prices[-ATR_PERIOD:]) * 0.015 # ATR is > 1.5% of average price
    is_crash_velocity = roc_20 < -12.0 # Slightly less extreme threshold for general crisis
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility_proxy) or is_crash_velocity or net_sentiment_score < -3.0

    # Capitulation Regime: An extreme subset of crisis, signaling a potential bottom
    is_deeply_oversold = rsi < 20 # More extreme oversold
    is_extreme_crash_velocity = roc_20 < -20.0 # More extreme crash
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold and net_sentiment_score < -2.0

    # --- 4. Decision Logic (Hierarchical) ---

    # Priority 1: ATR-BASED TRAILING STOP (Risk Management)
    # This is a dynamic stop-loss that adapts to volatility.
    # If the current price drops significantly below a recent high, adjusted by ATR.
    atr_stop_level = donchian_high_20 - (atr * ATR_STOP_MULTIPLIER)
    if current_price < atr_stop_level:
        return "SELL"

    # Priority 2: CONTRARIAN CAPITULATION (High Conviction Buy)
    # Buy when there is blood in the streets, but only if momentum shows signs of turning.
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # Priority 3: CRISIS AVERSION (Defensive Stance)
    # If in a general crisis (but not a specific capitulation buy signal), be defensive.
    # This ensures we prioritize being in cash during uncertain times.
    if is_crisis_regime:
        # If already in a downtrend or momentum is negative, sell. Otherwise, hold cash.
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD" # Hold cash and wait for the storm to pass.

    # REGIME 4: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC ---
    # Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 2.0 # Lower threshold for selling
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 80 # Slightly less extreme than V2's 82, to take profits earlier
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 75 # Slightly lower threshold to avoid buying into very extended rallies
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.0 # Higher threshold for buying
    is_sufficient_volatility = atr > (np.mean(all_prices[-ATR_PERIOD:]) * 0.005) # Avoids entering dead, sideways markets.

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"