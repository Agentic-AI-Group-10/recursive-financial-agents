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
        # Ensure data is a Series for ewm method
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback pure-python EMA calculation
        if len(data_arr) == 0:
            return np.array([])
        
        ema_values = np.zeros(len(data_arr), dtype=float)
        multiplier = 2 / (period + 1)
        
        # Initial SMA for the first EMA value
        if len(data_arr) < period: # Not enough data for even one EMA
            return np.array([])
        
        ema_values[period - 1] = np.mean(data_arr[:period])
        
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[period-1:] # Return only valid EMA values from period-1 onwards


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
    
    # Initial average gain/loss over the first 'period' deltas
    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    
    avg_gain = seed_gains / period
    avg_loss = seed_losses / period
    
    # Wilder's smoothing for subsequent periods
    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta >= 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # Handle division by zero
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    # Align the series for subtraction
    # The EMA series start at index `period - 1` of the original prices.
    # We need to find the common starting point for both EMA series.
    
    # Calculate the effective start index for each EMA relative to original prices
    short_ema_start_idx = short_period - 1
    long_ema_start_idx = long_period - 1
    
    # The MACD line can only start when both EMAs are available.
    # This means the MACD series starts at max(short_ema_start_idx, long_ema_start_idx)
    
    # Adjust short_ema_series to align with long_ema_series
    # If short_ema_series is longer, trim its beginning
    if len(short_ema_series) > len(long_ema_series):
        short_ema_aligned = short_ema_series[len(short_ema_series) - len(long_ema_series):]
        long_ema_aligned = long_ema_series
    elif len(long_ema_series) > len(short_ema_series):
        long_ema_aligned = long_ema_series[len(long_ema_series) - len(short_ema_series):]
        short_ema_aligned = short_ema_series
    else:
        short_ema_aligned = short_ema_series
        long_ema_aligned = long_ema_series

    if len(short_ema_aligned) == 0 or len(long_ema_aligned) == 0:
        return None, None, None

    macd_line = short_ema_aligned - long_ema_aligned
    
    if len(macd_line) < signal_period:
        return macd_line, np.array([]), np.array([]) # Not enough data for signal line
    
    signal_line = calculate_ema_series(macd_line.tolist(), signal_period) # Convert to list for ema calculation
    
    # Align macd_line with signal_line for histogram calculation
    if len(macd_line) > len(signal_line):
        macd_line_aligned_for_hist = macd_line[len(macd_line) - len(signal_line):]
    else:
        macd_line_aligned_for_hist = macd_line # Should not happen if signal_period < len(macd_line)

    histogram = macd_line_aligned_for_hist - signal_line
    
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR). Requires 'period' + 1 prices for initial TR."""
    if len(prices) < period + 1:
        return None
    
    highs = np.array(prices, dtype=float) # Assuming close prices are used for simplicity, or true high/low if available
    lows = np.array(prices, dtype=float) # In a real system, these would be separate
    closes = np.array(prices, dtype=float)

    true_ranges = []
    for i in range(1, len(closes)):
        tr1 = closes[i] - closes[i-1] # Simplified TR for close-to-close volatility
        # In a full ATR, it would be max(high-low, abs(high-prev_close), abs(low-prev_close))
        true_ranges.append(abs(tr1))
    
    if len(true_ranges) < period:
        return None

    atr_series = calculate_ema_series(true_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version builds upon V2 by specifically addressing underperformance in stress regimes
    and enhancing risk management.
    1.  Crisis Recovery Buy: Introduces a new high-priority buy signal during general crisis
        conditions when RSI bounces from deeply oversold levels and MACD momentum turns positive,
        aiming to capture V-shaped recoveries missed previously.
    2.  Dynamic ATR-based Stop-Loss: Replaces the fixed percentage stop-loss with a volatility-adaptive
        ATR-based trailing stop, providing more robust capital preservation across varying market conditions.
    3.  Refined Sentiment: Adjusts weights for contrarian keywords (e.g., "short squeeze", "capitulation")
        to increase their impact, and adds more nuanced economic keywords to better gauge market sentiment.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 4.0, "capitulation": 3.5, "panic selling": 3.0, "extreme fear": 2.5,
        "strong jobs report": 0.5, # Ambiguous: good for economy, bad for inflation/rates
        "supply chain recovery": 1.5, "strong consumer spending": 1.0,

        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
        "supply chain disruption": -1.5, "consumer spending slowdown": -1.0, "inflationary pressures": -2.0,
        "interest rate concerns": -2.0,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "decline in"]
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
    ATR_SHORT_PERIOD = 10
    ATR_LONG_PERIOD = 50
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20 # For Donchian High
    ATR_STOP_MULTIPLIER = 3.0 # Multiplier for ATR in stop-loss calculation
    RSI_OVERSOLD_THRESHOLD = 30 # For crisis recovery buy

    # Ensure enough history for all indicators
    # MACD needs long_period for EMA, then signal_period for signal line.
    # RSI needs period + 1. ATR needs period + 1.
    # For rsi_prev, we need prices up to yesterday, so all_prices[:-1] needs period + 1.
    required_history_length = max(SMA_TREND_LONG, ATR_LONG_PERIOD, ROC_CRASH_PERIOD, RSI_PERIOD + 2, 26 + 9) # 26+9 for MACD
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    
    rsi_current = calculate_rsi(all_prices, RSI_PERIOD)
    rsi_prev = calculate_rsi(all_prices[:-1], RSI_PERIOD) # RSI value from yesterday's close
    
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT_PERIOD)
    long_atr = calculate_atr(all_prices, ATR_LONG_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi_current, rsi_prev, short_atr, long_atr, roc_20, donchian_high_20]) or \
       macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    # Crisis Regime: General high-risk environment
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    # Capitulation Regime: An extreme subset of crisis, signaling a potential bottom
    is_deeply_oversold = rsi_current < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # Priority 1: Dynamic ATR-based Stop-Loss (Risk Management)
    # Exit if price drops below a trailing stop based on recent high and ATR
    if donchian_high_20 is not None and short_atr is not None:
        atr_stop_level = donchian_high_20 - (ATR_STOP_MULTIPLIER * short_atr)
        if current_price < atr_stop_level:
            return "SELL"

    # Priority 2: Crisis Recovery / Contrarian Capitulation BUYs
    # Attempt to buy during extreme fear or when a recovery from oversold conditions begins
    is_crisis_recovery_buy_signal = (
        is_crisis_regime and
        rsi_prev < RSI_OVERSOLD_THRESHOLD and # Was oversold yesterday
        rsi_current > RSI_OVERSOLD_THRESHOLD and # Recovering today
        macd_hist_delta > 0 # Momentum turning positive
    )
    
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"
    elif is_crisis_recovery_buy_signal:
        return "BUY"

    # Priority 3: Crisis Aversion (If not buying into recovery/capitulation)
    # If in a general crisis, be defensive.
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD" # Hold cash and wait for the storm to pass.

    # Priority 4: Normal Market SELL Logic
    # Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 3.0
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi_current > 82
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # Priority 5: Normal Market BUY Logic
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi_current < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.0
    is_sufficient_volatility = short_atr > (long_atr * 0.6) # Avoids entering dead, sideways markets.

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"