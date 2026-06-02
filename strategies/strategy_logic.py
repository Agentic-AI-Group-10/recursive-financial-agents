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
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
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
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, high_prices=None, low_prices=None, period=14):
    """
    Calculates Average True Range (ATR).
    Uses close-to-close volatility as a fallback if high/low are not provided.
    """
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    # Using close-to-close volatility as a proxy for True Range
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_choppiness_index(prices, period=14):
    """
    Calculates the Choppiness Index (CHOP).
    Values closer to 100 indicate sideways movement (chop),
    while values closer to 0 indicate a strong trend (up or down).
    """
    if len(prices) < period + 1:
        return None
    
    sum_of_true_ranges = 0
    for i in range(len(prices) - period, len(prices)):
        true_range = abs(prices[i] - prices[i-1]) # Simplified TR using close prices
        sum_of_true_ranges += true_range

    highest_high = np.max(prices[-period:])
    lowest_low = np.min(prices[-period:])
    price_range = highest_high - lowest_low

    if price_range == 0 or sum_of_true_ranges == 0:
        return 100.0 # Max choppiness if no price movement

    chop = 100 * (math.log10(sum_of_true_ranges) - math.log10(price_range)) / math.log10(period)
    return chop

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances the successful V2 strategy with three key upgrades for adaptability and risk management:
    1.  Dynamic ATR-Based Stop-Loss: The fixed percentage stop-loss is replaced with a
        volatility-adjusted stop based on the Average True Range (ATR). This allows the
        stop to be tighter in calm markets and wider in volatile ones, improving capital preservation.
    2.  Sentiment Greed Filter: A specific sub-score for euphoric keywords ("mania", "bubble",
        "extreme greed") is introduced. If this score is excessively high, new BUY signals are
        blocked to avoid entering the market at points of maximum risk.
    3.  Choppiness Index Filter: A Choppiness Index is added to the BUY logic to detect
        sideways, non-trending markets. Trades are filtered out during high "chop" periods,
        reducing whipsaws and focusing capital on high-probability trend-following entries.
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
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0,
        "uncertainty": -1.5,
    }
    greed_keywords = {
        "euphoria": 3.0, "mania": 3.5, "irrational exuberance": 4.0, "extreme greed": 2.5, "bubble": 2.0
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    greed_score = 0.0

    for keyword_dict, score_var in [(sentiment_keywords, 'net_sentiment_score'), (greed_keywords, 'greed_score')]:
        current_score = 0
        for keyword, weight in keyword_dict.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, context_lower):
                pre_context = context_lower[max(0, match.start() - 30):match.start()]
                is_negated = any(neg_word in pre_context for neg_word in negation_words)
                current_score += -weight if is_negated else weight
        if score_var == 'net_sentiment_score':
            net_sentiment_score = current_score
        else:
            greed_score = current_score

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_STOP_PERIOD = 14
    CHOP_PERIOD = 14
    STOP_LOSS_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 2.5

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, period=ATR_STOP_PERIOD)
    chop_index = calculate_choppiness_index(all_prices, CHOP_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, chop_index, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_medium_term_downtrend = current_price < sma_50
    is_crisis_regime = is_long_term_downtrend and net_sentiment_score < -5.0

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CRISIS AVERSION (HIGHEST PRIORITY)
    # If in a long-term downtrend with very negative news, be defensive and sell.
    if is_crisis_regime:
        return "SELL"

    # REGIME 2: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR-Based Stop-Loss.
    stop_price = donchian_high_20 - (atr * ATR_STOP_MULTIPLIER)
    if current_price < stop_price:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_medium_term_downtrend and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_overbought = rsi > 80
    if is_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_medium_term_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 75
    is_trending_market = chop_index < 61.8 # Standard CHOP threshold for trending markets
    is_not_euphoric = greed_score < 4.0 # Greed filter
    is_sentiment_permissive = net_sentiment_score > -3.0

    if (is_medium_term_uptrend and
        is_momentum_confirming_up and
        is_not_overbought and
        is_trending_market and
        is_not_euphoric and
        is_sentiment_permissive):
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"