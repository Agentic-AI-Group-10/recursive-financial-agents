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
        return ema_values

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
    
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    
    macd_line = full_short_ema[long_period-1:] - full_long_ema[long_period-1:]
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line_full = calculate_ema_series(macd_line, signal_period)
    signal_line = signal_line_full[signal_period-1:]
    
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

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands for the latest price point."""
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
    This version enhances the V2 strategy with a focus on dynamic, volatility-adjusted
    signals for improved robustness and risk management.
    1.  Dynamic Volatility Bands: Introduces Bollinger Bands for more adaptive
        overbought detection and profit-taking signals, replacing static RSI levels.
    2.  ATR-Based Trailing Stop: Replaces the fixed percentage stop-loss with a
        dynamic ATR-based trailing stop for better risk management in varying
        volatility regimes.
    3.  Enhanced Trend Confirmation: Uses responsive EMAs (20, 50, 100) for trend
        definition and adds a short-term EMA(20) check to confirm entry momentum,
        reducing whipsaws.
    4.  Capped Sentiment Score: Clamps the net sentiment score to prevent extreme
        news from overriding strong technical signals, ensuring a balanced approach.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5,
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
    
    # Clamp sentiment score to prevent excessive influence
    net_sentiment_score = np.clip(net_sentiment_score, -7.0, 7.0)

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    EMA_TREND_LONG = 100
    EMA_TREND_MEDIUM = 50
    EMA_TREND_SHORT = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_STD_DEV = 2.0
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 3.0

    required_history_length = EMA_TREND_LONG + 1
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_20_series = calculate_ema_series(all_prices, EMA_TREND_SHORT)
    ema_50_series = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)
    ema_100_series = calculate_ema_series(all_prices, EMA_TREND_LONG)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr_14 = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    upper_bb, _, _ = calculate_bollinger_bands(all_prices, BB_PERIOD, BB_STD_DEV)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [rsi, atr_14, roc_20, upper_bb, donchian_high_20]) or \
       macd_hist_series is None or len(macd_hist_series) < 2 or \
       len(ema_20_series) < required_history_length or \
       len(ema_50_series) < required_history_length or \
       len(ema_100_series) < required_history_length:
        return "HOLD"

    # Extract latest values from series
    ema_20 = ema_20_series[-1]
    ema_50 = ema_50_series[-1]
    ema_100 = ema_100_series[-1]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < ema_100
    is_high_volatility = atr_14 > (calculate_atr(all_prices, 50) * 1.75) if len(all_prices) > 51 else False
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < ema_50:
            return "SELL"
        return "HOLD"

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR-Based Trailing Stop-Loss.
    if current_price < (donchian_high_20 - ATR_STOP_MULTIPLIER * atr_14):
        return "SELL"

    # Priority 2: Trend breakdown signal.
    is_medium_term_downtrend = current_price < ema_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 3.0
    if is_medium_term_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_overbought_by_bb = current_price > upper_bb
    if is_overbought_by_bb and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_uptrend_structure = current_price > ema_50 and ema_50 > ema_100
    is_short_term_momentum_up = current_price > ema_20
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.0

    if (is_uptrend_structure and is_short_term_momentum_up and 
        is_momentum_confirming_up and is_not_overbought and 
        is_sentiment_permissive_for_buy):
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"