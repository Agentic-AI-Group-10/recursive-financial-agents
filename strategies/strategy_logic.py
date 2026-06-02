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
        ema_values = np.zeros(len(data_arr), dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values[period-1:]

def calculate_sma_series(data, period):
    """Calculates a full series of Simple Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        import pandas as pd
        return pd.Series(data_arr).rolling(window=period).mean().to_numpy()
    except ImportError:
        sma_values = np.zeros(len(data_arr) - period + 1, dtype=float)
        for i in range(len(sma_values)):
            sma_values[i] = np.mean(data_arr[i:i+period])
        return sma_values

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
    # Ensure full series are returned for alignment
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    
    # Align series by taking the tail of the shorter period's EMA
    macd_line = full_short_ema[long_period-short_period:] - full_long_ema
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    # Pass only the MACD line to the EMA function
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align histogram
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Use EMA for ATR smoothing
    try:
        import pandas as pd
        atr_series = pd.Series(price_ranges).ewm(alpha=1/period, adjust=False).mean().to_numpy()
        return atr_series[-1] if len(atr_series) > 0 else None
    except ImportError:
        # Fallback calculation if pandas is not available
        atr_val = np.mean(price_ranges[:period])
        for i in range(period, len(price_ranges)):
            atr_val = ((atr_val * (period - 1)) + price_ranges[i]) / period
        return atr_val

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances the successful V2 strategy with three key upgrades:
    1.  Dynamic Volatility Stop-Loss: Replaces the fixed percentage stop with a
        Chandelier Exit (Highest High - N * ATR). This adapts risk management to
        current market volatility, tightening stops in calm markets and loosening
        them in volatile ones to avoid premature exits.
    2.  Trend Strength Confirmation: Introduces a filter based on the slope of the
        50-day SMA. A BUY signal now requires the underlying trend itself to be
        accelerating upwards, effectively filtering out entries in choppy,
        sideways markets where price may cross the SMA without true momentum.
    3.  Sentiment Veto System: Elevates sentiment analysis from a simple score
        modifier to a potential veto. Extremely negative news can now block new
        BUY signals, and overwhelmingly positive news can block non-stop-loss SELL
        signals, preventing trades against a powerful narrative.
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

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 22
    ATR_STOP_MULTIPLIER = 3.0
    SMA_SLOPE_PERIOD = 5

    required_history_length = max(SMA_TREND_LONG + SMA_SLOPE_PERIOD, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100_series = calculate_sma_series(all_prices, SMA_TREND_LONG)
    sma_50_series = calculate_sma_series(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_22 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [rsi, atr, roc_20, donchian_high_22]) or macd_hist_series is None or len(macd_hist_series) < 2 or len(sma_50_series) < SMA_SLOPE_PERIOD or len(sma_100_series) < 1:
        return "HOLD"

    # Derive state variables from indicators
    current_sma_100 = sma_100_series[-1]
    current_sma_50 = sma_50_series[-1]
    sma_50_slope = current_sma_50 - sma_50_series[-SMA_SLOPE_PERIOD]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime & Sentiment State Detection ---
    is_long_term_downtrend = current_price < current_sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # Sentiment Veto States
    has_sentiment_veto_buy = net_sentiment_score <= -3.5
    has_sentiment_veto_sell = net_sentiment_score >= 3.5

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0 and not has_sentiment_veto_buy:
        return "BUY"

    # REGIME 2: DYNAMIC RISK MANAGEMENT (SECOND HIGHEST PRIORITY)
    # Chandelier Exit: Sell if price closes below the highest high of the last N days minus a multiple of ATR.
    chandelier_exit_price = donchian_high_22 - (ATR_STOP_MULTIPLIER * atr)
    if current_price < chandelier_exit_price:
        return "SELL"

    # REGIME 3: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < current_sma_50:
            return "SELL"
        return "HOLD"

    # REGIME 4: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC ---
    # Priority 1: Trend breakdown signal.
    is_primary_downtrend = current_price < current_sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_primary_downtrend and is_momentum_confirming_down and not has_sentiment_veto_sell:
        return "SELL"

    # Priority 2: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading and not has_sentiment_veto_sell:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > current_sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_trend_strength_confirmed = sma_50_slope > 0 # Ensure the MA itself is trending up
    is_not_overbought = rsi < 75

    if is_primary_uptrend and is_momentum_confirming_up and is_trend_strength_confirmed and is_not_overbought and not has_sentiment_veto_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"