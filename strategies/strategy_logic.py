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
        ema_values = np.zeros_like(data_arr, dtype=float)
        ema_values[period-1] = np.mean(data_arr[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data_arr)):
            ema_values[i] = (data_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
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
    prices_arr = np.array(prices, dtype=float)
    short_ema_full = calculate_ema_series(prices_arr, short_period)
    long_ema_full = calculate_ema_series(prices_arr, long_period)
    macd_line = short_ema_full - long_ema_full
    if len(macd_line) < long_period + signal_period -1: # Ensure enough data for signal line
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line[long_period-1:], signal_period)
    histogram = macd_line[long_period-1:][len(macd_line[long_period-1:])-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    try:
        import pandas as pd
        atr_series = pd.Series(price_ranges).ewm(alpha=1/period, adjust=False).mean().to_numpy()
        return atr_series[-1] if len(atr_series) > 0 else None
    except ImportError:
        # Fallback calculation if pandas isn't available
        atr_val = np.mean(price_ranges[:period])
        for i in range(period, len(price_ranges)):
            atr_val = ((atr_val * (period - 1)) + price_ranges[i]) / period
        return atr_val

def calculate_chandelier_exit(prices, atr_val, period=22, multiplier=3.0):
    """Calculates the Chandelier Exit for a long position."""
    if len(prices) < period or atr_val is None:
        return None
    highest_high = np.max(prices[-period:])
    return highest_high - (atr_val * multiplier)

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances V2 with three key architectural improvements:
    1.  Adaptive Risk Management: Replaces the fixed-percentage stop-loss with a
        volatility-adjusted Chandelier Exit (ATR-based), which dynamically tightens
        or loosens based on market conditions.
    2.  Enhanced Signal Confirmation: Buy signals now require dual trend confirmation
        (price > short-term EMA-20 and mid-term SMA-50), filtering out noise and
        reducing entries during minor counter-trend bounces.
    3.  Dynamic, Sentiment-Modulated Thresholds: Sentiment score is normalized for
        news context length and used to dynamically adjust RSI thresholds. This makes
        the system more sensitive to oversold conditions during negative news cycles
        and more tolerant of overbought conditions during positive ones.
    """
    # --- 1. Sentiment Analysis (with Normalization) ---
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

    # Normalize score to account for news density
    num_words = len(context_lower.split())
    normalized_sentiment = net_sentiment_score / (1 + math.log1p(num_words))

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    EMA_SHORT = 20
    SMA_TREND_MEDIUM = 50
    SMA_TREND_LONG = 100
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    CHANDELIER_PERIOD = 22
    CHANDELIER_MULTIPLIER = 2.5

    required_history_length = max(SMA_TREND_LONG + 1, CHANDELIER_PERIOD + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_20_series = calculate_ema_series(np.array(all_prices), EMA_SHORT)
    ema_20 = ema_20_series[-1] if len(ema_20_series) > 0 else None
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    chandelier_exit = calculate_chandelier_exit(all_prices, atr, CHANDELIER_PERIOD, CHANDELIER_MULTIPLIER)

    # Null check for all indicators
    if any(v is None for v in [ema_20, sma_50, sma_100, rsi, atr, chandelier_exit]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection & Dynamic Thresholds ---
    is_long_term_downtrend = current_price < sma_100
    is_mid_term_downtrend = current_price < sma_50
    is_crisis_regime = is_long_term_downtrend and is_mid_term_downtrend

    # Dynamic RSI thresholds based on sentiment
    rsi_overbought_threshold = max(70, 78 - (normalized_sentiment * 2))
    rsi_oversold_threshold = min(30, 25 - (normalized_sentiment * 2))

    # --- 4. Decision Logic (Hierarchical: Exits -> Entries) ---

    # --- A. EXIT LOGIC (RISK MANAGEMENT FIRST) ---
    # Priority 1: Volatility-based Trailing Stop (Chandelier Exit)
    if current_price < chandelier_exit:
        return "SELL"

    # Priority 2: Crisis Aversion Exit
    if is_crisis_regime and macd_histogram < 0:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with FADING momentum.
    is_extremely_overbought = rsi > (rsi_overbought_threshold + 5) # e.g., > 83 if base is 78
    is_momentum_fading = macd_hist_delta < 0
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"
        
    # Priority 4: Standard trend breakdown signal.
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_mid_term_downtrend and is_momentum_confirming_down:
        return "SELL"

    # --- B. ENTRY LOGIC ---
    # Priority 1: Contrarian Capitulation Buy
    is_deeply_oversold = rsi < rsi_oversold_threshold
    is_momentum_reversing_up = macd_hist_delta > 0
    if is_deeply_oversold and is_momentum_reversing_up and is_crisis_regime:
        return "BUY"

    # Priority 2: Standard Trend-Following Buy
    is_uptrend_confirmed = current_price > sma_50 and current_price > ema_20
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < rsi_overbought_threshold
    
    # Avoid buying into a long-term bear market unless sentiment is strongly positive
    can_buy_in_downtrend = is_long_term_downtrend and normalized_sentiment > 1.5
    
    if is_uptrend_confirmed and is_momentum_confirming_up and is_not_overbought:
        if not is_long_term_downtrend or can_buy_in_downtrend:
            return "BUY"

    # --- C. DEFAULT ACTION ---
    return "HOLD"