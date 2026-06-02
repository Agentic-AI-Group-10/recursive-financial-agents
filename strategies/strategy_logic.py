import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators (Unchanged from Parent) ---

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
    
    ema_short_full = calculate_ema_series(prices, short_period)
    ema_long_full = calculate_ema_series(prices, long_period)
    
    # Align series by taking the tail of the shorter one
    macd_line = ema_short_full[-len(ema_long_full):] - ema_long_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
        
    signal_line_full = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[-len(signal_line_full):] - signal_line_full
    
    return macd_line, signal_line_full, histogram

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
    This version evolves the parent strategy by introducing a more robust, hybrid decision model.
    1.  Adaptive Volatility Stop-Loss: Replaces the fixed 7% stop with a dynamic ATR-based
        trailing stop (Donchian High - N * ATR). This adapts risk management to current market
        volatility, preventing premature exits in volatile uptrends.
    2.  Signal Scoring Engine: Moves beyond rigid if/else for normal markets. It now calculates
        a weighted score based on trend (EMA), momentum (MACD histogram & velocity), and sentiment.
        This requires signal confluence, providing more robust trade entries.
    3.  Low-Volatility Regime Filter: Explicitly identifies and filters out low-volatility,
        sideways markets ('chop zones') to reduce whipsaw trades and conserve capital.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "gamma squeeze": 3.0, "capitulation": 3.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "vix crush": 1.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "euphoria": -2.5, "mania": -3.0, "extreme greed": -2.5,
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
    EMA_TREND_LONG = 100
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_SHORT = 10
    ATR_LONG = 50
    ATR_STOP_PERIOD = 20
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 25

    required_history_length = max(EMA_TREND_LONG + 1, ATR_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100_series = calculate_ema_series(all_prices, EMA_TREND_LONG)
    ema_50_series = calculate_ema_series(all_prices, EMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    atr_stop = calculate_atr(all_prices, ATR_STOP_PERIOD)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [rsi, short_atr, long_atr, roc_20, atr_stop, donchian_high]) or \
       macd_hist_series is None or len(macd_hist_series) < 2 or \
       len(ema_100_series) == 0 or len(ema_50_series) == 0:
        return "HOLD"

    ema_100 = ema_100_series[-1]
    ema_50 = ema_50_series[-1]
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < ema_100
    is_high_volatility = short_atr > (long_atr * 1.8)
    is_low_volatility = short_atr < (long_atr * 0.75)
    is_crash_velocity = roc_20 is not None and roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    is_deeply_oversold = rsi is not None and rsi < 25
    is_capitulation_candidate = is_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    if is_crisis_regime:
        return "SELL" # In crisis, preserve capital. Exit all positions.

    # REGIME 3: LOW-VOLATILITY / CHOP
    if is_low_volatility:
        return "HOLD" # Avoid trading in sideways, trendless markets to prevent whipsaw.

    # REGIME 4: NORMAL MARKET (SCORE-BASED)

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Adaptive ATR Trailing Stop-Loss
    ATR_STOP_MULTIPLIER = 2.5
    trailing_stop_price = donchian_high - (atr_stop * ATR_STOP_MULTIPLIER)
    if current_price < trailing_stop_price:
        return "SELL"

    # --- Scoring Engine for Entries/Exits ---
    buy_score = 0.0
    sell_score = 0.0
    
    # Trend Analysis
    if current_price > ema_50: buy_score += 1.0
    else: sell_score += 1.0
    if current_price > ema_100: buy_score += 1.0
    else: sell_score += 1.0
    if ema_50 > ema_100: buy_score += 0.5
    else: sell_score += 0.5

    # Momentum Analysis
    if macd_histogram > 0: buy_score += 1.0
    else: sell_score += 1.0
    if macd_hist_delta > 0: buy_score += 1.5 # Accelerating momentum is a strong signal
    else: sell_score += 1.5

    # Overbought/Oversold Analysis
    if rsi is not None:
        if rsi > 78: sell_score += 1.0 # Condition for profit-taking
        if rsi < 35: buy_score += 1.0  # Condition for dip-buying in uptrend

    # Sentiment Integration
    buy_score += net_sentiment_score / 2.0
    sell_score -= net_sentiment_score / 2.0

    # --- Final Decision based on Score ---
    BUY_THRESHOLD = 4.0
    SELL_THRESHOLD = 4.0

    if buy_score >= BUY_THRESHOLD:
        return "BUY"
    
    if sell_score >= SELL_THRESHOLD:
        return "SELL"

    # Default action is to hold the current position.
    return "HOLD"