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
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        # Fallback pure-python EMA calculation
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
    This version refactors the successful V2 into a more robust, adaptive system.
    1.  Composite Scoring System: Replaces rigid boolean checks with a weighted
        scoring model (`bullish_score`, `bearish_score`) that combines trend,
        momentum, and sentiment, allowing for more nuanced trade confirmations.
    2.  Dynamic ATR-Based Stop-Loss: The fixed-percentage stop-loss is replaced
        with a trailing stop based on Average True Range (ATR). This adapts risk
        management to current market volatility, protecting capital more effectively.
    3.  Choppy Market Regime Filter: A new regime is introduced to detect sideways,
        low-volatility markets. In this regime, trade entry thresholds are raised
        significantly to prevent whipsaws and conserve capital for high-probability trends.
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
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    long_atr = calculate_atr(all_prices, 50) # For regime detection
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, long_atr, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    is_deeply_oversold = rsi < 25
    is_capitulation_candidate = is_crash_velocity and is_deeply_oversold

    is_low_volatility = atr < (long_atr * 0.8)
    is_low_momentum = abs(roc_20) < 5.0
    is_choppy_regime = is_low_volatility and is_low_momentum

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: DYNAMIC RISK MANAGEMENT (PRIMARY SELL SIGNALS)
    # Priority 1: Volatility-adjusted trailing stop-loss.
    stop_loss_level = donchian_high_20 - (atr * 2.5)
    if current_price < stop_loss_level:
        return "SELL"

    # Priority 2: Profit-taking on extreme overbought conditions with FADING momentum.
    if rsi > 82 and macd_hist_delta < 0:
        return "SELL"

    # REGIME 3: CRISIS AVERSION
    if is_crisis_regime and not is_capitulation_candidate:
        if current_price < sma_50 and macd_histogram < 0:
            return "SELL"
        return "HOLD" # Stay defensive in cash

    # --- 5. Composite Scoring for Normal & Choppy Regimes ---
    bullish_score = 0.0
    bearish_score = 0.0

    # Trend Component (Weight: 2.5)
    if current_price > sma_50: bullish_score += 1.5
    else: bearish_score += 1.5
    if current_price > sma_100: bullish_score += 1.0
    else: bearish_score += 1.0

    # Momentum Component (Weight: 3.0)
    if macd_histogram > 0: bullish_score += 1.5
    else: bearish_score += 1.5
    if macd_hist_delta > 0: bullish_score += 1.5
    else: bearish_score += 1.5

    # Oscillator Component (Weight: 1.5)
    if rsi < 40: bullish_score += (40 - rsi) / 10.0 # Scaled score for oversold
    if rsi > 60: bearish_score += (rsi - 60) / 10.0 # Scaled score for overbought

    # Sentiment Component (Weight: Dynamic)
    bullish_score += max(0, net_sentiment_score / 2.0)
    bearish_score += max(0, -net_sentiment_score / 2.0)

    # --- 6. Final Trade Decision ---
    buy_threshold = 4.0
    sell_threshold = 4.0

    # In a choppy market, be much more selective.
    if is_choppy_regime:
        buy_threshold = 5.5
        sell_threshold = 5.5

    # BUY Condition: Strong bullish conviction with weak bearish signals.
    if bullish_score >= buy_threshold and bearish_score < (buy_threshold / 2):
        # Final check: avoid buying into extreme overbought conditions
        if rsi < 78:
            return "BUY"

    # SELL Condition: Strong bearish conviction with weak bullish signals.
    if bearish_score >= sell_threshold and bullish_score < (sell_threshold / 2):
        return "SELL"

    # Default action is to hold the current position.
    return "HOLD"