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
    
    # Ensure EMAs are aligned to the end of the price series
    macd_line = short_ema_full[len(short_ema_full)-len(long_ema_full):] - long_ema_full
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    signal_line_full = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[len(macd_line)-len(signal_line_full):] - signal_line_full
    
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
    This version transitions from discrete rules to a more robust, holistic scoring
    model, addressing key weaknesses of the prior version.
    1.  Unified Scoring System: Replaces rigid boolean flags with a `bullish_score`
        and `bearish_score`. These scores aggregate signals from trend (SMAs),
        momentum (MACD, RSI), and sentiment, providing a more nuanced view of the
        market and reducing whipsaws caused by a single indicator flipping.
    2.  Dynamic ATR Trailing Stop: The fixed-percentage stop-loss is replaced with
        a volatility-aware ATR-based trailing stop. This adapts risk management to
        the current market environment, protecting capital more effectively.
    3.  Enhanced Capitulation Signal: The contrarian buy signal is fortified with a
        volatility spike condition (ATR ratio), ensuring it triggers only during
        genuine panic selling, not just sharp drops.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "soft landing": 2.5, "cooling inflation": 2.5,
        "dovish": 2.0, "stimulus": 2.0, "strong earnings": 2.0, "ai boom": 2.5,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "gdp beat": 2.0, "jobs miss": 1.5, # Good for rates
        # Contrarian Positive (buy fear)
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        # Ambiguous
        "strong jobs report": -0.5, # Ambiguous: good for economy, bad for inflation/rates
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "gdp miss": -2.0,
        # Contrarian Negative (sell greed)
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
    ATR_STOP_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 2.5
    ROC_CRASH_PERIOD = 20

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    long_atr = calculate_atr(all_prices, 50) # For volatility regime check
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    
    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, long_atr, roc_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = atr > (long_atr * 1.8)
    is_crash_velocity = roc_20 is not None and roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 is not None and roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold and is_high_volatility

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: DYNAMIC RISK MANAGEMENT (HIGHEST PRIORITY SELL)
    # ATR Trailing Stop: Sell if price drops more than ATR*multiplier from recent high.
    recent_high = np.max(all_prices[-ATR_STOP_LOOKBACK:])
    atr_stop_price = recent_high - (atr * ATR_STOP_MULTIPLIER)
    if current_price < atr_stop_price:
        return "SELL"

    # REGIME 2: CONTRARIAN CAPITULATION (HIGH PRIORITY BUY)
    # Buy extreme fear, but only on signs of a powerful reversal.
    is_reversal_power = macd_hist_delta > abs(prev_macd_histogram * 0.1)
    if is_capitulation_candidate and macd_histogram > prev_macd_histogram and is_reversal_power:
        return "BUY"

    # REGIME 3: SCORE-BASED NORMAL/CRISIS TRADING
    bullish_score = 0.0
    bearish_score = 0.0

    # Trend Analysis
    if current_price > sma_50: bullish_score += 1.5
    else: bearish_score += 1.5
    if sma_50 > sma_100: bullish_score += 1.0
    else: bearish_score += 1.0

    # Momentum Analysis
    if macd_histogram > 0: bullish_score += 1.0
    else: bearish_score += 1.0
    if macd_hist_delta > 0: bullish_score += 1.5 # Accelerating momentum is key
    else: bearish_score += 1.5

    # Oscillator Analysis
    if rsi > 55: bullish_score += 1.0
    if rsi < 45: bearish_score += 1.0
    if rsi > 78: bearish_score += 1.5 # Overbought adds to bearish case
    if rsi < 22: bullish_score += 1.5 # Oversold adds to bullish case

    # Sentiment
    bullish_score += max(0, net_sentiment_score / 2.0)
    bearish_score += max(0, -net_sentiment_score / 2.0)

    # Adjust thresholds based on regime
    buy_threshold = 4.5
    sell_threshold = 4.5
    if is_crisis_regime:
        buy_threshold = 5.5  # Be more certain before buying in a crisis
        sell_threshold = 4.0 # Be quicker to sell in a crisis

    # --- Final Decision ---
    if bullish_score >= buy_threshold and bearish_score < (bullish_score * 0.5):
        return "BUY"

    if bearish_score >= sell_threshold and bullish_score < (bearish_score * 0.5):
        return "SELL"

    # Default action is to hold the current position.
    return "HOLD"