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

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

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

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    prices_slice = prices[-period:]
    middle_band = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return middle_band, upper_band, lower_band

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period):
    """Calculates the Rate of Change (ROC)."""
    if len(prices) < period + 1:
        return None
    if prices[-1 - period] == 0:
        return 0.0
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_momentum_cohesion(prices, p1=8, p2=16, p3=24):
    """
    Calculates a score based on the alignment of short, medium, and long-term momentum.
    Returns a score from -3 (perfect bearish alignment) to +3 (perfect bullish alignment).
    """
    roc1 = calculate_roc(prices, p1)
    roc2 = calculate_roc(prices, p2)
    roc3 = calculate_roc(prices, p3)
    if any(r is None for r in [roc1, roc2, roc3]):
        return 0.0
    
    score = 0
    # Normalize by a typical daily move to make the threshold more stable
    atr_norm = calculate_atr(prices, 14)
    if atr_norm is None or atr_norm == 0:
        return 0.0
    
    daily_change_pct = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0
    threshold = (atr_norm / prices[-1]) * 100 * 0.1 # A small fraction of ATR

    if roc1 > threshold and daily_change_pct >= 0: score += 1
    if roc2 > threshold: score += 1
    if roc3 > threshold: score += 1
    if roc1 < -threshold and daily_change_pct <= 0: score -= 1
    if roc2 < -threshold: score -= 1
    if roc3 < -threshold: score -= 1
    return float(score)

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy using a Momentum Cohesion Score
    for robust trend identification and refined entry/exit logic.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Expanded Dictionary) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Strong Positive
        "fed pivot": 3.5, "rate cut": 3.0, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "capitulation": 2.0, "breakthrough": 2.5,
        # Moderate Positive
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "disinflation": 2.0, "market rally": 2.0, "vix crush": 2.0, "de-escalation": 2.0,
        # Mild Positive
        "beat estimates": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "easing tensions": 1.5, "consumer confidence": 1.5, "weak jobs report": 2.0, # Weak jobs -> Fed easing
        # Strong Negative
        "yield curve inversion": -3.5, "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "conflict": -3.0, "quantitative tightening": -3.0, "contagion": -3.5,
        # Moderate Negative
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "sanctions": -2.5, "credit crunch": -2.5, "cpi beat": -2.5, "euphoria": -2.0, "vix spike": -2.5,
        "debt ceiling": -2.5, "supply chain disruption": -2.0,
        # Mild Negative
        "hawkish": -2.0, "bearish": -2.0, "plunge": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "market turmoil": -2.0, "bubble": -2.0, "tightening": -1.5, "miss estimates": -1.5,
        "downgrade": -1.5, "tariff": -1.5, "uncertainty": -1.5, "strong jobs report": -2.0, # Strong jobs -> Fed tightening
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "avert"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Tunable Parameters
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    BB_PERIOD = 20
    ATR_REGIME_SHORT = 10
    ATR_REGIME_LONG = 50
    MOMENTUM_P1, MOMENTUM_P2, MOMENTUM_P3 = 8, 16, 24

    required_history_length = max(LONG_EMA_PERIOD + 9, ATR_REGIME_LONG + 1, MOMENTUM_P3 + 2)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    prev_rsi = calculate_rsi(all_prices[:-1], RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_REGIME_SHORT)
    long_atr = calculate_atr(all_prices, ATR_REGIME_LONG)
    momentum_cohesion = calculate_momentum_cohesion(all_prices, MOMENTUM_P1, MOMENTUM_P2, MOMENTUM_P3)

    if any(v is None for v in [short_ema, long_ema, rsi, prev_rsi, upper_band, lower_band, short_atr, long_atr]) or macd_hist_series is None or len(macd_hist_series) < 3:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # Regime Detection
    is_high_volatility = short_atr > (long_atr * 1.7) # Slightly higher threshold
    is_trending_market = abs(momentum_cohesion) >= 2.0

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction, sentiment-driven, volatility-filtered ===
        BULLISH_SENTIMENT_THRESHOLD = 4.0
        BEARISH_SENTIMENT_THRESHOLD = -4.0
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and short_ema > long_ema and macd_histogram > 0 and rsi < 80:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and short_ema < long_ema and macd_histogram < 0 and rsi > 20:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive with Momentum Cohesion Confirmation ===
        if is_trending_market:
            # Sub-Regime: Confirmed Trending Market
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema
            
            # Profit-taking / Exit logic
            is_momentum_fading_up = macd_histogram > 0 and macd_histogram < prev_macd_histogram
            if bullish_trend and (rsi > 78 or current_price > upper_band) and is_momentum_fading_up:
                return "SELL"
            
            is_momentum_fading_down = macd_histogram < 0 and macd_histogram > prev_macd_histogram
            if bearish_trend and (rsi < 22 or current_price < lower_band) and is_momentum_fading_down:
                return "BUY" # Cover short

            # Entry logic with strong confirmation
            if bullish_trend and momentum_cohesion >= 2.0 and macd_histogram > prev_macd_histogram and rsi < 75 and net_sentiment_score > -2.0:
                return "BUY"
            
            if bearish_trend and momentum_cohesion <= -2.0 and macd_histogram < prev_macd_histogram and rsi > 25 and net_sentiment_score < 2.0:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            
            # **IMPROVEMENT**: Add reversal confirmation to avoid catching falling knives.
            is_reversing_up = macd_histogram > prev_macd_histogram and current_price > all_prices[-2]
            if rsi < 32 and current_price < lower_band and \
               net_sentiment_score > -3.5 and is_reversing_up:
                return "BUY"
                
            is_reversing_down = macd_histogram < prev_macd_histogram and current_price < all_prices[-2]
            if rsi > 68 and current_price > upper_band and \
               net_sentiment_score < 3.5 and is_reversing_down:
                return "SELL"

    return "HOLD"