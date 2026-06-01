import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_with_history(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the latest two MACD histogram values for momentum decay analysis."""
    if len(prices) < long_period + signal_period:
        return None, None
    ema_short = calculate_ema_series(prices, short_period)
    ema_long = calculate_ema_series(prices, long_period)
    macd_line = ema_short[len(ema_short) - len(ema_long):] - ema_long
    if len(macd_line) < signal_period:
        return None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    if len(signal_line) < 2:
        return None, None
    histogram_series = macd_line[len(macd_line) - len(signal_line):] - signal_line
    if len(histogram_series) < 2:
        return None, None
    return histogram_series[-1], histogram_series[-2]

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
    """Calculates a simplified Average True Range (ATR) using only closing prices."""
    if len(prices) < period + 1:
        return None
    price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    if not price_changes:
        return None
    # Wilder's smoothing for ATR
    atr_val = np.mean(price_changes[:period])
    for i in range(period, len(price_changes)):
        atr_val = (atr_val * (period - 1) + price_changes[i]) / period
    return atr_val

def decide(current_price, price_history, news_context):
    """
    A self-improved strategy that introduces proactive profit-taking via momentum
    decay analysis and uses ATR for more robust regime detection to increase
    activity and reduce drawdowns.
    """
    # --- 1. Sentiment Analysis (Unchanged from successful parent) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0, "ai boom": 2.5,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.0, "market turmoil": -2.0, "credit crunch": -2.5,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "uncertainty": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    BB_PERIOD = 20
    MEDIUM_TERM_SMA_PERIOD = 50
    ATR_SHORT_PERIOD = 10
    ATR_LONG_PERIOD = 50

    required_history_length = max(LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD, ATR_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_hist, macd_hist_prev = calculate_macd_with_history(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, MACD_SIGNAL_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    short_atr = calculate_atr(all_prices, ATR_SHORT_PERIOD)
    long_atr = calculate_atr(all_prices, ATR_LONG_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, macd_hist, macd_hist_prev, upper_band, short_atr, long_atr]):
        return "HOLD"

    # --- 3. Proactive Exit Logic (Profit-Taking & Risk Reduction) ---
    is_in_uptrend = short_ema > long_ema
    momentum_is_fading = macd_hist < macd_hist_prev
    
    # If in a mature uptrend and momentum is clearly waning, take profits.
    # This is a critical addition to prevent holding through reversals.
    if is_in_uptrend and momentum_is_fading and rsi > 55:
        return "SELL"

    # --- 4. Multi-Regime Entry Logic ---
    # Improved Regime Detection using ATR
    is_high_volatility = short_atr > (long_atr * 1.7)

    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following (Parent logic was effective here) ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        RSI_OVERBOUGHT_CEILING = 65
        RSI_OVERSOLD_FLOOR = 35

        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_in_uptrend and macd_hist > 0 and rsi < RSI_OVERBOUGHT_CEILING:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and not is_in_uptrend and macd_hist < 0 and rsi > RSI_OVERSOLD_FLOOR:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive Logic with Increased Sensitivity ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.007 # Slightly wider range for choppy detection

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market (More sensitive entry)
            BULLISH_SENTIMENT_THRESHOLD = 0.5  # Lowered to increase activity
            BEARISH_SENTIMENT_THRESHOLD = -0.5 # Lowered to increase activity
            
            macd_turning_positive = macd_hist > 0 and macd_hist_prev <= 0
            macd_turning_negative = macd_hist < 0 and macd_hist_prev >= 0

            if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_in_uptrend and macd_turning_positive and rsi < 70:
                return "BUY"
            elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and not is_in_uptrend and macd_turning_negative and rsi > 30:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Parent logic was sound)
            MEAN_REVERSION_RSI_OVERSOLD = 30
            MEAN_REVERSION_RSI_OVERBOUGHT = 70
            
            medium_sma = calculate_sma(all_prices, MEDIUM_TERM_SMA_PERIOD)
            if medium_sma is None:
                return "HOLD"

            # Buy the dip if confirmed by RSI/BBands, sentiment isn't catastrophic, and major trend is up.
            if (rsi < MEAN_REVERSION_RSI_OVERSOLD and current_price < lower_band) and \
               (net_sentiment_score > -2.0) and (current_price > medium_sma):
                return "BUY"
            # Sell the rip if confirmed by RSI/BBands and sentiment isn't euphoric.
            elif (rsi > MEAN_REVERSION_RSI_OVERBOUGHT and current_price > upper_band) and \
                 (net_sentiment_score < 2.0):
                return "SELL"

    return "HOLD"