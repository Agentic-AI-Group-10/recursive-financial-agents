import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    # Using a simplified method that is common and avoids re-calculating the full series
    prices_arr = np.array(prices, dtype=float)
    multiplier = 2 / (period + 1)
    # The first EMA is the SMA
    ema = np.mean(prices_arr[:period])
    for i in range(period, len(prices_arr)):
        ema = (prices_arr[i] - ema) * multiplier + ema
    return ema

def calculate_rsi(prices, period):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(np.array(prices, dtype=float))
    if len(deltas) < period:
        return None

    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]

    # Use Wilder's smoothing method for RSI calculation
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def decide(current_price, price_history, news_context):
    """
    A self-improved, dual-logic trading strategy that adapts its core mechanism
    based on market volatility, addressing failures in low-volatility trending markets.

    - Crisis Regime (High Volatility): High-conviction, news-driven logic.
    - Normal Regime (Low Volatility): Pure trend-following logic.
    """
    # --- 1. Sentiment Analysis (Primarily for Crisis Regime) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Bullish
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "positive outlook": 1.5,
        # Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "hawkish": -2.0, "tightening": -1.5,
        "bearish": -2.0, "miss": -1.5, "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0,
        "downgrade": -1.5, "tariff": -1.5, "weak earnings": -2.0, "bankruptcy": -2.5,
        "negative outlook": -1.5, "contraction": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Volatility Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    LONG_TERM_TREND_PERIOD = 50 # New: For master trend confirmation
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOLATILITY_PERIOD + 1, LONG_TERM_TREND_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    long_term_sma = calculate_sma(all_prices, LONG_TERM_TREND_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)

    # Safeguard against None values from calculations
    if any(v is None for v in [short_ema, long_ema, long_term_sma, rsi]):
        return "HOLD"

    # Calculate volatility to determine market regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    volatility = np.std(log_returns[-VOLATILITY_PERIOD:])
    
    # --- 3. Adaptive Decision Logic based on Regime ---
    is_high_volatility = volatility > 0.02  # Threshold for high-volatility regime

    if is_high_volatility:
        # --- CRISIS REGIME LOGIC ---
        # This logic was successful in past stress tests. It requires a strong
        # news catalyst, confirmed by trend, while avoiding over-extended entries.
        BULLISH_SENTIMENT_THRESHOLD = 1.5
        BEARISH_SENTIMENT_THRESHOLD = -1.5
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD

        # BUY: Strong positive news AND an established bullish trend AND not overbought.
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and is_not_overbought:
            return "BUY"
        
        # SELL: Strong negative news AND an established bearish trend AND not oversold.
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and is_not_oversold:
            return "SELL"
    
    else:
        # --- NORMAL REGIME LOGIC ---
        # This logic is a pure trend-following system, addressing the failure of being
        # too passive and getting faked out by news noise in stable markets.
        # Sentiment and RSI are ignored as primary signals to avoid whipsaws.
        
        # Primary BUY Signal: A bullish EMA crossover confirmed by a long-term uptrend.
        is_bullish_crossover = short_ema > long_ema
        is_above_long_term_trend = current_price > long_term_sma
        
        if is_bullish_crossover and is_above_long_term_trend:
            return "BUY"
        
        # Primary SELL Signal: A bearish EMA crossover indicates the trend has reversed.
        # This acts as the exit signal for the trend-following system.
        is_bearish_crossover = short_ema < long_ema
        
        if is_bearish_crossover:
            return "SELL"

    # Default to HOLD if no high-conviction signal is generated in either regime.
    return "HOLD"