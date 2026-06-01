import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for a series of prices."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    ema_values = np.zeros(len(prices_arr) - period + 1, dtype=float)
    ema_values[0] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(1, len(ema_values)):
        price_index = period + i - 1
        ema_values[i] = (prices_arr[price_index] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use Wilder's smoothing method
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram for the latest price."""
    if len(prices) < long_period + signal_period:
        return None, None, None
        
    ema_short_full = calculate_ema(prices, short_period)
    ema_long_full = calculate_ema(prices, long_period)
    
    # Align the series by taking the tail of the shorter EMA series
    ema_short_aligned = ema_short_full[-(len(ema_long_full)):]
    
    macd_line = ema_short_aligned - ema_long_full
    
    if len(macd_line) < signal_period:
        return None, None, None
        
    signal_line_full = calculate_ema(macd_line, signal_period)
    macd_histogram = macd_line[-(len(signal_line_full)):] - signal_line_full
    
    return macd_line[-1], signal_line_full[-1], macd_histogram[-1]

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that incorporates MACD for momentum
    confirmation, enhancing the existing trend, volatility, and sentiment framework.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Expanded Keywords ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat estimates": 1.5, "growth accelerates": 1.5, "ai boom": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "hot inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "tightening": -1.5, "miss estimates": -1.5, "geopolitical risk": -1.5, "tariff": -1.5
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
    SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD + SIGNAL_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema_full = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema_full = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line, macd_hist = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, SIGNAL_PERIOD)

    # Safeguard against None values from calculations
    if short_ema_full is None or long_ema_full is None or rsi is None or macd_hist is None:
        return "HOLD"
    
    short_ema = short_ema_full[-1]
    long_ema = long_ema_full[-1]

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.6) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic with MACD Confirmation ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction, momentum-confirmed trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        
        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        # BUY only with strong positive sentiment, uptrend, and accelerating momentum
        if bullish_trend and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and macd_hist > 0:
            return "BUY"
        # SELL only with strong negative sentiment, downtrend, and accelerating downward momentum
        elif bearish_trend and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and macd_hist < 0:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.007 # Slightly increased threshold for chop

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market with Momentum Confirmation
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            RSI_OVERBOUGHT = 70
            RSI_OVERSOLD = 30
            
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema

            # BUY on uptrend with positive sentiment, non-overbought RSI, and positive momentum
            if bullish_trend and macd_hist > 0 and rsi < RSI_OVERBOUGHT and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
                return "BUY"
            # SELL on downtrend with negative sentiment, non-oversold RSI, and negative momentum
            elif bearish_trend and macd_hist < 0 and rsi > RSI_OVERSOLD and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            MEAN_REVERSION_RSI_OVERSOLD = 25
            MEAN_REVERSION_RSI_OVERBOUGHT = 75
            
            # Buy on extreme oversold conditions, if sentiment isn't catastrophic
            if rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -2.0:
                return "BUY"
            # Sell on extreme overbought conditions, if sentiment isn't euphoric
            elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 2.0:
                return "SELL"

    # Default action is to hold, preserving capital when no high-conviction signal is present.
    return "HOLD"