import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

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

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period + signal_period:
        return None, None, None
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    if len(macd_line) < signal_period:
        return None, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic_oscillator(prices, period=14):
    """Calculates the Stochastic Oscillator (%K)."""
    if len(prices) < period:
        return None
    
    price_slice = prices[-period:]
    lowest_low = np.min(price_slice)
    highest_high = np.max(price_slice)
    
    if highest_high == lowest_low:
        return 50.0 # Return a neutral value if price hasn't moved
        
    stoch_k = 100 * (prices[-1] - lowest_low) / (highest_high - lowest_low)
    return stoch_k

def decide(current_price, price_history, news_context):
    """
    A self-improved, three-regime trading strategy utilizing a Stochastic Oscillator
    for enhanced mean-reversion logic in ranging markets.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Expanded Keyword Set) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.5, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5, "market rally": 2.0, "quantitative easing": 2.5, "qe": 2.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.5, "war": -3.0, "conflict": -3.0, "sanctions": -2.5,
        "market turmoil": -2.0, "credit crunch": -2.5, "ai bubble": -2.0,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "supply chain disruption": -1.5, "uncertainty": -1.5, "weak jobs": -2.0,
        "quantitative tightening": -2.5, "qt": -2.5, "vix spike": -2.0
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Three-Regime Detection ---
    all_prices = price_history + [current_price]
    
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    STOCH_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + 9, VOL_LONG_PERIOD + 1, STOCH_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema_series = calculate_ema_series(all_prices, SHORT_EMA_PERIOD)
    short_ema = short_ema_series[-1]
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    stoch_k = calculate_stochastic_oscillator(all_prices, STOCH_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)

    if any(v is None for v in [short_ema, long_ema, rsi, stoch_k]) or macd_hist_series is None or len(macd_hist_series) < 3:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # Regime Detection
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    
    # Coefficient of variation for the short-term EMA to detect choppiness
    ema_std_dev = np.std(short_ema_series[-15:])
    ema_mean = np.mean(short_ema_series[-15:])
    ema_cv = ema_std_dev / ema_mean if ema_mean != 0 else 0

    market_regime = "TRENDING"
    if (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015):
        market_regime = "CRISIS"
    elif ema_cv < 0.004: # If EMA is very flat, it's a ranging market
        market_regime = "RANGING"

    # --- 3. Multi-Regime Decision Logic ---
    if market_regime == "CRISIS":
        # High-conviction trend-following with strong sentiment confirmation
        BULLISH_SENTIMENT_THRESHOLD = 3.0
        BEARISH_SENTIMENT_THRESHOLD = -3.0
        
        if short_ema > long_ema and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and macd_histogram > 0:
            return "BUY"
        elif short_ema < long_ema and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and macd_histogram < 0:
            return "SELL"
            
    elif market_regime == "TRENDING":
        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        # Proactive profit-taking / trend exhaustion signal
        is_momentum_fading_up = macd_histogram > 0 and macd_histogram < prev_macd_histogram
        if bullish_trend and (rsi > 78 or stoch_k > 90) and is_momentum_fading_up:
            return "SELL"

        # Entry requires 2-day momentum confirmation to avoid whipsaws
        if bullish_trend and macd_histogram > 0 and prev_macd_histogram > 0 and rsi < 75 and stoch_k < 85 and net_sentiment_score > -1.5:
            return "BUY"
        
        # Exit requires 2-day momentum confirmation
        if bearish_trend and macd_histogram < 0 and prev_macd_histogram < 0 and rsi > 25 and stoch_k > 15 and net_sentiment_score < 1.5:
            return "SELL"

    elif market_regime == "RANGING":
        # Mean-reversion logic using Stochastic Oscillator and RSI confirmation
        is_reversing_up = macd_histogram > prev_macd_histogram
        is_reversing_down = macd_histogram < prev_macd_histogram
        
        # Buy the dip if deeply oversold and momentum is turning up
        if stoch_k < 20 and rsi < 35 and is_reversing_up and net_sentiment_score > -2.5:
            return "BUY"
            
        # Sell the rip if deeply overbought and momentum is turning down
        if stoch_k > 80 and rsi > 65 and is_reversing_down and net_sentiment_score < 2.5:
            return "SELL"

    return "HOLD"