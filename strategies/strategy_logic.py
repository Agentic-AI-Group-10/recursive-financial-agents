import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    # This is a simplified calculation for the final value only.
    # A full series calculation is needed for indicators like MACD.
    ema = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for price in prices_arr[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi(prices, period):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    if len(deltas) < period:
        return None
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use Wilder's smoothing method
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        rs = np.inf
    else:
        rs = avg_gain / avg_loss
        
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """
    Calculates the MACD line, signal line, and histogram for the latest price.
    """
    if len(prices) < long_period + signal_period:
        return None, None, None

    prices_arr = np.array(prices, dtype=float)
    
    # Internal function to get EMA series efficiently
    def get_ema_series(data, period):
        ema_series = np.zeros(len(data) - period + 1)
        ema_series[0] = np.mean(data[:period])
        multiplier = 2 / (period + 1)
        for i in range(1, len(ema_series)):
            ema_series[i] = (data[i + period - 1] - ema_series[i-1]) * multiplier + ema_series[i-1]
        return ema_series

    short_ema_series = get_ema_series(prices_arr, short_period)
    long_ema_series = get_ema_series(prices_arr, long_period)
    
    # Align series and calculate MACD line
    macd_line_series = short_ema_series[long_period - short_period:] - long_ema_series
    
    if len(macd_line_series) < signal_period:
        return None, None, None
        
    # Calculate signal line
    signal_line_series = get_ema_series(macd_line_series, signal_period)
    
    latest_macd = macd_line_series[-1]
    latest_signal = signal_line_series[-1]
    histogram = latest_macd - latest_signal
    
    return latest_macd, latest_signal, histogram

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that incorporates momentum (MACD)
    for proactive risk management and more robust entry/exit signals.
    """
    # --- 1. Enhanced Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 2.0, "peak inflation": 2.0,
        "disinflation": 1.5, "beat estimates": 1.5, "jobs growth": 1.5,
        "strong consumer": 1.5, "recovery": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "persistent inflation": -2.0, "sell-off": -2.0,
        "weak earnings": -2.0, "geopolitical tension": -2.0, "tightening": -1.5,
        "miss estimates": -1.5, "downgrade": -1.5, "tariff": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to"]
    canceller_words = ["avoids", "prevents", "dodges", "misses"] # Words that can flip a negative
    
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            is_cancelled = any(can_word in pre_context for can_word in canceller_words)
            
            final_weight = weight
            if is_negated:
                final_weight = -weight
            if is_cancelled and weight < 0: # If a negative keyword is cancelled
                final_weight = -weight # Flip to positive
                
            net_sentiment_score += final_weight

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
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line, _ = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, SIGNAL_PERIOD)

    # Safeguard against None values from calculations
    if any(v is None for v in [short_ema, long_ema, rsi, macd_line, signal_line]):
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic with Momentum ---
    is_uptrend = short_ema > long_ema
    is_downtrend = short_ema < long_ema
    is_bullish_momentum = macd_line > signal_line
    is_bearish_momentum = macd_line < signal_line

    # Proactive Exit Signals (Risk Management)
    # Exit long if uptrend shows bearish momentum, unless news is euphoric
    if is_uptrend and is_bearish_momentum and net_sentiment_score < 2.5:
        return "SELL"
    # Exit short (i.e., buy) if downtrend shows bullish momentum, unless news is catastrophic
    if is_downtrend and is_bullish_momentum and net_sentiment_score > -2.5:
        return "BUY"

    # Entry Signals
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend & momentum following ===
        if net_sentiment_score >= 2.0 and is_uptrend and is_bullish_momentum and rsi < 70:
            return "BUY"
        if net_sentiment_score <= -2.0 and is_downtrend and is_bearish_momentum and rsi > 30:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend/Momentum vs. Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            if net_sentiment_score >= 1.0 and is_uptrend and is_bullish_momentum and rsi < 75:
                return "BUY"
            if net_sentiment_score <= -1.0 and is_downtrend and is_bearish_momentum and rsi > 25:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            if rsi < 25 and net_sentiment_score > -1.5:
                return "BUY"
            if rsi > 75 and net_sentiment_score < 1.5:
                return "SELL"

    return "HOLD"