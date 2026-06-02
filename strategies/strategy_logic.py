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

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC)."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy with a dedicated "Crash Protection"
    mode to ensure robustness during sustained, high-volatility downturns. This version
    incorporates a "slow bleed" stop-loss and a safer "Phoenix" recovery signal.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Enhanced with contrarian and crisis keywords) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Strong Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "productivity boom": 2.5,
        "goldilocks": 2.0, "breakthrough": 2.0,
        # Moderate Positive
        "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "disinflation": 2.0, "market rally": 2.0, "vix crush": 2.0,
        # Mild Positive
        "beat estimates": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "easing tensions": 1.5, "consumer confidence": 1.5, "weak jobs report": 1.5, # Nuanced: good for rates
        # Contrarian Bullish (Fear) - Used with caution
        "capitulation": 1.5, "panic selling": 1.0, "extreme fear": 1.0,
        # Strong Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "conflict": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit default": -3.5,
        # Moderate Negative
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "sanctions": -2.5, "credit crunch": -2.5, "cpi beat": -2.5, "vix spike": -2.5,
        # Mild Negative
        "hawkish": -2.0, "bearish": -2.0, "plunge": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "market turmoil": -2.0, "bubble": -2.0, "tightening": -1.5, "miss estimates": -1.5,
        "downgrade": -1.5, "tariff": -1.5, "uncertainty": -1.5, "strong jobs report": -1.5, # Nuanced: bad for rates
        # Contrarian Bearish (Greed)
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

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Indicator Periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    TREND_SMA_PERIOD = 20 # For trend confirmation and recovery signal
    LONG_TERM_SMA_PERIOD = 50
    RSI_PERIOD = 14
    BB_PERIOD = 20
    ROC_PERIOD = 20
    ATR_REGIME_SHORT = 10
    ATR_REGIME_LONG = 50

    required_history_length = max(LONG_EMA_PERIOD + 9, ATR_REGIME_LONG + 1, LONG_TERM_SMA_PERIOD + 1, ROC_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    sma_20 = calculate_sma(all_prices, TREND_SMA_PERIOD)
    sma_50 = calculate_sma(all_prices, LONG_TERM_SMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_REGIME_SHORT)
    long_atr = calculate_atr(all_prices, ATR_REGIME_LONG)
    roc = calculate_roc(all_prices, ROC_PERIOD)

    if any(v is None for v in [short_ema, long_ema, sma_20, sma_50, rsi, upper_band, lower_band, short_atr, long_atr, roc]) or macd_hist_series is None or len(macd_hist_series) < 3:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Enhanced Regime Detection ---
    is_high_volatility = short_atr > (long_atr * 1.7)
    is_long_term_bearish = current_price < sma_50
    is_velocity_crash = roc < -10.0 # A sharp 10% drop over the last month

    # CRASH_MODE is triggered by a structural breakdown (below 50-day SMA) combined
    # with high volatility OR a sudden, high-velocity drop.
    CRASH_MODE = (is_long_term_bearish and is_high_volatility) or is_velocity_crash

    # --- 4. Multi-Regime Decision Logic ---
    if CRASH_MODE:
        # === CRASH PROTECTION MODE: Prioritize capital preservation. Avoid buying bottoms. ===
        # SELL signal is aggressive: get out on any sign of weakness.
        if current_price < sma_20:
            return "SELL"

        # **IMPROVEMENT: The "Phoenix" Recovery Signal**
        # Replaces risky "V-bottom hunting". We wait for a confirmed recovery.
        # This signal requires the price to rise from the ashes and prove itself.
        price_reclaimed_sma20 = current_price > sma_20
        momentum_is_positive = macd_histogram > 0 and prev_macd_histogram < 0
        sentiment_supports_recovery = net_sentiment_score > 2.5 # Needs strong macro news (stimulus, etc.)

        if price_reclaimed_sma20 and momentum_is_positive and sentiment_supports_recovery:
            return "BUY"
        
        # Default action in a crash is to HOLD cash and wait for the Phoenix signal.
        return "HOLD"

    # --- Logic for all other non-crash scenarios (NORMAL MODE) ---
    bullish_trend = short_ema > long_ema and current_price > sma_50
    bearish_trend = short_ema < long_ema and current_price < sma_50

    # **IMPROVEMENT: "Slow Bleed" Stop-Loss Logic**
    # Addresses failure to exit moderately losing trades. If we are in what should be
    # a bullish trend but the price violates the short-term EMA, the thesis is wrong. Exit.
    if bullish_trend and current_price < short_ema:
        return "SELL"
    # Symmetrically, if we are in a bearish trend (shorting) and price breaks above, exit.
    if bearish_trend and current_price > short_ema:
        return "BUY" # Cover short, which translates to BUY in this system

    # --- Normal Trend-Following Logic ---
    if bullish_trend:
        # BUY signal: Enter on strength, confirmed by momentum and sentiment.
        is_momentum_accelerating = macd_histogram > prev_macd_histogram
        if macd_histogram > 0 and is_momentum_accelerating and rsi > 50 and rsi < 75 and net_sentiment_score > -1.0:
            return "BUY"
        
        # Profit-taking SELL signal: Exit on signs of exhaustion.
        is_overbought = rsi > 78 or current_price > upper_band
        is_momentum_fading = macd_histogram > 0 and macd_histogram < prev_macd_histogram
        if is_overbought and is_momentum_fading:
            return "SELL"

    if bearish_trend:
        # SELL signal: Enter on weakness, confirmed by momentum and sentiment.
        is_momentum_accelerating_down = macd_histogram < prev_macd_histogram
        if macd_histogram < 0 and is_momentum_accelerating_down and rsi < 50 and rsi > 25 and net_sentiment_score < 1.0:
            return "SELL"

        # Oversold bounce BUY signal: Cover short on signs of a reversal.
        is_oversold = rsi < 22 or current_price < lower_band
        is_momentum_reversing_up = macd_histogram < 0 and macd_histogram > prev_macd_histogram
        if is_oversold and is_momentum_reversing_up:
            return "BUY"

    # --- Ranging Market Logic (if not clearly trending) ---
    # This logic applies if neither bullish_trend nor bearish_trend is true.
    if not (bullish_trend or bearish_trend):
        # Mean-reversion BUY at support
        if rsi < 35 and current_price < lower_band and macd_histogram > prev_macd_histogram and net_sentiment_score > -2.0:
            return "BUY"
        
        # Mean-reversion SELL at resistance
        if rsi > 65 and current_price > upper_band and macd_histogram < prev_macd_histogram and net_sentiment_score < 2.0:
            return "SELL"

    return "HOLD"