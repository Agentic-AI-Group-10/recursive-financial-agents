import numpy as np
import re
import pandas as pd

# --- Helper Functions for Technical Indicators ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    # Using pandas for a robust and standard EMA calculation
    return pd.Series(data).ewm(span=period, adjust=False).mean().to_numpy()

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = deltas * (deltas > 0)
    losses = -deltas * (deltas < 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Use Wilder's smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None

    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    macd_line = short_ema_series - long_ema_series
    
    if len(prices) < long_period + signal_period:
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    
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

def calculate_adx_from_close(prices, period=14):
    """
    Calculates the Average Directional Index (ADX) using only closing prices.
    This is an adaptation, as true ADX requires High and Low prices.
    """
    if len(prices) < 2 * period:
        return None

    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)

    plus_dm = np.where(deltas > 0, deltas, 0)
    minus_dm = np.where(deltas < 0, -deltas, 0)
    
    # Approximate True Range using absolute price change
    tr = np.abs(deltas)

    # Use pandas for smoothed calculations, which is more robust
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy() / atr)

    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx)] = 0 # Handle division by zero

    # Calculate ADX
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    
    return adx[-1]

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy using ADX for robust trend
    identification and enhanced signal confirmation.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis (Refined Keywords) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.5, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat estimates": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5, "market rally": 2.0, "jobless claims fall": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation persists": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.5, "war": -3.0, "conflict": -3.0, "sanctions": -2.5,
        "market turmoil": -2.0, "credit crunch": -2.5, "ai bubble": -2.0,
        "tightening": -1.5, "miss estimates": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "supply chain disruption": -2.0, "uncertainty": -1.5, "weak jobs": -2.0, "jobless claims rise": -1.5
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
    
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    BB_PERIOD = 20
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + 9, VOL_LONG_PERIOD + 1, 2 * ADX_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    adx = calculate_adx_from_close(all_prices, ADX_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, upper_band, adx]) or macd_hist_series is None or len(macd_hist_series) < 3:
        return "HOLD"
    
    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices[1:]) / np.array(all_prices[:-1]))
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.6) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction, sentiment-driven trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 3.0
        BEARISH_SENTIMENT_THRESHOLD = -3.0
        
        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and macd_histogram > 0 and rsi < 70:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and macd_histogram < 0 and rsi > 30:
            return "SELL"
    else:
        # === NORMAL MODE: ADX-based sub-regime switching ===
        is_trending_market = adx > 25

        if is_trending_market:
            # Sub-Regime: Trending Market (Confirmed by ADX)
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema
            
            # Proactive profit-taking / trend exhaustion signal
            is_momentum_fading_up = macd_histogram > 0 and macd_histogram < prev_macd_histogram
            if bullish_trend and rsi > 80 and is_momentum_fading_up:
                return "SELL"

            # Entry requires 2-day momentum confirmation to avoid whipsaws
            if bullish_trend and macd_histogram > 0 and prev_macd_histogram > 0 and rsi < 78 and net_sentiment_score > -2.0:
                return "BUY"
            
            # Exit requires 2-day momentum confirmation
            if bearish_trend and macd_histogram < 0 and prev_macd_histogram < 0 and rsi > 22 and net_sentiment_score < 2.0:
                return "SELL"
        else:
            # Sub-Regime: Ranging / Choppy Market (ADX < 25)
            
            # Buy the dip if oversold and showing signs of reversal
            is_reversing_up = macd_histogram > prev_macd_histogram
            if (rsi < 28 and current_price < lower_band) and \
               (net_sentiment_score > -2.5) and is_reversing_up:
                return "BUY"
                
            # Sell the rip if overbought and showing signs of reversal
            is_reversing_down = macd_histogram < prev_macd_histogram
            if (rsi > 72 and current_price > upper_band) and \
                 (net_sentiment_score < 2.5) and is_reversing_down:
                return "SELL"

    return "HOLD"