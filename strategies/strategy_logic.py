import numpy as np
import pandas as pd
import re

# --- Helper Function for All Technical Indicators ---

def calculate_indicators(prices, short_ema_p=12, long_ema_p=26, rsi_p=14, macd_sig_p=9, vol_short_p=20, vol_long_p=100):
    """
    Calculates all necessary technical indicators in one pass using pandas for efficiency and correctness.
    Returns a dictionary of indicators or None if data is insufficient.
    """
    # A safe buffer for all calculations (e.g., long vol + macd signal line smoothing)
    required_len = vol_long_p + macd_sig_p
    if len(prices) < required_len:
        return None

    price_series = pd.Series(prices, dtype=float)

    # EMAs
    short_ema = price_series.ewm(span=short_ema_p, adjust=False).mean()
    long_ema = price_series.ewm(span=long_ema_p, adjust=False).mean()

    # MACD
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=macd_sig_p, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    # RSI (using Wilder's smoothing as is standard)
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=rsi_p - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_p - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Volatility of Log Returns
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    short_term_vol = log_returns.rolling(window=vol_short_p).std().iloc[-1]
    long_term_vol = log_returns.rolling(window=vol_long_p).std().iloc[-1]

    # Consolidate results into a dictionary
    indicators = {
        "short_ema": short_ema.iloc[-1],
        "long_ema": long_ema.iloc[-1],
        "rsi": rsi.iloc[-1],
        "rsi_prev": rsi.iloc[-2],
        "macd_histogram": macd_histogram.iloc[-1],
        "macd_histogram_prev": macd_histogram.iloc[-2],
        "short_term_vol": short_term_vol,
        "long_term_vol": long_term_vol,
    }
    
    # Final check for any NaN values that could result from insufficient data in rolling windows
    if any(pd.isna(v) for v in indicators.values()):
        return None

    return indicators

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that uses MACD for momentum
    confirmation and refined entry triggers to improve signal quality.

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
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "ai boom": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat": 1.5, "growth": 1.5, "easing": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "credit crunch": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "geopolitical risk": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "default risk": -2.0
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
    
    indicators = calculate_indicators(all_prices)
    if indicators is None:
        return "HOLD"

    # Unpack indicator values for clarity
    short_ema, long_ema = indicators["short_ema"], indicators["long_ema"]
    rsi, rsi_prev = indicators["rsi"], indicators["rsi_prev"]
    macd_hist, macd_hist_prev = indicators["macd_histogram"], indicators["macd_histogram_prev"]
    short_vol, long_vol = indicators["short_term_vol"], indicators["long_term_vol"]

    # Adaptive Volatility Regime
    is_high_volatility = (short_vol > long_vol * 1.5) and (short_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following with momentum confirmation ===
        BULLISH_SENTIMENT_THRESHOLD = 2.0
        BEARISH_SENTIMENT_THRESHOLD = -2.0
        
        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        # Buy on strong sentiment and confirmed upward trend & momentum
        if bullish_trend and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and macd_hist > 0:
            return "BUY"
        # Sell on strong negative sentiment and confirmed downward trend & momentum
        elif bearish_trend and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and macd_hist < 0:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market with accelerating momentum
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            RSI_OVERBOUGHT = 70
            RSI_OVERSOLD = 30
            
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema

            # Buy if trend is up, sentiment is positive, not overbought, AND momentum is accelerating
            if bullish_trend and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and rsi < RSI_OVERBOUGHT and macd_hist > macd_hist_prev:
                return "BUY"
            # Sell if trend is down, sentiment is negative, not oversold, AND momentum is accelerating downwards
            elif bearish_trend and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and rsi > RSI_OVERSOLD and macd_hist < macd_hist_prev:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion with RSI crossover)
            MEAN_REVERSION_RSI_OVERSOLD = 28
            MEAN_REVERSION_RSI_OVERBOUGHT = 72
            
            # Buy only when RSI crosses UP from oversold, confirming a reversal
            if rsi > MEAN_REVERSION_RSI_OVERSOLD and rsi_prev <= MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -1.5:
                return "BUY"
            # Sell only when RSI crosses DOWN from overbought, confirming a reversal
            elif rsi < MEAN_REVERSION_RSI_OVERBOUGHT and rsi_prev >= MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 1.5:
                return "SELL"

    # Default action is to hold, preserving capital when no high-conviction signal is present.
    return "HOLD"