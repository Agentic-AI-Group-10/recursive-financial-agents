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
        return ema_values

def calculate_ema(prices, period):
    """Calculates the latest Exponential Moving Average (EMA)."""
    if len(prices) < period:
        return None
    return calculate_ema_series(prices, period)[-1]

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
    macd_line = short_ema_series - long_ema_series
    if len(macd_line) < long_period + signal_period -1: # Ensure enough data for signal line
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line[long_period-1:], signal_period)
    histogram = macd_line[long_period-1:][len(macd_line[long_period-1:])-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_adx(prices, period=14):
    """Calculates the Average Directional Index (ADX)."""
    if len(prices) < 2 * period:
        return None
    
    prices_arr = np.array(prices, dtype=float)
    highs = prices_arr # Using close as proxy for H/L/C
    lows = prices_arr
    closes = prices_arr

    up_move = np.diff(highs)
    down_move = -np.diff(lows)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr1 = np.abs(highs[1:] - lows[1:])
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    true_range = np.maximum(tr1, tr2, tr3)
    
    atr = calculate_ema_series(true_range, period)
    plus_di = 100 * (calculate_ema_series(plus_dm, period) / atr)
    minus_di = 100 * (calculate_ema_series(minus_dm, period) / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = calculate_ema_series(dx, period)
    
    return adx[-1] if len(adx) > 0 else None

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances adaptability and risk management over its predecessor.
    1.  Enhanced Trend Detection: Replaces lagging SMAs with more responsive EMAs
        (50-day and 200-day) for quicker identification of trend shifts.
    2.  Whipsaw Reduction Filter: Introduces the Average Directional Index (ADX)
        to measure trend strength. Trades are only initiated in strongly trending
        markets (ADX > 22), filtering out noise from sideways, choppy periods.
    3.  Dynamic Volatility-Based Exits: The static percentage stop-loss is replaced
        with a dynamic ATR (Average True Range) trailing stop. This adapts the
        stop-loss level to current market volatility, protecting profits more
        effectively and avoiding premature exits during volatile swings.
    4.  Sentiment Circuit Breaker: A "veto" system is added for catastrophic news
        (e.g., "black swan", "systemic risk"), preventing any BUY signals during
        periods of extreme market fear, regardless of technical indicators.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -5.0, "systemic risk": -5.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
    }
    catastrophic_keywords = ["black swan", "systemic risk", "war", "market collapse", "credit crisis"]
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    
    net_sentiment_score = 0.0
    has_catastrophic_news = any(keyword in context_lower for keyword in catastrophic_keywords)

    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    EMA_TREND_LONG = 200
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    STOP_LOSS_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 2.5

    required_history_length = max(EMA_TREND_LONG + 1, ADX_PERIOD * 2, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_200 = calculate_ema(all_prices, EMA_TREND_LONG)
    ema_50 = calculate_ema(all_prices, EMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)
    highest_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_200, ema_50, rsi, atr, adx, highest_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_uptrend = current_price > ema_200
    is_medium_term_uptrend = current_price > ema_50
    is_trending_market = adx > 22

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: RISK MANAGEMENT (HIGHEST PRIORITY)
    
    # Priority 1: Dynamic ATR Trailing Stop-Loss
    # Sell if the price drops more than ATR_STOP_MULTIPLIER * ATR from the recent high.
    trailing_stop_price = highest_high_20 - (ATR_STOP_MULTIPLIER * atr)
    if current_price < trailing_stop_price:
        return "SELL"

    # Priority 2: Major Trend Breakdown
    # Sell if we cross below the long-term trendline with confirming momentum.
    if current_price < ema_200 and ema_50 < ema_200:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading and not is_trending_market:
        return "SELL"

    # REGIME 2: BUY LOGIC (Opportunity Seeking)

    # Priority 1: Sentiment Veto / Circuit Breaker
    # Do not enter new long positions if there is catastrophic news.
    if has_catastrophic_news:
        return "HOLD"

    # Priority 2: Primary Trend-Following Entry
    is_confirmed_uptrend = is_long_term_uptrend and is_medium_term_uptrend
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 75
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.0

    if is_confirmed_uptrend and is_trending_market and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"

    # Priority 3: Contrarian "Buy the Dip" in a confirmed long-term uptrend
    is_pullback_in_uptrend = is_long_term_uptrend and not is_medium_term_uptrend
    is_oversold_on_pullback = rsi < 35
    is_momentum_reversing_up = macd_hist_delta > 0
    
    if is_pullback_in_uptrend and is_oversold_on_pullback and is_momentum_reversing_up:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"