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

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_adx(prices, period=14):
    """
    Calculates the Average Directional Index (ADX) using only closing prices.
    This is a proxy as traditional ADX uses High, Low, and Close.
    """
    if len(prices) < 2 * period:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    up_moves = np.where(deltas > 0, deltas, 0)
    down_moves = np.where(deltas < 0, -deltas, 0)
    
    tr = np.abs(deltas)
    
    atr = calculate_ema_series(tr, period)
    smoothed_up = calculate_ema_series(up_moves, period)
    smoothed_down = calculate_ema_series(down_moves, period)

    if atr is None or len(atr) == 0 or smoothed_up is None or smoothed_down is None:
        return None
        
    plus_di = 100 * (smoothed_up / atr)
    minus_di = 100 * (smoothed_down / atr)
    
    # Ensure lengths match for DX calculation
    min_len = min(len(plus_di), len(minus_di))
    plus_di = plus_di[-min_len:]
    minus_di = minus_di[-min_len:]

    sum_di = plus_di + minus_di
    # Avoid division by zero
    sum_di[sum_di == 0] = 1e-10
    
    dx = 100 * (np.abs(plus_di - minus_di) / sum_di)
    
    if len(dx) < period:
        return None
        
    adx = calculate_ema_series(dx, period)
    return adx[-1] if len(adx) > 0 else None

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances the V2 strategy with four key upgrades for robustness:
    1.  Dynamic ATR Stop-Loss: Replaces the fixed percentage stop-loss with a
        dynamic, ATR-based trailing stop to adapt risk management to market volatility.
    2.  ADX Trend Strength Filter: Adds an ADX filter to BUY signals to ensure
        entries occur during strong directional trends, reducing whipsaw losses.
    3.  Enhanced Volatility Regime Detection: Uses the standard deviation of recent
        returns to more accurately detect volatility spikes and shifts into crisis regimes.
    4.  Sentiment Density Scoring: Normalizes the sentiment score by news context
        length, measuring sentiment "density" to avoid dilution by long, neutral articles.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0,
        "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0, "short squeeze": 3.5,
        "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "vix spike": -2.5,
        "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0, "bubble": -2.0,
        "uncertainty": -1.5, "euphoria": -2.5, "mania": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    raw_sentiment_score = 0.0
    keyword_count = 0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        matches = list(re.finditer(pattern, context_lower))
        if matches:
            keyword_count += len(matches)
            for match in matches:
                pre_context = context_lower[max(0, match.start() - 30):match.start()]
                is_negated = any(neg_word in pre_context for neg_word in negation_words)
                raw_sentiment_score += -weight if is_negated else weight
    
    # Normalize score to create a "density" metric
    word_count = len(context_lower.split())
    net_sentiment_score = (raw_sentiment_score / (word_count + 1)) * 100 if word_count > 0 else 0.0

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20
    VOLATILITY_PERIOD = 20

    required_history_length = max(SMA_TREND_LONG + 1, 2 * ADX_PERIOD, VOLATILITY_PERIOD + 2)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])
    
    # Volatility Index Calculation
    returns = np.diff(all_prices[-VOLATILITY_PERIOD-1:]) / all_prices[-VOLATILITY_PERIOD-2:-1]
    volatility_index = np.std(returns)
    volatility_index_ma = np.mean([np.std(np.diff(all_prices[i-VOLATILITY_PERIOD:i]) / all_prices[i-VOLATILITY_PERIOD-1:i-1]) for i in range(len(all_prices) - 5, len(all_prices))]) if len(all_prices) > VOLATILITY_PERIOD + 5 else volatility_index

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, adx, roc_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_volatility_spike = volatility_index > (volatility_index_ma * 2.0)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_volatility_spike) or is_crash_velocity

    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD"

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR-based Trailing Stop-Loss.
    stop_loss_price = donchian_high_20 - (atr * 3.0)
    if current_price < stop_loss_price:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 1.0 # Tighter threshold for density score
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 82
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -1.0 # Tighter threshold
    is_trending_strongly = adx > 22 # ADX filter for trend strength

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_trending_strongly:
        return "BUY"

    return "HOLD"