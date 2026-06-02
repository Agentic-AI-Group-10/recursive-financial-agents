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

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None, None, None
    prices_slice = prices[-period:]
    sma = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    band_width = ((upper_band - lower_band) / sma) * 100 if sma > 0 else 0
    return upper_band, sma, lower_band, band_width

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces market adaptivity and enhances risk management.
    1.  Adaptive Indicators: Bollinger Bands are integrated to provide dynamic
        overbought/oversold levels, replacing some fixed RSI thresholds.
    2.  Volatility Regime Detection: The strategy now identifies periods of low-
        volatility "squeezes" using Bollinger Band Width, preparing to enter on
        high-momentum breakouts.
    3.  Enhanced Sentiment Engine: The sentiment model now detects clusters of
        negative news, applying a multiplier to the score to better react to
        cascading bad news.
    4.  ATR-Adaptive Stop-Loss: The stop-loss percentage now dynamically widens
        based on recent ATR, reducing premature exits in volatile markets.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    negative_keyword_count = 0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            final_weight = -weight if is_negated else weight
            net_sentiment_score += final_weight
            if final_weight < 0:
                negative_keyword_count += 1
    
    # Amplify score if multiple negative keywords are present (news cluster)
    if negative_keyword_count >= 3:
        net_sentiment_score *= 1.5

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])
    upper_bb, middle_bb, lower_bb, bb_width = calculate_bollinger_bands(all_prices, BB_PERIOD)
    
    # Calculate historical BBW to detect squeezes
    bbw_history = []
    if len(all_prices) >= BB_PERIOD + 10: # Need some history for the squeeze check
        for i in range(10):
            _, _, _, hist_bbw = calculate_bollinger_bands(all_prices[:-(i+1)], BB_PERIOD)
            if hist_bbw is not None:
                bbw_history.append(hist_bbw)

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, roc_20, upper_bb]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = is_long_term_downtrend or is_crash_velocity

    is_deeply_oversold = rsi < 25 and current_price < lower_bb
    is_capitulation_candidate = is_crash_velocity and is_deeply_oversold

    is_volatility_squeeze = len(bbw_history) > 0 and bb_width < np.percentile(bbw_history, 10)

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    if is_crisis_regime and not is_capitulation_candidate:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD"

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: ATR-Adaptive Dynamic Stop-Loss. Base 7%, widens up to 9% in high vol.
    atr_pct_of_price = (atr / current_price) if current_price > 0 else 0
    adaptive_stop_pct = 0.93 - min(atr_pct_of_price * 1.5, 0.02) # ATR multiplier scales sensitivity
    if current_price < (donchian_high_20 * adaptive_stop_pct):
        return "SELL"

    # Priority 2: Trend breakdown signal.
    is_trend_breakdown = current_price < sma_50 and current_price < middle_bb
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_breakdown and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Profit-taking on exhaustion.
    is_overbought_exhaustion = rsi > 78 and current_price > upper_bb and macd_hist_delta < 0
    if is_overbought_exhaustion:
        return "SELL"

    # --- BUY LOGIC ---
    # Sentiment-modulated RSI threshold
    rsi_buy_ceiling = 75 + min(max(net_sentiment_score, -5), 5)

    # Condition 1: Standard Trend-Following Entry
    is_uptrend = current_price > sma_50 and current_price > middle_bb
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < rsi_buy_ceiling
    
    if is_uptrend and is_momentum_confirming_up and is_not_overbought:
        return "BUY"

    # Condition 2: Volatility Breakout Entry
    is_breakout_candidate = is_volatility_squeeze and current_price > upper_bb
    if is_breakout_candidate and is_momentum_confirming_up:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"