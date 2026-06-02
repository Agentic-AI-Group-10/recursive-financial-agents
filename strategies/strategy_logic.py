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
        return ema_values[period-1:]

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
    prices_arr = np.array(prices, dtype=float)
    short_ema_full = calculate_ema_series(prices_arr, short_period)
    long_ema_full = calculate_ema_series(prices_arr, long_period)
    macd_line = short_ema_full[long_period-short_period:] - long_ema_full
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
    # Using Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    try:
        import pandas as pd
        return pd.Series(price_ranges).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    except ImportError:
        atr_val = np.mean(price_ranges[:period])
        for i in range(period, len(price_ranges)):
            atr_val = ((atr_val * (period - 1)) + price_ranges[i]) / period
        return atr_val

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    prices_arr = np.array(prices[-period:], dtype=float)
    middle_band = np.mean(prices_arr)
    std_dev = np.std(prices_arr)
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, middle_band, lower_band

def calculate_adx(prices, period=14):
    """Calculates the Average Directional Index (ADX)."""
    if len(prices) < 2 * period:
        return None
    prices_arr = np.array(prices, dtype=float)
    highs = prices_arr # Using close as a proxy for high/low
    lows = prices_arr
    
    up_move = np.diff(highs)
    down_move = -np.diff(lows)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr = np.abs(np.diff(prices_arr)) # Simplified TR for close-only data
    
    # Using Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    try:
        import pandas as pd
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().to_numpy()
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy() / atr)
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        return adx
    except (ImportError, ZeroDivisionError):
        # Fallback calculation if pandas is not available or division by zero
        return None

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces two major architectural upgrades for robustness:
    1.  ADX Trend Strength Filter: A new core component, the Average Directional
        Index (ADX), is implemented. The strategy now filters out low-conviction
        trades by requiring a minimum trend strength (ADX > 22) before entering
        positions in normal market conditions. This is designed to significantly
        reduce whipsaw losses in sideways or choppy markets.
    2.  Dynamic Volatility-Based Exits (ATR Chandelier Stop): The fixed percentage
        stop-loss is replaced with a Chandelier Exit based on the Average True
        Range (ATR). This allows the stop-loss to dynamically adapt to market
        volatility—tightening in calm markets to protect profits and widening in
        volatile markets to avoid premature exits.
    3.  Bollinger Band Mean Reversion Signal: The contrarian "buy the dip" logic
        is refined to use Bollinger Bands, triggering a buy on extreme statistical
        deviations from the mean, providing a more robust signal than ROC alone.
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
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    SMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    BB_PERIOD = 20
    CHANDELIER_LOOKBACK = 22
    CHANDELIER_ATR_MULT = 3.0

    required_history_length = max(SMA_TREND_LONG + 1, 2 * ADX_PERIOD, CHANDELIER_LOOKBACK + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)
    bb_upper, _, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high_22 = np.max(all_prices[-CHANDELIER_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, atr, adx, bb_lower, donchian_high_22]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_crisis_regime = is_long_term_downtrend and current_price < sma_50

    # Capitulation Regime: Extreme oversold state, often a precursor to a sharp bounce.
    is_deeply_oversold_rsi = rsi < 25
    is_below_bollinger = current_price < bb_lower
    is_capitulation_candidate = is_deeply_oversold_rsi and is_below_bollinger

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    # Buy on signs of extreme fear, but only when momentum shows a nascent turn.
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    # If in a confirmed long-term downtrend, be defensive and exit positions.
    if is_crisis_regime:
        return "SELL"

    # REGIME 3: NORMAL MARKET CONDITIONS (TREND-FILTERED)

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR Chandelier Stop-Loss.
    chandelier_exit = donchian_high_22 - (atr * CHANDELIER_ATR_MULT)
    if current_price < chandelier_exit:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_trend_breakdown = current_price < sma_50
    is_momentum_crossing_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_breakdown and is_momentum_crossing_down:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_overbought = rsi > 78
    if is_overbought and is_momentum_fading:
        return "SELL"

    # --- ADX TREND STRENGTH FILTER ---
    # Before considering any new BUY signals, ensure a trend is established.
    is_trending = adx > 22
    if not is_trending:
        return "HOLD"

    # --- BUY LOGIC (Only in a confirmed trending market) ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_crossing_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 75
    is_sentiment_permissive = net_sentiment_score > -3.0

    if is_primary_uptrend and is_momentum_crossing_up and is_not_overbought and is_sentiment_permissive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"