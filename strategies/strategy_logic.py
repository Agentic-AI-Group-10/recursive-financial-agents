import numpy as np
import re
import math

# --- Helper Functions for Technical Indicators ---

def _calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages. Internal use."""
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

def calculate_ema(prices, period):
    """Calculates the latest Exponential Moving Average (EMA)."""
    if len(prices) < period:
        return None
    return _calculate_ema_series(prices, period)[-1]

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
    short_ema_series = _calculate_ema_series(prices, short_period)
    long_ema_series = _calculate_ema_series(prices, long_period)
    macd_line = short_ema_series - long_ema_series
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = _calculate_ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = _calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands."""
    if len(prices) < period:
        return None, None, None
    prices_arr = np.array(prices[-period:], dtype=float)
    sma = np.mean(prices_arr)
    std_dev = np.std(prices_arr)
    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    return upper_band, sma, lower_band

def calculate_adx(prices, period=14):
    """Calculates the Average Directional Index (ADX)."""
    if len(prices) < 2 * period:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    
    plus_dm = np.where((deltas > 0) & (deltas > -np.roll(deltas, 1)), deltas, 0)
    minus_dm = np.where((deltas < 0) & (-deltas > np.roll(deltas, 1)), -deltas, 0)
    
    tr = np.abs(np.diff(prices_arr)) # Simplified TR for close-only prices
    
    # Use pandas for robust rolling calculations if available
    try:
        import pandas as pd
        tr_series = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        plus_dm_series = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_series = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        
        plus_di = 100 * (plus_dm_series / tr_series)
        minus_di = 100 * (minus_dm_series / tr_series)
        
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.iloc[-1]
    except (ImportError, ZeroDivisionError):
        return 25 # Fallback to a neutral value if calculation fails

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces volatility-adaptive mechanisms and enhanced regime filtering.
    1.  Volatility-Adaptive Risk Management: Replaces the static stop-loss with a
        more dynamic ATR-based mechanism, which adjusts risk thresholds based on
        current market volatility to avoid premature exits.
    2.  Enhanced Regime Filtering: Implements the Average Directional Index (ADX)
        to identify and avoid trading in low-trend, "choppy" market conditions,
        reducing false signals and whipsaw losses.
    3.  Refined Entry/Exit Signals: Migrates from SMA to faster-reacting EMA for
        trend analysis and integrates Bollinger Bands with RSI for more robust,
        volatility-aware overbought/oversold signals.
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
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.5,
        "supply chain disruption": -2.5, "uncertainty": -1.5,
        "euphoria": -3.0, "mania": -3.5, "irrational exuberance": -3.5, "extreme greed": -3.0,
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
    EMA_TREND_LONG = 100
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    BB_PERIOD = 20
    STOP_LOSS_LOOKBACK = 25
    ATR_STOP_MULTIPLIER = 3.0

    required_history_length = max(EMA_TREND_LONG + 1, ADX_PERIOD * 2, 60)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_100 = calculate_ema(all_prices, EMA_TREND_LONG)
    ema_50 = calculate_ema(all_prices, EMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)
    bb_upper, _, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_100, ema_50, rsi, atr, adx, bb_upper, donchian_high]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]

    # --- 3. Regime Detection ---
    is_trending_market = adx > 25
    is_choppy_market = adx < 20
    is_long_term_downtrend = current_price < ema_100
    is_deeply_oversold = rsi < 30 and current_price < bb_lower

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: RISK MANAGEMENT (HIGHEST PRIORITY)
    # Priority 1: Volatility-Adjusted Trailing Stop-Loss.
    stop_price = donchian_high - (ATR_STOP_MULTIPLIER * atr)
    if current_price < stop_price:
        return "SELL"

    # REGIME 2: EXTREME OVERBOUGHT / PROFIT TAKING
    # Sell when extremely overbought (RSI > 75) AND price pierces the upper Bollinger Band,
    # especially if momentum is fading (MACD histogram declining).
    is_extremely_overbought = rsi > 75 and current_price > bb_upper
    is_momentum_fading = macd_histogram < prev_macd_histogram
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # REGIME 3: TREND BREAKDOWN
    # Sell on a clear trend and momentum breakdown, avoiding sells in deeply oversold conditions.
    is_trend_breakdown = current_price < ema_50
    is_momentum_breakdown = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 2.0
    if is_trend_breakdown and is_momentum_breakdown and not is_deeply_oversold and is_sentiment_permissive_for_sell:
        return "SELL"

    # REGIME 4: TREND-FOLLOWING BUY
    # Buy only in a confirmed trending market to avoid whipsaws.
    is_primary_uptrend = current_price > ema_50 and ema_50 > ema_100
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 70
    is_sentiment_permissive_for_buy = net_sentiment_score > -2.0

    if is_trending_market and is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy:
        return "BUY"
        
    # REGIME 5: CONTRARIAN "BUY THE DIP"
    # High-conviction buy in a long-term uptrend during a sharp, oversold pullback.
    is_long_term_uptrend = current_price > ema_100
    is_momentum_reversing_up = macd_histogram > prev_macd_histogram
    if is_long_term_uptrend and is_deeply_oversold and is_momentum_reversing_up:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"