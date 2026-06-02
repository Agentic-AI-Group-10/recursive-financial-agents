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
        # Using pandas is preferred for accuracy and standard implementation
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()
    except ImportError:
        # Fallback pure-python EMA calculation
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
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    macd_line = full_short_ema - full_long_ema
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line = calculate_ema_series(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Using EMA for ATR calculation is standard (also known as Wilder's Smoothing)
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a robust, adaptive framework evolving from V2:
    1.  Adaptive Risk Management: Replaces the fixed-percentage stop with a dynamic,
        ATR-based trailing stop (3x ATR from the 20-day high) to better adapt
        to changing market volatility.
    2.  Enhanced Trend & Regime Filtering: Upgrades trend analysis from SMA to more
        responsive EMAs (50 and 200). A strict regime filter is enforced:
        trend-following long positions are only considered above the 200-day EMA,
        preserving capital during major bear markets.
    3.  Volatility Breakout Confirmation: BUY signals are strengthened, now requiring
        a price breakout above the previous 20-day high, ensuring entry is backed
        by strong momentum and reducing trades in choppy, trendless markets.
    4.  Refined Sentiment Dictionary: Adds nuanced market structure terms like
        "gamma squeeze" and "liquidity crisis" for more accurate sentiment scoring.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Positive
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "gamma squeeze": 3.5, "short squeeze": 3.5, "capitulation": 3.0,
        "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5, # Ambiguous
        # Negative
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "liquidity crisis": -4.0, "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5,
        "geopolitical risk": -2.5, "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0,
        "bearish": -2.0, "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0,
        "bubble": -2.0, "uncertainty": -1.5,
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
    EMA_TREND_LONG = 200
    EMA_TREND_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ROC_CRASH_PERIOD = 20
    BREAKOUT_LOOKBACK = 20
    ATR_STOP_MULTIPLE = 3.0

    required_history_length = EMA_TREND_LONG + 1
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_200 = calculate_ema(all_prices, EMA_TREND_LONG)
    ema_50 = calculate_ema(all_prices, EMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    
    # Corrected lookback to prevent look-ahead bias
    donchian_high_20_prior = np.max(all_prices[-BREAKOUT_LOOKBACK-1:-1]) if len(all_prices) > BREAKOUT_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_200, ema_50, rsi, atr, roc_20, donchian_high_20_prior]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_bull_regime = current_price > ema_200
    is_bear_regime = not is_bull_regime

    # Capitulation Regime: An extreme event that can override the main regime filter
    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    # Buy extreme fear, but only when momentum shows signs of turning.
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR Stop-Loss. Sell if price drops 3x ATR from the 20-day high.
    atr_stop_level = donchian_high_20_prior - (ATR_STOP_MULTIPLE * atr)
    if current_price < atr_stop_level:
        return "SELL"

    # Priority 2: Bear Regime Exit. If we enter a bear market, liquidate long positions.
    if is_bear_regime and current_price < ema_50:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC (Only in Bull Regime) ---
    if is_bull_regime:
        is_medium_term_uptrend = current_price > ema_50
        is_momentum_positive = macd_histogram > 0
        is_breakout_confirmed = current_price > donchian_high_20_prior
        is_not_overextended = rsi < 78
        is_sentiment_permissive = net_sentiment_score > -3.0

        if is_medium_term_uptrend and is_momentum_positive and is_breakout_confirmed and is_not_overextended and is_sentiment_permissive:
            return "BUY"

    # Default action is to hold the current position.
    return "HOLD"