import numpy as np
import re
import math

# Using pandas is strongly preferred for robust, industry-standard indicator calculations.
# The fallback methods are retained but may have slight differences from standard implementations.
try:
    import pandas as pd
except ImportError:
    pd = None

# --- Helper Functions for Technical Indicators ---

def calculate_ema(data, period):
    """Calculates the Exponential Moving Average (EMA) for a series."""
    if pd:
        return pd.Series(data).ewm(span=period, adjust=False).mean().to_numpy()
    # Fallback pure-python EMA
    if len(data) < period: return np.array([])
    ema_values = np.zeros_like(data, dtype=float)
    ema_values[period-1] = np.mean(data[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(data)):
        ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period: return None
    return np.mean(prices[-period:])

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1: return None
    if pd:
        delta = pd.Series(prices).diff(1)
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return (100 - (100 / (1 + rs))).iloc[-1]
    # Fallback pure-python RSI
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the latest MACD line, signal line, and histogram value."""
    if len(prices) < long_period + signal_period: return None, None, None
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    if len(ema_short) == 0 or len(ema_long) == 0: return None, None, None
    macd_line_series = ema_short - ema_long
    signal_line_series = calculate_ema(macd_line_series, signal_period)
    if len(signal_line_series) < 2: return None, None, None
    histogram_series = macd_line_series - signal_line_series
    return macd_line_series[-1], signal_line_series[-1], histogram_series[-2:]

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility for simplicity."""
    if len(prices) < period + 1: return None
    price_ranges = np.abs(np.diff(np.array(prices)))
    atr_series = calculate_ema(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands."""
    if len(prices) < period: return None, None, None
    sma = np.mean(prices[-period:])
    std_dev = np.std(prices[-period:])
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces three architectural upgrades for enhanced robustness:
    1.  Dynamic ATR Trailing Stop: Replaces the fixed-percentage stop-loss with a
        volatility-adjusted trailing stop (Donchian High - N*ATR). This adapts
        risk management to the current market volatility, preventing premature
        exits in volatile uptrends and tightening stops in calm periods.
    2.  Multi-Indicator Consensus Model: Moves beyond simple MACD crossovers for
        entries. A new high-conviction BUY signal requires consensus from a trend-
        following EMA crossover (20 > 50), positive momentum (RSI > 50), and a
        positive MACD, significantly reducing whipsaw trades in sideways markets.
    3.  Bollinger Band Exhaustion Signal: Profit-taking is refined. Instead of a
        static RSI level, a SELL is triggered when the price exceeds the upper
        Bollinger Band while momentum (MACD histogram) is actively decelerating,
        identifying points of likely exhaustion with greater precision.
    """
    # --- 1. Configuration & Constants ---
    # Trend Indicators
    EMA_SHORT_PERIOD = 20
    EMA_LONG_PERIOD = 50
    SMA_CRISIS_PERIOD = 200
    # Momentum/Oscillator Indicators
    RSI_PERIOD = 14
    MACD_SHORT, MACD_LONG, MACD_SIGNAL = 12, 26, 9
    # Volatility Indicators
    ATR_PERIOD = 14
    BB_PERIOD = 20
    # Risk Management
    STOP_LOSS_LOOKBACK = 25
    ATR_STOP_MULTIPLIER = 2.5

    # --- 2. Data Preparation & History Check ---
    all_prices = price_history + [current_price]
    required_history_length = max(SMA_CRISIS_PERIOD, EMA_LONG_PERIOD, BB_PERIOD) + MACD_SIGNAL
    if len(all_prices) < required_history_length:
        return "HOLD"

    # --- 3. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "soft landing": 2.5, "cooling inflation": 2.5,
        "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "de-escalation": 2.0,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "black swan": -4.0, "systemic risk": -4.0,
        "credit crunch": -3.5, "rate hike": -2.5, "bankruptcy": -2.5, "hawkish": -2.0,
        "bearish": -2.0, "sell-off": -2.0, "bubble": -2.0, "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "fail to", "without", "avoids", "prevent"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', context_lower):
            # Basic negation check for simplicity
            pre_context = context_lower.split(keyword, 1)[0][-30:]
            if not any(neg_word in pre_context for neg_word in negation_words):
                net_sentiment_score += weight

    # --- 4. Technical Indicator Calculation ---
    prices_arr = np.array(all_prices, dtype=float)
    ema_short = calculate_ema(prices_arr, EMA_SHORT_PERIOD)
    ema_long = calculate_ema(prices_arr, EMA_LONG_PERIOD)
    sma_crisis = calculate_sma(all_prices, SMA_CRISIS_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd(all_prices, MACD_SHORT, MACD_LONG, MACD_SIGNAL)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    bb_upper, _, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [sma_crisis, rsi, atr, bb_upper, macd_hist_series]) or len(ema_short) < 1 or len(ema_long) < 1:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    macd_hist_delta = macd_hist_series[-1] - macd_hist_series[-2]

    # --- 5. Regime Detection ---
    is_crisis_regime = current_price < sma_crisis
    is_deeply_oversold_for_contrarian_buy = rsi < 28 and current_price < bb_lower

    # --- 6. Decision Logic (Hierarchical) ---

    # REGIME 1: CRISIS AVERSION (HIGHEST PRIORITY)
    # If in a long-term bear market, the primary goal is capital preservation.
    # Only sell or hold cash; buying is disabled until the trend improves.
    if is_crisis_regime:
        return "SELL"

    # REGIME 2: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR Trailing Stop-Loss.
    # Sell if price drops more than ATR_MULTIPLIER * ATR from the recent high.
    trailing_stop_price = donchian_high - (atr * ATR_STOP_MULTIPLIER)
    if current_price < trailing_stop_price:
        return "SELL"

    # Priority 2: Exhaustion / Profit-Taking Signal.
    # Sell if price is overextended (above BB) and momentum is clearly fading.
    is_overextended = current_price > bb_upper
    is_momentum_fading_sharply = macd_hist_delta < 0 and macd_histogram < macd_hist_series[-2]
    is_overbought = rsi > 78
    if is_overextended and is_overbought and is_momentum_fading_sharply:
        return "SELL"

    # Priority 3: Trend Breakdown Signal.
    # Sell if the short-term trend crosses below the long-term trend.
    if ema_short[-1] < ema_long[-1] and ema_short[-2] >= ema_long[-2]:
        return "SELL"

    # --- BUY LOGIC ---
    # Priority 1: Contrarian "Buy the Dip" Signal.
    # High-conviction buy on extreme oversold conditions IF momentum is turning positive.
    if is_deeply_oversold_for_contrarian_buy and macd_hist_delta > 0:
        return "BUY"

    # Priority 2: High-Conviction Trend-Following Entry.
    # Buy only when multiple signals align, confirming a healthy uptrend.
    is_uptrend_confirmed = ema_short[-1] > ema_long[-1]
    is_positive_momentum = rsi > 52
    is_accelerating_momentum = macd_histogram > 0
    is_sentiment_supportive = net_sentiment_score > -2.0
    # Trigger condition: a fresh crossover into a confirmed uptrend.
    is_fresh_ema_cross = ema_short[-1] > ema_long[-1] and ema_short[-2] <= ema_long[-2]

    if is_fresh_ema_cross and is_uptrend_confirmed and is_positive_momentum and is_accelerating_momentum and is_sentiment_supportive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"