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
        ema_values = np.zeros_like(data_arr, dtype=float)
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
    if period == 0: return None # Avoid division by zero
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
    short_ema_full = calculate_ema_series(prices, short_period)
    long_ema_full = calculate_ema_series(prices, long_period)
    macd_line = short_ema_full[len(short_ema_full)-len(long_ema_full):] - long_ema_full
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line_full = calculate_ema_series(macd_line, signal_period)
    signal_line = signal_line_full
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

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    prices_slice = prices[-period:]
    sma = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces dynamic, volatility-aware mechanisms for enhanced robustness.
    1.  Dynamic Overbought/Oversold Levels: Replaces static RSI thresholds with
        Bollinger Bands. Sell signals are now triggered when the price exceeds the
        upper band and momentum fades, providing a market-adaptive exit point.
    2.  Volatility-Adjusted Stop-Loss: The fixed percentage stop-loss is upgraded to an
        ATR-based trailing stop. This allows for wider stops in volatile markets and
        tighter stops in calm markets, optimizing risk management.
    3.  Dual-Confirmation Trend Filter: A faster EMA (20) is used alongside the
        primary SMA (50) to confirm trend entries, reducing false signals and
        whipsaws in choppy, directionless markets.
    4.  Sentiment Veto System: A "black swan" filter is implemented. An extremely
        negative sentiment score now acts as a veto, preventing new BUY signals
        regardless of technical strength, preserving capital during news-driven crashes.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5, # Ambiguous
        "recession": -3.0, "crisis": -3.5, "stagflation": -3.5, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -5.0, "systemic risk": -5.0, "contagion": -4.0, "credit crunch": -4.0,
        "liquidity crisis": -4.5, "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5,
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
    EMA_TREND_FAST = 20
    SMA_TREND_MEDIUM = 50
    SMA_TREND_LONG = 100
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = max(SMA_TREND_LONG + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    ema_20_series = calculate_ema_series(all_prices, EMA_TREND_FAST)
    ema_20 = ema_20_series[-1] if len(ema_20_series) > 0 else None
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    bb_upper, _, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [ema_20, sma_50, sma_100, rsi, atr, bb_upper, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection & Conditions ---
    # Crisis Regime: High-risk environment defined by long-term trend and volatility.
    is_long_term_downtrend = current_price < sma_100
    is_crisis_regime = is_long_term_downtrend and (current_price < sma_50)

    # Capitulation Regime: Extreme oversold state, potential for sharp reversal.
    is_deeply_oversold = rsi < 28
    is_far_below_bb = current_price < bb_lower
    is_capitulation_candidate = is_deeply_oversold and is_far_below_bb

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CRISIS AVERSION & CONTRARIAN BUYS
    if is_crisis_regime:
        # Contrarian Buy: Look for capitulation, but only if momentum is turning.
        if is_capitulation_candidate and macd_hist_delta > 0:
            return "BUY"
        # Otherwise, in a crisis, the default is to be defensive and sell.
        return "SELL"

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Volatility-Adjusted Trailing Stop-Loss
    # Sell if price drops more than 2.5x ATR from the 20-day high.
    if current_price < (donchian_high_20 - (2.5 * atr)):
        return "SELL"

    # Priority 2: Profit-taking on over-extension with fading momentum.
    # Sell if price is above the upper Bollinger Band and momentum is decreasing.
    is_overextended = current_price > bb_upper
    is_momentum_fading = macd_hist_delta < 0
    if is_overextended and is_momentum_fading:
        return "SELL"

    # Priority 3: Trend Breakdown Signal (Dual Confirmation)
    # Sell if the fast EMA crosses below the medium SMA, confirming a trend change.
    prev_ema_20 = ema_20_series[-2] if len(ema_20_series) > 1 else ema_20
    prev_sma_50 = calculate_sma(all_prices[:-1], SMA_TREND_MEDIUM)
    if prev_sma_50 is not None and prev_ema_20 > prev_sma_50 and ema_20 <= sma_50:
        return "SELL"

    # --- BUY LOGIC ---
    # Priority 1: Sentiment Veto
    # Do not enter new long positions if news is catastrophically bad.
    if net_sentiment_score < -7.0:
        return "HOLD" # Vetoes BUY, but doesn't force a SELL on its own.

    # Priority 2: Primary Trend Entry Signal (Dual Confirmation)
    is_primary_uptrend = current_price > sma_50 and ema_20 > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and macd_hist_delta > 0
    is_not_overbought = rsi < 75 # A simple check to avoid buying at the absolute top.

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"