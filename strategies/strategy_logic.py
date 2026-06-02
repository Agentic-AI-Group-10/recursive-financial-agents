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
        return ema_values

def calculate_tema(prices, period):
    """Calculates the Triple Exponential Moving Average (TEMA)."""
    if len(prices) < 3 * period - 2: # TEMA requires significant history
        return None
    ema1 = calculate_ema_series(prices, period)
    ema2 = calculate_ema_series(ema1, period)
    ema3 = calculate_ema_series(ema2, period)
    tema = 3 * ema1[-1] - 3 * ema2[-1] + ema3[-1]
    return tema

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
    # Ensure we get the full series for subsequent calculations
    full_short_ema = calculate_ema_series(prices, short_period)
    full_long_ema = calculate_ema_series(prices, long_period)
    
    # Align the series by taking the tail of the shorter EMA series
    macd_line = full_short_ema[len(full_short_ema)-len(full_long_ema):] - full_long_ema
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    # Calculate signal line on the MACD line itself
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align histogram calculation
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    # Use EMA series calculation for ATR smoothing
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
    This version enhances responsiveness and risk management with three key upgrades:
    1.  Upgraded Trend Indicator: Replaces lagging SMAs with a more responsive Triple
        Exponential Moving Average (TEMA) for primary trend definition, enabling
        quicker reaction to market shifts.
    2.  Adaptive Risk Management: Implements a dynamic ATR-based trailing stop-loss,
        which adjusts the stop level based on market volatility, providing more
        intelligent risk control than a fixed percentage.
    3.  Enhanced Signal Logic: The primary BUY signal is refined to trigger on
        pullbacks within a confirmed TEMA-defined uptrend, aiming to improve entry
        quality and reduce whipsaws from simple momentum crossovers.
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
    TEMA_TREND_PERIOD = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = 2.5
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    required_history_length = 3 * TEMA_TREND_PERIOD # TEMA requires the most data
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    tema_50 = calculate_tema(all_prices, TEMA_TREND_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [tema_50, rsi, atr, roc_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_uptrend = current_price > tema_50
    is_downtrend = not is_uptrend
    
    # Crisis Regime: High-velocity drop, deep oversold conditions
    is_crash_velocity = roc_20 < -15.0
    is_deeply_oversold = rsi < 25
    is_crisis_regime = is_crash_velocity and is_downtrend

    # Capitulation Candidate: An extreme subset of crisis, signaling a potential bottom
    is_capitulation_candidate = is_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    # Buy when there is blood in the streets, but only if momentum shows signs of turning.
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    # If in a general crisis (but not a specific capitulation buy signal), be defensive.
    if is_crisis_regime:
        return "SELL" # Exit all positions during a confirmed high-velocity crash

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: ATR Trailing Stop-Loss.
    trailing_stop_level = donchian_high_20 - (atr * ATR_STOP_MULTIPLIER)
    if current_price < trailing_stop_level:
        return "SELL"

    # Priority 2: Trend breakdown signal. Price crosses below the TEMA.
    if is_downtrend:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    # Buy on pullbacks in a confirmed uptrend.
    is_momentum_positive = macd_histogram > 0
    is_not_overbought = rsi < 75
    is_sentiment_permissive = net_sentiment_score > -3.0

    if is_uptrend and is_momentum_positive and is_not_overbought and is_sentiment_permissive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"