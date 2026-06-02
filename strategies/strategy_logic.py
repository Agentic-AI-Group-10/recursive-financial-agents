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
    short_ema_full = calculate_ema_series(prices, short_period)
    long_ema_full = calculate_ema_series(prices, long_period)
    macd_line = short_ema_full[long_period-short_period:] - long_ema_full
    if len(macd_line) < signal_period:
        return macd_line, None, None
    signal_line_full = calculate_ema_series(macd_line, signal_period)
    signal_line = signal_line_full[signal_period-1:]
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

def calculate_kama(prices, period=10, fast_ema_period=2, slow_ema_period=30):
    """Calculates Kaufman's Adaptive Moving Average (KAMA)."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    change = np.abs(prices_arr[period:] - prices_arr[:-period])
    volatility = np.array([np.sum(np.abs(np.diff(prices_arr[i:i+period]))) for i in range(len(prices_arr) - period)])
    
    # Avoid division by zero
    volatility[volatility == 0] = 1e-10
    
    efficiency_ratio = change / volatility
    
    fast_sc = 2 / (fast_ema_period + 1)
    slow_sc = 2 / (slow_ema_period + 1)
    
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros_like(prices_arr)
    kama[period-1] = np.mean(prices_arr[:period])
    
    for i in range(period, len(prices_arr)):
        sc_index = i - period
        if sc_index < len(smoothing_constant):
            kama[i] = kama[i-1] + smoothing_constant[sc_index] * (prices_arr[i] - kama[i-1])
        else: # Should not happen with correct indexing, but as a safeguard
            kama[i] = kama[i-1]
            
    return kama[-1]

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a sophisticated adaptive framework and enhanced risk management:
    1.  Adaptive Trend Filtering: Replaces static SMAs with Kaufman's Adaptive Moving
        Average (KAMA), which dynamically adjusts to market volatility. This reduces
        whipsaws in choppy markets and responds faster to new trends.
    2.  Volatility-Based Risk Management: The fixed percentage stop-loss is replaced
        by a dynamic ATR (Average True Range) trailing stop. This gives trades more
        room to breathe during high volatility and tightens stops in quiet periods.
    3.  Proactive Pullback Entries: Instead of waiting for lagging MACD crossovers,
        the strategy now identifies pullbacks within an established KAMA-defined uptrend,
        allowing for earlier entries with better risk/reward profiles.
    4.  Enhanced Sentiment Conviction: A dedicated "Fear & Greed" score is calculated
        from extreme sentiment keywords to act as a powerful confirmation filter for
        contrarian and trend-following signals.
    """
    # --- 1. Configuration & Parameters ---
    KAMA_PERIOD = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = 2.5
    STOP_LOSS_LOOKBACK = 25
    ROC_CRASH_PERIOD = 20
    
    # --- 2. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "soft landing": 2.5, "cooling inflation": 2.5,
        "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0, "short squeeze": 3.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "black swan": -4.0, "systemic risk": -4.0,
        "contagion": -3.5, "credit crunch": -3.5, "rate hike": -2.5, "bankruptcy": -2.5,
        "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0, "bubble": -2.0, "uncertainty": -1.5,
    }
    fear_greed_keywords = {
        "capitulation": 5.0, "panic selling": 4.0, "extreme fear": 3.0, # Buy signals
        "euphoria": -5.0, "mania": -4.0, "irrational exuberance": -4.0, "extreme greed": -3.0 # Sell signals
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    
    net_sentiment_score = 0.0
    fear_greed_score = 0.0

    def score_keywords(text, keywords):
        score = 0.0
        for keyword, weight in keywords.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, text):
                pre_context = text[max(0, match.start() - 30):match.start()]
                is_negated = any(neg_word in pre_context for neg_word in negation_words)
                score += -weight if is_negated else weight
        return score

    net_sentiment_score = score_keywords(context_lower, sentiment_keywords)
    fear_greed_score = score_keywords(context_lower, fear_greed_keywords)

    # --- 3. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]
    required_history_length = max(KAMA_PERIOD + 2, RSI_PERIOD + 2, ATR_PERIOD + 2, ROC_CRASH_PERIOD + 2, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    kama = calculate_kama(all_prices, period=KAMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [kama, rsi, atr, donchian_high]) or macd_hist_series is None or len(macd_hist_series) < 3:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 4. Regime Detection ---
    is_long_term_uptrend = current_price > kama
    is_long_term_downtrend = not is_long_term_uptrend
    
    # Capitulation Regime: An extreme event signaling a potential bottom
    roc_crash = ((all_prices[-1] - all_prices[-1 - ROC_CRASH_PERIOD]) / all_prices[-1 - ROC_CRASH_PERIOD]) * 100
    is_deeply_oversold = rsi < 28
    is_extreme_crash = roc_crash is not None and roc_crash < -15.0
    is_capitulation_candidate = is_extreme_crash and is_deeply_oversold

    # --- 5. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    # Buy when there is blood in the streets, but only if momentum shows signs of turning.
    if is_capitulation_candidate and macd_hist_delta > 0 and fear_greed_score > 2.0:
        return "BUY"

    # REGIME 2: DOWNTREND / CRISIS AVERSION
    if is_long_term_downtrend:
        return "SELL" # Be in cash during major KAMA-defined downtrends.

    # REGIME 3: UPTREND / NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR Trailing Stop-Loss.
    atr_stop_price = donchian_high - (ATR_STOP_MULTIPLIER * atr)
    if current_price < atr_stop_price:
        return "SELL"

    # Priority 2: Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading and fear_greed_score < -2.0:
        return "SELL"

    # --- BUY LOGIC (Pullback in Uptrend) ---
    # Look for a dip (RSI pullback) within a confirmed uptrend.
    is_pullback = rsi < 65 and np.max(all_prices[-10:-1]) > current_price # Simple pullback check
    is_momentum_recovering = macd_hist_delta > 0
    is_not_overbought = rsi < 75
    is_sentiment_permissive = net_sentiment_score > -3.0

    if is_long_term_uptrend and is_pullback and is_momentum_recovering and is_not_overbought and is_sentiment_permissive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"