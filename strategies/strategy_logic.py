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
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        # Fallback pure-python EMA calculation
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
    """
    Calculates Average True Range (ATR) using close-to-close volatility.
    Note: A simplification as true ATR requires High and Low prices.
    """
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version introduces a more robust, regime-aware framework.
    1.  Macro Regime Filter: A 200-day SMA is used to classify the market into
        a 'Macro Bull' or 'Macro Bear' state. The strategy becomes more
        aggressive in bull markets and more defensive in bear markets,
        requiring higher conviction for buy signals during downturns.
    2.  Dynamic ATR Stop-Loss: The fixed percentage stop-loss is replaced with
        a dynamic stop based on Average True Range (ATR), making it adaptive
        to market volatility for better capital preservation.
    3.  Consolidated Scoring System: Decision logic is refactored into a
        scoring model that weighs multiple factors (trend, momentum, sentiment,
        RSI) to generate a final signal, reducing whipsaws and false entries.
    """
    # --- 1. Parameters & Configuration ---
    all_prices = price_history + [current_price]
    
    # Indicator Periods
    SMA_MACRO = 200
    SMA_MEDIUM = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    STOP_LOSS_LOOKBACK = 20
    ATR_MULTIPLIER = 3.0

    required_history_length = SMA_MACRO + 1
    if len(all_prices) < required_history_length:
        return "HOLD"

    # --- 2. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "soft landing": 2.5, "cooling inflation": 2.5,
        "ai boom": 2.5, "stimulus": 2.0, "dovish": 2.0, "record high": 2.0, "bullish": 2.0,
        "strong earnings": 2.0, "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "vix spike": -2.5,
        "hawkish": -2.0, "bearish": -2.0, "sell-off": -2.0, "uncertainty": -1.5,
        "euphoria": -2.5, "mania": -3.0, "extreme greed": -2.5,
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]
    net_sentiment_score = 0.0
    panic_score = 0
    for keyword, weight in sentiment_keywords.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', context_lower):
            net_sentiment_score += weight
            if weight <= -3.5: # High-impact negative keywords
                panic_score += 1
    
    is_panic_sentiment = panic_score >= 2 # Trigger if 2 or more severe keywords appear

    # --- 3. Technical Indicator Calculation ---
    sma_200 = calculate_sma(all_prices, SMA_MACRO)
    sma_50 = calculate_sma(all_prices, SMA_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    donchian_high = np.max(all_prices[-STOP_LOSS_LOOKBACK:])

    # Null check for all indicators
    if any(v is None for v in [sma_200, sma_50, rsi, atr, donchian_high]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 4. State & Regime Detection ---
    is_macro_bull = current_price > sma_200
    is_macro_bear = not is_macro_bull
    
    is_capitulation_candidate = rsi < 25 and (current_price < sma_50 * 0.90) # Deeply oversold and far from mean

    # --- 5. Decision Logic (Hierarchical) ---

    # 5.1: HIGH-PRIORITY OVERRIDES (RISK MANAGEMENT & SPECIAL CASES)
    
    # OVERRIDE 1: DYNAMIC ATR STOP-LOSS
    stop_loss_price = donchian_high - (ATR_MULTIPLIER * atr)
    if current_price < stop_loss_price:
        return "SELL"

    # OVERRIDE 2: CONTRARIAN CAPITULATION BUY
    if is_capitulation_candidate and macd_hist_delta > 0: # Momentum must be turning
        # Be more cautious buying dips in a macro bear market
        if is_macro_bull or (is_macro_bear and rsi < 20): # Require extreme oversold in bear market
            return "BUY"

    # OVERRIDE 3: PANIC SENTIMENT SELL
    if is_panic_sentiment:
        return "SELL"

    # 5.2: SCORING SYSTEM FOR NORMAL CONDITIONS
    buy_score = 0
    sell_score = 0

    # Trend Analysis
    if current_price > sma_50: buy_score += 2
    else: sell_score += 2
    if is_macro_bull: buy_score += 2
    else: sell_score += 2
    
    # Momentum Analysis
    if macd_histogram > 0 and prev_macd_histogram <= 0: buy_score += 3  # Bullish Crossover
    if macd_histogram < 0 and prev_macd_histogram >= 0: sell_score += 3 # Bearish Crossover
    if macd_hist_delta > 0: buy_score += 1 # Accelerating Momentum
    if macd_hist_delta < 0: sell_score += 1 # Decelerating Momentum

    # Oscillator Analysis (RSI)
    if rsi > 75: sell_score += 2
    if rsi > 82: sell_score += 1 # Extra weight for extreme overbought
    if rsi < 35: buy_score += 1

    # Sentiment Analysis
    if net_sentiment_score > 1.5: buy_score += 2
    if net_sentiment_score < -1.5: sell_score += 2

    # 5.3: FINAL DECISION BASED ON SCORES
    buy_threshold = 5
    sell_threshold = 5

    # Adjust thresholds based on macro regime
    if is_macro_bear:
        buy_threshold = 7  # Higher conviction needed to buy in a bear market
        sell_threshold = 4 # Lower conviction needed to sell
    
    if buy_score >= buy_threshold and sell_score < 4:
        return "BUY"
    
    if sell_score >= sell_threshold:
        return "SELL"

    return "HOLD"