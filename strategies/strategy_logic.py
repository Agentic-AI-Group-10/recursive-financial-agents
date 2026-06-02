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
        # Initialize the first EMA with a simple moving average
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

    # Initial average gain and loss over the first 'period'
    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    avg_gain = seed_gains / period
    avg_loss = seed_losses / period

    # Wilder's smoothing for subsequent periods
    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta >= 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0 # If no losses, RSI is 100. If no gains and no losses, it's neutral.
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period:
        return None, None, None
    
    # Ensure enough data for EMA calculations
    if len(prices) < long_period + short_period - 1: # Minimum for both EMAs to start
        return None, None, None

    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)

    # MACD line is the difference between the two EMAs.
    # Align the series by taking the latest common length.
    min_len = min(len(short_ema_series), len(long_ema_series))
    macd_line = short_ema_series[-min_len:] - long_ema_series[-min_len:]

    if len(macd_line) < signal_period:
        return macd_line, np.array([]), np.array([]) # Not enough data for signal line yet

    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Histogram is MACD line minus Signal line. Align again.
    min_len_hist = min(len(macd_line), len(signal_line))
    histogram = macd_line[-min_len_hist:] - signal_line[-min_len_hist:]
    
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1: # Need at least period + 1 prices to calculate first TR
        return None
    
    # True Range (TR) calculation
    true_ranges = []
    for i in range(1, len(prices)):
        high_low = abs(prices[i] - prices[i-1]) # Simplified: using close-to-close for volatility
        true_ranges.append(high_low)
    
    # ATR is the EMA of True Ranges
    atr_series = calculate_ema_series(true_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    # ROC = ((Current Price - Price 'period' periods ago) / Price 'period' periods ago) * 100
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version builds upon the successful V2 by addressing the primary limitation of low trade count
    and aiming for improved performance in normal market conditions, while preserving its strength
    in crisis regimes.

    Key Enhancements:
    1.  **Relaxed Sentiment for Capitulation Buy:** The highest-conviction "buy the dip" signal
        (extreme ROC decline, low RSI, momentum turning positive) no longer requires a positive
        sentiment score. During true capitulation, news is inherently negative, and this change
        ensures the strategy can enter at absolute bottoms without being blocked by overwhelming
        negative sentiment.
    2.  **New "Pullback Buy" in Uptrends:** Introduces a more frequent buy signal for normal markets.
        It identifies pullbacks (RSI < 45) within an established uptrend (above SMA_50) when
        momentum (MACD histogram) shows signs of turning upwards, aiming to capitalize on healthy
        dips.
    3.  **More Sensitive Profit-Taking:** The "SELL" condition for overbought markets is adjusted
        to trigger at a lower RSI threshold (RSI > 70, from 82) when momentum is fading. This
        allows for quicker profit realization and potentially increases trade frequency by
        exiting overextended positions sooner.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5, # Ambiguous: good for economy, bad for inflation/rates
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
    ATR_SHORT = 10
    ATR_LONG = 50
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20

    # Ensure sufficient history for all indicators
    # MACD requires long_period + signal_period - 1 for histogram to be meaningful
    # RSI needs period + 1
    # ATR needs period + 1
    # ROC needs period + 1
    # SMA needs period
    required_history_length = max(SMA_TREND_LONG, ATR_LONG + 1, ROC_CRASH_PERIOD + 1, 26 + 9 - 1 + 1) # 26+9-1 for MACD histogram, +1 for current price
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    # macd_hist_series needs at least 2 elements for macd_hist_delta
    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_20, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram # Change in MACD histogram (momentum velocity)

    # --- 3. Regime Detection ---
    # Crisis Regime: General high-risk environment
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    # Capitulation Regime: An extreme subset of crisis, signaling a potential bottom
    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CONTRARIAN CAPITULATION (HIGHEST PRIORITY)
    # Buy when there is blood in the streets, but only if momentum shows signs of turning.
    # Removed sentiment filter here to allow entry during extreme negative news.
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    # If in a general crisis (but not a specific capitulation buy signal), be defensive.
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD" # Hold cash and wait for the storm to pass.

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic Stop-Loss. Tighter 7% drop from 20-day high.
    if current_price < (donchian_high_20 * 0.93):
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0 # MACD histogram crosses zero downwards
    is_sentiment_permissive_for_sell = net_sentiment_score < 3.0
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_overbought_for_profit_taking = rsi > 70 # Adjusted from 82 to 70 for more sensitive profit-taking
    if is_overbought_for_profit_taking and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0 # MACD histogram crosses zero upwards
    is_not_overbought_for_entry = rsi < 78 # Still conservative for initial entry
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.0
    is_sufficient_volatility = short_atr > (long_atr * 0.6) # Avoids entering dead, sideways markets.

    # Existing Trend-Following Buy
    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought_for_entry and is_sentiment_permissive_for_buy and is_sufficient_volatility:
        return "BUY"

    # NEW: Pullback Buy in an Uptrend (to increase trade frequency in normal markets)
    is_pullback_oversold = rsi < 45 # RSI indicates a significant pullback, but not extreme oversold like capitulation
    is_momentum_recovering_from_dip = macd_hist_delta > 0 # Momentum turning positive after a dip
    if is_primary_uptrend and is_pullback_oversold and is_momentum_recovering_from_dip and is_sentiment_permissive_for_buy:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"