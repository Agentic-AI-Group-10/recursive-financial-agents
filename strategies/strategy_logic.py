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
    
    # Ensure both EMA series are long enough and aligned
    if len(short_ema_series) == 0 or len(long_ema_series) == 0:
        return None, None, None
        
    # Align the series by taking the latest common length
    min_len = min(len(short_ema_series), len(long_ema_series))
    macd_line = short_ema_series[-min_len:] - long_ema_series[-min_len:]
    
    if len(macd_line) < signal_period:
        return macd_line, None, None
    
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    if len(signal_line) == 0:
        return macd_line, None, None

    # Align histogram calculation
    min_len_hist = min(len(macd_line), len(signal_line))
    histogram = macd_line[-min_len_hist:] - signal_line[-min_len_hist:]
    return macd_line, signal_line, histogram

def calculate_atr(prices, period=14):
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    # True Range (TR) = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
    # Since we only have closing prices, we approximate TR as abs(CurrentClose - PrevClose)
    # This is a simplification and not a true ATR, but consistent with available data.
    price_ranges = np.abs(np.diff(prices_arr))
    if len(price_ranges) == 0:
        return None
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_std_dev(prices, period):
    """Calculates the Standard Deviation for the latest period."""
    if len(prices) < period:
        return None
    return np.std(prices[-period:])

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    if len(prices) < period:
        return None, None, None
    sma = calculate_sma(prices, period)
    std_dev = calculate_std_dev(prices, period)
    if sma is None or std_dev is None:
        return None, None, None
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version addresses the low trade count and aims to improve the win rate
    in normal market conditions by integrating Bollinger Bands and refining
    entry/exit logic.

    Key Enhancements:
    1.  Bollinger Bands Integration: Introduced Bollinger Bands to identify
        mean-reversion opportunities within trends, allowing for more frequent
        and potentially higher-conviction trades on pullbacks (buy near lower band)
        and profit-taking (sell near upper band).
    2.  Refined Normal Regime Logic: Buy signals now combine trend, momentum,
        and either a MACD crossover or a bounce from the lower Bollinger Band.
        Sell signals are enhanced with Bollinger Band breaches and fading momentum
        for quicker profit-taking and trend reversal detection.
    3.  Adjusted RSI Thresholds: Slightly relaxed RSI thresholds for normal
        market conditions (e.g., RSI < 70 for not overbought) to allow more
        trading opportunities without entering extreme overbought/oversold states.
    4.  Maintained Robustness: The successful Contrarian Capitulation, Crisis Aversion,
        and dynamic ATR-based stop-loss mechanisms are preserved due to their
        proven effectiveness in managing risk and capturing extreme market turns.
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
    BB_PERIOD = 20
    BB_STD_DEV = 2

    required_history_length = max(SMA_TREND_LONG + 1, ATR_LONG + 1, ROC_CRASH_PERIOD + 1, BB_PERIOD + 1, 50)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(all_prices, BB_PERIOD, BB_STD_DEV)

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_20, donchian_high_20, bb_upper, bb_middle, bb_lower]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

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
    if is_capitulation_candidate and macd_hist_delta > 0 and net_sentiment_score > -2.0: # Added sentiment filter for higher conviction
        return "BUY"

    # REGIME 2: CRISIS AVERSION
    # If in a general crisis (but not a specific capitulation buy signal), be defensive.
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50 or net_sentiment_score < -1.0: # Added sentiment to crisis sell
            return "SELL"
        return "HOLD" # Hold cash and wait for the storm to pass.

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic Stop-Loss. Tighter 7% drop from 20-day high.
    if current_price < (donchian_high_20 * 0.93):
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 3.0
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with FADING momentum or Bollinger Band breach.
    is_overbought_rsi = rsi > 75 # Adjusted from 82
    is_momentum_fading = macd_hist_delta < 0
    is_above_upper_bb = current_price > bb_upper

    if (is_overbought_rsi and is_momentum_fading) or (is_above_upper_bb and is_momentum_fading):
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 70 # Adjusted from 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.0
    is_sufficient_volatility = short_atr > (long_atr * 0.6) # Avoids entering dead, sideways markets.
    is_below_lower_bb = current_price < bb_lower

    # New Buy Condition: Trend continuation or Bollinger Band bounce
    if is_primary_uptrend and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility:
        # Option 1: Standard momentum confirmation (MACD crossover)
        if is_momentum_confirming_up:
            return "BUY"
        # Option 2: Bollinger Band bounce within an uptrend (buy the dip)
        # Price is below lower band, but momentum is turning up from oversold RSI
        if is_below_lower_bb and rsi < 35 and macd_hist_delta > 0:
            return "BUY"

    # Default action is to hold the current position.
    return "HOLD"