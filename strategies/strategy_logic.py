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
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
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
    """Calculates Average True Range (ATR) using close-to-close volatility."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    """Calculates the Rate of Change (ROC) over a given period."""
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_bollinger_bandwidth_series(prices, period=20, num_std_dev=2):
    """Calculates a series of Bollinger Bandwidths."""
    if len(prices) < period:
        return None
    try:
        import pandas as pd
        prices_series = pd.Series(prices)
        rolling_sma = prices_series.rolling(window=period).mean()
        rolling_std = prices_series.rolling(window=period).std()
        upper_band = rolling_sma + (rolling_std * num_std_dev)
        lower_band = rolling_sma - (rolling_std * num_std_dev)
        # Avoid division by zero in the bandwidth calculation
        safe_rolling_sma = rolling_sma.replace(0, 1e-9)
        bandwidth_series = ((upper_band - lower_band) / safe_rolling_sma).to_numpy()
        return bandwidth_series[period-1:]
    except ImportError:
        return None # Pandas is required for this robust calculation

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances the V2 strategy with three major upgrades for robustness:
    1.  Volatility Squeeze Detection: Implements a Bollinger Band Squeeze filter.
        In low-volatility, choppy markets, the system will HOLD, preventing
        whipsaw trades and waiting for a decisive breakout.
    2.  Dynamic ATR-Based Stop-Loss: Replaces the fixed percentage stop-loss with
        a Chandelier Exit (ATR-based), which adapts the risk level to current
        market volatility for more intelligent capital preservation.
    3.  Sentiment-Modulated Thresholds: RSI overbought/oversold levels are now
        dynamically adjusted based on the news sentiment score, making the system
        more attuned to market psychology and less prone to premature signals.
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
        "uncertainty": -1.5, "jobless claims rise": -1.5, "supply chain disruption": -2.0,
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
    ATR_STOP = 20
    ROC_CRASH_PERIOD = 20
    STOP_LOSS_LOOKBACK = 20
    BB_SQUEEZE_LOOKBACK = 60

    required_history_length = max(SMA_TREND_LONG + 1, BB_SQUEEZE_LOOKBACK + 1, 75)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    sma_50 = calculate_sma(all_prices, SMA_TREND_MEDIUM)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, ATR_SHORT)
    long_atr = calculate_atr(all_prices, ATR_LONG)
    atr_for_stop = calculate_atr(all_prices, ATR_STOP)
    roc_20 = calculate_roc(all_prices, ROC_CRASH_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:])
    bandwidth_series = calculate_bollinger_bandwidth_series(all_prices, period=20)

    # Null check for all indicators
    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, atr_for_stop, roc_20, donchian_high_20, bandwidth_series]) or macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection (Hierarchical) ---

    # REGIME 1: VOLATILITY SQUEEZE (HIGHEST PRIORITY FILTER)
    # If volatility is extremely low, stay out to avoid whipsaws.
    current_bandwidth = bandwidth_series[-1]
    historical_low_bandwidth = np.percentile(bandwidth_series[-BB_SQUEEZE_LOOKBACK:], 10)
    if current_bandwidth < historical_low_bandwidth:
        return "HOLD"

    # REGIME 2: CRISIS & CAPITULATION
    is_long_term_downtrend = current_price < sma_100
    is_high_volatility = short_atr > (long_atr * 1.75)
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    # Dynamic RSI threshold for capitulation, more sensitive in negative sentiment
    capitulation_rsi_threshold = 28 + net_sentiment_score
    is_deeply_oversold = rsi < capitulation_rsi_threshold
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    # --- 4. Decision Logic (Hierarchical) ---

    # CONTRARIAN CAPITULATION BUY
    if is_capitulation_candidate and macd_hist_delta > 0:
        return "BUY"

    # CRISIS AVERSION
    if is_crisis_regime:
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        return "HOLD"

    # REGIME 3: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Dynamic ATR Stop-Loss (Chandelier Exit).
    chandelier_exit = donchian_high_20 - (3 * atr_for_stop)
    if current_price < chandelier_exit:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_primary_downtrend and is_momentum_confirming_down and net_sentiment_score < 3.0:
        return "SELL"

    # Priority 3: Profit-taking on overbought conditions with FADING momentum.
    # Dynamic RSI threshold, holds longer in positive sentiment.
    overbought_rsi_threshold = 80 + net_sentiment_score
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > overbought_rsi_threshold
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    # Dynamic RSI threshold, more permissive in positive sentiment.
    buy_rsi_ceiling = 78 + net_sentiment_score
    is_not_overbought = rsi < buy_rsi_ceiling
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.5
    is_sufficient_volatility = short_atr > (long_atr * 0.6)

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"