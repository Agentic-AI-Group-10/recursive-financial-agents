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

def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    """Calculates Bollinger Bands and Bandwidth."""
    if len(prices) < period:
        return None, None, None, None
    prices_arr = np.array(prices, dtype=float)
    try:
        import pandas as pd
        series = pd.Series(prices_arr)
        middle_band_series = series.rolling(window=period).mean()
        std_series = series.rolling(window=period).std()
        upper_band_series = middle_band_series + (std_series * std_dev)
        lower_band_series = middle_band_series - (std_series * std_dev)
        bbw_series = ((upper_band_series - lower_band_series) / middle_band_series).to_numpy()
        return upper_band_series.iloc[-1], middle_band_series.iloc[-1], lower_band_series.iloc[-1], bbw_series
    except ImportError:
        # Fallback without pandas, cannot calculate historical BBW series
        sma = np.mean(prices_arr[-period:])
        std = np.std(prices_arr[-period:])
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower, None

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED STRATEGY V3:
    This version enhances the successful V2 strategy with three key architectural upgrades:
    1.  Bollinger Band Squeeze Breakout: Replaces the generic volatility filter with a
        high-probability entry signal. BUY signals are now triggered on MACD crossovers
        that coincide with a price breakout above the upper Bollinger Band immediately
        following a period of low-volatility consolidation (a "squeeze"). This
        significantly reduces whipsaw trades in sideways markets.
    2.  Adaptive ATR Stop-Loss: The fixed percentage stop-loss is replaced with a dynamic
        stop based on a multiple of the Average True Range (ATR). This allows the stop
        to be tighter in low-volatility markets and wider during volatile periods,
        improving risk management and preventing premature exits.
    3.  Capped Sentiment Score: The influence of the news sentiment score is now capped
        at a maximum and minimum value. This prevents extreme, potentially noisy,
        headlines from single-handedly overriding the technical model, making the
        strategy more robust against sentiment overreactions.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "productivity growth": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.5,
        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,
        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,
        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,
        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,
        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "supply chain disruption": -2.5, "uncertainty": -1.5,
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
    
    # Capped sentiment to prevent extreme influence
    net_sentiment_score = np.clip(net_sentiment_score, -5.0, 5.0)

    # --- 2. Technical Indicators & State Calculation ---
    all_prices = price_history + [current_price]

    # Indicator Periods
    SMA_TREND_LONG = 100
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_SQUEEZE_LOOKBACK = 50
    STOP_LOSS_LOOKBACK = 20
    ATR_STOP_MULTIPLIER = 3.0

    required_history_length = max(SMA_TREND_LONG + 1, BB_PERIOD + BB_SQUEEZE_LOOKBACK)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    sma_100 = calculate_sma(all_prices, SMA_TREND_LONG)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, ATR_PERIOD)
    upper_bb, middle_bb, lower_bb, bbw_series = calculate_bollinger_bands(all_prices, BB_PERIOD)
    donchian_high_20 = np.max(all_prices[-STOP_LOSS_LOOKBACK:]) if len(all_prices) >= STOP_LOSS_LOOKBACK else None

    # Null check for all indicators
    if any(v is None for v in [sma_100, rsi, atr, upper_bb, middle_bb, donchian_high_20]) or macd_hist_series is None or len(macd_hist_series) < 2 or bbw_series is None or len(bbw_series) < BB_SQUEEZE_LOOKBACK:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    # --- 3. Regime Detection ---
    is_long_term_downtrend = current_price < sma_100
    is_crisis_regime = is_long_term_downtrend and current_price < middle_bb

    # --- 4. Decision Logic (Hierarchical) ---

    # REGIME 1: CRISIS AVERSION (HIGHEST PRIORITY)
    # If in a long-term downtrend, be extremely defensive.
    if is_crisis_regime:
        if macd_histogram < 0:
            return "SELL"
        return "HOLD" # Hold cash and wait for the storm to pass.

    # REGIME 2: NORMAL MARKET CONDITIONS

    # --- SELL LOGIC (Risk Management First) ---
    # Priority 1: Adaptive ATR Stop-Loss.
    stop_price = donchian_high_20 - (ATR_STOP_MULTIPLIER * atr)
    if current_price < stop_price:
        return "SELL"

    # Priority 2: Standard trend breakdown signal.
    is_trend_breakdown = current_price < middle_bb
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    if is_trend_breakdown and is_momentum_confirming_down:
        return "SELL"

    # Priority 3: Profit-taking on extreme overbought conditions with FADING momentum.
    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 80
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    # --- BUY LOGIC ---
    # Condition 1: Trend must be positive (above 20-period moving average)
    is_uptrend = current_price > middle_bb
    
    # Condition 2: Momentum must be turning positive (MACD cross)
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    
    # Condition 3: Volatility Squeeze setup
    # Check if current bandwidth is the lowest in the lookback period (the "squeeze")
    is_squeeze = bbw_series[-2] < np.min(bbw_series[-BB_SQUEEZE_LOOKBACK:-1])
    # Check if price is now breaking out of the upper band
    is_breakout = current_price > upper_bb
    
    # Condition 4: Not already extremely overbought
    is_not_overbought = rsi < 75
    
    # Condition 5: Sentiment is not catastrophically bad
    is_sentiment_permissive = net_sentiment_score > -4.0

    # High-probability BUY signal: A breakout from a volatility squeeze
    if is_uptrend and is_momentum_confirming_up and is_squeeze and is_breakout and is_not_overbought and is_sentiment_permissive:
        return "BUY"

    # Default action is to hold the current position.
    return "HOLD"