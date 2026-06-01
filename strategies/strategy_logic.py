import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    # The following is a more efficient way to calculate the latest EMA value
    # without computing the entire series, but the full calculation is more robust
    # for demonstration.
    ema_values = np.zeros_like(prices_arr, dtype=float)
    ema_values[period - 1] = np.mean(prices_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values[-1]

def calculate_rsi(prices, period):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    deltas = np.diff(prices_arr)
    if len(deltas) < period:
        return None
    
    # Use the last 'period' deltas for calculation
    relevant_deltas = deltas[-period:]
    gains = np.where(relevant_deltas > 0, relevant_deltas, 0)
    losses = np.where(relevant_deltas < 0, -relevant_deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, period, num_std_dev):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    prices_slice = prices[-period:]
    middle_band = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, middle_band, lower_band

def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy that retains the successful "Crisis Mode" logic
    while implementing a mean-reverting strategy for "Normal Mode" to address
    whipsaws and improve trade timing.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Enhanced Keywords ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # Bullish
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "positive outlook": 1.5,
        # Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "hawkish": -2.0, "tightening": -1.5,
        "bearish": -2.0, "miss": -1.5, "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0,
        "downgrade": -1.5, "tariff": -1.5, "weak earnings": -2.0, "bankruptcy": -2.5,
        "negative outlook": -1.5, "contraction": -1.5
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Volatility Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    LONG_SMA_PERIOD = 50
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20
    BBAND_PERIOD = 20
    BBAND_STD_DEV = 2

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOLATILITY_PERIOD + 1, LONG_SMA_PERIOD, BBAND_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate volatility to determine market regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    volatility = np.std(log_returns[-VOLATILITY_PERIOD:])
    is_high_volatility = volatility > 0.02  # Threshold for high-volatility regime

    # --- 3. Adaptive Decision Logic based on Regime ---
    if is_high_volatility:
        # --- CRISIS MODE (Proven Success) ---
        # Be more selective. Use stricter thresholds and confluence of signals.
        BULLISH_SENTIMENT_THRESHOLD = 1.5
        BEARISH_SENTIMENT_THRESHOLD = -1.5
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        # Calculate indicators needed for Crisis Mode
        short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
        long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
        rsi = calculate_rsi(all_prices, RSI_PERIOD)
        if short_ema is None or long_ema is None or rsi is None:
            return "HOLD"

        # Define trend and momentum signals
        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD

        # BUY: Strong positive sentiment AND a bullish trend AND not overbought
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and is_not_overbought:
            return "BUY"
        
        # SELL: Strong negative sentiment AND a bearish trend AND not oversold
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and is_not_oversold:
            return "SELL"
    
    else:
        # --- NORMAL MODE (Self-Improved Mean-Reversion Strategy) ---
        # Goal: Avoid whipsaws from slow crossovers. Buy dips in uptrends, sell rallies in downtrends.
        
        # Define thresholds for Normal Mode
        SENTIMENT_CONFIRMATION_BULLISH = 1.0
        SENTIMENT_CONFIRMATION_BEARISH = -1.0

        # Calculate indicators needed for Normal Mode
        long_sma = calculate_sma(all_prices, LONG_SMA_PERIOD)
        upper_bband, _, lower_bband = calculate_bollinger_bands(all_prices, BBAND_PERIOD, BBAND_STD_DEV)
        if long_sma is None or upper_bband is None:
            return "HOLD"

        # Determine primary trend using the long-term SMA
        is_primary_uptrend = current_price > long_sma

        # BUY Signal: In a primary uptrend, buy when price pulls back to the lower Bollinger Band.
        # Sentiment acts as a soft filter to avoid buying into very negative news.
        if is_primary_uptrend and current_price <= lower_bband and net_sentiment_score > SENTIMENT_CONFIRMATION_BEARISH:
            return "BUY"
        
        # SELL Signal: In a primary downtrend, sell when price rallies to the upper Bollinger Band.
        # Sentiment acts as a soft filter to avoid selling into very positive news.
        elif not is_primary_uptrend and current_price >= upper_bband and net_sentiment_score < SENTIMENT_CONFIRMATION_BULLISH:
            return "SELL"
        
    # Default to HOLD if no high-conviction signal is found in either regime
    return "HOLD"