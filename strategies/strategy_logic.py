import numpy as np
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    prices_arr = np.array(prices, dtype=float)
    # This is an iterative calculation, so we compute the full series to get the last value
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
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use Wilder's smoothing method for RSI
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        rs = np.inf
    else:
        rs = avg_gain / avg_loss
        
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd_crossover(prices, short_period=12, long_period=26, signal_period=9):
    """
    Calculates the last two values of the MACD and Signal lines to detect a crossover.
    Returns (prev_macd, current_macd, prev_signal, current_signal).
    """
    required_len = long_period + signal_period
    if len(prices) < required_len:
        return None, None, None, None

    prices_arr = np.array(prices, dtype=float)
    
    # Calculate full EMA series
    def _get_ema_series(data, period):
        ema_series = np.full_like(data, np.nan)
        ema_series[period - 1] = np.mean(data[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema_series[i] = (data[i] - ema_series[i-1]) * multiplier + ema_series[i-1]
        return ema_series

    ema_short_series = _get_ema_series(prices_arr, short_period)
    ema_long_series = _get_ema_series(prices_arr, long_period)
    
    macd_line = ema_short_series - ema_long_series
    
    # Calculate signal line from valid MACD values
    macd_line_valid = macd_line[long_period-1:]
    if len(macd_line_valid) < signal_period:
        return None, None, None, None
        
    signal_line_valid = _get_ema_series(macd_line_valid, signal_period)
    
    # Get the last two points for crossover detection
    current_macd = macd_line[-1]
    prev_macd = macd_line[-2]
    current_signal = signal_line_valid[-1]
    prev_signal = signal_line_valid[-2]
    
    return prev_macd, current_macd, prev_signal, current_signal

def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy with a dynamic volatility regime switch.
    - Crisis Mode (High Volatility): Uses a high-conviction model requiring confluence
      of strong sentiment and EMA trend, a proven successful strategy.
    - Normal Mode (Low Volatility): Uses a more responsive MACD crossover strategy to
      capture trends more effectively, addressing the lag of the previous model.
    """
    # --- 1. Sentiment Analysis (Unchanged) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "rate cut": 2.5, "stimulus": 2.0, "dovish": 2.0, "easing": 1.5, "record high": 2.0,
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "strong earnings": 2.0,
        "recovery": 1.5, "upgrade": 1.5, "expansion": 1.5, "positive outlook": 1.5,
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
    RSI_PERIOD = 14
    VOLATILITY_PERIOD = 20
    MACD_SHORT = 12
    MACD_LONG = 26
    MACD_SIGNAL = 9

    # Ensure enough data for volatility calculation
    if len(all_prices) < VOLATILITY_PERIOD + 1:
        return "HOLD"

    # Calculate volatility to determine market regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    volatility = np.std(log_returns[-VOLATILITY_PERIOD:])
    is_high_volatility = volatility > 0.02  # Threshold for high-volatility regime

    # --- 3. Adaptive Decision Logic based on Regime ---
    if is_high_volatility:
        # --- CRISIS MODE ---
        # This mode was successful. Logic is preserved. It requires strong confluence
        # of sentiment and a clear trend to make high-conviction trades.
        required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1)
        if len(all_prices) < required_history_length:
            return "HOLD"

        short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
        long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
        rsi = calculate_rsi(all_prices, RSI_PERIOD)

        if short_ema is None or long_ema is None or rsi is None:
            return "HOLD"

        BULLISH_SENTIMENT_THRESHOLD = 1.5
        BEARISH_SENTIMENT_THRESHOLD = -1.5
        RSI_OVERBOUGHT = 65
        RSI_OVERSOLD = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD

        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and is_not_overbought:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and is_not_oversold:
            return "SELL"
        
    else:
        # --- NORMAL MODE ---
        # This mode is overhauled to use a more responsive MACD crossover, addressing
        # the lag and drawdown issues of the previous EMA-based model.
        required_history_length = MACD_LONG + MACD_SIGNAL
        if len(all_prices) < required_history_length:
            return "HOLD"
        
        rsi = calculate_rsi(all_prices, RSI_PERIOD)
        prev_macd, current_macd, prev_signal, current_signal = calculate_macd_crossover(
            all_prices, MACD_SHORT, MACD_LONG, MACD_SIGNAL
        )

        if rsi is None or current_macd is None:
            return "HOLD"

        RSI_OVERBOUGHT = 70
        RSI_OVERSOLD = 30
        
        # Define signals based on MACD crossover
        bullish_crossover = prev_macd < prev_signal and current_macd > current_signal
        bearish_crossover = prev_macd > prev_signal and current_macd < current_signal
        
        # RSI filter to avoid buying tops and selling bottoms
        is_not_overbought = rsi < RSI_OVERBOUGHT
        is_not_oversold = rsi > RSI_OVERSOLD
        
        # Softer sentiment check: act on technicals unless news is strongly contradictory
        has_positive_sentiment_bias = net_sentiment_score >= 0
        has_negative_sentiment_bias = net_sentiment_score <= 0

        # BUY: Bullish MACD crossover, not overbought, and no strong negative news
        if bullish_crossover and is_not_overbought and has_positive_sentiment_bias:
            return "BUY"
        
        # SELL: Bearish MACD crossover, not oversold, and no strong positive news
        elif bearish_crossover and is_not_oversold and has_negative_sentiment_bias:
            return "SELL"

    # Default to HOLD if no high-conviction signal is found in either regime
    return "HOLD"