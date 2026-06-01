import numpy as np
import re
import pandas as pd

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
    # Using pandas for a robust and standard EMA calculation.
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return None
    
    series = pd.Series(prices)
    delta = series.diff(1)
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD histogram for the latest price."""
    if len(prices) < long_period + signal_period:
        return None, None

    series = pd.Series(prices)
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    histogram = macd_line.iloc[-1] - signal_line.iloc[-1]
    prev_histogram = macd_line.iloc[-2] - signal_line.iloc[-2] if len(macd_line) > 1 else 0
    
    return histogram, prev_histogram

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    
    series = pd.Series(prices)
    middle_band = series.rolling(window=period).mean().iloc[-1]
    std_dev = series.rolling(window=period).std().iloc[-1]
    
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return middle_band, upper_band, lower_band

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy with a dedicated, more
    responsive exit logic to address passivity and reduce drawdowns.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis with Expanded Keywords ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.0, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.0, "market turmoil": -2.0, "credit crunch": -2.5,
        "tightening": -1.5, "miss": -1.5, "downgrade": -1.5, "tariff": -1.5,
        "supply chain disruption": -1.5, "uncertainty": -1.5, "weak jobs": -2.0
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 2. Technical Indicators & Adaptive Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    BB_PERIOD = 20
    EXIT_SMA_PERIOD = 20 # Faster SMA for exit logic
    MEDIUM_TERM_SMA_PERIOD = 50
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD, VOL_LONG_PERIOD + 1)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_histogram, _ = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, MACD_SIGNAL_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    exit_sma = calculate_sma(all_prices, EXIT_SMA_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, macd_histogram, upper_band, exit_sma]):
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic with Dedicated Exits ---
    buy_signal = False
    sell_signal = False

    # === Universal Exit Logic (Trend Failure) ===
    # This logic is checked regardless of regime to cut losses or take profits quickly.
    # It's more sensitive than the entry logic.
    exit_long_signal = current_price < exit_sma and macd_histogram < 0
    # (Exit short signal is a BUY, handled in the buy logic section)
    exit_short_signal = current_price > exit_sma and macd_histogram > 0

    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        RSI_OVERBOUGHT_CEILING = 65
        RSI_OVERSOLD_FLOOR = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        # High-conviction entry signals
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and macd_histogram > 0 and rsi < RSI_OVERBOUGHT_CEILING:
            buy_signal = True
        if net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and macd_histogram < 0 and rsi > RSI_OVERSOLD_FLOOR:
            sell_signal = True
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            
            bullish_trend = short_ema > long_ema
            bearish_trend = short_ema < long_ema

            # High-conviction entry signals
            if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and macd_histogram > 0 and rsi < 70:
                buy_signal = True
            if net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and macd_histogram < 0 and rsi > 30:
                sell_signal = True
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion)
            medium_sma = calculate_sma(all_prices, MEDIUM_TERM_SMA_PERIOD)
            if medium_sma is not None:
                # Buy the dip only if confirmed and medium-term trend is intact.
                if (rsi < 30 and current_price < lower_band) and \
                   (net_sentiment_score > -2.0) and (current_price > medium_sma):
                    buy_signal = True
                # Sell the rip only if confirmed.
                elif (rsi > 70 and current_price > upper_band) and \
                     (net_sentiment_score < 2.0):
                    sell_signal = True

    # --- 4. Final Decision ---
    # A BUY signal can be a new entry or an exit from a short position.
    if buy_signal or exit_short_signal:
        return "BUY"
    
    # A SELL signal can be a new short entry or an exit from a long position.
    if sell_signal or exit_long_signal:
        return "SELL"

    return "HOLD"