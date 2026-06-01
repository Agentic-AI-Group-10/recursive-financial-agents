import numpy as np
import pandas as pd
import re

# --- Helper Functions for Technical Indicators ---

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    return pd.Series(data).ewm(span=period, adjust=False).mean().to_numpy()

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return None
    
    delta = pd.Series(prices).diff(1)
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
        return None
    
    prices_series = pd.Series(prices)
    short_ema = prices_series.ewm(span=short_period, adjust=False).mean()
    long_ema = prices_series.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line.iloc[-1] - signal_line.iloc[-1]
    return histogram

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    
    prices_series = pd.Series(prices)
    middle_band = prices_series.rolling(window=period).mean().iloc[-1]
    std_dev = prices_series.rolling(window=period).std().iloc[-1]
    
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return middle_band, upper_band, lower_band

def calculate_adx(prices, period=14):
    """Calculates the Average Directional Index (ADX)."""
    if len(prices) < 2 * period:
        return None

    df = pd.DataFrame({'close': prices})
    df['high'] = df['close'] # Using close as proxy for H/L/C
    df['low'] = df['close']

    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()

    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    df['+di'] = 100 * (df['+dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr'])
    df['-di'] = 100 * (df['-dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr'])
    
    df['dx'] = 100 * (abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']))
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
    
    return df['adx'].iloc[-1]

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy using ADX for trend analysis
    and a faster, dedicated exit logic to reduce drawdowns.

    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.

    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0, "quantitative easing": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "ai boom": 2.0, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5, "strong jobs": 2.0,
        "consumer confidence": 1.5,
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "hot inflation": -2.5, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.0, "market turmoil": -2.0, "credit crunch": -2.5, "vix spike": -2.0,
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
    MEDIUM_SMA_PERIOD = 50
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    BB_PERIOD = 20
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + 9, VOL_LONG_PERIOD + 1, 2 * ADX_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    medium_sma = calculate_sma(all_prices, MEDIUM_SMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_histogram = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    adx = calculate_adx(all_prices, ADX_PERIOD)

    if any(v is None for v in [short_ema, medium_sma, long_ema, rsi, macd_histogram, upper_band, adx]):
        return "HOLD"

    # --- 3. DEDICATED EXIT LOGIC (to reduce drawdowns) ---
    # This is a faster signal to sell an existing long position, preventing large drawdowns.
    # It's more sensitive than a full bearish entry signal.
    if short_ema < medium_sma:
        return "SELL"

    # --- 4. Regime Detection & Entry Logic ---
    log_returns = np.log(np.array(all_prices[1:]) / np.array(all_prices[:-1]))
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)
    
    is_trending_market = adx > 25

    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following only ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        
        bullish_trend = short_ema > long_ema and current_price > medium_sma
        bearish_trend = short_ema < long_ema
        
        if bullish_trend and macd_histogram > 0 and rsi < 65 and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
            return "BUY"
        elif bearish_trend and macd_histogram < 0 and rsi > 35 and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion via ADX) ===
        if is_trending_market:
            # Sub-Regime: Normal Trending Market
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            
            bullish_trend = short_ema > long_ema and macd_histogram > 0
            bearish_trend = short_ema < long_ema and macd_histogram < 0

            if bullish_trend and rsi < 70 and net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
                return "BUY"
            elif bearish_trend and rsi > 30 and net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD:
                return "SELL"
        else:
            # Sub-Regime: Ranging Market (Mean-Reversion)
            # Safer mean-reversion: only buy dips if the longer-term trend is still up.
            if current_price > medium_sma:
                if rsi < 30 and current_price < lower_band and net_sentiment_score > -1.5:
                    return "BUY"
            
            # Sell rips if overbought, regardless of medium-term trend.
            if rsi > 70 and current_price > upper_band and net_sentiment_score < 1.5:
                return "SELL"

    return "HOLD"