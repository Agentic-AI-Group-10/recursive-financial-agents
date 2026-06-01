import numpy as np
import pandas as pd
import re

# --- Helper Functions for Technical Indicators ---

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for a series of prices."""
    if len(prices) < period:
        return None
    # Using pandas for a robust, standard EMA calculation
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1:
        return None
    
    series = pd.Series(prices)
    delta = series.diff(1)
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates MACD line and Signal line."""
    if len(prices) < long_period:
        return None, None
    
    prices_series = pd.Series(prices)
    ema_short = prices_series.ewm(span=short_period, adjust=False).mean()
    ema_long = prices_series.ewm(span=long_period, adjust=False).mean()
    
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_adx(prices, high_prices, low_prices, period=14):
    """Calculates the Average Directional Index (ADX)."""
    # Note: This ADX calculation requires high and low prices.
    # We will approximate them from closing prices for this implementation.
    if len(prices) < period * 2:
        return None

    df = pd.DataFrame({'close': prices, 'high': high_prices, 'low': low_prices})
    
    # Calculate True Range (TR)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()

    # Calculate Directional Movement (+DM, -DM)
    df['up_move'] = df['high'].diff()
    df['down_move'] = df['low'].diff() * -1
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    df['+di'] = 100 * (df['+dm'].ewm(span=period, adjust=False).mean() / df['atr'])
    df['-di'] = 100 * (df['-dm'].ewm(span=period, adjust=False).mean() / df['atr'])

    # Calculate ADX
    df['dx'] = 100 * (abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']))
    adx = df['dx'].ewm(span=period, adjust=False).mean().iloc[-1]
    
    return adx

def decide(current_price, price_history, news_context):
    """
    A self-improved, multi-regime trading strategy that uses ADX for trend strength
    and MACD for trend confirmation, in addition to volatility-based regime switching.
    """
    # --- 1. Sentiment Analysis ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        # High-Impact Bullish
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.0, "ai boom": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 1.5, "disinflation": 1.5,
        "beat expectations": 1.5, "growth": 1.5, "recovery": 1.5, "upgrade": 1.5,
        # High-Impact Bearish
        "rate hike": -2.5, "recession": -2.5, "crisis": -2.5, "bankruptcy": -2.5,
        "hard landing": -2.5, "stagflation": -2.5, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "hot inflation": -2.0, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical tension": -1.5, "miss expectations": -1.5, "downgrade": -1.5, "tariff": -1.5
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
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    # Ensure enough data for all indicators
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1, VOL_LONG_PERIOD + 1, ADX_PERIOD * 2)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Approximate high/low for ADX calculation from close prices (a simplification)
    # A more advanced version would pass OHLC data.
    all_highs = [p * 1.01 for p in all_prices] # Approximation
    all_lows = [p * 0.99 for p in all_prices]  # Approximation

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_line, signal_line = calculate_macd(all_prices)
    adx = calculate_adx(all_prices, all_highs, all_lows, ADX_PERIOD)

    # Safeguard against None values
    if any(v is None for v in [short_ema, long_ema, rsi, macd_line, signal_line, adx]):
        return "HOLD"

    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 3. Multi-Regime Decision Logic ---
    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following with MACD confirmation ===
        BULLISH_SENTIMENT_THRESHOLD = 2.0
        BEARISH_SENTIMENT_THRESHOLD = -2.0
        
        is_bullish_trend = short_ema > long_ema and macd_line > signal_line
        is_bearish_trend = short_ema < long_ema and macd_line < signal_line
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_bullish_trend and rsi < 70:
            return "BUY"
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and is_bearish_trend and rsi > 30:
            return "SELL"
    else:
        # === NORMAL MODE: ADX-based adaptive strategy ===
        is_trending_market = adx > 25
        is_choppy_market = adx < 20

        if is_trending_market:
            # Sub-Regime: Normal Trending Market with MACD confirmation
            BULLISH_SENTIMENT_THRESHOLD = 1.0
            BEARISH_SENTIMENT_THRESHOLD = -1.0
            
            is_bullish_trend = short_ema > long_ema and macd_line > signal_line
            is_bearish_trend = short_ema < long_ema and macd_line < signal_line

            if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and is_bullish_trend and rsi < 75:
                return "BUY"
            elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and is_bearish_trend and rsi > 25:
                return "SELL"
        elif is_choppy_market:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion Logic)
            MEAN_REVERSION_RSI_OVERSOLD = 25
            MEAN_REVERSION_RSI_OVERBOUGHT = 75
            
            if rsi < MEAN_REVERSION_RSI_OVERSOLD and net_sentiment_score > -1.5:
                return "BUY"
            elif rsi > MEAN_REVERSION_RSI_OVERBOUGHT and net_sentiment_score < 1.5:
                return "SELL"

    return "HOLD"