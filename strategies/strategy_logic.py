import numpy as np
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
    data_arr = np.array(data, dtype=float)
    ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
    ema_values[0] = np.mean(data_arr[:period])
    multiplier = 2 / (period + 1)
    for i in range(1, len(ema_values)):
        ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]
    return ema_values

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    ema_s = calculate_ema_series(prices, period)
    return ema_s[-1] if len(ema_s) > 0 else None

def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI) using Wilder's smoothing method.
    """
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD histogram for the latest price."""
    if len(prices) < long_period + signal_period:
        return None

    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line) < signal_period:
        return None
        
    signal_line_series = calculate_ema_series(macd_line, signal_period)
    
    if len(signal_line_series) == 0:
        return None
        
    histogram = macd_line[-1] - signal_line_series[-1]
    return histogram

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    """Calculates the Bollinger Bands for the latest price."""
    if len(prices) < period:
        return None, None, None
    
    prices_slice = prices[-period:]
    middle_band = np.mean(prices_slice)
    std_dev = np.std(prices_slice)
    
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return middle_band, upper_band, lower_band

def decide(current_price, price_history, news_context):
    """
    SELF-IMPROVED VERSION
    This strategy addresses the parent's passivity and slow exits by introducing a master
    risk-off signal and relaxing entry criteria in normal trending markets.

    Improvements:
    1.  Master Risk-Off Signal: A price cross below the 50-day SMA triggers a "SELL" to
        aggressively cut losses and prevent large drawdowns, a key failure of the parent.
    2.  Less Passive Trend Following: In normal trending markets, the sentiment filter is
        changed from a strict positive requirement to a veto on only strongly negative news.
        This allows the strategy to participate in more technical trends.
    3.  Preservation of Strengths: The successful volatility-based regime switching and
        multi-factor confirmation logic are retained.
    """
    # --- 1. Sentiment Analysis (Unchanged - Robust) ---
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

    # --- 2. Technical Indicators & Regime Detection ---
    all_prices = price_history + [current_price]
    
    # Define periods
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    BB_PERIOD = 20
    RISK_SMA_PERIOD = 50  # Key period for risk management
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + MACD_SIGNAL_PERIOD, VOL_LONG_PERIOD + 1, RISK_SMA_PERIOD)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # Calculate core indicators
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    macd_histogram = calculate_macd(all_prices, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, MACD_SIGNAL_PERIOD)
    _, upper_band, lower_band = calculate_bollinger_bands(all_prices, BB_PERIOD)
    risk_sma = calculate_sma(all_prices, RISK_SMA_PERIOD)

    if any(v is None for v in [short_ema, long_ema, rsi, macd_histogram, upper_band, risk_sma]):
        return "HOLD"

    # --- 3. MASTER RISK-OFF SIGNAL (NEW) ---
    # This rule provides a fast exit to prevent large drawdowns, a key weakness of the parent.
    # If the price closes below the 50-day SMA, exit any long position.
    if current_price < risk_sma:
        return "SELL"

    # --- 4. Multi-Regime Entry Logic ---
    # Adaptive Volatility Regime
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    is_high_volatility = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    if is_high_volatility:
        # === CRISIS MODE: High-conviction trend-following (Unchanged - Proven effective) ===
        BULLISH_SENTIMENT_THRESHOLD = 2.5
        BEARISH_SENTIMENT_THRESHOLD = -2.5
        RSI_OVERBOUGHT_CEILING = 65
        RSI_OVERSOLD_FLOOR = 35

        bullish_trend = short_ema > long_ema
        bearish_trend = short_ema < long_ema
        
        if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_trend and macd_histogram > 0 and rsi < RSI_OVERBOUGHT_CEILING:
            return "BUY"
        # The master risk-off signal handles exits. This SELL is for initiating a short in a crisis.
        elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_trend and macd_histogram < 0 and rsi > RSI_OVERSOLD_FLOOR:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive (Trend-Following or Mean-Reversion) ===
        trend_strength = abs(short_ema - long_ema) / long_ema
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market (IMPROVED - Less Passive)
            SENTIMENT_VETO_THRESHOLD = -1.5  # Veto buy only on strongly negative news
            RSI_OVERBOUGHT = 70
            
            bullish_trend = short_ema > long_ema
            
            # Entry is now primarily technical, with sentiment as a safety veto.
            # This addresses the parent's weakness of being faked out by minor negative news.
            if bullish_trend and macd_histogram > 0 and rsi < RSI_OVERBOUGHT and net_sentiment_score > SENTIMENT_VETO_THRESHOLD:
                return "BUY"
        else:
            # Sub-Regime: Choppy / Ranging Market (Mean-Reversion - Unchanged)
            # This logic is sound and is now protected by the master risk-off signal.
            MEAN_REVERSION_RSI_OVERSOLD = 30
            MEAN_REVERSION_RSI_OVERBOUGHT = 70
            
            # Buy the dip only if confirmed by RSI/Bollinger and not in a major downtrend.
            if (rsi < MEAN_REVERSION_RSI_OVERSOLD and current_price < lower_band) and \
               (net_sentiment_score > -2.0): # Ensure news isn't catastrophic
                return "BUY"
            # Sell the rip. The master risk-off signal provides the primary downside protection.
            elif (rsi > MEAN_REVERSION_RSI_OVERBOUGHT and current_price > upper_band) and \
                 (net_sentiment_score < 2.0):
                return "SELL"

    return "HOLD"