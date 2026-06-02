import numpy as np
import re

# --- Helper Functions for Technical Indicators (Unchanged from robust parent) ---

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

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period + signal_period:
        return None, None, None

    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    # Align series by slicing the longer one
    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line) < signal_period:
        return None, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    # Align histogram to the signal line
    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    
    return macd_line, signal_line, histogram

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
    SELF-IMPROVED VERSION 2.0
    This strategy enhances the successful parent model by introducing more robust
    confirmation signals and special logic for extreme market conditions.
    
    Improvements:
    1.  Refactored into a `market_state` dictionary for clarity.
    2.  Enhanced mean-reversion logic with RSI momentum confirmation.
    3.  Added "V-shape recovery" and "Blow-off top" signals for high-volatility regimes.
    4.  Updated sentiment keyword library for contemporary relevance.
    """
    # --- 1. Data and Parameter Setup ---
    all_prices = price_history + [current_price]
    
    # Indicator Parameters
    SHORT_EMA_PERIOD = 12
    LONG_EMA_PERIOD = 26
    RSI_PERIOD = 14
    BB_PERIOD = 20
    VOL_SHORT_PERIOD = 20
    VOL_LONG_PERIOD = 100

    required_history_length = max(LONG_EMA_PERIOD + 9, VOL_LONG_PERIOD + 1, RSI_PERIOD + 2)
    if len(all_prices) < required_history_length:
        return "HOLD"

    # --- 2. Sentiment Analysis (Updated Keywords) ---
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "stimulus": 2.0, "soft landing": 2.5,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "surge": 2.0,
        "strong earnings": 2.0, "cooling inflation": 2.0, "inflation eases": 2.0,
        "ai boom": 2.5, "technological breakthrough": 2.0, "easing tensions": 1.5,
        "beat expectations": 1.5, "growth accelerates": 2.0, "recovery": 1.5, "upgrade": 1.5,
        "strong jobs": 2.0, "consumer confidence": 1.5, "market rally": 2.0, "geopolitical stability": 2.0,
        "rate hike": -2.5, "recession": -3.0, "crisis": -3.0, "bankruptcy": -3.0,
        "hard landing": -2.5, "stagflation": -3.0, "hawkish": -2.0, "bearish": -2.0,
        "plunge": -2.0, "inflation spike": -2.5, "sell-off": -2.0, "weak earnings": -2.0,
        "geopolitical risk": -2.5, "war": -3.5, "conflict": -3.0, "sanctions": -2.5,
        "market turmoil": -2.0, "credit crunch": -3.0, "contagion": -3.5, "defaults": -3.0,
        "tightening": -1.5, "miss expectations": -1.5, "downgrade": -1.5, "tariff": -2.0,
        "supply chain disruption": -2.0, "uncertainty": -1.5, "weak jobs": -2.5, "cyberattack": -3.0
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 30):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight

    # --- 3. Technical Indicator Calculation & State Assembly ---
    market_state = {}
    
    # Core Indicators
    market_state['short_ema'] = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    market_state['long_ema'] = calculate_ema(all_prices, LONG_EMA_PERIOD)
    market_state['rsi'] = calculate_rsi(all_prices, RSI_PERIOD)
    market_state['prev_rsi'] = calculate_rsi(all_prices[:-1], RSI_PERIOD) # For momentum check
    _, market_state['upper_band'], market_state['lower_band'] = calculate_bollinger_bands(all_prices, BB_PERIOD)
    _, _, macd_hist_series = calculate_macd_series(all_prices)

    if any(v is None for v in market_state.values()) or macd_hist_series is None or len(macd_hist_series) < 3:
        return "HOLD"
    
    market_state['macd_hist'] = macd_hist_series[-1]
    market_state['prev_macd_hist'] = macd_hist_series[-2]

    # Regime Detection
    log_returns = np.log(np.array(all_prices)[1:] / np.array(all_prices)[:-1])
    short_term_vol = np.std(log_returns[-VOL_SHORT_PERIOD:])
    long_term_vol = np.std(log_returns[-VOL_LONG_PERIOD:])
    market_state['is_high_volatility'] = (short_term_vol > long_term_vol * 1.5) and (short_term_vol > 0.015)

    # --- 4. Multi-Regime Decision Logic ---
    if market_state['is_high_volatility']:
        # === CRISIS MODE: High-conviction trend-following and extreme reversal plays ===
        
        # **NEW**: V-Shape Recovery Signal (Catching panic bottoms)
        if market_state['prev_rsi'] < 20 and market_state['rsi'] > market_state['prev_rsi'] and \
           market_state['macd_hist'] > market_state['prev_macd_hist']:
            return "BUY"
            
        # **NEW**: Blow-off Top Signal (Selling into euphoria peaks)
        if market_state['prev_rsi'] > 80 and market_state['rsi'] < market_state['prev_rsi'] and \
           market_state['macd_hist'] < market_state['prev_macd_hist']:
            return "SELL"

        # Standard Crisis Trend-Following (High sentiment confirmation)
        bullish_trend = market_state['short_ema'] > market_state['long_ema']
        bearish_trend = market_state['short_ema'] < market_state['long_ema']
        
        if net_sentiment_score >= 3.0 and bullish_trend and market_state['macd_hist'] > 0 and market_state['rsi'] < 70:
            return "BUY"
        elif net_sentiment_score <= -3.0 and bearish_trend and market_state['macd_hist'] < 0 and market_state['rsi'] > 30:
            return "SELL"
    else:
        # === NORMAL MODE: Adaptive trend-following and mean-reversion ===
        trend_strength = abs(market_state['short_ema'] - market_state['long_ema']) / market_state['long_ema']
        is_choppy_market = trend_strength < 0.005

        if not is_choppy_market:
            # Sub-Regime: Normal Trending Market
            bullish_trend = market_state['short_ema'] > market_state['long_ema']
            bearish_trend = market_state['short_ema'] < market_state['long_ema']
            
            # Proactive profit-taking / trend exhaustion signal
            is_momentum_fading_up = market_state['macd_hist'] > 0 and market_state['macd_hist'] < market_state['prev_macd_hist']
            if bullish_trend and market_state['rsi'] > 78 and is_momentum_fading_up:
                return "SELL"

            # Entry with 2-day momentum confirmation
            if bullish_trend and market_state['macd_hist'] > 0 and market_state['prev_macd_hist'] > 0 and market_state['rsi'] < 75 and net_sentiment_score > -1.5:
                return "BUY"
            
            # Exit with 2-day momentum confirmation
            if bearish_trend and market_state['macd_hist'] < 0 and market_state['prev_macd_hist'] < 0 and market_state['rsi'] > 25 and net_sentiment_score < 1.5:
                return "SELL"
        else:
            # Sub-Regime: Choppy / Ranging Market (IMPROVED Mean-Reversion Logic)
            
            # Buy the dip with stronger confirmation (RSI must also be reversing)
            is_reversing_up = market_state['macd_hist'] > market_state['prev_macd_hist']
            is_rsi_reversing_up = market_state['rsi'] > market_state['prev_rsi']
            if (market_state['rsi'] < 30 and current_price < market_state['lower_band']) and \
               (net_sentiment_score > -2.5) and is_reversing_up and is_rsi_reversing_up:
                return "BUY"
                
            # Sell the rip with stronger confirmation (RSI must also be reversing)
            is_reversing_down = market_state['macd_hist'] < market_state['prev_macd_hist']
            is_rsi_reversing_down = market_state['rsi'] < market_state['prev_rsi']
            if (market_state['rsi'] > 70 and current_price > market_state['upper_band']) and \
               (net_sentiment_score < 2.5) and is_reversing_down and is_rsi_reversing_down:
                return "SELL"

    return "HOLD"