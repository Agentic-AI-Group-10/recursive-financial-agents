import numpy as np
import re
import math
import pandas as pd

# --- Helper Functions for Technical Indicators ---

def calculate_ema_series(data, period):
    """Calculates a full series of Exponential Moving Averages."""
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    try:
        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]
    except ImportError:
        ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)
        if len(data_arr) >= period:
            ema_values[0] = np.mean(data_arr[:period])
            multiplier = 2 / (period + 1)
            for i in range(1, len(ema_values)):
                ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values

def calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):
    """Calculates the MACD line, signal line, and histogram series."""
    if len(prices) < long_period: 
        return None, None, None
    
    short_ema_series = calculate_ema_series(prices, short_period)
    long_ema_series = calculate_ema_series(prices, long_period)
    
    if len(short_ema_series) == 0 or len(long_ema_series) == 0:
        return None, None, None

    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series
    
    if len(macd_line) < signal_period: 
        return macd_line, None, None
        
    signal_line = calculate_ema_series(macd_line, signal_period)
    
    if len(signal_line) == 0:
        return macd_line, None, None

    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line
    
    return macd_line, signal_line, histogram

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
        return 100.0 # No losses, RSI is 100
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

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
    if prices[-1 - period] == 0: 
        return 0.0
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculates Bollinger Bands."""
    if len(prices) < period:
        return None, None
    sma = calculate_sma(prices, period)
    std = np.std(prices[-period:])
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    return upper_band, lower_band

def calculate_stochastic_oscillator(prices, period=14):
    """Calculates the Stochastic Oscillator."""
    if len(prices) < period:
        return None
    lowest_low = np.min(prices[-period:])
    highest_high = np.max(prices[-period:])
    return ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100

def calculate_force_index(prices, volume):
    """Calculates the Force Index (FI)."""
    if len(prices) < 2 or len(volume) < 2:
        return None
    return np.sum(np.diff(prices) * np.diff(volume)) / len(volume)

def calculate_sma(prices, period):
    """Calculates the Simple Moving Average (SMA) for the latest price."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def decide(current_price, price_history, news_context):
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,
        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,
        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,
        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,
        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,
        "strong jobs report": 0.0, 
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

    all_prices = price_history + [current_price]
    all_volumes = [0.0] * len(price_history) + [0.0]  # Replace with actual volume data

    # Dynamic scaling of indicators based on price history length
    sma_trend_long = max(100, len(all_prices) // 2)
    sma_trend_medium = max(50, len(all_prices) // 4)
    rsi_period = 14
    atr_short = 10
    atr_long = 50
    roc_crash_period = 20
    stop_loss_lookback = 30 
    fi_period = 20

    required_history_length = max(sma_trend_long + 1, atr_long + 1, roc_crash_period + 1, rsi_period + 1, 
                                  26 + 9 + 1, stop_loss_lookback + 1, fi_period + 1) 
    if len(all_prices) < required_history_length:
        return "HOLD"

    sma_100 = calculate_sma(all_prices, sma_trend_long)
    sma_50 = calculate_sma(all_prices, sma_trend_medium)
    rsi = calculate_rsi(all_prices, rsi_period)
    macd_line, signal_line, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, atr_short)
    long_atr = calculate_atr(all_prices, atr_long)
    roc_20 = calculate_roc(all_prices, roc_crash_period)
    donchian_high_30 = np.max(all_prices[-stop_loss_lookback:]) if len(all_prices) >= stop_loss_lookback else None
    upper_band, lower_band = calculate_bollinger_bands(all_prices)
    stochastic_oscillator = calculate_stochastic_oscillator(all_prices)
    fi = calculate_force_index(all_prices, all_volumes)

    if any(v is None for v in [sma_100, sma_50, rsi, short_atr, long_atr, roc_20, donchian_high_30, upper_band, lower_band, stochastic_oscillator, fi]) or \
       macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    is_high_volatility = short_atr > (long_atr * 1.75)
    is_extreme_volatility = short_atr > (long_atr * 2.0) 

    is_long_term_downtrend = current_price < sma_100
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    is_deeply_oversold = rsi < 30 
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    if is_capitulation_candidate and macd_hist_delta > 0 and stochastic_oscillator < 20: 
        return "BUY"

    if is_crisis_regime:
        is_recovering_from_oversold = rsi > 35 and macd_hist_delta > 0 and stochastic_oscillator > 80
        
        if is_recovering_from_oversold and net_sentiment_score > -1.0: 
            return "BUY"
        
        if macd_histogram < 0 or current_price < sma_50:
            return "SELL"
        
        return "HOLD"

    base_stop_loss_factor = 0.88 
    if is_extreme_volatility: 
        stop_loss_factor = 0.80 
    elif is_high_volatility:
        stop_loss_factor = 0.85 
    else:
        stop_loss_factor = base_stop_loss_factor

    if current_price < (donchian_high_30 * stop_loss_factor): 
        return "SELL"

    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = net_sentiment_score < 3.0
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 82
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = net_sentiment_score > -3.0
    is_sufficient_volatility = short_atr > (long_atr * 0.6) 
    is_price_in_bollinger_band = current_price > lower_band and current_price < upper_band

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility and is_price_in_bollinger_band and stochastic_oscillator > 20:
        return "BUY"

    return "HOLD"