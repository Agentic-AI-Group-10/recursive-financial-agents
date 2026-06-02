import numpy as np
import re
import pandas as pd

def calculate_ema_series(data, period):
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

def calculate_atr(prices, period=14):
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    price_ranges = np.abs(np.diff(prices_arr))
    atr_series = calculate_ema_series(price_ranges, period)
    return atr_series[-1] if len(atr_series) > 0 else None

def calculate_roc(prices, period=20):
    if len(prices) < period + 1:
        return None
    if prices[-1 - period] == 0: 
        return 0.0
    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100

def calculate_donchian_channel(prices, period=20):
    if len(prices) < period:
        return None, None
    donchian_high = np.max(prices[-period:])
    donchian_low = np.min(prices[-period:])
    upper_band = donchian_high
    lower_band = donchian_low
    return upper_band, lower_band

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    if len(prices) < period:
        return None, None
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    return upper_band, lower_band

def calculate_stochastic_oscillator(prices, period=14):
    if len(prices) < period:
        return None
    lowest_low = np.min(prices[-period:])
    highest_high = np.max(prices[-period:])
    return ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100

def calculate_force_index(prices, volume):
    if len(prices) < 2 or len(volume) < 2:
        return None
    return np.sum(np.diff(prices) * np.diff(volume)) / len(volume)

def calculate_keltner_channel(prices, period=20):
    if len(prices) < period:
        return None, None
    sma = np.mean(prices[-period:])
    atr = calculate_atr(prices, period)
    upper_band = sma + (atr * 2)
    lower_band = sma - (atr * 2)
    return upper_band, lower_band

def calculate_adx(prices, period=14):
    if len(prices) < period + 1:
        return None
    prices_arr = np.array(prices, dtype=float)
    high = prices_arr[-period-1:]
    low = prices_arr[-period-1:]
    close = prices_arr[-period:]
    delta_high = np.diff(high)
    delta_low = np.diff(low)
    delta_close = np.diff(close)
    tr = np.maximum(delta_high, delta_low)
    tr = np.maximum(tr, np.abs(delta_close))
    tr_smoothed = calculate_ema_series(tr, period)
    up_move = high[1:] - np.roll(high, 1)[1:]
    down_move = np.roll(low, 1)[1:] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm_smoothed = calculate_ema_series(plus_dm, period)
    minus_dm_smoothed = calculate_ema_series(minus_dm, period)
    di_plus = 100 * (plus_dm_smoothed / tr_smoothed)
    di_minus = 100 * (minus_dm_smoothed / tr_smoothed)
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = calculate_ema_series(dx, period)
    return adx[-1] if len(adx) > 0 else None

def calculate_sentiment_score(news_context):
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.5, "rate cut": 3.0, "quantitative easing": 2.8, "soft landing": 2.7,
        "cooling inflation": 2.6, "cpi miss": 2.5, "ai boom": 2.4, "stimulus": 2.2,
        "dovish": 2.1, "record high": 2.0, "bullish": 1.9, "strong earnings": 1.8,
        "beat estimates": 1.6, "recovery": 1.5, "de-escalation": 1.7,
        "short squeeze": 3.8, "capitulation": 3.2, "panic selling": 2.7, "extreme fear": 2.1,
        "strong jobs report": 0.0, 
        "recession": -3.5, "crisis": -3.2, "stagflation": -3.1, "hot inflation": -3.0,
        "war": -3.3, "yield curve inversion": -3.7, "quantitative tightening": -2.7,
        "black swan": -4.2, "systemic risk": -4.1, "contagion": -3.6, "credit crunch": -3.5,
        "rate hike": -2.8, "bankruptcy": -2.6, "hard landing": -2.7, "geopolitical risk": -2.6,
        "cpi beat": -2.7, "vix spike": -2.6, "hawkish": -2.2, "bearish": -2.1,
        "sell-off": -2.1, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,
        "uncertainty": -1.6,
        "euphoria": -2.7, "mania": -3.2, "irrational exuberance": -3.1, "extreme greed": -2.6,
        "market rebound": 2.7, "rebound potential": 2.2, "safe haven": 1.6,
        "economic recovery": 2.1, "bull market": 2.0, "bear market": -2.1,
        "inflation concerns": -1.2, "deflation risk": -2.1
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "unlikely", "avoid", "no signs of", "unlikely to"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 50):match.start()]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            net_sentiment_score += -weight if is_negated else weight
    return net_sentiment_score

def decide(current_price, price_history, news_context):
    context_lower = news_context.lower()
    sentiment_score = calculate_sentiment_score(news_context)
    all_prices = price_history + [current_price]
    all_volumes = [0.0] * len(price_history) + [0.0]

    if len(all_prices) < 50:
        return "HOLD"

    sma_50 = np.mean(all_prices[-50:])
    sma_200 = np.mean(all_prices[-200:])
    ema_12 = calculate_ema_series(all_prices, 12)[-1] if len(all_prices) >= 12 else None
    ema_26 = calculate_ema_series(all_prices, 26)[-1] if len(all_prices) >= 26 else None
    ema_9 = calculate_ema_series(all_prices, 9)[-1] if len(all_prices) >= 9 else None
    rsi = calculate_rsi(all_prices, 14)
    macd_line, signal_line, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, 10)
    long_atr = calculate_atr(all_prices, 50)
    roc_20 = calculate_roc(all_prices, 20)
    donchian_high_30, donchian_low_30 = calculate_donchian_channel(all_prices, 30)
    upper_band, lower_band = calculate_bollinger_bands(all_prices)
    stochastic_oscillator = calculate_stochastic_oscillator(all_prices)
    fi = calculate_force_index(all_prices, all_volumes)
    keltner_upper_band, keltner_lower_band = calculate_keltner_channel(all_prices)
    adx = calculate_adx(all_prices)

    if any(v is None for v in [sma_50, sma_200, ema_12, ema_26, ema_9, rsi, short_atr, long_atr, roc_20, donchian_high_30, donchian_low_30, upper_band, lower_band, stochastic_oscillator, fi, adx]) or \
       macd_hist_series is None or len(macd_hist_series) < 2:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram

    is_high_volatility = short_atr > (long_atr * 1.75)
    is_extreme_volatility = short_atr > (long_atr * 2.0)

    is_long_term_downtrend = current_price < sma_200
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity

    is_deeply_oversold = rsi < 28
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    if is_capitulation_candidate and macd_hist_delta > 0 and stochastic_oscillator < 18 and ema_12 > ema_26 and ema_9 > ema_12 and adx > 25:
        return "BUY"

    if is_crisis_regime:
        is_recovering_from_oversold = rsi > 38 and macd_hist_delta > 0 and stochastic_oscillator > 85 and adx > 20
        if is_recovering_from_oversold and sentiment_score > -0.8:
            return "BUY"
        if macd_histogram < -0.5 or current_price < sma_50 * 0.98:
            return "SELL"
        return "HOLD"

    base_stop_loss_factor = 0.88
    if is_extreme_volatility:
        stop_loss_factor = 0.78
    elif is_high_volatility:
        stop_loss_factor = 0.83
    else:
        stop_loss_factor = base_stop_loss_factor

    if current_price < (donchian_high_30 * stop_loss_factor) and adx > 22:
        return "SELL"

    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < -0.3 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = sentiment_score < 2.5
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell and adx > 20:
        return "SELL"

    is_momentum_fading = macd_hist_delta < -0.2
    is_extremely_overbought = rsi > 84
    if is_extremely_overbought and is_momentum_fading and adx > 25:
        return "SELL"

    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0.3 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 74
    is_sentiment_permissive_for_buy = sentiment_score > -2.8
    is_sufficient_volatility = short_atr > (long_atr * 0.55)
    is_price_in_bollinger_band = current_price > lower_band * 1.01 and current_price < upper_band * 0.99
    is_price_in_keltner_channel = current_price > keltner_lower_band * 1.02 and current_price < keltner_upper_band * 0.98
    is_ema_crossover = ema_12 is not None and ema_26 is not None and ema_12 > ema_26 * 1.005 and ema_9 > ema_12 * 1.003
    is_adx_trending = adx > 25

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility and is_price_in_bollinger_band and is_price_in_keltner_channel and stochastic_oscillator > 25 and is_ema_crossover and is_adx_trending:
        return "BUY"

    return "HOLD"