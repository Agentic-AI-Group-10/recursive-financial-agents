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

def calculate_ichimoku_cloud(prices, period=26):
    if len(prices) < period:
        return None, None, None, None, None
    tenkan_sen = np.mean(prices[-9:])
    kijun_sen = np.mean(prices[-26:])
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = np.mean(prices[-52:])
    chikou_span = prices[-26]
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_sentiment_score(news_context):
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
    phrases = ["fed pivot", "rate cut", "quantitative easing", "soft landing", "cooling inflation"]
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        for match in re.finditer(pattern, context_lower):
            net_sentiment_score += 2.5
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
    ema_20 = calculate_ema_series(all_prices, 20)[-1] if len(all_prices) >= 20 else None
    ema_50 = calculate_ema_series(all_prices, 50)[-1] if len(all_prices) >= 50 else None
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
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = calculate_ichimoku_cloud(all_prices)

    if any(v is None for v in [sma_50, sma_200, ema_20, ema_50, rsi, short_atr, long_atr, roc_20, donchian_high_30, donchian_low_30, upper_band, lower_band, stochastic_oscillator, fi]) or \
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

    is_deeply_oversold = rsi < 30
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold

    if is_capitulation_candidate and macd_hist_delta > 0 and stochastic_oscillator < 20:
        return "BUY"

    if is_crisis_regime:
        is_recovering_from_oversold = rsi > 35 and macd_hist_delta > 0 and stochastic_oscillator > 80
        if is_recovering_from_oversold and sentiment_score > -1.0:
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
    is_sentiment_permissive_for_sell = sentiment_score < 3.0
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    is_momentum_fading = macd_hist_delta < 0
    is_extremely_overbought = rsi > 82
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    is_primary_uptrend = current_price > sma_50
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < 78
    is_sentiment_permissive_for_buy = sentiment_score > -3.0
    is_sufficient_volatility = short_atr > (long_atr * 0.6)
    is_price_in_bollinger_band = current_price > lower_band and current_price < upper_band
    is_price_in_keltner_channel = current_price > keltner_lower_band and current_price < keltner_upper_band
    is_ema_crossover = ema_20 is not None and ema_50 is not None and ema_20 > ema_50

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility and is_price_in_bollinger_band and is_price_in_keltner_channel and stochastic_oscillator > 20 and is_ema_crossover:
        return "BUY"

    return "HOLD"