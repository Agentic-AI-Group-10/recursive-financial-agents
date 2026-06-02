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

def calculate_bollinger_bands(prices, period=20):
    if len(prices) < period:
        return None, None, None
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, lower, sma

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

def calculate_stochastic_oscillator(prices, period=14):
    if len(prices) < period:
        return None
    lowest_low = np.min(prices[-period:])
    highest_high = np.max(prices[-period:])
    return ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100

def calculate_keltner_channel(prices, period=20):
    if len(prices) < period:
        return None, None
    sma = np.mean(prices[-period:])
    atr = calculate_atr(prices, period)
    upper_band = sma + (atr * 2)
    lower_band = sma - (atr * 2)
    return upper_band, lower_band

def calculate_sentiment_score(news_context):
    context_lower = news_context.lower()
    sentiment_keywords = {
        "fed pivot": 3.5, "rate cut": 3.0, "quantitative easing": 2.8, "soft landing": 2.8,
        "cooling inflation": 2.7, "cpi miss": 2.6, "ai boom": 2.7, "stimulus": 2.2,
        "dovish": 2.2, "record high": 2.1, "bullish": 2.1, "strong earnings": 2.1,
        "beat estimates": 1.6, "recovery": 1.6, "upgrade": 1.6, "de-escalation": 2.2,
        "short squeeze": 3.8, "capitulation": 3.3, "panic selling": 2.7, "extreme fear": 2.2,
        "strong jobs report": 2.0, "recession": -3.2, "crisis": -3.2, "stagflation": -3.1,
        "hot inflation": -3.1, "war": -3.2, "yield curve inversion": -3.6, "quantitative tightening": -2.6,
        "black swan": -4.2, "systemic risk": -4.2, "contagion": -3.6, "credit crunch": -3.6,
        "rate hike": -2.6, "bankruptcy": -2.6, "hard landing": -2.6, "geopolitical risk": -2.6,
        "cpi beat": -2.6, "vix spike": -2.6, "hawkish": -2.1, "bearish": -2.1,
        "sell-off": -2.1, "weak earnings": -2.1, "market turmoil": -2.1, "bubble": -2.1,
        "economic slowdown": -2.1, "market correction": -2.1, "regime shift": -3.2,
        "uncertainty": -1.6, "euphoria": -2.6, "mania": -3.2, "irrational exuberance": -3.2,
        "extreme greed": -2.6, "market rebound": 2.7, "rebound potential": 2.2, "safe haven": 1.6,
        "economic recovery": 2.2, "bull market": 2.2, "bear market": -2.2, "inflation concerns": -1.1,
        "deflation risk": -2.2, "market breadth": 1.1, "geopolitical stability": 2.2, "market resilience": 2.2,
        "central bank intervention": 2.8, "monetary easing": 2.6, "fiscal stimulus": 2.4,
        "market rotation": 1.8, "risk on": 2.3, "risk off": -2.3, "safe haven demand": 2.0,
        "economic expansion": 2.1, "growth acceleration": 2.5, "policy uncertainty": -2.0,
        "sector rotation": 1.9, "valuation expansion": 1.7, "valuation contraction": -1.7,
        "market breadth expansion": 1.8, "liquidity surge": 2.4, "portfolio rebalancing": 1.5,
        "technical breakout": 2.3, "short covering": 2.5, "liquidity crunch": -2.8,
        "margin squeeze": -2.7, "deleveraging": -2.5, "credit expansion": 2.3,
        "inflation easing": 2.9, "market stability": 2.4, "policy clarity": 2.3,
        "market consolidation": 1.4, "volatility surge": -2.4, "liquidity expansion": 2.1,
        "risk parity": 1.9, "portfolio diversification": 1.7, "safe haven rotation": 2.3,
        "economic resilience": 2.5, "policy support": 2.6, "market confidence": 2.2,
        "bullish momentum": 2.4, "bearish momentum": -2.4, "market breadth contraction": -1.7,
        "valuation peak": -2.8, "valuation trough": 2.8, "liquidity contraction": -2.7,
        "portfolio concentration": -1.8, "risk parity": 1.9
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "unlikely", "avoid", "no signs of", "unlikely to", "lack", "absence", "never", "none", "neglect", "without", "lack of", "fail to", "struggle to", "prevent", "avoid", "unlikely", "neglect", "no longer", "never again", "no longer", "lack of", "fail to", "struggle to", "prevent", "avoid", "unlikely", "neglect", "no longer", "without any", "lack any", "fail any", "struggle any", "prevent any", "avoid any", "unlikely any", "neglect any"]
    net_sentiment_score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'(?<!\S)(?i)' + re.escape(keyword) + r'(?!\S)'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start() - 200):match.start()]
            post_context = context_lower[match.end():match.end() + 200]
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            if any(neg_word in post_context for neg_word in negation_words):
                is_negated = not is_negated
            if is_negated:
                weight *= 0.03  # Increased negation penalty
            net_sentiment_score += -weight if is_negated else weight
    return net_sentiment_score

def decide(current_price, price_history, news_context):
    context_lower = news_context.lower()
    sentiment_score = calculate_sentiment_score(news_context)
    all_prices = price_history + [current_price]
    price_len = len(all_prices)

    if price_len < 50:
        return "HOLD"

    sma_50 = np.mean(all_prices[-50:])
    sma_200 = np.mean(all_prices[-200:]) if price_len >= 200 else None
    
    volatility_ratio = calculate_atr(all_prices, 20) / calculate_atr(all_prices, 50)
    ema_short = 12 if volatility_ratio < 1.5 else 8
    ema_long = 26 if volatility_ratio < 1.5 else 18
    ema_signal = 9 if volatility_ratio < 1.5 else 6
    
    ema_12 = calculate_ema_series(all_prices, ema_short)[-1] if price_len >= ema_short else None
    ema_26 = calculate_ema_series(all_prices, ema_long)[-1] if price_len >= ema_long else None
    ema_9 = calculate_ema_series(all_prices, ema_signal)[-1] if price_len >= ema_signal else None
    
    rsi = calculate_rsi(all_prices, 14)
    macd_line, signal_line, macd_hist_series = calculate_macd_series(all_prices)
    short_atr = calculate_atr(all_prices, 20)
    long_atr = calculate_atr(all_prices, 50)
    roc_20 = calculate_roc(all_prices, 20)
    donchian_high_30, donchian_low_30 = calculate_donchian_channel(all_prices, 30)
    stochastic_oscillator = calculate_stochastic_oscillator(all_prices)
    keltner_upper_band, keltner_lower_band = calculate_keltner_channel(all_prices)
    bollinger_upper, bollinger_lower, bollinger_sma = calculate_bollinger_bands(all_prices)

    if any(v is None for v in [sma_50, ema_12, ema_26, ema_9, rsi, short_atr, long_atr, roc_20, donchian_high_30, donchian_low_30, stochastic_oscillator]) or \
       macd_hist_series is None or len(macd_hist_series) < 2 or bollinger_upper is None:
        return "HOLD"

    macd_histogram = macd_hist_series[-1]
    prev_macd_histogram = macd_hist_series[-2]
    macd_hist_delta = macd_histogram - prev_macd_histogram
    macd_hist_acceleration = macd_hist_delta - (macd_hist_series[-3] - macd_hist_series[-2]) if len(macd_hist_series) >= 3 else 0

    is_high_volatility = volatility_ratio > 1.75
    is_extreme_volatility = volatility_ratio > 2.0

    is_long_term_downtrend = current_price < sma_200 if sma_200 is not None else False
    is_crash_velocity = roc_20 < -15.0
    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity or ("yield curve inversion" in context_lower)

    is_deeply_oversold = rsi < 25
    is_extreme_crash_velocity = roc_20 < -18.0
    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold and current_price < donchian_low_30 and (short_atr > long_atr * 1.2) and current_price < keltner_lower_band

    if is_capitulation_candidate and macd_hist_delta > 0 and stochastic_oscillator < 15 and ema_12 > ema_26 and sentiment_score > -1.5 and macd_hist_acceleration > 0 and signal_line[-1] < macd_histogram:
        return "BUY"

    if is_crisis_regime:
        is_recovering_from_oversold = rsi > 35 and macd_hist_delta > 0 and stochastic_oscillator > 85 and sentiment_score > -0.5
        if is_recovering_from_oversold:
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

    atr_stop = keltner_lower_band + (short_atr * 0.5)
    if current_price < atr_stop and current_price < donchian_high_30 * stop_loss_factor:
        return "SELL"

    is_primary_downtrend = current_price < sma_50
    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0
    is_sentiment_permissive_for_sell = sentiment_score < 2.0
    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:
        return "SELL"

    is_momentum_fading = macd_hist_delta < 0
    overbought_threshold = 85 if is_high_volatility else 82
    is_extremely_overbought = rsi > overbought_threshold
    if is_extremely_overbought and is_momentum_fading:
        return "SELL"

    is_primary_uptrend = current_price > sma_50 and (sma_200 is None or current_price > sma_200)
    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0
    is_not_overbought = rsi < (75 if is_high_volatility else 72)
    is_sentiment_permissive_for_buy = sentiment_score > -2.5
    is_sufficient_volatility = short_atr > (long_atr * 0.6)
    is_price_in_keltner_channel = current_price > keltner_lower_band and current_price < keltner_upper_band
    is_ema_crossover = ema_12 is not None and ema_26 is not None and ema_12 > ema_26
    is_bollinger_in_range = current_price > bollinger_lower and current_price < bollinger_upper

    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility and is_price_in_keltner_channel and is_bollinger_in_range and stochastic_oscillator > 30 and is_ema_crossover and macd_hist_acceleration > 0:
        return "BUY"

    return "HOLD"