import numpy as np
import re
import pandas as pd

def calculate_ema_series(data, period):
    if len(data) < period:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]

def calculate_bollinger_bands(prices, period=20):
    if len(prices) < period:
        return None, None, None
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    return sma + (std * 2), sma - (std * 2), sma

def calculate_macd_series(prices):
    if len(prices) < 26:
        return None, None, None
    short_ema = calculate_ema_series(prices, 12)
    long_ema = calculate_ema_series(prices, 26)
    macd = short_ema[-len(long_ema):] - long_ema
    signal = calculate_ema_series(macd, 9)
    return macd[-1], signal[-1], macd[-1] - signal[-1]

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(np.array(prices, dtype=float))
    seed_gains = deltas[:period][deltas[:period] >= 0].sum()
    seed_losses = -deltas[:period][deltas[:period] < 0].sum()
    avg_gain, avg_loss = seed_gains/period, seed_losses/period
    for i in range(period, len(deltas)):
        delta = deltas[i]
        avg_gain = (avg_gain*(period-1) + max(delta,0))/period
        avg_loss = (avg_loss*(period-1) + max(-delta,0))/period
    return 100 - (100/(1 + avg_gain/avg_loss)) if avg_loss != 0 else 100

def calculate_atr(prices, period=14):
    if len(prices) < period + 1:
        return None
    return calculate_ema_series(np.abs(np.diff(np.array(prices, dtype=float))), period)[-1]

def calculate_sentiment_score(news):
    score = 0.0
    keywords = {
        "rate cut": 3.0, "quantitative easing": 2.8, "soft landing": 2.5, "recession": -3.0,
        "bullish": 2.2, "bearish": -2.2, "market crash": -3.5, "systemic risk": -3.2,
        "inflation easing": 2.8, "geopolitical risk": -2.6, "economic recovery": 2.4,
        "central bank easing": 3.0, "monetary stimulus": 2.8, "fiscal expansion": 2.6
    }
    negations = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids"]
    for word, weight in keywords.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        for match in re.finditer(pattern, news.lower()):
            pre_context = news.lower()[max(0, match.start()-50):match.start()]
            is_negated = any(neg in pre_context for neg in negations)
            score += -weight if is_negated else weight
    return score

def decide(current_price, price_history, news_context):
    all_prices = price_history + [current_price]
    if len(all_prices) < 50:
        return "HOLD"
    
    sma_50 = np.mean(all_prices[-50:])
    sma_200 = np.mean(all_prices[-200:]) if len(all_prices) >= 200 else None
    volatility = calculate_atr(all_prices, 20)
    ema_short = 12 if volatility < 1.5 else 8
    ema_long = 26 if volatility < 1.5 else 18
    
    ema_12 = calculate_ema_series(all_prices, ema_short)[-1] if len(all_prices) >= ema_short else None
    ema_26 = calculate_ema_series(all_prices, ema_long)[-1] if len(all_prices) >= ema_long else None
    rsi = calculate_rsi(all_prices)
    macd_line, signal_line, histogram = calculate_macd_series(all_prices)
    
    if any(v is None for v in [sma_50, ema_12, ema_26, rsi, macd_line, signal_line, histogram]):
        return "HOLD"
    
    is_uptrend = current_price > sma_50 and (sma_200 is None or current_price > sma_200)
    is_downtrend = current_price < sma_50 and (sma_200 is None or current_price < sma_200)
    is_momentum_up = macd_line > signal_line and histogram > 0
    is_momentum_down = macd_line < signal_line and histogram < 0
    
    sentiment_score = calculate_sentiment_score(news_context)
    crisis_keywords = ["yield curve inversion", "banking crisis", "systemic risk", "market crash"]
    is_crisis = any(k in news_context.lower() for k in crisis_keywords) or rsi < 20 and volatility > 2.5
    
    if is_crisis:
        if rsi > 40 and sentiment_score > -1.0 and is_momentum_up:
            return "BUY"
        if is_momentum_down and current_price < sma_50:
            return "SELL"
        return "HOLD"
    
    if is_uptrend and is_momentum_up and rsi < 70 and sentiment_score > -2.0:
        return "BUY"
    if is_downtrend and is_momentum_down and rsi > 80 and sentiment_score < -1.5:
        return "SELL"
    
    return "HOLD"