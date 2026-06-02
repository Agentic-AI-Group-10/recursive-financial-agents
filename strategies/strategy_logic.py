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

def calculate_macd_series(prices):
    if len(prices) < 26: return None, None, None
    short_ema = calculate_ema_series(prices, 12)
    long_ema = calculate_ema_series(prices, 26)
    if len(short_ema) < 2 or len(long_ema) < 1: return None, None, None
    macd_line = short_ema[-1] - long_ema[-1]
    signal_line = calculate_ema_series([macd_line], 9)[-1] if len(prices)>=9 else None
    histogram = macd_line - signal_line if signal_line else 0
    return macd_line, signal_line, histogram

def calculate_rsi(prices, period=14):
    if len(prices) < period+1: return None
    deltas = np.diff(np.array(prices, dtype=float))
    gains = deltas[deltas>0].sum()
    losses = -deltas[deltas<0].sum()
    avg_gain = gains/period
    avg_loss = losses/period
    for i in range(period, len(deltas)):
        delta = deltas[i]
        avg_gain = (avg_gain*(period-1) + max(0,delta))/period
        avg_loss = (avg_loss*(period-1) + max(0,-delta))/period
    return 100 - (100/(1+(avg_gain/avg_loss))) if avg_loss else 100

def calculate_atr(prices, period=14):
    if len(prices) < period+1: return None
    return np.mean(np.abs(np.diff(np.array(prices, dtype=float)))[-period:])

def calculate_sentiment_score(news_context):
    context_lower = news_context.lower()
    keywords = {
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
    score = 0.0
    for keyword, weight in keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start()-30):match.start()]
            is_negated = any(neg in pre_context for neg in negation_words)
            score += (-weight if is_negated else weight)
    return score

def decide(current_price, price_history, news_context):
    all_prices = price_history + [current_price]
    if len(all_prices) < 50: return "HOLD"
    
    sma_50 = np.mean(all_prices[-50:])
    sma_200 = np.mean(all_prices[-200:])
    ema_20 = calculate_ema_series(all_prices, 20)[-1] if len(all_prices)>=20 else None
    ema_50 = calculate_ema_series(all_prices, 50)[-1] if len(all_prices)>=50 else None
    
    rsi_period = 14 if len(all_prices) > 100 else 7
    rsi = calculate_rsi(all_prices, rsi_period)
    
    macd_line, signal_line, _ = calculate_macd_series(all_prices)
    atr = calculate_atr(all_prices, 14)
    roc_20 = ((current_price - all_prices[-21]) / all_prices[-21]) * 100 if len(all_prices)>=21 else 0
    
    sentiment = calculate_sentiment_score(news_context)
    volatility_factor = atr / np.mean(calculate_atr(all_prices[-100:], 14)) if len(all_prices)>=100 else 1
    
    is_crisis = (current_price < sma_200 and atr > 1.5*np.mean(all_prices[-20:]) or roc_20 < -10)
    is_oversold = rsi < 30 if rsi else False
    is_overbought = rsi > 70 if rsi else False
    
    if is_crisis:
        if is_oversold and macd_line > signal_line and sentiment > -2:
            return "BUY"
        if is_overbought and macd_line < signal_line and sentiment < -1:
            return "SELL"
        return "HOLD"
    
    if (current_price > sma_50 and ema_20 and ema_20 > ema_50 and 
        rsi and rsi < 65 and macd_line and macd_line > 0 and 
        sentiment > -3 and volatility_factor < 1.5):
        return "BUY"
    
    if (current_price < sma_50 and ema_20 and ema_20 < ema_50 and 
        rsi and rsi > 35 and macd_line and macd_line < 0 and 
        sentiment < 1 and volatility_factor > 1.2):
        return "SELL"
    
    return "HOLD"