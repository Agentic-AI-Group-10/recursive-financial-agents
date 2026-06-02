import numpy as np
import re

def calculate_ema(data, period):
    if len(data) < period: return np.array([])
    return np.convolve(data, np.repeat(2/(period+1), period), mode='valid')

def calculate_macd(data):
    if len(data) < 26: return None
    short = calculate_ema(data, 12)
    long = calculate_ema(data, 26)
    signal = calculate_ema(np.array(short) - np.array(long), 9)
    return np.array(short[-len(signal):]) - np.array(long[-len(signal):]), signal, np.array(short) - np.array(long)

def calculate_rsi(data, period=14):
    if len(data) < period+1: return None
    deltas = np.diff(data)
    avg_gain = np.mean(deltas[deltas>0][:period])
    avg_loss = -np.mean(deltas[deltas<0][:period])
    for i in range(period, len(deltas)):
        delta = deltas[i]
        avg_gain = (avg_gain*(period-1) + max(0, delta))/period
        avg_loss = (avg_loss*(period-1) + max(0, -delta))/period
    return 100 - (100/(1 + (avg_gain/avg_loss) if avg_loss else 100000))

def calculate_atr(prices, period=14):
    if len(prices) < period+1: return None
    return np.mean(np.abs(np.diff(prices))[-period:])

def calculate_obv(prices, volumes):
    if len(prices) < 2: return None
    obv = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]: obv += volumes[i]
        elif prices[i] < prices[i-1]: obv -= volumes[i]
    return obv

def calculate_sentiment(news):
    keywords = {
        "fed pivot": 3.5, "rate cut": 3.0, "quantitative easing": 2.8, "soft landing": 2.8,
        "recession": -3.2, "crisis": -3.2, "yield curve inversion": -3.6, "market crash": -3.5,
        "bull market": 2.2, "bear market": -2.2, "inflation easing": 2.9, "economic recovery": 2.5,
        "geopolitical stability": 2.4, "systemic risk": -4.2, "central bank easing": 3.0,
        "monetary stimulus": 2.8, "fiscal expansion": 2.6, "market breadth": 1.1,
        "liquidity surge": 2.4, "technical breakout": 2.6, "short covering": 2.8,
        "liquidity crunch": -2.9, "margin squeeze": -2.8, "deleveraging": -2.6,
        "economic resilience": 2.7, "bullish momentum": 2.5, "bearish momentum": -2.5
    }
    negations = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "unlikely"]
    score = 0.0
    context = news.lower()
    for keyword, weight in keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context):
            pre = context[max(0, match.start()-100):match.start()]
            post = context[match.end():match.end()+100]
            negated = any(n in pre for n in negations)
            if any(n in post for n in negations): negated = not negated
            if negated: weight *= 0.1
            recency = 1.0 - (match.start()/max(1, len(context)))
            weight *= (1.0 + recency*0.5)
            score += -weight if negated else weight
    return score

def decide(current_price, price_history, news_context):
    prices = np.array(price_history + [current_price])
    if len(prices) < 50: return "HOLD"
    
    # Core indicators
    ema12 = calculate_ema(prices, 12)[-1] if len(prices)>=12 else None
    ema26 = calculate_ema(prices, 26)[-1] if len(prices)>=26 else None
    macd_line, signal_line, _ = calculate_macd(prices)
    rsi = calculate_rsi(prices)
    atr = calculate_atr(prices)
    sma50 = np.mean(prices[-50:])
    
    # Volatility adaptive parameters
    volatility_ratio = atr / np.mean(calculate_atr(prices[-100:], 50) if len(prices)>=100 else atr)
    ema_short = 8 if volatility_ratio > 1.5 else 12
    ema_long = 18 if volatility_ratio > 1.5 else 26
    
    # Sentiment processing
    sentiment_score = calculate_sentiment(news_context)
    crisis_keywords = ["yield curve inversion", "market crash", "systemic risk", "banking crisis"]
    is_crisis = any(k in news_context.lower() for k in crisis_keywords)
    
    # Decision logic
    if is_crisis:
        if rsi is not None and rsi < 25 and macd_line is not None and macd_line[-1] > 0 and sentiment_score > -1.5:
            return "BUY"
        return "SELL" if macd_line is not None and macd_line[-1] < 0 else "HOLD"
    
    # Buy conditions
    if (ema12 is not None and ema26 is not None and ema12 > ema26 and 
        rsi is not None and rsi < 65 and 
        macd_line is not None and macd_line[-1] > 0 and 
        prices[-1] > sma50 and 
        sentiment_score > -2.0):
        return "BUY"
    
    # Sell conditions
    if (rsi is not None and rsi > 85 and 
        (macd_line is not None and macd_line[-1] < 0) and 
        prices[-1] < sma50 and 
        sentiment_score < -1.5):
        return "SELL"
    
    return "HOLD"