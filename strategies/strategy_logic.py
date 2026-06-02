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
    return sma + 2*std, sma - 2*std, sma

def calculate_macd_series(prices):
    if len(prices) < 26:
        return None, None, None
    short_ema = calculate_ema_series(prices, 12)
    long_ema = calculate_ema_series(prices, 26)
    macd_line = short_ema[-len(long_ema):] - long_ema
    signal_line = calculate_ema_series(macd_line, 9)
    histogram = macd_line[-len(signal_line):] - signal_line
    return macd_line[-1], signal_line[-1], histogram[-1]

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    seed_gain = np.sum(deltas[:period][deltas[:period] > 0])
    seed_loss = np.sum(-deltas[:period][deltas[:period] < 0])
    avg_gain, avg_loss = seed_gain/period, seed_loss/period
    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0
        avg_gain = (avg_gain * (period-1) + gain) / period
        avg_loss = (avg_loss * (period-1) + loss) / period
    return 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss != 0 else 100

def calculate_atr(prices, period=14):
    if len(prices) < period + 1:
        return None
    ranges = np.abs(np.diff(prices))
    return calculate_ema_series(ranges, period)[-1]

def calculate_obv(prices, volumes):
    if len(prices) < 2:
        return None
    obv = np.zeros(len(prices))
    obv[0] = volumes[0]
    for i in range(1, len(prices)):
        obv[i] = obv[i-1] + volumes[i] if prices[i] > prices[i-1] else (obv[i-1] - volumes[i] if prices[i] < prices[i-1] else obv[i-1])
    return obv[-1]

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
        "portfolio concentration": -1.8, "risk parity": 1.9, "inflation easing": 2.9,
        "central bank easing": 3.0, "monetary stimulus": 2.8, "fiscal expansion": 2.6,
        "geopolitical stability": 2.4, "market breadth expansion": 2.2, "liquidity surge": 2.5,
        "technical breakout": 2.6, "short covering": 2.8, "liquidity crunch": -2.9,
        "margin squeeze": -2.8, "deleveraging": -2.6, "credit expansion": 2.4,
        "policy clarity": 2.5, "market stability": 2.6, "economic resilience": 2.7,
        "bullish momentum": 2.5, "bearish momentum": -2.5, "market breadth contraction": -1.8,
        "supply chain issues": -2.4, "interest rate pause": 2.3, "geopolitical tension": -2.3,
        "market crash": -3.5, "equity plunge": -3.4, "liquidity freeze": -3.3, "debt ceiling": -3.1
    }
    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "unlikely", "avoid", "no signs of", "unlikely to", "lack", "absence", "never", "none", "neglect", "without", "lack of", "fail to", "struggle to", "prevent", "avoid", "unlikely", "neglect", "no longer", "never again", "no longer", "lack of", "fail to", "struggle to", "prevent", "avoid", "unlikely", "neglect", "no longer", "without any", "lack any", "fail any", "struggle any", "prevent any", "avoid any", "unlikely any", "neglect any", "no longer any", "lack of any", "fail of any", "struggle of any", "prevent of any", "avoid of any", "unlikely of any", "neglect of any", "doesn't", "doesn't show", "doesn't indicate", "doesn't suggest", "isn't showing", "isn't indicating", "isn't suggesting", "lacks", "fails to", "struggles to", "avoids", "prevents", "unlikely to", "avoiding", "lacking", "failing to", "struggling to", "preventing", "avoiding", "unlikely showing", "lacking any", "failing any", "struggling any", "preventing any", "avoiding any", "unlikely any"]
    score = 0.0
    for keyword, weight in sentiment_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, context_lower):
            pre_context = context_lower[max(0, match.start()-100):match.start()]
            post_context = context_lower[match.end():match.end()+100]
            is_negated = any(neg in pre_context for neg in negation_words)
            if any(neg in post_context for neg in negation_words):
                is_negated = not is_negated
            if is_negated:
                weight *= 0.2
            recency_factor = 1.0 - (match.start() / max(1, len(context_lower)))
            weight *= (1.0 + recency_factor * 0.5)
            score += -weight if is_negated else weight
    return score

def decide(current_price, price_history, news_context):
    all_prices = price_history + [current_price]
    price_len = len(all_prices)
    
    if price_len < 50:
        return "HOLD"
    
    sma_50 = np.mean(all_prices[-50:])
    sma_200 = np.mean(all_prices[-200:]) if price_len >= 200 else None
    
    volatility_ratio = calculate_atr(all_prices, 20) / calculate_atr(all_prices, 50)
    ema_short = 12 if volatility_ratio < 1.5 else 8
    ema_long = 26 if volatility_ratio < 1.5 else 18
    
    ema_12 = calculate_ema_series(all_prices, ema_short)[-1] if price_len >= ema_short else None
    ema_26 = calculate_ema_series(all_prices, ema_long)[-1] if price_len >= ema_long else None
    
    rsi_period = 14 if volatility_ratio < 1.75 else 7
    rsi = calculate_rsi(all_prices, rsi_period)
    
    macd_line, signal_line, histogram = calculate_macd_series(all_prices)
    
    if any(v is None for v in [sma_50, ema_12, ema_26, rsi, macd_line, signal_line, histogram]):
        return "HOLD"
    
    is_high_volatility = volatility_ratio > 1.75
    is_extreme_volatility = volatility_ratio > 2.0
    
    is_long_term_downtrend = current_price < sma_200 if sma_200 is not None else False
    is_crisis_regime = is_long_term_downtrend and is_high_volatility or "yield curve inversion" in news_context.lower()
    
    sentiment_score = calculate_sentiment_score(news_context)
    
    if is_crisis_regime:
        if rsi < 25 and histogram > 0 and current_price > sma_50 and sentiment_score > -1.5:
            return "BUY"
        if rsi > 75 and histogram < 0 and current_price < sma_50 and sentiment_score < -2.5:
            return "SELL"
        return "HOLD"
    
    if rsi < 25 and histogram > 0 and ema_12 > ema_26 and current_price > sma_50 and sentiment_score > -1.5:
        return "BUY"
    
    if rsi > 75 and histogram < 0 and ema_12 < ema_26 and current_price < sma_50 and sentiment_score < -1.5:
        return "SELL"
    
    return "HOLD"