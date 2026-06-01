# ==============================================================================
# Baseline Strategy Logic
# Location: /home/ow9800/recursive-financial-agents/strategies/strategy_logic.py
# ==============================================================================
# This file is dynamically evolved by the Recursive Architect.
# The backtesting engine dynamically imports and calls the decide() function.

def decide(current_price, price_history, news_context):
    """
    Baseline trading strategy that analyzes daily news context and historical prices.
    
    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.
        
    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    context_lower = news_context.lower()
    
    # Sentiment Keywords
    bullish_keywords = ["bullish", "beat", "surge", "growth", "gdp rise", "rate cut", "upgrade", "stimulus", "recovery"]
    bearish_keywords = ["bearish", "miss", "plunge", "recession", "rate hike", "inflation", "downgrade", "crisis", "tariff"]
    
    bullish_score = sum(1 for word in bullish_keywords if word in context_lower)
    bearish_score = sum(1 for word in bearish_keywords if word in context_lower)
    
    # Also add a simple technical analysis filter (5-day simple moving average)
    if len(price_history) >= 5:
        sma_5 = sum(price_history[-5:]) / 5
        is_above_sma = current_price > sma_5
    else:
        is_above_sma = True
        
    # Decision Logic
    if bullish_score > bearish_score and is_above_sma:
        return "BUY"
    elif bearish_score > bullish_score or not is_above_sma:
        return "SELL"
    else:
        return "HOLD"
