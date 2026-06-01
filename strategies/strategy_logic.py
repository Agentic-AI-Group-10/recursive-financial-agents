import numpy as np
import re # For more robust keyword matching with word boundaries

# Helper functions for technical indicators
def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA) for the latest price."""
    if len(prices) < period:
        return None
    
    prices_arr = np.array(prices, dtype=float)
    
    # Initialize EMA with SMA for the first 'period' values
    # Ensure there are enough prices for the initial SMA
    if len(prices_arr) < period:
        return None
    
    ema_values = np.zeros_like(prices_arr, dtype=float)
    ema_values[period - 1] = np.mean(prices_arr[:period])
    
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices_arr)):
        ema_values[i] = (prices_arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        
    return ema_values[-1] # Return the latest EMA value

def calculate_rsi(prices, period):
    """Calculates the Relative Strength Index (RSI) for the latest price."""
    if len(prices) < period + 1: # Need at least period + 1 prices to calculate first change
        return None

    prices_arr = np.array(prices, dtype=float)
    
    # Calculate price changes
    deltas = np.diff(prices_arr)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0) # Make losses positive
    
    # Calculate initial average gain and loss
    # Ensure there are enough deltas for the initial period
    if len(gains) < period:
        return None

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Calculate initial RS and RSI
    if avg_loss == 0:
        rs = np.inf # Avoid division by zero, implies no losses
    else:
        rs = avg_gain / avg_loss
    
    rsi = 100 - (100 / (1 + rs))

    # For subsequent periods, use Wilder's smoothing
    # Note: The provided prices list already contains all historical prices up to current.
    # We need to calculate RSI based on the *entire* relevant history.
    # The loop below correctly updates avg_gain and avg_loss for the latest period.
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
        
    return rsi

def decide(current_price, price_history, news_context):
    """
    Self-improved trading strategy incorporating enhanced sentiment analysis,
    EMA crossovers, and RSI to make more robust decisions.
    
    Parameters:
        current_price (float): The current day's closing price for SPY.
        price_history (list of float): List of historical closing prices up to yesterday.
        news_context (str): Combined news headlines from the last 24 hours.
        
    Returns:
        str: "BUY", "SELL", or "HOLD"
    """
    
    # --- 1. Enhanced Sentiment Analysis ---
    context_lower = news_context.lower()
    
    # Define sentiment keywords with weights and negation patterns
    sentiment_keywords = {
        "bullish": 2.0, "beat": 1.5, "surge": 2.0, "growth": 1.5, "gdp rise": 2.0, 
        "rate cut": 2.5, "upgrade": 1.5, "stimulus": 2.0, "recovery": 1.5,
        "strong earnings": 2.0, "positive outlook": 1.5, "expansion": 1.5,
        "optimistic": 1.0, "record high": 2.0, "breakout": 1.5, "acquisition": 1.0,

        "bearish": -2.0, "miss": -1.5, "plunge": -2.0, "recession": -2.5, 
        "rate hike": -2.5, "inflation": -2.0, "downgrade": -1.5, "crisis": -2.5, 
        "tariff": -1.5, "weak earnings": -2.0, "negative outlook": -1.5, "contraction": -1.5,
        "pessimistic": -1.0, "sell-off": -2.0, "decline": -1.5, "bankruptcy": -2.5
    }
    
    negation_words = ["not", "no", "lack of", "fail to", "decline in", "without", "struggle to"]
    
    net_sentiment_score = 0.0
    
    for keyword, weight in sentiment_keywords.items():
        # Use regex for whole word/phrase matching
        # re.escape handles special characters in keywords
        pattern = r'\b' + re.escape(keyword) + r'\b'
        
        # Find all occurrences of the keyword
        for match in re.finditer(pattern, context_lower):
            match_index = match.start()
            
            # Check for negation words preceding the keyword within a small window
            pre_context = context_lower[max(0, match_index - 30):match_index] # Check up to 30 chars before
            is_negated = any(neg_word in pre_context for neg_word in negation_words)
            
            if is_negated:
                net_sentiment_score -= weight # Negated bullish becomes bearish, negated bearish becomes bullish
            else:
                net_sentiment_score += weight
    
    # --- 2. Improved Technical Indicators ---
    
    # Combine current price with history for indicator calculations
    all_prices = price_history + [current_price]
    
    # Define periods for indicators
    SHORT_EMA_PERIOD = 10
    LONG_EMA_PERIOD = 20
    RSI_PERIOD = 14
    
    # Ensure enough data for robust indicator calculations
    # RSI needs at least period + 1 prices for initial calculation
    required_history_length = max(LONG_EMA_PERIOD, RSI_PERIOD + 1)
    if len(all_prices) < required_history_length:
        # Not enough data for robust indicators, default to HOLD
        return "HOLD"

    # Calculate EMAs
    short_ema = calculate_ema(all_prices, SHORT_EMA_PERIOD)
    long_ema = calculate_ema(all_prices, LONG_EMA_PERIOD)
    
    # Calculate RSI
    rsi = calculate_rsi(all_prices, RSI_PERIOD)
    
    # Safeguard against any unexpected None from indicator calculations (should be handled by length check)
    if short_ema is None or long_ema is None or rsi is None:
        return "HOLD" 

    # --- 3. Refined Decision Logic ---
    
    # Define thresholds for sentiment and RSI
    # RELAXED SENTIMENT THRESHOLDS to increase trade frequency
    BULLISH_SENTIMENT_THRESHOLD = 1.0 
    BEARISH_SENTIMENT_THRESHOLD = -1.0 
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Initialize technical signals
    bullish_tech_signal = False
    bearish_tech_signal = False
    
    # EMA Crossover (Simplified: Removed current_price confirmation for less lag and more activity)
    # Bullish: Short EMA above Long EMA
    if short_ema > long_ema:
        bullish_tech_signal = True
    # Bearish: Short EMA below Long EMA
    elif short_ema < long_ema:
        bearish_tech_signal = True
        
    # RSI as a filter: Prevent buying into overbought conditions or selling into oversold conditions
    # This filter remains crucial for risk management.
    if bullish_tech_signal and rsi >= RSI_OVERBOUGHT:
        bullish_tech_signal = False # Overbought, do not buy
    if bearish_tech_signal and rsi <= RSI_OVERSOLD:
        bearish_tech_signal = False # Oversold, do not sell
        
    # Combine all signals for final decision
    # The AND logic is retained, but individual conditions are less strict,
    # aiming for more frequent, yet still confirmed, trades.
    
    # BUY condition: Moderately bullish sentiment AND confirmed bullish technicals (EMA crossover, not overbought)
    if net_sentiment_score >= BULLISH_SENTIMENT_THRESHOLD and bullish_tech_signal:
        return "BUY"
    
    # SELL condition: Moderately bearish sentiment AND confirmed bearish technicals (EMA crossover, not oversold)
    elif net_sentiment_score <= BEARISH_SENTIMENT_THRESHOLD and bearish_tech_signal:
        return "SELL"
    
    # Default to HOLD if conditions are not met or conflicting, reducing noise
    else:
        return "HOLD"