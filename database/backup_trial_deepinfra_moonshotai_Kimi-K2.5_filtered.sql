--
-- PostgreSQL database dump
--

\restrict MVdTOiRhbOCTSLqLasRUrh6YEHiDEaMGPVcazE8gt89ER2ptbPUVSrkJNgpLKLs

-- Dumped from database version 14.23 (Ubuntu 14.23-1.pgdg22.04+1)
-- Dumped by pg_dump version 14.23 (Ubuntu 14.23-0ubuntu0.22.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: lessons_learned; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lessons_learned (
    lesson_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    strategy_id uuid,
    run_id uuid,
    summary text NOT NULL,
    sentiment_pattern text NOT NULL,
    key_failure_cause text,
    embedding public.vector(768),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.lessons_learned OWNER TO postgres;

--
-- Name: runs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.runs (
    run_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    strategy_id uuid,
    regime character varying(100) NOT NULL,
    initial_portfolio_value numeric(15,2) NOT NULL,
    final_portfolio_value numeric(15,2) NOT NULL,
    total_return_pct numeric(10,4) NOT NULL,
    benchmark_return_pct numeric(10,4) NOT NULL,
    alpha_vs_benchmark numeric(10,4) NOT NULL,
    max_drawdown_pct numeric(10,4) NOT NULL,
    sharpe_ratio_annualized numeric(10,4) NOT NULL,
    win_rate_pct numeric(10,2) NOT NULL,
    total_trades_executed integer NOT NULL,
    total_closed_trades integer NOT NULL,
    elapsed_seconds numeric(12,4) NOT NULL,
    total_energy_joules numeric(15,2) NOT NULL,
    measured_via_hardware boolean DEFAULT false NOT NULL,
    input_tokens_consumed integer DEFAULT 0 NOT NULL,
    output_tokens_consumed integer DEFAULT 0 NOT NULL,
    total_llm_calls integer DEFAULT 0 NOT NULL,
    token_cost_usd numeric(15,6) DEFAULT 0.000000 NOT NULL,
    energy_cost_usd numeric(15,6) DEFAULT 0.000000 NOT NULL,
    total_compute_cost_usd numeric(15,6) DEFAULT 0.000000 NOT NULL,
    net_profit_loss_usd numeric(15,2) NOT NULL,
    efficiency_score_aroi numeric(18,4) NOT NULL,
    trade_log_json jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.runs OWNER TO postgres;

--
-- Name: strategies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.strategies (
    strategy_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    generation integer NOT NULL,
    branch_name character varying(255) NOT NULL,
    code_content text NOT NULL,
    code_hash character varying(64) NOT NULL,
    rationale text NOT NULL,
    parent_strategy_id uuid,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.strategies OWNER TO postgres;

--
-- Data for Name: lessons_learned; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.lessons_learned (lesson_id, strategy_id, run_id, summary, sentiment_pattern, key_failure_cause, embedding, created_at) FROM stdin;
\.


--
-- Data for Name: runs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.runs (run_id, strategy_id, regime, initial_portfolio_value, final_portfolio_value, total_return_pct, benchmark_return_pct, alpha_vs_benchmark, max_drawdown_pct, sharpe_ratio_annualized, win_rate_pct, total_trades_executed, total_closed_trades, elapsed_seconds, total_energy_joules, measured_via_hardware, input_tokens_consumed, output_tokens_consumed, total_llm_calls, token_cost_usd, energy_cost_usd, total_compute_cost_usd, net_profit_loss_usd, efficiency_score_aroi, trade_log_json, created_at) FROM stdin;
512c10e9-dade-4910-ac31-acf4f8e76e63	2bbff88e-cfac-465a-9376-6ef80f329fba	Normal_Filtered	10000.00	11691.11	16.9111	14.2622	2.6489	-2.0817	2.3202	100.00	2	1	3.7429	243.29	f	0	0	0	0.000000	0.000024	0.000024	1691.11	16911079.8890	[{"fee": 5.0, "cash": 0.0, "date": "2017-07-13", "price": 212.9791, "action": "BUY", "shares": 46.9295, "portfolio_value": 9995.0}, {"fee": 5.8485, "cash": 11691.11, "date": "2018-01-24", "price": 249.2454, "action": "SELL", "shares": 0.0, "portfolio_value": 11691.11}]	2026-06-02 18:55:02.543557-04
e965c63f-fc3a-49bf-9fa8-a2ea1b0fee51	2bbff88e-cfac-465a-9376-6ef80f329fba	Covid_Filtered	10000.00	8901.28	-10.9872	23.1527	-34.1399	-13.3839	-1.1137	0.00	4	2	6.0985	396.41	f	0	0	0	0.000000	0.000040	0.000040	-1098.72	-10987181.3647	[{"fee": 5.0, "cash": 0.0, "date": "2019-04-02", "price": 257.6118, "action": "BUY", "shares": 38.7987, "portfolio_value": 9995.0}, {"fee": 4.9771, "cash": 9949.13, "date": "2019-06-06", "price": 256.5578, "action": "SELL", "shares": 0.0, "portfolio_value": 9949.13}, {"fee": 4.9746, "cash": 0.0, "date": "2020-02-07", "price": 303.5869, "action": "BUY", "shares": 32.7556, "portfolio_value": 9944.16}, {"fee": 4.4529, "cash": 8901.28, "date": "2020-02-27", "price": 271.8847, "action": "SELL", "shares": 0.0, "portfolio_value": 8901.28}]	2026-06-02 18:55:09.084925-04
\.


--
-- Data for Name: strategies; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.strategies (strategy_id, generation, branch_name, code_content, code_hash, rationale, parent_strategy_id, created_at) FROM stdin;
2bbff88e-cfac-465a-9376-6ef80f329fba	0	main	import numpy as np\nimport re\nimport pandas as pd\n\ndef calculate_ema_series(data, period):\n    if len(data) < period:\n        return np.array([])\n    data_arr = np.array(data, dtype=float)\n    try:\n        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]\n    except ImportError:\n        ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)\n        if len(data_arr) >= period:\n            ema_values[0] = np.mean(data_arr[:period])\n            multiplier = 2 / (period + 1)\n            for i in range(1, len(ema_values)):\n                ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]\n        return ema_values\n\ndef calculate_bollinger_bands(prices, period=20):\n    if len(prices) < period:\n        return None, None, None\n    sma = np.mean(prices[-period:])\n    std = np.std(prices[-period:])\n    upper = sma + (std * 2)\n    lower = sma - (std * 2)\n    return upper, lower, sma\n\ndef calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):\n    if len(prices) < long_period: \n        return None, None, None\n    short_ema_series = calculate_ema_series(prices, short_period)\n    long_ema_series = calculate_ema_series(prices, long_period)\n    if len(short_ema_series) == 0 or len(long_ema_series) == 0:\n        return None, None, None\n    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series\n    if len(macd_line) < signal_period: \n        return macd_line, None, None\n    signal_line = calculate_ema_series(macd_line, signal_period)\n    if len(signal_line) == 0:\n        return None, None, None\n    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line\n    return macd_line, signal_line, histogram\n\ndef calculate_rsi(prices, period=14):\n    if len(prices) < period + 1:\n        return None\n    prices_arr = np.array(prices, dtype=float)\n    deltas = np.diff(prices_arr)\n    seed_gains = deltas[:period][deltas[:period] >= 0].sum()\n    seed_losses = -deltas[:period][deltas[:period] < 0].sum()\n    avg_gain = seed_gains / period\n    avg_loss = seed_losses / period\n    for i in range(period, len(deltas)):\n        delta = deltas[i]\n        gain = delta if delta >= 0 else 0.0\n        loss = -delta if delta < 0 else 0.0\n        avg_gain = (avg_gain * (period - 1) + gain) / period\n        avg_loss = (avg_loss * (period - 1) + loss) / period\n    if avg_loss == 0:\n        return 100.0\n    rs = avg_gain / avg_loss\n    return 100.0 - (100.0 / (1.0 + rs))\n\ndef calculate_atr(prices, period=14):\n    if len(prices) < period + 1:\n        return None\n    prices_arr = np.array(prices, dtype=float)\n    price_ranges = np.abs(np.diff(prices_arr))\n    atr_series = calculate_ema_series(price_ranges, period)\n    return atr_series[-1] if len(atr_series) > 0 else None\n\ndef calculate_roc(prices, period=20):\n    if len(prices) < period + 1:\n        return None\n    if prices[-1 - period] == 0: \n        return 0.0\n    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100\n\ndef calculate_donchian_channel(prices, period=20):\n    if len(prices) < period:\n        return None, None\n    donchian_high = np.max(prices[-period:])\n    donchian_low = np.min(prices[-period:])\n    upper_band = donchian_high\n    lower_band = donchian_low\n    return upper_band, lower_band\n\ndef calculate_stochastic_oscillator(prices, period=14):\n    if len(prices) < period:\n        return None\n    lowest_low = np.min(prices[-period:])\n    highest_high = np.max(prices[-period:])\n    return ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100\n\ndef calculate_keltner_channel(prices, period=20):\n    if len(prices) < period:\n        return None, None\n    sma = np.mean(prices[-period:])\n    atr = calculate_atr(prices, period)\n    upper_band = sma + (atr * 2)\n    lower_band = sma - (atr * 2)\n    return upper_band, lower_band\n\ndef calculate_obv(prices, volumes):\n    if len(prices) < 2 or len(volumes) < 2:\n        return None\n    obv = 0\n    for i in range(1, len(prices)):\n        if prices[i] > prices[i-1]:\n            obv += volumes[i]\n        elif prices[i] < prices[i-1]:\n            obv -= volumes[i]\n    return obv\n\ndef calculate_sentiment_score(news_context):\n    context_lower = news_context.lower()\n    sentiment_keywords = {\n        "fed pivot": 3.5, "rate cut": 3.0, "quantitative easing": 2.8, "soft landing": 2.8,\n        "cooling inflation": 2.7, "cpi miss": 2.6, "ai boom": 2.7, "stimulus": 2.2,\n        "dovish": 2.2, "record high": 2.1, "bullish": 2.1, "strong earnings": 2.1,\n        "beat estimates": 1.6, "recovery": 1.6, "upgrade": 1.6, "de-escalation": 2.2,\n        "short squeeze": 3.8, "capitulation": 3.3, "panic selling": 2.7, "extreme fear": 2.2,\n        "strong jobs report": 2.0, "recession": -3.2, "crisis": -3.2, "stagflation": -3.1,\n        "hot inflation": -3.1, "war": -3.2, "yield curve inversion": -3.6, "quantitative tightening": -2.6,\n        "black swan": -4.2, "systemic risk": -4.2, "contagion": -3.6, "credit crunch": -3.6,\n        "rate hike": -2.6, "bankruptcy": -2.6, "hard landing": -2.6, "geopolitical risk": -2.6,\n        "cpi beat": -2.6, "vix spike": -2.6, "hawkish": -2.1, "bearish": -2.1,\n        "sell-off": -2.1, "weak earnings": -2.1, "market turmoil": -2.1, "bubble": -2.1,\n        "economic slowdown": -2.1, "market correction": -2.1, "regime shift": -3.2,\n        "uncertainty": -1.6, "euphoria": -2.6, "mania": -3.2, "irrational exuberance": -3.2,\n        "extreme greed": -2.6, "market rebound": 2.7, "rebound potential": 2.2, "safe haven": 1.6,\n        "economic recovery": 2.2, "bull market": 2.2, "bear market": -2.2, "inflation concerns": -1.1,\n        "deflation risk": -2.2, "market breadth": 1.1, "geopolitical stability": 2.2, "market resilience": 2.2,\n        "central bank intervention": 2.8, "monetary easing": 2.6, "fiscal stimulus": 2.4,\n        "market rotation": 1.8, "risk on": 2.3, "risk off": -2.3, "safe haven demand": 2.0,\n        "economic expansion": 2.1, "growth acceleration": 2.5, "policy uncertainty": -2.0,\n        "sector rotation": 1.9, "valuation expansion": 1.7, "valuation contraction": -1.7,\n        "market breadth expansion": 1.8, "liquidity surge": 2.4, "portfolio rebalancing": 1.5,\n        "technical breakout": 2.3, "short covering": 2.5, "liquidity crunch": -2.8,\n        "margin squeeze": -2.7, "deleveraging": -2.5, "credit expansion": 2.3,\n        "inflation easing": 2.9, "market stability": 2.4, "policy clarity": 2.3,\n        "market consolidation": 1.4, "volatility surge": -2.4, "liquidity expansion": 2.1,\n        "risk parity": 1.9, "portfolio diversification": 1.7, "safe haven rotation": 2.3,\n        "economic resilience": 2.5, "policy support": 2.6, "market confidence": 2.2,\n        "bullish momentum": 2.4, "bearish momentum": -2.4, "market breadth contraction": -1.7,\n        "valuation peak": -2.8, "valuation trough": 2.8, "liquidity contraction": -2.7,\n        "portfolio concentration": -1.8, "risk parity": 1.9, "inflation easing": 2.9,\n        "central bank easing": 3.0, "monetary stimulus": 2.8, "fiscal expansion": 2.6,\n        "geopolitical stability": 2.4, "market breadth expansion": 2.2, "liquidity surge": 2.5,\n        "technical breakout": 2.6, "short covering": 2.8, "liquidity crunch": -2.9,\n        "margin squeeze": -2.8, "deleveraging": -2.6, "credit expansion": 2.4,\n        "policy clarity": 2.5, "market stability": 2.6, "economic resilience": 2.7,\n        "bullish momentum": 2.5, "bearish momentum": -2.5, "market breadth contraction": -1.8,\n        "supply chain issues": -2.4, "interest rate pause": 2.3, "geopolitical tension": -2.3,\n        "market crash": -3.5, "equity plunge": -3.4, "liquidity freeze": -3.3, "debt ceiling": -3.1,\n        "rebound potential": 2.3, "risk parity": 1.9, "safe haven": 2.1, "central bank easing": 3.0,\n        "monetary stimulus": 2.8, "fiscal expansion": 2.6, "geopolitical stability": 2.4,\n        "market breadth expansion": 2.2, "liquidity surge": 2.5, "technical breakout": 2.6,\n        "short covering": 2.8, "liquidity crunch": -2.9, "margin squeeze": -2.8,\n        "deleveraging": -2.6, "credit expansion": 2.4, "policy clarity": 2.5,\n        "market stability": 2.6, "economic resilience": 2.7, "bullish momentum": 2.5,\n        "bearish momentum": -2.5, "market breadth contraction": -1.8, "supply chain issues": -2.4,\n        "interest rate pause": 2.3, "geopolitical tension": -2.3, "market crash": -3.5,\n        "equity plunge": -3.4, "liquidity freeze": -3.3, "debt ceiling": -3.1\n    }\n    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent", "unlikely", "avoid", "no signs of", "unlikely to", "lack", "absence", "never", "none", "neglect", "without", "lack of", "fail to", "struggle to", "prevent", "avoid", "unlikely", "neglect", "no longer", "never again", "no longer", "lack of", "fail to", "struggle to", "prevent", "avoid", "unlikely", "neglect", "no longer", "without any", "lack any", "fail any", "struggle any", "prevent any", "avoid any", "unlikely any", "neglect any", "no longer any", "lack of any", "fail of any", "struggle of any", "prevent of any", "avoid of any", "unlikely of any", "neglect of any", "doesn't", "doesn't show", "doesn't indicate", "doesn't suggest", "isn't showing", "isn't indicating", "isn't suggesting", "lacks", "fails to", "struggles to", "avoids", "prevents", "unlikely to", "avoiding", "lacking", "failing to", "struggling to", "preventing", "avoiding", "unlikely showing", "lacking any", "failing any", "struggling any", "preventing any", "avoiding any", "unlikely any"]\n    net_sentiment_score = 0.0\n    for keyword, weight in sentiment_keywords.items():\n        pattern = r'\\b' + re.escape(keyword) + r'\\b'\n        for match in re.finditer(pattern, context_lower):\n            pre_context = context_lower[max(0, match.start() - 100):match.start()]\n            post_context = context_lower[match.end():match.end() + 100]\n            is_negated = any(neg_word in pre_context for neg_word in negation_words)\n            if any(neg_word in post_context for neg_word in negation_words):\n                is_negated = not is_negated\n            if is_negated:\n                weight *= 0.1\n            recency_factor = 1.0 - (match.start() / max(1, len(context_lower)))\n            weight *= (1.0 + recency_factor * 0.5)\n            net_sentiment_score += -weight if is_negated else weight\n    return net_sentiment_score\n\ndef calculate_ichimoku(prices, conversion_period=9, base_period=26, leading_span_b_period=52, lagging_span_period=26):\n    if len(prices) < max(conversion_period, base_period, leading_span_b_period, lagging_span_period):\n        return None, None, None, None, None\n    prices_arr = np.array(prices, dtype=float)\n    conversion_line = (np.max(prices_arr[-conversion_period:]) + np.min(prices_arr[-conversion_period:])) / 2\n    base_line = (np.max(prices_arr[-base_period:]) + np.min(prices_arr[-base_period:])) / 2\n    leading_span_a = (conversion_line + base_line) / 2\n    leading_span_b = (np.max(prices_arr[-leading_span_b_period:]) + np.min(prices_arr[-leading_span_b_period:])) / 2\n    lagging_span = prices_arr[-lagging_span_period]\n    return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span\n\ndef decide(current_price, price_history, news_context):\n    context_lower = news_context.lower()\n    sentiment_score = calculate_sentiment_score(news_context)\n    all_prices = price_history + [current_price]\n    price_len = len(all_prices)\n\n    if price_len < 50:\n        return "HOLD"\n\n    sma_50 = np.mean(all_prices[-50:])\n    sma_200 = np.mean(all_prices[-200:]) if price_len >= 200 else None\n    \n    volatility_ratio = calculate_atr(all_prices, 20) / calculate_atr(all_prices, 50)\n    ema_short = 12 if volatility_ratio < 1.5 else 8\n    ema_long = 26 if volatility_ratio < 1.5 else 18\n    ema_signal = 9 if volatility_ratio < 1.5 else 6\n    \n    ema_12 = calculate_ema_series(all_prices, ema_short)[-1] if price_len >= ema_short else None\n    ema_26 = calculate_ema_series(all_prices, ema_long)[-1] if price_len >= ema_long else None\n    ema_9 = calculate_ema_series(all_prices, ema_signal)[-1] if price_len >= ema_signal else None\n    \n    rsi_period = 14 if volatility_ratio < 1.75 else 7\n    rsi = calculate_rsi(all_prices, rsi_period)\n    macd_line, signal_line, macd_hist_series = calculate_macd_series(all_prices)\n    short_atr = calculate_atr(all_prices, 20)\n    long_atr = calculate_atr(all_prices, 50)\n    roc_20 = calculate_roc(all_prices, 20)\n    donchian_high_30, donchian_low_30 = calculate_donchian_channel(all_prices, 30)\n    stochastic_oscillator = calculate_stochastic_oscillator(all_prices)\n    keltner_upper_band, keltner_lower_band = calculate_keltner_channel(all_prices)\n    bollinger_upper, bollinger_lower, bollinger_sma = calculate_bollinger_bands(all_prices)\n    ichimoku_conversion, ichimoku_base, ichimoku_a, ichimoku_b, ichimoku_lagging = calculate_ichimoku(all_prices)\n\n    if any(v is None for v in [sma_50, ema_12, ema_26, ema_9, rsi, short_atr, long_atr, roc_20, donchian_high_30, donchian_low_30, stochastic_oscillator]) or \\\n       macd_hist_series is None or len(macd_hist_series) < 2 or bollinger_upper is None or ichimoku_a is None:\n        return "HOLD"\n\n    macd_histogram = macd_hist_series[-1]\n    prev_macd_histogram = macd_hist_series[-2]\n    macd_hist_delta = macd_histogram - prev_macd_histogram\n    macd_hist_acceleration = macd_hist_delta - (macd_hist_series[-3] - macd_hist_series[-2]) if len(macd_hist_series) >= 3 else 0\n\n    is_high_volatility = volatility_ratio > 1.75\n    is_extreme_volatility = volatility_ratio > 2.0\n\n    is_long_term_downtrend = current_price < sma_200 if sma_200 is not None else False\n    is_crash_velocity = roc_20 < -18.0\n    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity or ("yield curve inversion" in context_lower) or "banking crisis" in context_lower or "sovereign debt" in context_lower or "financial crisis" in context_lower or "systemic risk" in context_lower or "market crash" in context_lower\n\n    is_deeply_oversold = rsi < 25\n    is_extreme_crash_velocity = roc_20 < -22.0\n    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold and current_price < donchian_low_30 and (short_atr > long_atr * 1.2) and current_price < keltner_lower_band and current_price > keltner_lower_band - (short_atr * 0.5)\n\n    if is_capitulation_candidate and macd_hist_delta > 0 and stochastic_oscillator < 15 and ema_12 > ema_26 and sentiment_score > -1.5 and macd_hist_acceleration > 0 and signal_line[-1] < macd_histogram and ichimoku_a > ichimoku_b and current_price > ichimoku_a and current_price > ichimoku_lagging:\n        return "BUY"\n\n    if is_crisis_regime:\n        is_recovering_from_oversold = rsi > 40 and macd_hist_delta > 0 and stochastic_oscillator > 85 and sentiment_score > -0.5 and ichimoku_a > ichimoku_b and current_price > ichimoku_a and current_price > ichimoku_lagging\n        if is_recovering_from_oversold:\n            return "BUY"\n        if macd_histogram < 0 or current_price < sma_50:\n            return "SELL"\n        return "HOLD"\n\n    base_stop_loss_factor = 0.88\n    if is_extreme_volatility:\n        stop_loss_factor = 0.78\n    elif is_high_volatility:\n        stop_loss_factor = 0.83\n    else:\n        stop_loss_factor = base_stop_loss_factor\n\n    atr_stop = keltner_lower_band + (short_atr * 0.5)\n    if current_price < atr_stop and current_price < donchian_high_30 * stop_loss_factor:\n        return "SELL"\n\n    is_primary_downtrend = current_price < sma_50\n    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0\n    is_sentiment_permissive_for_sell = sentiment_score < 2.0\n    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:\n        return "SELL"\n\n    is_momentum_fading = macd_hist_delta < 0\n    overbought_threshold = 85 if is_high_volatility else 82\n    is_extremely_overbought = rsi > overbought_threshold\n    if is_extremely_overbought and is_momentum_fading:\n        return "SELL"\n\n    is_primary_uptrend = current_price > sma_50 and (sma_200 is None or current_price > sma_200)\n    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0\n    is_not_overbought = rsi < (75 if is_high_volatility else 72)\n    is_sentiment_permissive_for_buy = sentiment_score > -2.5\n    is_sufficient_volatility = short_atr > (long_atr * 0.6)\n    is_price_in_keltner_channel = current_price > keltner_lower_band and current_price < keltner_upper_band\n    is_ema_crossover = ema_12 is not None and ema_26 is not None and ema_12 > ema_26\n    is_bollinger_in_range = current_price > bollinger_lower and current_price < bollinger_upper\n    is_ichimoku_cloud_positive = ichimoku_a > ichimoku_b and current_price > ichimoku_a\n    is_lagging_positive = current_price > ichimoku_lagging\n\n    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility and is_price_in_keltner_channel and is_bollinger_in_range and stochastic_oscillator > 30 and is_ema_crossover and macd_hist_acceleration > 0 and is_ichimoku_cloud_positive and is_lagging_positive:\n        return "BUY"\n\n    return "HOLD"	ba7209e71e93974a5faa1e9c9442048768e788569ccb50ef0ebe83bad04d608f	Initial human-written rule-based baseline strategy.	\N	2026-06-02 18:54:57.976135-04
\.


--
-- Name: lessons_learned lessons_learned_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lessons_learned
    ADD CONSTRAINT lessons_learned_pkey PRIMARY KEY (lesson_id);


--
-- Name: runs runs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_pkey PRIMARY KEY (run_id);


--
-- Name: strategies strategies_branch_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.strategies
    ADD CONSTRAINT strategies_branch_name_key UNIQUE (branch_name);


--
-- Name: strategies strategies_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.strategies
    ADD CONSTRAINT strategies_pkey PRIMARY KEY (strategy_id);


--
-- Name: idx_lessons_embedding_cosine; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lessons_embedding_cosine ON public.lessons_learned USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_lessons_run_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lessons_run_id ON public.lessons_learned USING btree (run_id);


--
-- Name: idx_lessons_strategy_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lessons_strategy_id ON public.lessons_learned USING btree (strategy_id);


--
-- Name: idx_runs_aroi; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_runs_aroi ON public.runs USING btree (efficiency_score_aroi);


--
-- Name: idx_runs_regime; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_runs_regime ON public.runs USING btree (regime);


--
-- Name: idx_runs_strategy_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_runs_strategy_id ON public.runs USING btree (strategy_id);


--
-- Name: idx_runs_total_return; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_runs_total_return ON public.runs USING btree (total_return_pct);


--
-- Name: idx_strategies_generation; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_strategies_generation ON public.strategies USING btree (generation);


--
-- Name: idx_strategies_parent_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_strategies_parent_id ON public.strategies USING btree (parent_strategy_id);


--
-- Name: lessons_learned lessons_learned_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lessons_learned
    ADD CONSTRAINT lessons_learned_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.runs(run_id) ON DELETE CASCADE;


--
-- Name: lessons_learned lessons_learned_strategy_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lessons_learned
    ADD CONSTRAINT lessons_learned_strategy_id_fkey FOREIGN KEY (strategy_id) REFERENCES public.strategies(strategy_id) ON DELETE CASCADE;


--
-- Name: runs runs_strategy_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.runs
    ADD CONSTRAINT runs_strategy_id_fkey FOREIGN KEY (strategy_id) REFERENCES public.strategies(strategy_id) ON DELETE CASCADE;


--
-- Name: strategies strategies_parent_strategy_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.strategies
    ADD CONSTRAINT strategies_parent_strategy_id_fkey FOREIGN KEY (parent_strategy_id) REFERENCES public.strategies(strategy_id) ON DELETE SET NULL;


--
-- PostgreSQL database dump complete
--

\unrestrict MVdTOiRhbOCTSLqLasRUrh6YEHiDEaMGPVcazE8gt89ER2ptbPUVSrkJNgpLKLs

