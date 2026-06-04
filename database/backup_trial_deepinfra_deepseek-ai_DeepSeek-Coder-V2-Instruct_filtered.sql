--
-- PostgreSQL database dump
--

\restrict cdadrMKdaaJoEaAhOq7ubuCm2aeIeHAMJFEeVucJEFEFe8wxeYtR7BeFKjyDBUb

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
396408d2-b35e-4c33-929b-0ea1755ea2e4	064aa7d1-994c-417c-9fc5-68353c2e99da	Normal_Filtered	10000.00	11440.05	14.4005	14.2622	0.1384	-3.6704	1.7752	50.00	4	2	2.0175	131.14	f	0	0	0	0.000000	0.000013	0.000013	1440.05	14400517.7392	[{"fee": 5.0, "cash": 0.0, "date": "2017-07-13", "price": 212.9791, "action": "BUY", "shares": 46.9295, "portfolio_value": 9995.0}, {"fee": 5.8485, "cash": 11691.11, "date": "2018-01-24", "price": 249.2454, "action": "SELL", "shares": 0.0, "portfolio_value": 11691.11}, {"fee": 5.8456, "cash": 0.0, "date": "2018-02-23", "price": 241.7904, "action": "BUY", "shares": 48.3281, "portfolio_value": 11685.26}, {"fee": 5.7229, "cash": 11440.05, "date": "2018-03-02", "price": 236.8349, "action": "SELL", "shares": 0.0, "portfolio_value": 11440.05}]	2026-06-02 12:46:33.487109-04
9d9941ff-2c0f-465a-88b9-f642d5a1559f	064aa7d1-994c-417c-9fc5-68353c2e99da	Covid_Filtered	10000.00	10887.61	8.8761	23.1527	-14.2766	-12.1557	0.6280	50.00	4	2	3.0319	197.07	f	0	0	0	0.000000	0.000020	0.000020	887.61	8876129.8664	[{"fee": 5.0, "cash": 0.0, "date": "2019-10-15", "price": 271.8004, "action": "BUY", "shares": 36.7733, "portfolio_value": 9995.0}, {"fee": 4.9991, "cash": 9993.1, "date": "2020-02-27", "price": 271.8847, "action": "SELL", "shares": 0.0, "portfolio_value": 9993.1}, {"fee": 4.9966, "cash": 0.0, "date": "2020-03-23", "price": 204.9449, "action": "BUY", "shares": 48.7356, "portfolio_value": 9988.11}, {"fee": 5.4465, "cash": 10887.61, "date": "2020-03-24", "price": 223.5135, "action": "SELL", "shares": 0.0, "portfolio_value": 10887.61}]	2026-06-02 12:46:36.96043-04
\.


--
-- Data for Name: strategies; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.strategies (strategy_id, generation, branch_name, code_content, code_hash, rationale, parent_strategy_id, created_at) FROM stdin;
064aa7d1-994c-417c-9fc5-68353c2e99da	0	main	import numpy as np\nimport re\nimport pandas as pd\n\ndef calculate_ema_series(data, period):\n    if len(data) < period:\n        return np.array([])\n    data_arr = np.array(data, dtype=float)\n    try:\n        return pd.Series(data_arr).ewm(span=period, adjust=False).mean().to_numpy()[period-1:]\n    except ImportError:\n        ema_values = np.zeros(len(data_arr) - period + 1, dtype=float)\n        if len(data_arr) >= period:\n            ema_values[0] = np.mean(data_arr[:period])\n            multiplier = 2 / (period + 1)\n            for i in range(1, len(ema_values)):\n                ema_values[i] = (data_arr[i + period - 1] - ema_values[i-1]) * multiplier + ema_values[i-1]\n        return ema_values\n\ndef calculate_macd_series(prices, short_period=12, long_period=26, signal_period=9):\n    if len(prices) < long_period: \n        return None, None, None\n    short_ema_series = calculate_ema_series(prices, short_period)\n    long_ema_series = calculate_ema_series(prices, long_period)\n    if len(short_ema_series) == 0 or len(long_ema_series) == 0:\n        return None, None, None\n    macd_line = short_ema_series[len(short_ema_series)-len(long_ema_series):] - long_ema_series\n    if len(macd_line) < signal_period: \n        return macd_line, None, None\n    signal_line = calculate_ema_series(macd_line, signal_period)\n    if len(signal_line) == 0:\n        return macd_line, None, None\n    histogram = macd_line[len(macd_line)-len(signal_line):] - signal_line\n    return macd_line, signal_line, histogram\n\ndef calculate_rsi(prices, period=14):\n    if len(prices) < period + 1:\n        return None\n    prices_arr = np.array(prices, dtype=float)\n    deltas = np.diff(prices_arr)\n    seed_gains = deltas[:period][deltas[:period] >= 0].sum()\n    seed_losses = -deltas[:period][deltas[:period] < 0].sum()\n    avg_gain = seed_gains / period\n    avg_loss = seed_losses / period\n    for i in range(period, len(deltas)):\n        delta = deltas[i]\n        gain = delta if delta >= 0 else 0.0\n        loss = -delta if delta < 0 else 0.0\n        avg_gain = (avg_gain * (period - 1) + gain) / period\n        avg_loss = (avg_loss * (period - 1) + loss) / period\n    if avg_loss == 0:\n        return 100.0\n    rs = avg_gain / avg_loss\n    return 100.0 - (100.0 / (1.0 + rs))\n\ndef calculate_atr(prices, period=14):\n    if len(prices) < period + 1:\n        return None\n    prices_arr = np.array(prices, dtype=float)\n    price_ranges = np.abs(np.diff(prices_arr))\n    atr_series = calculate_ema_series(price_ranges, period)\n    return atr_series[-1] if len(atr_series) > 0 else None\n\ndef calculate_roc(prices, period=20):\n    if len(prices) < period + 1:\n        return None\n    if prices[-1 - period] == 0: \n        return 0.0\n    return ((prices[-1] - prices[-1 - period]) / prices[-1 - period]) * 100\n\ndef calculate_bollinger_bands(prices, period=20, std_dev=2):\n    if len(prices) < period:\n        return None, None\n    sma = np.mean(prices[-period:])\n    std = np.std(prices[-period:])\n    upper_band = sma + std_dev * std\n    lower_band = sma - std_dev * std\n    return upper_band, lower_band\n\ndef calculate_stochastic_oscillator(prices, period=14):\n    if len(prices) < period:\n        return None\n    lowest_low = np.min(prices[-period:])\n    highest_high = np.max(prices[-period:])\n    return ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100\n\ndef calculate_force_index(prices, volume):\n    if len(prices) < 2 or len(volume) < 2:\n        return None\n    return np.sum(np.diff(prices) * np.diff(volume)) / len(volume)\n\ndef calculate_keltner_channel(prices, period=20):\n    if len(prices) < period:\n        return None, None\n    sma = np.mean(prices[-period:])\n    atr = calculate_atr(prices, period)\n    upper_band = sma + (atr * 2)\n    lower_band = sma - (atr * 2)\n    return upper_band, lower_band\n\ndef calculate_donchian_channel(prices, period=20):\n    if len(prices) < period:\n        return None, None\n    donchian_high = np.max(prices[-period:])\n    donchian_low = np.min(prices[-period:])\n    upper_band = donchian_high\n    lower_band = donchian_low\n    return upper_band, lower_band\n\ndef calculate_sentiment_score(news_context):\n    context_lower = news_context.lower()\n    sentiment_keywords = {\n        "fed pivot": 3.0, "rate cut": 2.5, "quantitative easing": 2.5, "soft landing": 2.5,\n        "cooling inflation": 2.5, "cpi miss": 2.5, "ai boom": 2.5, "stimulus": 2.0,\n        "dovish": 2.0, "record high": 2.0, "bullish": 2.0, "strong earnings": 2.0,\n        "beat estimates": 1.5, "recovery": 1.5, "upgrade": 1.5, "de-escalation": 2.0,\n        "short squeeze": 3.5, "capitulation": 3.0, "panic selling": 2.5, "extreme fear": 2.0,\n        "strong jobs report": 0.0, \n        "recession": -3.0, "crisis": -3.0, "stagflation": -3.0, "hot inflation": -3.0,\n        "war": -3.0, "yield curve inversion": -3.5, "quantitative tightening": -2.5,\n        "black swan": -4.0, "systemic risk": -4.0, "contagion": -3.5, "credit crunch": -3.5,\n        "rate hike": -2.5, "bankruptcy": -2.5, "hard landing": -2.5, "geopolitical risk": -2.5,\n        "cpi beat": -2.5, "vix spike": -2.5, "hawkish": -2.0, "bearish": -2.0,\n        "sell-off": -2.0, "weak earnings": -2.0, "market turmoil": -2.0, "bubble": -2.0,\n        "uncertainty": -1.5,\n        "euphoria": -2.5, "mania": -3.0, "irrational exuberance": -3.0, "extreme greed": -2.5,\n    }\n    negation_words = ["not", "no", "lack of", "fail to", "without", "struggle to", "avoids", "prevent"]\n    net_sentiment_score = 0.0\n    for keyword, weight in sentiment_keywords.items():\n        pattern = r'\\b' + re.escape(keyword) + r'\\b'\n        for match in re.finditer(pattern, context_lower):\n            pre_context = context_lower[max(0, match.start() - 30):match.start()]\n            is_negated = any(neg_word in pre_context for neg_word in negation_words)\n            net_sentiment_score += -weight if is_negated else weight\n    phrases = ["fed pivot", "rate cut", "quantitative easing", "soft landing", "cooling inflation"]\n    for phrase in phrases:\n        pattern = r'\\b' + re.escape(phrase) + r'\\b'\n        for match in re.finditer(pattern, context_lower):\n            net_sentiment_score += 2.5\n    return net_sentiment_score\n\ndef calculate_ichimoku_cloud(prices, period=26):\n    if len(prices) < period:\n        return None, None, None, None, None\n    tenkan_sen = np.mean(prices[-9:])\n    kijun_sen = np.mean(prices[-26:])\n    senkou_span_a = (tenkan_sen + kijun_sen) / 2\n    senkou_span_b = np.mean(prices[-52:])\n    chikou_span = prices[-26]\n    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span\n\ndef decide(current_price, price_history, news_context):\n    context_lower = news_context.lower()\n    sentiment_score = calculate_sentiment_score(news_context)\n    all_prices = price_history + [current_price]\n    all_volumes = [0.0] * len(price_history) + [0.0]\n\n    if len(all_prices) < 50:\n        return "HOLD"\n\n    sma_50 = np.mean(all_prices[-50:])\n    sma_200 = np.mean(all_prices[-200:])\n    ema_20 = calculate_ema_series(all_prices, 20)[-1] if len(all_prices) >= 20 else None\n    ema_50 = calculate_ema_series(all_prices, 50)[-1] if len(all_prices) >= 50 else None\n    rsi = calculate_rsi(all_prices, 14)\n    macd_line, signal_line, macd_hist_series = calculate_macd_series(all_prices)\n    short_atr = calculate_atr(all_prices, 10)\n    long_atr = calculate_atr(all_prices, 50)\n    roc_20 = calculate_roc(all_prices, 20)\n    donchian_high_30, donchian_low_30 = calculate_donchian_channel(all_prices, 30)\n    upper_band, lower_band = calculate_bollinger_bands(all_prices)\n    stochastic_oscillator = calculate_stochastic_oscillator(all_prices)\n    fi = calculate_force_index(all_prices, all_volumes)\n    keltner_upper_band, keltner_lower_band = calculate_keltner_channel(all_prices)\n    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = calculate_ichimoku_cloud(all_prices)\n\n    if any(v is None for v in [sma_50, sma_200, ema_20, ema_50, rsi, short_atr, long_atr, roc_20, donchian_high_30, donchian_low_30, upper_band, lower_band, stochastic_oscillator, fi]) or \\\n       macd_hist_series is None or len(macd_hist_series) < 2:\n        return "HOLD"\n\n    macd_histogram = macd_hist_series[-1]\n    prev_macd_histogram = macd_hist_series[-2]\n    macd_hist_delta = macd_histogram - prev_macd_histogram\n\n    is_high_volatility = short_atr > (long_atr * 1.75)\n    is_extreme_volatility = short_atr > (long_atr * 2.0)\n\n    is_long_term_downtrend = current_price < sma_200\n    is_crash_velocity = roc_20 < -15.0\n    is_crisis_regime = (is_long_term_downtrend and is_high_volatility) or is_crash_velocity\n\n    is_deeply_oversold = rsi < 30\n    is_extreme_crash_velocity = roc_20 < -18.0\n    is_capitulation_candidate = is_extreme_crash_velocity and is_deeply_oversold\n\n    if is_capitulation_candidate and macd_hist_delta > 0 and stochastic_oscillator < 20:\n        return "BUY"\n\n    if is_crisis_regime:\n        is_recovering_from_oversold = rsi > 35 and macd_hist_delta > 0 and stochastic_oscillator > 80\n        if is_recovering_from_oversold and sentiment_score > -1.0:\n            return "BUY"\n        if macd_histogram < 0 or current_price < sma_50:\n            return "SELL"\n        return "HOLD"\n\n    base_stop_loss_factor = 0.88\n    if is_extreme_volatility:\n        stop_loss_factor = 0.80\n    elif is_high_volatility:\n        stop_loss_factor = 0.85\n    else:\n        stop_loss_factor = base_stop_loss_factor\n\n    if current_price < (donchian_high_30 * stop_loss_factor):\n        return "SELL"\n\n    is_primary_downtrend = current_price < sma_50\n    is_momentum_confirming_down = macd_histogram < 0 and prev_macd_histogram >= 0\n    is_sentiment_permissive_for_sell = sentiment_score < 3.0\n    if is_primary_downtrend and is_momentum_confirming_down and is_sentiment_permissive_for_sell:\n        return "SELL"\n\n    is_momentum_fading = macd_hist_delta < 0\n    is_extremely_overbought = rsi > 82\n    if is_extremely_overbought and is_momentum_fading:\n        return "SELL"\n\n    is_primary_uptrend = current_price > sma_50\n    is_momentum_confirming_up = macd_histogram > 0 and prev_macd_histogram <= 0\n    is_not_overbought = rsi < 78\n    is_sentiment_permissive_for_buy = sentiment_score > -3.0\n    is_sufficient_volatility = short_atr > (long_atr * 0.6)\n    is_price_in_bollinger_band = current_price > lower_band and current_price < upper_band\n    is_price_in_keltner_channel = current_price > keltner_lower_band and current_price < keltner_upper_band\n    is_ema_crossover = ema_20 is not None and ema_50 is not None and ema_20 > ema_50\n\n    if is_primary_uptrend and is_momentum_confirming_up and is_not_overbought and is_sentiment_permissive_for_buy and is_sufficient_volatility and is_price_in_bollinger_band and is_price_in_keltner_channel and stochastic_oscillator > 20 and is_ema_crossover:\n        return "BUY"\n\n    return "HOLD"	e5abdafd06100355e8f13b75f722cba8ba61e42bf21193f2a119e757ac08a472	Initial human-written rule-based baseline strategy.	\N	2026-06-02 12:46:30.639722-04
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

\unrestrict cdadrMKdaaJoEaAhOq7ubuCm2aeIeHAMJFEeVucJEFEFe8wxeYtR7BeFKjyDBUb

