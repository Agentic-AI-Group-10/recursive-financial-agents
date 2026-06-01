#!/usr/bin/env python3
# ==============================================================================
# Cloud API Connectors (Google AI Studio & DeepInfra)
# Location: /home/ow9800/recursive-financial-agents/engine/api_connectors.py
# ==============================================================================
# This module provides robust, zero-dependency REST clients for query execution
# against Google AI Studio (Gemini 1.5 Flash) and DeepInfra (Llama-3/Mistral).
# It features automated fallback to standard urllib, precise token accounting,
# real-world pricing cost estimators, and rate-limiting pacing guards.

import os
import time
import json
import urllib.request
import urllib.error

# Real-world pricing definitions (USD per 1 Million tokens)
PRICING_MODELS = {
    "gemini-2.5-flash": {
        "input_cost_per_m": 0.075,
        "output_cost_per_m": 0.300
    },
    "gemini-2.5-pro": {
        "input_cost_per_m": 1.250,
        "output_cost_per_m": 5.000
    },
    "gemini-1.5-flash": {
        "input_cost_per_m": 0.075,
        "output_cost_per_m": 0.300
    },
    "gemini-1.5-pro": {
        "input_cost_per_m": 1.250,
        "output_cost_per_m": 5.000
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "input_cost_per_m": 0.055,
        "output_cost_per_m": 0.055
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "input_cost_per_m": 0.590,
        "output_cost_per_m": 0.790
    }
}

def load_env_variables(filepath=".env"):
    """Manually parses a local .env file to load variables into os.environ.
    This avoids requiring the third-party python-dotenv package.
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

# Load environment variables on import
load_env_variables()


class APIConnector:
    def __init__(self, provider="google", model=None, pacing_delay=4.0):
        """
        Initializes the API connector.
        
        Args:
            provider (str): 'google' (AI Studio) or 'deepinfra' (OpenAI-compatible)
            model (str): Optional. Model identifier (defaults based on provider).
            pacing_delay (float): Seconds to sleep between calls to avoid rate limits (e.g. 15 RPM on Gemini Free Tier).
        """
        self.provider = provider.lower()
        self.pacing_delay = pacing_delay
        self.last_call_time = 0.0
        
        # Telemetry State
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_calls = 0

        # Provider configurations
        if self.provider == "google":
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.model = model or "gemini-2.5-flash"
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set. Check your .env file.")
        elif self.provider == "deepinfra":
            self.api_key = os.getenv("DEEPINFRA_API_KEY")
            self.model = model or "meta-llama/Meta-Llama-3-8B-Instruct"
            if not self.api_key:
                raise ValueError("DEEPINFRA_API_KEY environment variable is not set. Check your .env file.")
        else:
            raise ValueError(f"Unknown API provider: {provider}. Choose 'google' or 'deepinfra'.")

    def _pace_request(self):
        """Enforces a physical sleep delay between requests to guarantee rate compliance."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.pacing_delay:
            sleep_duration = self.pacing_delay - elapsed
            time.sleep(sleep_duration)
        self.last_call_time = time.time()

    def _post_request(self, url, headers, data):
        """Executes a POST request using Python's standard library urllib.request."""
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP Error {e.code}: {e.reason}. Response details: {error_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to execute API call: {e}")

    def query(self, prompt, temperature=0.2, max_tokens=500):
        """
        Sends a text prompt to the configured model.
        
        Args:
            prompt (str): Text prompt to feed the LLM.
            temperature (float): Model sampling temperature.
            max_tokens (int): Limits maximum output generation length (Default: 500, recommended for thinking models).
            
        Returns:
            dict: Structured decision output containing:
                - 'text': Content string returned.
                - 'tokens_in': Tracked input token count.
                - 'tokens_out': Tracked output token count.
                - 'cost': Computed pricing in USD.
        """
        self._pace_request()
        self.total_calls += 1
        
        if self.provider == "google":
            # 1. Google AI Studio REST Endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }
            
            response = self._post_request(url, headers, payload)
            
            # Extract content text
            try:
                # Handle cases where thinking consumed the entire token budget, leaving no content parts
                candidate = response["candidates"][0]
                if candidate.get("finishReason") == "MAX_TOKENS" and "parts" not in candidate.get("content", {}):
                    text_content = "[ERROR: Model reached output token limit during thinking phase. Try increasing max_tokens.]"
                else:
                    text_content = candidate["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(f"Failed to parse Gemini response payload: {json.dumps(response)}")
                
            # Token accounting (Gemini API provides exact metadata or fallback estimate)
            # Standard estimation: ~1.3 tokens per word if API metadata is missing
            tokens_in = int(len(prompt.split()) * 1.3)
            tokens_out = int(len(text_content.split()) * 1.3)
            
            # If Google API metadata exists, grab it
            if "usageMetadata" in response:
                tokens_in = response["usageMetadata"].get("promptTokenCount", tokens_in)
                tokens_out = response["usageMetadata"].get("candidatesTokenCount", tokens_out)

        elif self.provider == "deepinfra":
            # 2. DeepInfra OpenAI-Compatible REST Endpoint
            url = "https://api.deepinfra.com/v1/openai/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = self._post_request(url, headers, payload)
            
            try:
                text_content = response["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(f"Failed to parse DeepInfra response payload: {json.dumps(response)}")
                
            # Grab exact OpenAI usage statistics
            tokens_in = response.get("usage", {}).get("prompt_tokens", int(len(prompt.split()) * 1.3))
            tokens_out = response.get("usage", {}).get("completion_tokens", int(len(text_content.split()) * 1.3))

        # 3. Cost Estimations
        pricing = PRICING_MODELS.get(self.model, {"input_cost_per_m": 0.055, "output_cost_per_m": 0.055})
        cost = (tokens_in / 1000000.0 * pricing["input_cost_per_m"]) + \
               (tokens_out / 1000000.0 * pricing["output_cost_per_m"])
               
        # Cumulative telemetry tracking
        self.total_input_tokens += tokens_in
        self.total_output_tokens += tokens_out
        self.total_cost_usd += cost
        
        return {
            "text": text_content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost
        }


# Quick Verification Sandbox
if __name__ == "__main__":
    print("Testing API Connector implementations locally...")
    
    # 1. Test Google AI Studio
    try:
        google_client = APIConnector(provider="google")
        print("\nSending Test Prompt to Gemini 1.5 Flash...")
        res = google_client.query("Print 'SPY_BUY' or 'SPY_SELL' followed by a one-sentence reasoning for S&P 500.", max_tokens=500)
        print(f"Response: {res['text']}")
        print(f"Tokens In: {res['tokens_in']} | Tokens Out: {res['tokens_out']} | Cost: ${res['cost']:.6f}")
    except Exception as e:
        print(f"⚠️ Google AI Studio Test Postponed/Failed: {e}")
        
    # 2. Test DeepInfra (If key provided)
    if os.getenv("DEEPINFRA_API_KEY"):
        try:
            di_client = APIConnector(provider="deepinfra")
            print("\nSending Test Prompt to DeepInfra Llama-3-8B...")
            res = di_client.query("Print 'SPY_BUY' or 'SPY_SELL' followed by a one-sentence reasoning.", max_tokens=500)
            print(f"Response: {res['text']}")
            print(f"Tokens In: {res['tokens_in']} | Tokens Out: {res['tokens_out']} | Cost: ${res['cost']:.6f}")
        except Exception as e:
            print(f"⚠️ DeepInfra Test Failed: {e}")
