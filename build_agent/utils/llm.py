# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 


import os
import time
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Global API key, set from main.py via set_api_key()
_amd_api_key = None

def set_api_key(api_key):
    """Set the AMD LLM API Gateway key globally."""
    global _amd_api_key
    _amd_api_key = api_key


def _get_api_key():
    """Resolve API key from global state or environment variable."""
    if _amd_api_key:
        return _amd_api_key
    env_key = os.environ.get("AMD_LLM_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "No AMD LLM API key found. Provide --api-key CLI argument "
        "or set AMD_LLM_API_KEY environment variable."
    )


def _call_openai(model, messages, temperature, n, max_tokens):
    """Call OpenAI-compatible API (GPT models)."""
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    return content, usage


@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(5))
def _call_amd_claude(model, messages, temperature, max_tokens):
    """Call Claude via AMD's internal LLM API Gateway."""
    api_key = _get_api_key()
    server = "https://llm-api.amd.com/claude3"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
    }

    max_tokens = min(max_tokens, 16000)

    body = {
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_completion_tokens": max_tokens,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        url=f"{server}/{model}/chat/completions",
        json=body,
        headers=headers,
        timeout=600,
    )

    if response.status_code != 200:
        raise ValueError(f"AMD LLM API returned status {response.status_code}: {response.text}")

    result = response.json()

    # Extract content -- gateway may return Anthropic or OpenAI format
    if "content" in result and len(result["content"]) > 0:
        content = result["content"][0]["text"]
    elif "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unexpected response format: {result}")

    # Extract usage if available
    if "usage" in result:
        u = result["usage"]
        usage = {
            "prompt_tokens": u.get("prompt_tokens", u.get("input_tokens", 0)),
            "completion_tokens": u.get("completion_tokens", u.get("output_tokens", 0)),
            "total_tokens": u.get("total_tokens",
                                  u.get("prompt_tokens", u.get("input_tokens", 0))
                                  + u.get("completion_tokens", u.get("output_tokens", 0))),
        }
    else:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return content, usage


def get_llm_response(model: str, messages, temperature=0.0, n=1, max_tokens=4096):
    max_retry = 5
    count = 0
    while count < max_retry:
        try:
            if "claude" in model.lower():
                content, usage = _call_amd_claude(model, messages, temperature, max_tokens)
            else:
                content, usage = _call_openai(model, messages, temperature, n, max_tokens)
            return [content], usage
        except Exception as e:
            print(f"Error: {e}")
            count += 1
            time.sleep(3)
    return None, None