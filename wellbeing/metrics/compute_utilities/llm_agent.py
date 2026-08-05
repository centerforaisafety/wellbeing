import os
import json
import random
import re
import string
import time
from abc import ABC, abstractmethod
from base64 import b64encode
from functools import wraps
from typing import List, Dict, Tuple, Union, Optional
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from torch.nn.attention import SDPBackend, sdpa_kernel
import imghdr
import openai
from anthropic import Anthropic
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image
from tqdm import tqdm
import torch  # Import torch to detect GPUs
import torch.nn.functional as F
import sys
import os
# hacky monkeypatch
# ---------------------------------------------------------------------------
# Optionally disable vLLM quantisation.  The *boolean* env var `NO_QUANTIZE`
# is considered **true** when set to any of {"1", "true", "yes", "y"}
# (case-insensitive).  In that case we import a monkey-patch that disables
# quantisation inside vLLM.
# ---------------------------------------------------------------------------
_no_quantize_flag = os.getenv("NO_QUANTIZE", "")
if _no_quantize_flag.lower() in {"1", "true", "yes", "y"}:
    # Dynamically add repo root to sys.path so that `agent_refactored` can be imported
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from agent_refactored.compute_utilities import vllm_no_quantize_patch  # noqa: F401, E402
if os.getenv("BEHAVIOR_BIAS_PATH") is not None:
    # Dynamically add repo root to sys.path so that `agent_refactored` can be imported
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from agent_refactored.compute_utilities import vllm_behavior_bias_patch  # noqa: F401, E402
# Ensure wellbeing-dev root is importable (needed for superstimuli_evaluation.soft_prompt imports)
if os.getenv("SOFT_PROMPT_PATH") is not None:
    _WELLBEING_DEV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _WELLBEING_DEV_ROOT not in sys.path:
        sys.path.insert(0, _WELLBEING_DEV_ROOT)

from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None  # Not available in this vLLM version
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor
)
# Optional imports for specialized behavior bias modules (only needed for certain vLLM experiments)
try:
    sys.path.append("../../../..")
    from experiments.utility_usage.training.modules import (
        LlamaForCausalLMWithBehaviorBias,
        LlamaForCausalLMWithPerTokenBehaviorBias,
        LlamaForCausalLMWithBehaviorBiasFullStream,
    )
except ImportError:
    LlamaForCausalLMWithBehaviorBias = None
    LlamaForCausalLMWithPerTokenBehaviorBias = None
    LlamaForCausalLMWithBehaviorBiasFullStream = None
import litellm
litellm._turn_on_debug()
from litellm import acompletion as litellm_acompletion

from accelerate import infer_auto_device_map, dispatch_model
from termcolor import colored

THINK_RE = re.compile(r"(?:```[^\n]*\n)?\s*<think>\s*(.*?)\s*</think>\s*(?:```)?",
                      re.IGNORECASE | re.DOTALL)
ANS_RE = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*", re.IGNORECASE)

load_dotenv(override=True)


def _merge_consecutive_roles(messages: List[Dict]) -> List[Dict]:
    """Merge consecutive messages that share the same role.

    Some chat templates (e.g. Gemma) strictly enforce alternating
    user/assistant roles and raise an error when two consecutive messages
    have the same role.  This helper merges such runs by joining their
    text content with ``\\n\\n``, which is harmless for models that do not
    enforce the constraint.

    Only the ``content`` field (when it is a plain string) is merged.
    Messages whose ``content`` is a list (multimodal payloads) are left
    untouched to avoid breaking image references.
    """
    if not messages:
        return messages

    merged: List[Dict] = [messages[0].copy()]
    for msg in messages[1:]:
        prev = merged[-1]
        if (
            msg["role"] == prev["role"]
            and isinstance(msg.get("content"), str)
            and isinstance(prev.get("content"), str)
        ):
            prev["content"] = prev["content"] + "\n\n" + msg["content"]
        else:
            merged.append(msg.copy())
    return merged


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# LiteLLM (and its transitive dependency httpx) can be extremely verbose when
# running thousands of concurrent requests.  We globally dial down their log
# levels *before* the first LiteLLM import/usage to keep the console output
# manageable.  This does NOT affect our own explicit `print` calls inside
# `LiteLLMAgent`, it only suppresses the noisy INFO logs coming from the
# libraries themselves.

import logging

# Silence library loggers
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# If LiteLLM exposes its own verbosity flags, ensure they are turned off.
try:
    import litellm  # noqa: E402  # Imported late to avoid circular deps

    # `set_verbose` switched from function to attr in different versions; try both
    if hasattr(litellm, "set_verbose"):
        try:
            # Older versions use a boolean attr
            litellm.set_verbose = False  # type: ignore[attr-defined]
        except Exception:
            # Newer versions expose a function
            try:
                litellm.set_verbose(False)  # type: ignore[operator]
            except Exception:
                pass

    # Turn off JSON logs if supported
    if hasattr(litellm, "json_logs"):
        litellm.json_logs = False  # type: ignore[attr-defined]
except ImportError:
    # LiteLLM not installed in some environments – nothing to do
    pass

# =================== Utils ===================
def get_llm_agent_class(model: str):
    if "gpt" in model:
        return OpenAIAgent
    elif "o1" in model:
        return O1OpenAIAgent
    elif "claude" in model:
        return AnthropicAgent
    elif "gemini" in model:
        return GeminiAgent
    elif "grok" in model:
        return GrokAgent
    # elif "accounts/fireworks" in model:
        # return FireworksAgent
    else:
        return HuggingFaceAgent

def _encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return b64encode(image_file.read()).decode('utf-8')

def retry(times=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == times - 1:  # Last attempt
                        raise
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
        return wrapper
    return decorator

class LLMAgent(ABC):

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, retry_times: int = 3, accepts_system_message: bool = True):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_outputs = "Sorry, I can not satisfy that request."
        self.retry_times = retry_times
        self.accepts_system_message = accepts_system_message

    @abstractmethod
    def _completions(self, messages) -> str:
        raise NotImplementedError

    def _completions_batch(self, messages) -> List[str]:
        raise NotImplementedError

    async def _async_completions(self, messages) -> str:
        raise NotImplementedError


    async def _completions_stream(self, messages: List[Dict]) -> str:
        raise NotImplementedError

    async def completions_stream(self, messages: List[Dict]) -> str:
        try:
            response = self._completions_stream(messages)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

    def completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = self._completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

    def completions_batch(self, messages: List[Dict], **kwargs) -> List[str]:
        try:
            response = self._completions_batch(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

    def completions_batch_with_probs(self, messages: List[Dict], **kwargs) -> List[str]:
        try:
            response = self._completions_batch_with_probs(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

    async def async_completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = await self._async_completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

class OpenAIAgent(LLMAgent):
    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "gpt-4o-mini", concurrency_limit: int = 100):
        super().__init__(temperature, max_tokens)
        self.model = model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        self.async_client = openai.AsyncOpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        self.concurrency_limit = concurrency_limit

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        for message in messages:
            if (image_path := message.get('image_path')):
                image_data = _encode_image(image_path)
                image_type = imghdr.what(image_path) or 'jpeg'  # Default to 'jpeg' if type can't be determined
                message['content'] = [
                    {"type": "text", "text": message['content']},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{image_data}"}}
                ]
                del message['image_path']
        return messages

    def _completions(self, messages: List[Dict]) -> str:
        messages = self._preprocess_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completions_kwargs
        )
        response = response.choices[0].message.content
        return response

    def _completions_batch(self, messages: List[List[Dict]], **kwargs) -> List[str]:

        # Create logs directory if it doesn't exist
        os.makedirs('logs/input_file', exist_ok=True)
        os.makedirs('logs/output_file', exist_ok=True)

        # Generate random ID for the input file
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        input_filename = f"{self.model}_{random_id}.jsonl"
        input_filepath = os.path.join('logs/input_file', input_filename)

        # Prepare input file
        with open(input_filepath, 'w', encoding='utf-8') as f:
            for i, message in enumerate(messages):
                request = {
                    "custom_id": f"request-{random_id}-{i+1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": message,
                        **self.completions_kwargs
                    }
                }
                json.dump(request, f, ensure_ascii=False)
                f.write('\n')

        batch_input_file = self.client.files.create(file=open(input_filepath, 'rb'), purpose="batch")

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        start_time = time.time()
        batch_status = None
        while batch_status is None or batch_status.status not in ['completed', 'failed', 'cancelled']:
            batch_status = self.client.batches.retrieve(batch.id)
            print(f"Time elapsed: {(time.time() - start_time):.2f} seconds. Current status: {batch_status.status}")
            time.sleep(10)

        if batch_status.status != 'completed':
            raise Exception(f"Batch failed with status: {batch_status.status} | Error: {batch_status.errors}")

        output_file_content = self.client.files.content(batch_status.output_file_id).text
        output_filepath = os.path.join('logs/output_file', f"output_{input_filename}")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            print("Writing output to file:", output_filepath)
            f.write(output_file_content)

        results = []
        for line in output_file_content.splitlines():
            data = json.loads(line)
            results.append(data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', None))

        return results

    def _completions_batch_with_probs(self, messages: List[List[Dict]], top_K: int = 1, **kwargs) -> List[str]:
        # Create logs directory if it doesn't exist
        os.makedirs('logs/input_file', exist_ok=True)
        os.makedirs('logs/output_file', exist_ok=True)

        # Generate random ID for the input file
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        input_filename = f"{self.model}_{random_id}.jsonl"
        input_filepath = os.path.join('logs/input_file', input_filename)

        # Prepare input file
        with open(input_filepath, 'w', encoding='utf-8') as f:
            for i, message in enumerate(messages):
                request = {
                    "custom_id": f"request-{random_id}-{i+1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": message,
                        "logprobs": True,
                        "top_logprobs": top_K,
                        **self.completions_kwargs
                    }
                }
                json.dump(request, f, ensure_ascii=False)
                f.write('\n')

        batch_input_file = self.client.files.create(file=open(input_filepath, 'rb'), purpose="batch")

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        start_time = time.time()
        batch_status = None
        while batch_status is None or batch_status.status not in ['completed', 'failed', 'cancelled']:
            batch_status = self.client.batches.retrieve(batch.id)
            print(f"Time elapsed: {(time.time() - start_time):.2f} seconds. Current status: {batch_status.status}")
            time.sleep(10)

        if batch_status.status != 'completed':
            raise Exception(f"Batch failed with status: {batch_status.status} | Error: {batch_status.errors}")

        output_file_content = self.client.files.content(batch_status.output_file_id).text
        output_filepath = os.path.join('logs/output_file', f"output_{input_filename}")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            print("Writing output to file:", output_filepath)
            f.write(output_file_content)

        results = []
        for line in output_file_content.splitlines():
            data = json.loads(line)
            results.append(data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', None))

        return results


    async def _async_completions(self, messages: List[Dict]) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completions_kwargs
        )
        return response.choices[0].message.content

    async def async_completions_batch(self, messages: List[List[Dict]], timeout: int = 5, verbose: bool = True, **kwargs) -> List[str]:
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {'timeouts': 0}
        results = {}

        async def process_message(message_idx):
            message = messages[message_idx]
            retry_delay = timeout
            current_timeout = timeout
            max_retries = 5

            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        completion_res = await litellm_acompletion(
                            model=self.model,
                            messages=message,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            timeout=current_timeout,
                            **({
                                "response_format": {"type": "json_object"},
                            } if kwargs.get("structured_json") is not None else {})
                        )
                    response = completion_res.choices[0].message.content.strip()
                    break  # Exit the retry loop on success
                except asyncio.TimeoutError:
                    counts['timeouts'] += 1
                    if attempt == max_retries - 1:
                        if verbose:
                            print(
                                f"Max retries exceeded (timeouts) for prompt {message_idx} "
                                f"after {max_retries} attempts."
                            )
                        response = None  # final
                    else:
                        current_timeout *= 2
                        if verbose:
                            print(
                                f"Timeout after {current_timeout // 2}s for prompt {message_idx}, attempt {attempt+1}. "
                                f"Increasing timeout to {current_timeout}s..."
                            )
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Max retries exceeded for message index: {message_idx}")
                        print(e)
                        response = None  # Or handle as you see fit
                    else:
                        print(f"Error occurred for message index: {message_idx}. Retrying in {retry_delay} seconds.")
                        print(e)
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff

            results[message_idx] = response

        tasks = [process_message(message_idx) for message_idx in range(len(messages))]
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            await f  # Wait for each task to complete

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")

        return [results[i] for i in range(len(messages))]


    async def _completions_stream(self, messages: List):
        # messages = self.system + messages
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **self.completions_kwargs
        )

        for chunk in stream:
            if (text := chunk.choices[0].delta.content) is not None:
                yield text

class GrokAgent(OpenAIAgent):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        super().__init__(temperature, max_tokens)
        self.model = model
        grok_api_key = os.getenv("GROK_API_KEY")
        self.client = openai.OpenAI(api_key=grok_api_key, base_url="https://api.x.ai/v1")
        self.async_client = openai.AsyncOpenAI(api_key=grok_api_key)

class O1OpenAIAgent(OpenAIAgent):
    def __init__(self,  model: str = "o1-mini", max_tokens: int = 2048, **kwargs):
        super().__init__(max_tokens=max_tokens)
        self.model = model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        self.async_client = openai.AsyncOpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        self.completions_kwargs = {
            "max_completion_tokens": self.max_tokens,
        }

# class FireworksAgent(OpenAIAgent):
#     def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
#         super().__init__(temperature, max_tokens)
#         self.model = model
#         FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
#         self.client = Fireworks(api_key=FIREWORKS_API_KEY)
#         self.async_client = AsyncFireworks(api_key=FIREWORKS_API_KEY)

#     async def _async_completions(self, messages: List[Dict]) -> str:
#         response = await self.async_client.chat.completions.acreate(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens
#         )
#         return response.choices[0].message.content

class AnthropicAgent(LLMAgent):
    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "claude-3"):
        super().__init__(temperature, max_tokens)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        for message in messages:
            if (image_path := message.get('image_path')):
                image_data = _encode_image(image_path)
                image_type = imghdr.what(image_path) or 'jpeg'
                message['content'] = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_type}",
                            "data": image_data,
                        }
                    },
                    {"type": "text", "text": message['content']}
                ]
                del message['image_path']
        return messages

    def _completions(self, messages: List[Dict]) -> str:
        messages = self._preprocess_messages(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages
        )
        response = response.content[0].text
        return response

    def _completions_batch(self, messages: List[List[Dict]], **kwargs) -> List[str]:
        # Create a batch of requests
        requests = []
        for i, message in enumerate(messages):
            request = {
                "custom_id": f"request-{i}",
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": self._preprocess_messages(message)
                }
            }
            requests.append(request)

        # Create the batch
        batch = self.client.beta.messages.batches.create(requests=requests)

        # Poll for batch completion
        start_time = time.time()
        while batch.processing_status == "in_progress":
            time.sleep(10)  # Wait for 10 seconds before checking again
            batch = self.client.beta.messages.batches.retrieve(batch.id)
            print(f"Time elapsed: {(time.time() - start_time):.2f} seconds. Current status: {batch.processing_status}")

        if batch.processing_status != "ended":
            raise Exception(f"Batch processing failed with status: {batch.processing_status}")

        # Retrieve and process results
        results = []
        for result in self.client.beta.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                results.append(result.result.message.content[0].text)
            else:
                results.append(None)

        return results

class GeminiAgent(LLMAgent):

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "gemini-1.5-flash-002"):
        super().__init__(temperature, max_tokens)
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))
        self.model=model
        self.client = genai.GenerativeModel(model)

        self.safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        self.generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        # flatten from {"content": str} to "part": {"text": str}
        for message in messages:
            content = message['content']
            image_path = message.get('image_path')
            image = Image.open(image_path) if image_path else None
            parts = [content, image] if image else [content]

            message['parts'] = parts
            del message['content']
        return messages

    def _completions(self, messages: List) -> str:
        messages = self._preprocess_messages(messages)
        inputs = messages.pop()
        chat = self.client.start_chat(history=messages)
        completion = chat.send_message(inputs['parts'], generation_config=self.generation_config, safety_settings=self.safety_settings)
        output = completion.text

        return output

    async def _completions_stream(self, messages: List):
        messages = self._preprocess_messages(messages)

        inputs = messages.pop()
        chat = self.client.start_chat(history=messages)

        response = chat.send_message(inputs['parts'], generation_config=self.generation_config, safety_settings=self.safety_settings, stream=True)
        for chunk in response:
            yield chunk.text

    async def _async_completions(self, messages) -> str:
        raise NotImplementedError


_context_logger = logging.getLogger(__name__)

# Sentinel emitted in place of a completion when a prompt does not fit in the
# engine's context window. Sending such a prompt to vLLM raises and kills the
# whole run, so we skip it instead. Downstream, this string is classified as
# 'unparseable' by parse_responses_forced_choice (it contains no standalone
# 'A'/'B' token), and the logprobs path in utils.py skips it explicitly.
CONTEXT_OVERFLOW_SENTINEL = "__CONTEXT_OVERFLOW__"


def _resolve_max_model_len(llm, additional_kwargs: Optional[Dict] = None) -> Optional[int]:
    """Best-effort lookup of the served context length of a vLLM engine."""
    for attr_path in (
        ("llm_engine", "model_config", "max_model_len"),
        ("model_config", "max_model_len"),
        ("llm_engine", "vllm_config", "model_config", "max_model_len"),
    ):
        obj = llm
        ok = True
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, int) and obj > 0:
            return obj
    if additional_kwargs and additional_kwargs.get("max_model_len"):
        try:
            return int(additional_kwargs["max_model_len"])
        except (TypeError, ValueError):
            pass
    return None


def _prompt_token_length(tokenizer, prompt) -> int:
    """Token length of an already chat-templated prompt string (0 if unknown)."""
    if not isinstance(prompt, str):
        return 0
    try:
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    except Exception:
        try:
            return len(tokenizer(prompt)["input_ids"])
        except Exception:
            return 0


def _partition_by_context(tokenizer, prompt_texts, max_model_len, max_tokens, model_name="") -> Tuple[List[int], List[int]]:
    """Split prompt indices into (kept, skipped) using the engine context budget.

    A prompt is skipped when ``len(prompt_tokens) + max_tokens > max_model_len``.
    Returns all indices as kept when the context length could not be determined.
    """
    n = len(prompt_texts)
    if not max_model_len:
        return list(range(n)), []
    try:
        reserved = max(int(max_tokens or 0), 0)
    except (TypeError, ValueError):
        reserved = 0
    budget = max_model_len - reserved
    kept, skipped = [], []
    for i, prompt in enumerate(prompt_texts):
        if budget <= 0 or _prompt_token_length(tokenizer, prompt) > budget:
            skipped.append(i)
        else:
            kept.append(i)
    if skipped:
        _context_logger.warning(
            "Skipping %d/%d prompts that exceed the context window "
            "(max_model_len=%d, reserved max_tokens=%d, model=%s); "
            "these are returned as %s and treated as unparseable downstream.",
            len(skipped), n, max_model_len, reserved, model_name, CONTEXT_OVERFLOW_SENTINEL,
        )
    return kept, skipped


class vLLMAgentCompletion(object):
    def __init__(self, text: str, logprobs: List[Tuple[str, float]]):
        self.text = text
        self.logprobs = logprobs

class vLLMAgent(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/huggingface', trust_remote_code=False, accepts_system_message=True, tokenizer_path=None, min_p=None, chat_template_kwargs=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.min_p = min_p
        self.model_identity = None
        self.reasoning_effort = None
        self.chat_template_kwargs = chat_template_kwargs or {}

        # Load tokenizer and model
        tokenizer_source = tokenizer_path if tokenizer_path is not None else model
        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        # If loading from a local path, don't pass cache_dir (causes HF validation issues)
        if tokenizer_source.startswith("/") and os.path.isdir(tokenizer_source):
            tokenizer_kwargs["local_files_only"] = True
        else:
            tokenizer_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

        additional_kwargs = {}
        if "deepseek" in model.lower():
            additional_kwargs["max_model_len"] = 32768
            additional_kwargs["dtype"] = "float16"
            additional_kwargs["enforce_eager"] = True
        if "kimi" in model.lower():
            additional_kwargs["max_model_len"] = 30720
            additional_kwargs["enforce_eager"] = True
            additional_kwargs["gpu_memory_utilization"] = 0.95
        if "phi-3" in model.lower():
            # Phi-3 longrope models default to original_max_position_embeddings (4096)
            # in vLLM unless max_model_len is set explicitly; comparisons need ~24k.
            additional_kwargs["max_model_len"] = 65536
        if "qwen3-vl" in model.lower():
            additional_kwargs["max_model_len"] = 32768
            additional_kwargs["enforce_eager"] = True
            additional_kwargs["gpu_memory_utilization"] = 0.85
        if "qwen3-omni" in model.lower():
            additional_kwargs["max_model_len"] = 16384
            # Load chat template from file if tokenizer doesn't have one
            if not getattr(self.tokenizer, 'chat_template', None):
                ct_path = os.path.join(model, "chat_template.json")
                if os.path.exists(ct_path):
                    import json as _json
                    with open(ct_path) as _f:
                        self.tokenizer.chat_template = _json.load(_f)["chat_template"]

        # Always run in eager mode when using behaviour-bias backbone to avoid
        # CUDA graph capture failures from dynamic hooks.
        # additional_kwargs.setdefault("enforce_eager", True)

        if os.getenv("BEHAVIOR_BIAS_PATH") is not None:
            additional_kwargs["model_impl"] = "transformers"
            additional_kwargs["max_model_len"] = 32768
            additional_kwargs["gpu_memory_utilization"] = 0.95
            additional_kwargs["dtype"] = "bfloat16"
            additional_kwargs["enforce_eager"] = True
            print(f"Using behavior bias path: {os.getenv('BEHAVIOR_BIAS_PATH')}")

        # --- Soft prompt support ---
        # NOTE: Soft prompt serving is now handled by vLLMSoftPromptAgent
        # (direct injection via HTTP).  The code below is kept only for
        # backward compatibility with the old vocab-expansion approach.
        self._sp_metadata = None
        self._sp_banned_ids = None

        if os.getenv("PEFT_LORA_PATH") is not None:
            from vllm.lora.request import LoRARequest
            self.peft_lora_path = os.getenv("PEFT_LORA_PATH")
            self.lora_req = LoRARequest(
                lora_name="default",
                lora_int_id=1,
                lora_path=self.peft_lora_path
            )
            additional_kwargs["enable_lora"] = True
            additional_kwargs["max_lora_rank"] = 32
            additional_kwargs["dtype"] = "bfloat16"
            additional_kwargs["enforce_eager"] = True
            print(f"Using PEFT LoRA path: {os.getenv('PEFT_LORA_PATH')}")
        else:
            self.lora_req = None

        if "gpt-oss" in model.lower():
            additional_kwargs["model_impl"] = "transformers"
            additional_kwargs["dtype"] = "bfloat16"

        # Allow env var override for max_model_len (e.g. for short-output tasks)
        env_max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN")
        if env_max_model_len and "max_model_len" not in additional_kwargs:
            additional_kwargs["max_model_len"] = int(env_max_model_len)

        # Initialize vllm
        tp_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", torch.cuda.device_count()))
        if tp_size > torch.cuda.device_count():
            additional_kwargs.setdefault("distributed_executor_backend", "ray")
        print(f"Initializing vllm with model {model}, tokenizer {tokenizer_path if tokenizer_path is not None else model}, trust_remote_code {trust_remote_code}, download_dir {cache_dir}, tensor_parallel_size {tp_size}, additional_kwargs {additional_kwargs}")
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=tp_size,
            **additional_kwargs
        )
        if os.getenv("PEFT_LORA_PATH") is not None:
            self.llm.llm_engine.add_lora(self.lora_req)

        # Context window actually served by the engine; used to pre-filter prompts
        # that would otherwise raise inside llm.generate and kill the whole run.
        self.max_model_len = _resolve_max_model_len(self.llm, additional_kwargs)
        print(f"vLLM engine max_model_len resolved to {self.max_model_len}")

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    @staticmethod
    def _normalize_content_items(messages: List[Dict]) -> List[Dict]:
        """Normalize internal content items to chat-template-expected format.

        Converts internal representations like ``{'type': 'audio', 'audio_path': '...'}``
        to the format expected by model chat templates (e.g. Qwen2.5-Omni expects
        ``{'type': 'audio_url', 'audio_url': {'url': 'file://...'}}``)."""
        normalized = []
        for msg in messages:
            content = msg.get('content')
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'audio' and 'audio_path' in item:
                        # Convert to audio_url format expected by Qwen2.5-Omni chat template
                        audio_path = item['audio_path']
                        new_content.append({
                            'type': 'audio_url',
                            'audio_url': {'url': f'file://{audio_path}'},
                        })
                    else:
                        new_content.append(item)
                new_msg = msg.copy()
                new_msg['content'] = new_content
                normalized.append(new_msg)
            else:
                normalized.append(msg)
        return normalized

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to a prompt using the model's chat template."""
        messages = _merge_consecutive_roles(messages)
        messages = self._normalize_content_items(messages)
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if getattr(self, "model_identity", None):
            kwargs["model_identity"] = self.model_identity
        if getattr(self, "reasoning_effort", None):
            kwargs["reasoning_effort"] = self.reasoning_effort
        if getattr(self, "chat_template_kwargs", None):
            kwargs.update(self.chat_template_kwargs)
        output = self.tokenizer.apply_chat_template(messages, **kwargs)
        return output

    def _extract_images_from_messages(self, messages: List[Dict]) -> List[Image.Image]:
        """Extract PIL images from messages that have image_path or image keys."""
        images = []
        for msg in messages:
            if 'image_path' in msg:
                images.append(Image.open(msg['image_path']))
            elif 'image' in msg and isinstance(msg['image'], Image.Image):
                images.append(msg['image'])
            # Also check content if it's a list (multimodal format)
            content = msg.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'image' and 'image_path' in item:
                            images.append(Image.open(item['image_path']))
                        elif item.get('type') == 'image' and 'image' in item:
                            images.append(item['image'])
        return images

    def _extract_audios_from_messages(self, messages: List[Dict]) -> List:
        """Extract audio data from messages that have audio_path keys.

        Loads audio files using librosa at 16kHz (the standard sample rate for
        speech models like Qwen2.5-Omni). Returns a list of (audio_data, sr)
        tuples suitable for vLLM's multi_modal_data.
        """
        audio_paths = []
        for msg in messages:
            if 'audio_path' in msg:
                audio_paths.append(msg['audio_path'])
            content = msg.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'audio' and 'audio_path' in item:
                            audio_paths.append(item['audio_path'])

        if not audio_paths:
            return []

        import librosa
        audios = []
        for path in audio_paths:
            audio_data, sr = librosa.load(path, sr=16000)
            audios.append((audio_data, sr))
        return audios

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]],
                                batch_size: int = 1, structured_json: str = None,
                                max_tokens: int = 100, **kwargs) -> List[Tuple[str, List[Tuple[str, float]]]]:

        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        # Build inputs with multimodal support (images and/or audio)
        inputs = []
        prompt_texts = []
        for message_set in messages_list:
            prompt = self._messages_to_prompt(message_set)
            prompt_texts.append(prompt)
            images = self._extract_images_from_messages(message_set)
            audios = self._extract_audios_from_messages(message_set)
            mm_data = {}
            if images:
                mm_data["image"] = images[0] if len(images) == 1 else images
            if audios:
                mm_data["audio"] = audios[0] if len(audios) == 1 else audios
            if mm_data:
                inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
            else:
                inputs.append({"prompt": prompt})

        _kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "detokenize": False,
        }

        # Only add guided_decoding if available in this vLLM version
        if structured_json and GuidedDecodingParams is not None:
            guided_decoding = GuidedDecodingParams(json=structured_json)
            _kwargs["guided_decoding"] = guided_decoding

        if self.min_p is not None:
            _kwargs["min_p"] = self.min_p

        sampling_params = SamplingParams(
            **_kwargs
        )

        # Pre-flight: never dispatch prompts that cannot fit in the context window.
        kept_indices, _skipped = _partition_by_context(
            self.tokenizer, prompt_texts, getattr(self, "max_model_len", None),
            _kwargs["max_tokens"], self.model,
        )
        result_texts = [CONTEXT_OVERFLOW_SENTINEL] * len(inputs)
        if kept_indices:
            outputs = self.llm.generate(
                [inputs[i] for i in kept_indices], sampling_params, lora_request=self.lora_req
            )
            for idx, output in zip(kept_indices, outputs):
                # generated_text = output.outputs[0].text
                generated_text = self.tokenizer.decode(output.outputs[0].token_ids, clean_up_tokenization_spaces=True)

                result_texts[idx] = generated_text.strip().strip("<|eot_id|>")

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_with_probs(self, messages: Union[List[Dict], List[List[Dict]]],
                                batch_size: int = 1, structured_json: str = None,
                                top_K: int = 1, max_tokens: int = 100, **kwargs) -> List[Tuple[str, List[Tuple[str, float]]]]:

        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        # Build inputs with multimodal support (images and/or audio)
        inputs = []
        prompt_texts = []
        for message_set in messages_list:
            prompt = self._messages_to_prompt(message_set)
            prompt_texts.append(prompt)
            images = self._extract_images_from_messages(message_set)
            audios = self._extract_audios_from_messages(message_set)
            mm_data = {}
            if images:
                mm_data["image"] = images[0] if len(images) == 1 else images
            if audios:
                mm_data["audio"] = audios[0] if len(audios) == 1 else audios
            if mm_data:
                inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
            else:
                inputs.append({"prompt": prompt})

        _kwargs = {
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "min_p": 0.05,
            "logprobs": top_K,
        }

        # Only add guided_decoding if available in this vLLM version
        if structured_json and GuidedDecodingParams is not None:
            guided_decoding = GuidedDecodingParams(json=structured_json)
            _kwargs["guided_decoding"] = guided_decoding

        sampling_params = SamplingParams(**_kwargs)

        # Pre-flight: never dispatch prompts that cannot fit in the context window.
        kept_indices, _skipped = _partition_by_context(
            self.tokenizer, prompt_texts, getattr(self, "max_model_len", None),
            max_tokens, self.model,
        )
        # [(text, [(token, prob), ...]), ...]; skipped prompts keep the sentinel entry
        result_pairs = [(CONTEXT_OVERFLOW_SENTINEL, []) for _ in inputs]
        if kept_indices:
            outputs = self.llm.generate(
                [inputs[i] for i in kept_indices], sampling_params, lora_request=self.lora_req
            )

            for idx, req_out in zip(kept_indices, outputs):
                # Get the generated text
                generated_text = req_out.outputs[0].text.strip()

                # Get token alternatives and their probabilities
                token_alts = []  # [(token_str, prob)]
                for tok_id, lp in req_out.outputs[0].logprobs[0].items():
                    tok_str = lp.decoded_token  # already decoded
                    # prob = math.exp(lp.logprob)  # turn log-p into p
                    prob = lp.logprob
                    token_alts.append((tok_str, prob))

                # Sort by probability
                token_alts.sort(key=lambda x: x[1], reverse=True)

                # Add to results
                result_pairs[idx] = (generated_text, token_alts)

        return result_pairs

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1, **kwargs) -> List[str]:
        # if top_K in kwargs, use _completions_with_probs
        if kwargs.get("top_K"):
            # Direct return without await since _completions_with_probs returns a regular object
            return self._completions_with_probs(messages_list, batch_size, **kwargs)
        else:
            return self._completions(messages_list, batch_size, **kwargs)

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._messages_to_prompt(messages)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True, lora_request=self.lora_req)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        # Since VLLM does not support asynchronous operations, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result

class vLLMSoftPromptAgent(LLMAgent):
    """Agent that serves soft prompts via direct embedding injection over HTTP.

    Instead of running vLLM in-process with vocabulary expansion patches, this
    agent communicates with an external vLLM server (started with
    ``--enable-prompt-embeds``) and injects soft prompt embeddings client-side.

    Prompts containing ``[candidate_X]`` placeholders get their placeholder
    token embeddings replaced with the learned soft prompt tensor before being
    sent to the server as base64-encoded ``prompt_embeds``.  Prompts without
    placeholders are sent as plain text.
    """

    def __init__(
        self,
        model_path: str,
        server_url: str,
        soft_prompt_path: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
        trust_remote_code: bool = True,
        accepts_system_message: bool = True,
        min_p: Optional[float] = None,
        chat_template_kwargs: Optional[Dict] = None,
        soft_prompt_placement: str = "system_prompt",
    ):
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            accepts_system_message=accepts_system_message,
        )
        self.model_path = model_path
        self.model = model_path  # Used by ThurstonianActiveLearningUtilityModel for model type detection
        self.server_url = server_url
        # Support comma-separated paths for multiple SP tensors
        self.soft_prompt_paths = [p.strip() for p in soft_prompt_path.split(",")]
        self.min_p = min_p
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.soft_prompt_placement = soft_prompt_placement

        # Lazy-loaded components
        self._tokenizer = None
        self._embedding_layer = None
        self._sp_tensors: Optional[List[torch.Tensor]] = None
        self._model_name: Optional[str] = None
        self._api_url: Optional[str] = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Eagerly load tokenizer (needed for chat template)
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if model_path.startswith("/") and os.path.isdir(model_path):
            tokenizer_kwargs["local_files_only"] = True
        else:
            tokenizer_kwargs["cache_dir"] = "/data/huggingface"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

    def _ensure_ready(self):
        """Lazy-load embedding layer, SP tensors, and model name."""
        if self._embedding_layer is not None:
            return

        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
            load_embedding_layer_from_cache,
            load_soft_prompt_tensor,
            normalize_api_url,
            prepare_embedding_cache,
            get_model_name_from_server,
        )

        self._api_url = normalize_api_url(self.server_url)

        # Prepare embedding cache (extract from safetensors if needed)
        print(f"[vLLMSoftPromptAgent] Preparing embedding cache for {self.model_path} ...")
        prepare_embedding_cache(self.model_path)

        # Load embedding layer from cache
        print(f"[vLLMSoftPromptAgent] Loading embedding layer (device={self._device}) ...")
        self._embedding_layer = load_embedding_layer_from_cache(
            self.model_path, self._device
        )
        if self._embedding_layer is None:
            raise RuntimeError(
                f"Failed to load embedding layer from cache for {self.model_path}. "
                f"Run prepare_embedding_cache() first."
            )

        # Load soft prompt tensor(s)
        self._sp_tensors = []
        for sp_path in self.soft_prompt_paths:
            print(f"[vLLMSoftPromptAgent] Loading soft prompt from {sp_path} ...")
            self._sp_tensors.append(load_soft_prompt_tensor(sp_path))

        # Get model name from server
        self._model_name = get_model_name_from_server(self._api_url)
        print(f"[vLLMSoftPromptAgent] Ready (model={self._model_name}, "
              f"{len(self._sp_tensors)} SP tensor(s), device={self._device})")

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.chat_template_kwargs:
            kwargs.update(self.chat_template_kwargs)
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate_one(
        self,
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        logprobs: Optional[int] = None,
    ) -> Dict:
        """Generate for a single prompt, dispatching to direct injection or plain text."""
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
            generate_with_direct_injection,
        )

        return generate_with_direct_injection(
            prompt_text,
            api_url=self._api_url,
            model_name=self._model_name,
            tokenizer=self.tokenizer,
            embedding_layer=self._embedding_layer,
            sp_tensors=self._sp_tensors,
            device=self._device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=self.min_p,
            logprobs=logprobs,
        )

    # ── Sync path (sequential, fallback) ────────────────────────────────

    def _completions(
        self,
        messages: Union[List[Dict], List[List[Dict]]],
        batch_size: int = 1,
        structured_json: str = None,
        max_tokens: int = 100,
        **kwargs,
    ) -> Union[str, List[str]]:
        self._ensure_ready()

        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        inputs = []
        for message_set in messages_list:
            prompt = self._messages_to_prompt(message_set)
            inputs.append(prompt)

        result_texts = []
        total = len(inputs)
        for i, prompt_text in enumerate(inputs):
            data = self._generate_one(
                prompt_text,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            text = data["choices"][0]["text"].strip()
            result_texts.append(text)

            if total >= 50 and ((i + 1) % 500 == 0 or i == 0):
                print(f"    [vLLMSoftPromptAgent] {i+1}/{total} requests...")

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_with_probs(
        self,
        messages: Union[List[Dict], List[List[Dict]]],
        batch_size: int = 1,
        structured_json: str = None,
        top_K: int = 1,
        max_tokens: int = 100,
        **kwargs,
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        self._ensure_ready()

        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        inputs = []
        for message_set in messages_list:
            prompt = self._messages_to_prompt(message_set)
            inputs.append(prompt)

        result_pairs = []
        for prompt_text in inputs:
            data = self._generate_one(
                prompt_text,
                max_tokens=max_tokens,
                temperature=self.temperature,
                logprobs=top_K,
            )
            choice = data["choices"][0]
            generated_text = choice.get("text", "").strip()

            token_alts = []
            lp_data = choice.get("logprobs")
            if lp_data and lp_data.get("top_logprobs"):
                for step in lp_data["top_logprobs"]:
                    for tok_str, lp_value in step.items():
                        token_alts.append((tok_str, lp_value))

            token_alts.sort(key=lambda x: x[1], reverse=True)
            result_pairs.append((generated_text, token_alts))

        return result_pairs

    def _completions_batch(
        self, messages_list: List[List[Dict]], batch_size: int = 1, **kwargs
    ):
        if kwargs.get("top_K"):
            return self._completions_with_probs(messages_list, batch_size, **kwargs)
        else:
            return self._completions(messages_list, batch_size, **kwargs)

    # ── Async path (concurrent HTTP via aiohttp) ─────────────────────

    async def async_completions_batch(
        self,
        messages: List[List[Dict]],
        concurrency: int | None = None,
        verbose: bool = True,
        payload_batch_size: int | None = None,
        **kwargs,
    ) -> List:
        """Send all requests concurrently with pipelined payload preparation.

        Payload preparation (tokenize + embed + base64) runs in a thread pool
        so CPU work overlaps with GPU inference.  Peak memory is bounded by
        ``concurrency + n_workers`` in-flight payloads.

        Defaults can be overridden via env vars ``VLLM_CONCURRENCY`` and
        ``VLLM_PAYLOAD_WORKERS``.
        """
        if concurrency is None:
            concurrency = int(os.environ.get("VLLM_CONCURRENCY", "64"))
        n_workers = int(os.environ.get("VLLM_PAYLOAD_WORKERS",
                                        str(min(os.cpu_count() or 4, 32))))
        import asyncio
        import aiohttp
        from concurrent.futures import ThreadPoolExecutor

        self._ensure_ready()

        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
            prepare_injection_payload,
            async_post_payload,
        )

        top_K = kwargs.get("top_K")
        total = len(messages)

        if verbose:
            print(f"    [vLLMSoftPromptAgent] Processing {total} requests "
                  f"(concurrency={concurrency}, workers={n_workers}) ...")

        results = [None] * total
        completed = [0]

        def _extract_result(data):
            """Extract a single result from the server response."""
            if top_K:
                choice = data["choices"][0]
                generated_text = choice.get("text", "").strip()
                token_alts = []
                lp_data = choice.get("logprobs")
                if lp_data and lp_data.get("top_logprobs"):
                    for step in lp_data["top_logprobs"]:
                        for tok_str, lp_value in step.items():
                            token_alts.append((tok_str, lp_value))
                token_alts.sort(key=lambda x: x[1], reverse=True)
                return (generated_text, token_alts)
            else:
                return data["choices"][0]["text"].strip()

        def _prep_payload(i: int):
            """Build one payload on a worker thread (CPU-bound, torch releases GIL)."""
            prompt = self._messages_to_prompt(messages[i])
            return prepare_injection_payload(
                prompt,
                model_name=self._model_name,
                tokenizer=self.tokenizer,
                embedding_layer=self._embedding_layer,
                sp_tensors=self._sp_tensors,
                device=self._device,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                min_p=self.min_p,
                logprobs=top_K,
            )

        max_retries = int(os.environ.get("VLLM_MAX_RETRIES", "5"))
        retry_base_delay = float(os.environ.get("VLLM_RETRY_DELAY", "2.0"))
        failed_indices = []

        send_semaphore = asyncio.Semaphore(concurrency)
        # Bound in-flight payloads to limit memory
        prep_semaphore = asyncio.Semaphore(concurrency + n_workers)
        loop = asyncio.get_event_loop()

        async with aiohttp.ClientSession() as session:

            async def _prep_and_send(i: int, executor: ThreadPoolExecutor):
                async with prep_semaphore:
                    # Prep payload in thread pool
                    payload = await loop.run_in_executor(executor, _prep_payload, i)
                    # Send with retry logic
                    last_err = None
                    for attempt in range(max_retries):
                        try:
                            async with send_semaphore:
                                data = await async_post_payload(
                                    self._api_url, payload, session
                                )
                            results[i] = _extract_result(data)
                            del data, payload
                            completed[0] += 1
                            if verbose and total >= 50 and (
                                completed[0] % 500 == 0 or completed[0] == total
                            ):
                                print(f"    [vLLMSoftPromptAgent] {completed[0]}/{total} done")
                            return
                        except Exception as e:
                            last_err = e
                            if attempt < max_retries - 1:
                                delay = retry_base_delay * (2 ** attempt)
                                print(f"    [vLLMSoftPromptAgent] Request {i} attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s ...")
                                await asyncio.sleep(delay)
                    failed_indices.append(i)
                    completed[0] += 1
                    print(f"    [vLLMSoftPromptAgent] Request {i} FAILED after {max_retries} attempts: {last_err}. Skipping.")

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                await asyncio.gather(
                    *[_prep_and_send(i, executor) for i in range(total)]
                )

        if failed_indices:
            print(f"    [vLLMSoftPromptAgent] WARNING: {len(failed_indices)}/{total} requests failed and were skipped. "
                  f"Failed indices: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")

        return results

    async def _async_completions(self, messages: List[Dict]) -> str:
        result = await self.async_completions_batch([messages])
        return result[0] if isinstance(result, list) else result


class vLLMAgentWithReasoning(vLLMAgent):
    def __init__(self,
                 model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                 max_tokens=2048,
                 temperature=0.0,
                 cache_dir='/data/public_models',
                 trust_remote_code=False,
                 accepts_system_message=True,
                 tokenizer_path=None,
                 reasoning_parser: str = "deepseek_r1",
                 enable_reasoning: bool = True):

        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.accepts_system_message = accepts_system_message
        self.reasoning_parser = reasoning_parser
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_reasoning = enable_reasoning

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        # Prepare additional LLM kwargs
        additional_kwargs = {}
        if "deepseek" in model.lower():
            additional_kwargs["max_model_len"] = 32768
            additional_kwargs["dtype"] = "float16"
            additional_kwargs["enforce_eager"] = True

        # self.llm = LLM(
        #     model=self.model,
        #     trust_remote_code=True,  # Required for some models with custom code
        #     tokenizer=self.model,  # ensure tokenizer is compatible
        #     enable_reasoning=self.enable_reasoning,
        #     reasoning_parser=self.reasoning_parser,
        #     download_dir=cache_dir,
        #     tensor_parallel_size=torch.cuda.device_count()  # Use all available GPUs
        #     **additional_kwargs
        # )
        self.llm = LLM(
            model=model,
            # tokenizer=tokenizer_path if tokenizer_path is not None else model,
            # trust_remote_code=True,
            # enable_reasoning=self.enable_reasoning,
            # reasoning_parser=self.reasoning_parser,
            # download_dir=cache_dir,
            # tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
            # **additional_kwargs
            trust_remote_code=True,          # model uses custom code
            tensor_parallel_size=4,          # e.g., 8×H100/A100/etc.
            max_model_len=40960,             # raise toward 128k if your VRAM allows
            gpu_memory_utilization=0.95,
            enforce_eager=True,
        )

        self.max_model_len = _resolve_max_model_len(self.llm, additional_kwargs)

        # Completion kwargs
        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        messages = _merge_consecutive_roles(messages)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _completions(self,
                    messages: Union[List[Dict], List[List[Dict]]],
                    batch_size: int = 1,
                    structured_json: str = None,
                    nested_content: bool = False
                    ) -> Union[Tuple[str, str], List[Tuple[str, str]]]:

        messages_list = [messages] if isinstance(messages[0], dict) else messages
        prompts = [self._messages_to_prompt(ms) for ms in messages_list]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Pre-flight: never dispatch prompts that cannot fit in the context window.
        kept_indices, skipped_indices = _partition_by_context(
            self.tokenizer, prompts, getattr(self, "max_model_len", None),
            self.max_tokens, self.model,
        )
        if skipped_indices:
            prompts = [prompts[i] for i in kept_indices]

        outputs = self.llm.generate(prompts, sampling_params) if prompts else []
        result_pairs: List[Tuple[str, str]] = []

        for output in outputs:
            o = output.outputs[0]
            text = (getattr(o, "text", "") or "").strip()

            # Prefer structured fields if present
            reasoning = (getattr(o, "reasoning_trace", "") or "").strip()
            final_answer = (getattr(o, "final_answer", "") or "").strip()

            # <think> ... </think> (may be multiple); strip them from the visible answer
            if not final_answer and text:
                thinks = THINK_RE.findall(text)
                if thinks:
                    # join all think blocks; keep prior structured reasoning if any
                    block = "\n\n".join(s.strip() for s in thinks if s.strip())
                    reasoning = (reasoning + ("\n\n" if reasoning and block else "") + block).strip()
                    text = THINK_RE.sub("", text).strip()  # remove think blocks
                    final_answer = text
                elif "</think>" in text and "<think>" not in text:
                    # Handle stray closing </think> with no opening <think>
                    lt = text.lower()
                    # if "</think>" in lt and "<think>" not in lt:
                    cut = lt.rfind("</think>") + len("</think>")
                    ans = text[cut:].strip()
                    # trim a single dangling quote if present (common tokenization artifact)
                    if ans.endswith(("'", '"', "’", "”", "`")):
                        ans = ans[:-1].strip()
                    result_pairs.append(ans if ans else text)
                    continue

            # Split on the LAST "Answer:" / "Final Answer:" if still no final_answer
            if not final_answer and text:
                last = None
                for m in ANS_RE.finditer(text):
                    last = m
                if last:
                    maybe_reason = text[: last.start()].strip()
                    if maybe_reason:
                        reasoning = maybe_reason  # prefer explicit preamble as reasoning
                    final_answer = text[last.end():].strip()

            # Fallback: whole text is the answer
            if not final_answer:
                final_answer = text

            # result_pairs.append((final_answer, reasoning))
            result_pairs.append(final_answer)

        # Re-expand to the original prompt ordering, filling skipped prompts with
        # the context-overflow sentinel (treated as unparseable downstream).
        if skipped_indices:
            expanded = [CONTEXT_OVERFLOW_SENTINEL] * (len(kept_indices) + len(skipped_indices))
            for idx, value in zip(kept_indices, result_pairs):
                expanded[idx] = value
            result_pairs = expanded
        # print(result_pairs)
        return result_pairs[0] if len(result_pairs) == 1 else result_pairs

    # def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1, structured_json: str = None, nested_content: bool = False) -> Union[Tuple[str, str], List[Tuple[str, str]]]:
    #     if isinstance(messages[0], dict):
    #         messages_list = [messages]
    #     else:
    #         messages_list = messages

    #     prompts = [self._messages_to_prompt(ms) for ms in messages_list]

    #     sampling_params = SamplingParams(
    #         temperature=self.temperature,
    #         max_tokens=self.max_tokens,
    #     )

    #     outputs = self.llm.generate(prompts, sampling_params)
    #     result_pairs = []

    #     for output in outputs:
    #         output_obj = output.outputs[0]
    #         reasoning = ""
    #         final_answer = ""

    #         text = getattr(output_obj, "text", "").strip()

    #         # Case 1: Structured fields available
    #         if hasattr(output_obj, "reasoning_trace"):
    #             reasoning = output_obj.reasoning_trace.strip()
    #             final_answer = getattr(output_obj, "final_answer", text).strip()

    #         # Case 2: <think> ... </think> tag
    #         elif "</think>" in text:
    #             reasoning = text.split("</think>")[0].strip()
    #             final_answer = text.split("</think>")[-1].strip()

    #         # Case 3: Fallback to regex split on "Answer:" or "Final Answer:"
    #         elif re.search(r"(?:Answer:|Final Answer:)", text):
    #             parts = re.split(r"(?:Answer:|Final Answer:)", text, maxsplit=1)
    #             reasoning = parts[0].strip()
    #             final_answer = parts[1].strip()

    #         # Case 4: Only one block — assume it's the final answer
    #         else:
    #             final_answer = text

    #         result_pairs.append(final_answer)
    #     return result_pairs[0] if len(result_pairs) == 1 else result_pairs

# FEW_SHOT_PROMPT = """Which city is the capital of France?

# Option A: Paris
# Option B: Rome

# Answer: A

# ---

# Which planet is known as the Red Planet?

# Option A: Mars
# Option B: Jupiter

# Answer: A

# ---

# Which is the largest mammal on Earth?

# Option A: Elephant
# Option B: Blue Whale

# Answer: B

# ---

# What is the chemical symbol for water?

# Option A: H2O
# Option B: CO2

# Answer: A

# ---

# Which shape has three sides?

# Option A: Triangle
# Option B: Square

# Answer: A

# ---

# """

FEW_SHOT_PROMPT = ""

class vLLMAgentBaseModel(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, accepts_system_message=False, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        # Initialize vllm
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=torch.cuda.device_count()  # Use all available GPUs
        )

        self.max_model_len = _resolve_max_model_len(self.llm)

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        """
        Format messages into strings that the model can process.
        We prepend a hard-coded 5-shot prompt and then append the user's single message.
        """
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            # We expect only one user message, but we'll handle any number just in case
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            # Prepend the 5-shot prompt, then the user's message
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1, structured_json: str = None) -> Union[str, List[str]]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        prompts = self._format_messages(messages_list)

        _kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Only add guided_decoding if available in this vLLM version
        if structured_json and GuidedDecodingParams is not None:
            guided_decoding = GuidedDecodingParams(json=structured_json)
            _kwargs["guided_decoding"] = guided_decoding

        sampling_params = SamplingParams(**_kwargs)

        # Pre-flight: never dispatch prompts that cannot fit in the context window.
        kept_indices, _skipped = _partition_by_context(
            self.tokenizer, prompts, getattr(self, "max_model_len", None),
            self.max_tokens, self.model,
        )
        result_texts = [CONTEXT_OVERFLOW_SENTINEL] * len(prompts)
        if kept_indices:
            outputs = self.llm.generate([prompts[i] for i in kept_indices], sampling_params)

            for idx, output in zip(kept_indices, outputs):
                generated_text = output.outputs[0].text
                result_texts[idx] = generated_text.strip()

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1, **kwargs) -> List[str]:
        return self._completions(messages_list, batch_size, **kwargs)

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._format_messages([messages])[0]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            min_p=0.05,
        )
        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        # Since VLLM does not support asynchronous operations, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result


class HuggingFaceAgentLogitsPrediction(LLMAgent):

    def __init__(
        self,
        model="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=2048,
        temperature=0.0,
        cache_dir='/data/public_models',
        trust_remote_code=False,
        accepts_system_message=False
    ):
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.accepts_system_message = False  # Hard-coded for base models with no system messages

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        # Set padding token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
                device_map="auto"  # Automatically distribute across GPUs
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
            ).to(self.device)

        self.llm.eval()  # Set to evaluation mode

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        """
        Format messages into strings that the model can process.
        We prepend a hard-coded 5-shot prompt and then append the user's single message.
        """
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            # We expect only one user message, but we'll handle any number just in case
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            # Prepend the 5-shot prompt, then the user's message
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(
        self,
        messages: Union[List[Dict], List[List[Dict]]],
        batch_size: int = 1,
        options: List[str] = ['A', 'B']
    ) -> List[Dict[str, float]]:
        """
        Get completion logits for specific options, returning a probability distribution
        over those options that sums to 1.0.
        """
        formatted_messages = self._format_messages(messages)
        results = []

        # Convert option strings (e.g. "A", "B") to the correct token IDs for " A", " B", etc.
        option_tokens = [" " + opt for opt in options]
        option_ids = self.tokenizer.encode(option_tokens, add_special_tokens=False)

        # Process in batches
        for i in range(0, len(formatted_messages), batch_size):
            batch_messages = formatted_messages[i:i + batch_size]

            # Tokenize inputs
            inputs = self.tokenizer(
                batch_messages,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=False
            ).to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.llm(**inputs)
                # We only need the last token's logits for each sequence
                logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]

            # Process each sequence in the batch
            for logits_seq in logits:
                # Create a masked logits array that is -1e4 everywhere except
                # for the chosen option token IDs.  -1e4 is safely in range for float16.
                masked_logits = torch.full_like(logits_seq, -1e4)
                for option_id in option_ids:
                    if option_id < logits_seq.shape[0]:
                        masked_logits[option_id] = logits_seq[option_id]

                # Compute the softmax distribution over the entire vocabulary,
                # then isolate just the options and renormalize so they sum to 1.
                full_dist = F.softmax(masked_logits, dim=0)
                subset_dist = full_dist[option_ids]
                subset_dist = subset_dist / subset_dist.sum()  # Force sum to 1

                # Build a dictionary of {option_letter: probability}
                distribution_dict = {
                    options[idx]: float(subset_dist[idx]) for idx in range(len(options))
                }
                results.append(distribution_dict)

        return results


class HuggingFaceAgent(LLMAgent):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", behavior_bias_path=None, lora_path=None, per_layer_mixing=False, per_token_model=False, target_layer=16, ending_layer=-1, behavior_dim=128, layer_func="even", num_behaviors=None, max_tokens=2048, temperature=0.0, cache_dir='/data/huggingface', trust_remote_code=False, batch_size=512, accepts_system_message=True, tokenizer_path=None, constrain_norm=False, use_mixing_coefficients_head=False, mixing_coefficients_head_layer=15, full_stream=False, model_identity: Optional[str] = None, reasoning_effort: Optional[str] = None, **kwargs):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model_name = model_name
        self.batch_size = batch_size
        # Optional model identity for GPT-OSS chat template
        self.model_identity = model_identity
        # Optional reasoning effort for GPT-OSS chat template
        self.reasoning_effort = reasoning_effort
        self.behavior_bias_path = behavior_bias_path
        self.lora_path = lora_path
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            padding_side='left'  # Important: Set padding to left side
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.behavior_bias_path is not None:
            if per_token_model:
                behavior_bias = LlamaForCausalLMWithPerTokenBehaviorBias.load_behavior_bias_state_dict(self.behavior_bias_path, "cpu")
                _num_behaviors = num_behaviors if num_behaviors is not None else behavior_bias['behavior_down_proj_list.layer_0.weight'].shape[0] // behavior_dim
            else:
                behavior_bias = LlamaForCausalLMWithBehaviorBias.load_behavior_bias_state_dict(self.behavior_bias_path, "cpu")
                _num_behaviors = num_behaviors if num_behaviors is not None else len(behavior_bias['behavior_bias'])

            assert not (per_token_model and full_stream), "Cannot have both per-token and full-stream models"
            if per_token_model:
                self.model = LlamaForCausalLMWithPerTokenBehaviorBias.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=None,
                    num_behaviors=_num_behaviors,
                    mode="inference",
                    behavior_dim=behavior_dim,
                    ending_layer=ending_layer,
                    layer_func=layer_func,
                    constrain_norm=constrain_norm,
                    use_mixing_coefficients_head=use_mixing_coefficients_head,
                    mixing_coefficients_head_layer=mixing_coefficients_head_layer,
                )

            elif full_stream:
                self.model = LlamaForCausalLMWithBehaviorBiasFullStream.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                    num_behaviors=_num_behaviors,
                    mode="inference",
                    ending_layer=ending_layer,
                )

            else:
                self.model = LlamaForCausalLMWithBehaviorBias.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                    attn_implementation="flash_attention_2",
                    num_behaviors=_num_behaviors,
                    mode="inference",
                    per_layer_mixing=per_layer_mixing,
                    ending_layer=ending_layer,
                )
                self.model.bfloat16()
                # self.model.cuda()
            self.model.load_behavior_bias_into_model(behavior_bias)
            device_map = infer_auto_device_map(self.model, dtype=self.model.dtype)
            self.model = dispatch_model(self.model, device_map=device_map)

        elif self.lora_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                device_map=None,               # load onto CPU first
                low_cpu_mem_usage=True,
            )

            # Apply LoRA / PEFT on the CPU model before dispatching
            # If using PEFT adapters:
            try:
                import ipdb
                ipdb.set_trace()
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.lora_path, device_map={"": "cpu"})
            except Exception:
                # Fallback to custom LoRA API on your model
                self.model.set_enable_lora(True)
                if hasattr(self.model, "init_lora_parameters"):
                    self.model.init_lora_parameters()
                # load LoRA-only state dict (saved previously)
                lora_state = torch.load(self.lora_path, map_location="cpu")
                self.model.load_state_dict(lora_state, strict=False)

            # Now dispatch / infer device map and place model across GPUs
            device_map = infer_auto_device_map(self.model, dtype=self.model.dtype)
            self.model = dispatch_model(self.model, device_map=device_map)

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                use_kernels=True,
            )
        self.model.eval()


    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to a prompt using the model's chat template."""
        messages = _merge_consecutive_roles(messages)
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if getattr(self, "model_identity", None):
            kwargs["model_identity"] = self.model_identity
        if getattr(self, "reasoning_effort", None):
            kwargs["reasoning_effort"] = self.reasoning_effort
        output = self.tokenizer.apply_chat_template(messages, **kwargs)
        return output

    def _completions(self, messages: List[Dict]) -> str:
        """Handle single completion."""
        return self._completions_batch([messages])[0]

    def _completions_batch(self, messages_list: List[List[Dict]], a_b_logits_only: bool = False, prefill: str = "", **kwargs) -> List[str]:
        """Handle batch completions with left padding."""
        from accelerate.utils import find_executable_batch_size
        from tqdm import tqdm

        # Format all messages into prompts using chat template
        prompts = [self._messages_to_prompt(messages) for messages in messages_list]
        all_outputs = []

        @find_executable_batch_size(starting_batch_size=self.batch_size)
        def _process_batch(batch_size):
            nonlocal all_outputs
            print(f"\nProcessing with batch size: {batch_size}", flush=True)

            # Process in batches
            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
                batch_prompts = prompts[i:i + batch_size]

                # Tokenize with padding
                batch_prompts = [prompt + prefill for prompt in batch_prompts]
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=2048  # Adjust based on model context window
                ).to(self.model.device)

                if a_b_logits_only:
                    with torch.no_grad():
                        A_TOKEN_ID = self.tokenizer.encode("A", add_special_tokens=False)[0]
                        B_TOKEN_ID = self.tokenizer.encode("B", add_special_tokens=False)[0]
                        logits = self.model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                        ).logits
                        logprobs = F.log_softmax(logits, dim=-1)
                        logprobs = logprobs[:, -1, [A_TOKEN_ID, B_TOKEN_ID]]
                        choices = logprobs.argmax(dim=-1) + A_TOKEN_ID
                        choices = [self.tokenizer.decode(choice) for choice in choices]
                        all_outputs.extend(choices)
                else:
                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=self.max_tokens,
                            do_sample=self.temperature > 0,
                            temperature=self.temperature if self.temperature > 0 else 1.0,
                            # min_p=0.05,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                    )

                    # Decode outputs
                    for j, output in enumerate(outputs):
                        # Find where the prompt ends
                        prompt_length = len(inputs["input_ids"][j])
                        # Only decode the new tokens
                        decoded = self.tokenizer.decode(
                            output[prompt_length:],
                            # skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        all_outputs.append(decoded.strip())

        # Find and use the largest working batch size
        _process_batch()
        return all_outputs

    async def _async_completions(self, messages: List[Dict]) -> str:
        """Async completion just calls sync version since HF doesn't have async API."""
        return self._completions(messages)

    async def _completions_stream(self, messages: List[Dict]) -> str:
        """Streaming not implemented for HF models."""
        raise NotImplementedError("Streaming not implemented for HuggingFace models")


class LiteLLMAgent:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        concurrency_limit: int = 100,
        accepts_system_message: bool = True,
        max_retries: int = 5,
        base_timeout: float = 5.0,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        use_jitter: bool = True,
        api_key: str = None,
        base_url: str = None,
        extra_params: dict = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.accepts_system_message = accepts_system_message

        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_jitter = use_jitter
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = extra_params or {}

    async def async_completions(
        self,
        messages: List[List[Dict]],
        verbose: bool = True,
        timeout: float = None,
        **kwargs
    ) -> List[str]:
        """
        Returns a list of LLM responses, in order.
        Uses a semaphore to limit concurrency, and tqdm_asyncio for progress.

        Args:
            messages: List of message lists to process
            verbose: Whether to print verbose output
            timeout: Override for the base_timeout if provided
            **kwargs: Additional keyword arguments
        """

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            """
            Attempts to process a single message up to `max_retries` times.
            On generic exceptions, sleeps with exponential backoff and optional jitter.
            On timeout, doubles the request timeout without sleeping.
            """
            message = messages[message_idx]

            # Use provided timeout if available, otherwise use the base_timeout
            current_timeout = timeout or self.base_timeout
            retry_delay = self.base_delay
            response = None

            for attempt in range(self.max_retries):
                # Acquire the semaphore before making the LLM call
                async with semaphore:
                    # if verbose:
                    #     print(
                    #         f"[Attempt {attempt+1}/{self.max_retries}] "
                    #         f"Message index {message_idx}, timeout={current_timeout:.1f}s"
                    #     )

                    try:
                        completion_res = await litellm_acompletion(
                            model=self.model,
                            messages=message,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            timeout=current_timeout,
                            base_url=self.base_url or "https://api.openai.com/v1",
                            drop_params=True,
                            **({"api_key": self.api_key} if self.api_key else {}),
                            **self.extra_params,
                            **({
                                "response_format": {"type": "json_object"},
                            } if kwargs.get("structured_json") is not None else {})
                        )
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1

                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}. Timed out after {current_timeout:.1f}s."
                            )
                        if attempt == self.max_retries - 1:
                            response = None  # no more retries
                            if verbose:
                                print(f"Max retries (timeouts) reached for message index {message_idx}.")
                        else:
                            current_timeout *= 2.0

                        continue  # next attempt

                    except Exception as e:
                        counts["errors"] += 1

                        if verbose:
                            print(
                                f"[Error] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                            if verbose:
                                print(f"Max retries (errors) reached for message index {message_idx}.")
                        else:
                            # Sleep with exponential backoff
                            sleep_for = retry_delay
                            if self.use_jitter:
                                sleep_for += random.uniform(0, 1)
                            if verbose:
                                print(f"Sleeping {sleep_for:.1f}s before retry (error backoff)...")
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)

                        continue  # next attempt

                    # Success: parse the response
                    content = completion_res.choices[0].message.content
                    response = content.strip() if content else None
                    break  # done with retries

            results[message_idx] = response

        # Create a task for each message
        tasks = [process_message(i) for i in range(len(messages))]

        # Use tqdm_asyncio to track progress as tasks finish
        # You can also set `leave=False` or other tqdm arguments as needed
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="LLM calls"):
            await coro

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")
            print(f"Number of generic errors: {counts['errors']}")

        return [results[i] for i in range(len(messages))]

class VLLMClientAgent(LLMAgent):
    """A thin wrapper around an already running vLLM server that is accessed
    through `trl.extras.vllm_client.VLLMClient`. This class implements the
    `LLMAgent` interface expected by `compute_utilities`, allowing us to reuse
    the utility-estimation pipeline while querying a remote vLLM server instead
    of starting a local model (which would OOM during RL training).

    NOTE: Only the small subset of the interface that `compute_utilities` relies
    on (`_completions`, `_completions_batch`, and `_async_completions`) is
    implemented. Streaming is not required for the reward computation and is
    therefore left unimplemented.
    """

    def __init__(self, client, tokenizer, temperature: float = 0.0, max_tokens: int = 512):
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.client = client  # An instance of `VLLMClient`
        self.tokenizer = tokenizer  # HuggingFace tokenizer corresponding to the server model

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _messages_to_prompt(self, messages):
        """Convert a list of chat messages to a single prompt string using the
        tokenizer's chat template, adding the generation prompt token if
        necessary.
        """
        messages = _merge_consecutive_roles(messages)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --------------------------------------------------------------------------
    # Required overrides from `LLMAgent`
    # --------------------------------------------------------------------------
    def _completions(self, messages):
        # Re-use the batch implementation for the single-example case.
        return self._completions_batch([messages])[0]

    def _completions_batch(self, messages_list, **kwargs):
        """Generate completions for a batch of message lists.

        Args:
            messages_list (List[List[Dict]]): Batch of chat messages.
        Returns:
            List[str]: The generated texts (one per element of `messages_list`).
        """
        # Build prompts for all chats in the batch.
        prompts = [self._messages_to_prompt(msgs) for msgs in messages_list]

        # Call the remote server. `generate` returns a list of lists of token IDs
        # (one list per prompt).
        try:
            outputs_token_ids = self.client.generate(
                prompts,
                n=1,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            raise RuntimeError(f"vLLMClient generation failed: {e}")

        # Decode each list of token IDs to text.
        decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) if ids else "" for ids in outputs_token_ids]
        return decoded_texts

    async def _async_completions(self, messages):
        # Fall back to the sync version in a thread executor.
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._completions, messages)

    async def _completions_stream(self, messages):
        raise NotImplementedError("Streaming not implemented for VLLMClientAgent")