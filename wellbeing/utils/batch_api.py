"""
Provider Batch API support for the direct API agents in api_agents.py.

Opt-in per model via `use_batch_api: true` in configs/models.yaml. When enabled,
a whole wave of prompts is submitted as one (or a few) provider batch jobs
instead of thousands of live requests, at ~50% of the live price:

  - OpenAI    : /v1/batches (JSONL upload -> batch -> output file)
  - Anthropic : client.messages.batches (inline requests -> streamed results)
  - Gemini    : google-genai batch mode (JSONL upload via the File API)

Every request in a wave gets its own custom_id ("req-{index}"), so K samples per
prompt are K separate batch entries; results are re-assembled by custom_id, not
by arrival order. Items that fail/expire/error are resubmitted in a follow-up
batch and, if they still fail, fall back to live calls -- they are never
silently dropped.

The returned list is aligned with the input `messages_list` and uses None for
unrecoverable failures, exactly like DirectAPIAgent's live path.

Crash recovery is opt-in. Each run stamps its batches with a fresh random nonce,
so a run can reattach to its own in-flight batches but never to another run's --
identical resampling replicates share a payload signature, and adopting the
earlier replicate's finished batch would silently return old samples as new
data. To resume a genuinely crashed run, set WELLBEING_BATCH_RESUME=1, which
adopts the newest state file (and its nonce) for the wave. Never set it while
another replicate of the same wave is still in flight.
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

# Default polling / timeout behaviour. Provider batches routinely take hours,
# and both OpenAI and Anthropic use a 24h completion window, so we must not give
# up before that window has actually elapsed.
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_MAX_WAIT = 30 * 3600  # 30 hours
DEFAULT_RETRY_ROUNDS = 1
SUBMIT_MAX_ATTEMPTS = 3
FETCH_MAX_ATTEMPTS = 3

# Where submitted-batch state is persisted so a killed process can reattach to
# an in-flight (already paid for) batch instead of resubmitting it.
DEFAULT_STATE_DIR = os.environ.get(
    "WELLBEING_BATCH_STATE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".batch_state"),
)


# ---------------------------------------------------------------------------
#  Providers
# ---------------------------------------------------------------------------

class _BaseBatchProvider:
    """Adapts one provider's batch API to submit/poll/fetch.

    All three methods are synchronous (the provider SDKs are sync) and are run
    in a worker thread by the async driver below.
    """

    name = "base"
    # Conservative margins under the documented per-batch limits.
    max_requests = 10_000
    max_bytes = 100 * 1024 * 1024

    def __init__(self, agent):
        self.agent = agent

    def build_payload(self, custom_id: str, messages: List[Dict]) -> Dict[str, Any]:
        raise NotImplementedError

    def submit(self, payloads: List[Dict[str, Any]], tag: str = None) -> str:
        raise NotImplementedError

    def find_batch(self, tag: str) -> Optional[str]:
        """Find an already-created batch carrying `tag`, if the SDK allows it.

        Used to recover a batch whose create() call succeeded server-side but
        whose response we lost, so a submit retry does not double-bill.
        Providers that cannot tag or list batches return None.
        """
        return None

    def poll(self, batch_id: str):
        """Returns (done: bool, status: str, progress: str)."""
        raise NotImplementedError

    def fetch(self, batch_id: str) -> Dict[str, Optional[str]]:
        """Returns {custom_id: content_or_None} for *delivered* items only.

        custom_ids absent from the returned dict are treated as failures and
        are retried by the driver. A present-but-None value means the provider
        returned a successful response with empty content, which the live path
        also maps to None -- those are not retried.
        """
        raise NotImplementedError


class OpenAIBatchProvider(_BaseBatchProvider):
    """OpenAI /v1/batches. Limits: 50k requests and 200MB per input file."""

    name = "openai"
    max_requests = 45_000
    max_bytes = 180 * 1024 * 1024

    def build_payload(self, custom_id: str, messages: List[Dict]) -> Dict[str, Any]:
        prepared = self.agent._preprocess_messages(messages)
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.agent.model,
                "messages": prepared,
                **self.agent.generation_config,
            },
        }

    def submit(self, payloads: List[Dict[str, Any]], tag: str = None) -> str:
        client = self.agent.client
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for payload in payloads:
                f.write(json.dumps(payload) + "\n")
            path = f.name
        try:
            with open(path, "rb") as fh:
                uploaded = client.files.create(file=fh, purpose="batch")
            create_kwargs = dict(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            if tag:
                create_kwargs["metadata"] = {"wellbeing_tag": tag}
            batch = client.batches.create(**create_kwargs)
        finally:
            os.unlink(path)
        return batch.id

    def find_batch(self, tag: str) -> Optional[str]:
        if not tag:
            return None
        for batch in self.agent.client.batches.list(limit=100):
            metadata = getattr(batch, "metadata", None) or {}
            if metadata.get("wellbeing_tag") != tag:
                continue
            if batch.status in ("failed", "cancelled", "expired"):
                continue
            return batch.id
        return None

    def poll(self, batch_id: str):
        batch = self.agent.client.batches.retrieve(batch_id)
        counts = batch.request_counts
        progress = ""
        if counts is not None:
            progress = f"{counts.completed or 0}/{counts.total or 0} done, {counts.failed or 0} failed"
        done = batch.status in ("completed", "failed", "expired", "cancelled")
        return done, batch.status, progress

    def fetch(self, batch_id: str) -> Dict[str, Optional[str]]:
        client = self.agent.client
        batch = client.batches.retrieve(batch_id)
        results: Dict[str, Optional[str]] = {}
        if not getattr(batch, "output_file_id", None):
            return results
        text = client.files.content(batch.output_file_id).text
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id is None:
                continue
            if record.get("error"):
                continue  # failure -> omit so the driver retries it
            response = record.get("response") or {}
            if response.get("status_code") != 200:
                continue
            body = response.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                results[custom_id] = None
                continue
            content = (choices[0].get("message") or {}).get("content")
            results[custom_id] = content.strip() if content else None
        return results


class AnthropicBatchProvider(_BaseBatchProvider):
    """Anthropic Message Batches. Limits: 100k requests and 256MB per batch."""

    name = "anthropic"
    max_requests = 90_000
    max_bytes = 200 * 1024 * 1024

    @staticmethod
    def _strip_cache_control(obj):
        """Remove cache_control blocks (recursively) from a message structure.

        The live agent marks the final user block ephemeral, which pays off when
        many requests share a prefix. In a batch the final blocks are unique per
        request, so every entry would be a 1.25x cache *write* with no read --
        eroding the 50% batch discount. Live behaviour is untouched; this only
        applies to payloads built for the Batch API.
        """
        if isinstance(obj, dict):
            return {k: AnthropicBatchProvider._strip_cache_control(v)
                    for k, v in obj.items() if k != "cache_control"}
        if isinstance(obj, list):
            return [AnthropicBatchProvider._strip_cache_control(v) for v in obj]
        return obj

    def build_payload(self, custom_id: str, messages: List[Dict]) -> Dict[str, Any]:
        system, prepared = self.agent._preprocess_messages(messages)
        params: Dict[str, Any] = {
            "model": self.agent.model,
            "messages": self._strip_cache_control(prepared),
            **self.agent.generation_config,
        }
        if system is not None:
            params["system"] = self._strip_cache_control(system)
        params.setdefault("max_tokens", 20)
        return {"custom_id": custom_id, "params": params}

    def submit(self, payloads: List[Dict[str, Any]], tag: str = None) -> str:
        # The Message Batches API has no metadata/display-name field, so a
        # create whose response is lost cannot be located by tag; the driver
        # logs loudly instead (see find_batch's default None).
        batch = self.agent.client.messages.batches.create(requests=payloads)
        return batch.id

    def poll(self, batch_id: str):
        batch = self.agent.client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        progress = ""
        if counts is not None:
            progress = (
                f"{counts.succeeded} succeeded, {counts.processing} processing, "
                f"{counts.errored} errored, {counts.expired} expired, {counts.canceled} canceled"
            )
        status = batch.processing_status
        return status == "ended", status, progress

    def fetch(self, batch_id: str) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        for item in self.agent.client.messages.batches.results(batch_id):
            result = item.result
            if getattr(result, "type", None) != "succeeded":
                continue  # errored / expired / canceled -> retry
            message = result.message
            blocks = getattr(message, "content", None) or []
            # Mirror the live path, which reads the last content block.
            content = getattr(blocks[-1], "text", None) if blocks else None
            results[item.custom_id] = content.strip() if content else None
        return results


class GeminiBatchProvider(_BaseBatchProvider):
    """Gemini batch mode via the native google-genai SDK (File API + batches).

    The repo's `gemini_direct` agent talks to Gemini's OpenAI-compatible
    endpoint, which has no batch support, so batch mode uses the native SDK
    with the same GEMINI_API_KEY. Vertex AI (`vertex_gemini_direct`) is not
    supported here because Vertex batches require GCS/BigQuery I/O.
    """

    name = "gemini"
    max_requests = 45_000
    max_bytes = 1024 * 1024 * 1024

    def __init__(self, agent):
        super().__init__(agent)
        from google import genai  # noqa: F401  (import error surfaces early)

        api_key_env = getattr(agent, "api_key_env", None) or "GEMINI_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Gemini batch mode needs an API key in ${api_key_env} (native google-genai SDK)."
            )
        self.client = genai.Client(api_key=api_key)
        # Native SDK wants a bare model id ("gemini-2.5-flash"), not a
        # provider-prefixed one.
        model = self.agent.model
        for prefix in ("google/", "gemini/", "models/"):
            if model.startswith(prefix):
                model = model[len(prefix):]
        self.model = model

    @staticmethod
    def _to_gemini_messages(messages: List[Dict]):
        system_texts = []
        contents = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                system_texts.append(content if isinstance(content, str) else json.dumps(content))
                continue
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"text": item["text"]})
                    else:
                        raise ValueError(
                            f"Gemini batch mode only supports text content, got item type "
                            f"{item.get('type')!r}. Use live mode for multimodal prompts."
                        )
            else:
                parts = [{"text": str(content)}]
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
        system_instruction = None
        if system_texts:
            system_instruction = {"parts": [{"text": "\n\n".join(system_texts)}]}
        return system_instruction, contents

    def _generation_config(self) -> Dict[str, Any]:
        config = dict(self.agent.generation_config)
        gen: Dict[str, Any] = {}
        if "temperature" in config:
            gen["temperature"] = config["temperature"]
        max_tokens = config.get("max_tokens", config.get("max_completion_tokens"))
        if max_tokens is not None:
            gen["max_output_tokens"] = max_tokens
        return gen

    def build_payload(self, custom_id: str, messages: List[Dict]) -> Dict[str, Any]:
        system_instruction, contents = self._to_gemini_messages(messages)
        request: Dict[str, Any] = {"contents": contents}
        gen = self._generation_config()
        if gen:
            request["generation_config"] = gen
        if system_instruction is not None:
            request["system_instruction"] = system_instruction
        return {"key": custom_id, "request": request}

    @staticmethod
    def _display_name(tag: str) -> str:
        return f"wellbeing-{tag}"

    def submit(self, payloads: List[Dict[str, Any]], tag: str = None) -> str:
        from google.genai import types

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for payload in payloads:
                f.write(json.dumps(payload) + "\n")
            path = f.name
        try:
            uploaded = self.client.files.upload(
                file=path,
                config=types.UploadFileConfig(
                    display_name="wellbeing-batch-requests", mime_type="jsonl"
                ),
            )
            config = {"display_name": self._display_name(tag)} if tag else None
            job = self.client.batches.create(model=self.model, src=uploaded.name, config=config)
        finally:
            os.unlink(path)
        return job.name

    def find_batch(self, tag: str) -> Optional[str]:
        if not tag:
            return None
        wanted = self._display_name(tag)
        for job in self.client.batches.list(config={"page_size": 100}):
            if getattr(job, "display_name", None) != wanted:
                continue
            if self._state_name(job) in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                continue
            return job.name
        return None

    @staticmethod
    def _state_name(job) -> str:
        state = getattr(job, "state", None)
        return getattr(state, "name", str(state))

    def poll(self, batch_id: str):
        job = self.client.batches.get(name=batch_id)
        state = self._state_name(job)
        done = state in (
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_PARTIALLY_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        )
        stats = getattr(job, "completion_stats", None)
        progress = ""
        if stats is not None:
            progress = (
                f"{getattr(stats, 'successful_count', '?')} succeeded, "
                f"{getattr(stats, 'failed_count', '?')} failed"
            )
        return done, state, progress

    @staticmethod
    def _text_from_response(response: Dict[str, Any]) -> Optional[str]:
        candidates = response.get("candidates") or []
        if not candidates:
            return None
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts)
        return text.strip() if text else None

    def fetch(self, batch_id: str) -> Dict[str, Optional[str]]:
        job = self.client.batches.get(name=batch_id)
        results: Dict[str, Optional[str]] = {}
        dest = getattr(job, "dest", None)
        if dest is None:
            return results
        if getattr(dest, "file_name", None):
            data = self.client.files.download(file=dest.file_name)
            text = data.decode("utf-8") if isinstance(data, bytes) else data
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = record.get("key")
                if key is None or record.get("error"):
                    continue
                response = record.get("response")
                if not response:
                    continue
                results[key] = self._text_from_response(response)
        elif getattr(dest, "inlined_responses", None):
            # Defensive: small jobs may come back inline. Inline responses carry
            # the request metadata when set, otherwise they preserve input order.
            for item in dest.inlined_responses:
                metadata = getattr(item, "metadata", None) or {}
                key = metadata.get("key")
                if key is None or getattr(item, "error", None):
                    continue
                response = getattr(item, "response", None)
                if response is None:
                    continue
                as_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                results[key] = self._text_from_response(as_dict)
        return results


def get_batch_provider(agent) -> _BaseBatchProvider:
    """Pick the batch provider for a BaseLLMAgent, or raise if unsupported."""
    provider = getattr(agent, "provider", None)
    mapping = {
        "openai": OpenAIBatchProvider,
        "anthropic": AnthropicBatchProvider,
        "gemini": GeminiBatchProvider,
    }
    if provider not in mapping:
        raise ValueError(
            f"use_batch_api is not supported for provider {provider!r}. "
            "Batch mode is implemented for OpenAI, Anthropic, and native Gemini only "
            "(Vertex AI, xAI, OpenRouter, and OpenAI-compatible proxies have no batch endpoint here)."
        )
    return mapping[provider](agent)


def batch_api_supported(agent) -> bool:
    return getattr(agent, "provider", None) in ("openai", "anthropic", "gemini")


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

class _WaveState:
    """Crash-safe record of the batches submitted for one wave of payloads.

    A wave is identified by a stable hash of its payloads PLUS a per-run nonce.
    The hash alone is deliberately not enough: we run identical resampling
    replicates (same prompts, same K, same params), which share a signature, and
    adopting the previous replicate's finished batch would silently return old
    samples as new data. Every run therefore mints a fresh nonce, which goes
    into both the filename and the batch tags, so cross-run adoption is
    impossible -- a fresh run can only ever match batches it created itself.

    Reattaching to a previous process's batches is therefore OPT-IN, via
    WELLBEING_BATCH_RESUME=1. Default (off): always a fresh nonce and a fresh
    submit; state files are still written, so an operator can recover an
    orphaned batch manually. With resume on, the newest state file for this
    signature is adopted (nonce and all), which is what you want after a crash
    and what you must NOT use while another replicate of the same wave is in
    flight.

    The file is written BEFORE the create call (state 'submitting') so a process
    killed mid-create still leaves a trace, and is deleted only once the whole
    wave -- every retry round and the live fallback -- has been assembled.
    """

    def __init__(self, state_dir: str, signature: str, provider_name: str, model: str,
                 num_requests: int, resume: bool = False):
        import glob
        import uuid

        self.signature = signature
        existing_path = None
        candidates = sorted(
            glob.glob(os.path.join(state_dir, f"batch_wave_{signature}_*.json")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if candidates and not resume:
            print(f"[batch state] {len(candidates)} state file(s) exist for wave {signature} "
                  f"(e.g. {candidates[0]}), but WELLBEING_BATCH_RESUME is not set: treating this "
                  "as a new run with a fresh nonce. Nothing from the earlier run will be reused.")
        elif candidates:
            existing_path = candidates[0]

        self.data = None
        if existing_path:
            try:
                with open(existing_path) as f:
                    loaded = json.load(f)
                if loaded.get("wave_signature") == signature and loaded.get("nonce"):
                    self.data = loaded
                    self.path = existing_path
                    print(f"[batch state] WELLBEING_BATCH_RESUME set: adopting {existing_path} "
                          f"(nonce {loaded['nonce']})")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[batch state] ignoring unreadable state file {existing_path}: {e}")

        if self.data is None:
            nonce = uuid.uuid4().hex[:8]
            self.data = {
                "provider": provider_name,
                "model": model,
                "wave_signature": signature,
                "nonce": nonce,
                "num_requests": num_requests,
                "created_at": time.time(),
                "chunks": {},
            }
            self.path = os.path.join(state_dir, f"batch_wave_{signature}_{nonce}.json")

    @property
    def nonce(self) -> str:
        return self.data["nonce"]

    def tag(self, chunk_index: int) -> str:
        return f"{self.signature}-{self.nonce}-c{chunk_index}"

    @staticmethod
    def compute_signature(provider_name: str, model: str, payloads: List[Dict[str, Any]]) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"{provider_name}\0{model}\0".encode("utf-8"))
        for payload in payloads:
            hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
            hasher.update(b"\0")
        return hasher.hexdigest()[:32]

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp_path, self.path)

    def get_chunk(self, chunk_index: int) -> Optional[Dict[str, Any]]:
        return self.data["chunks"].get(str(chunk_index))

    def mark_submitting(self, chunk_index: int, custom_ids: List[str]):
        self.data["chunks"][str(chunk_index)] = {
            "state": "submitting",
            "batch_id": None,
            "num_requests": len(custom_ids),
            "custom_ids": custom_ids,
        }
        self.save()

    def mark_submitted(self, chunk_index: int, batch_id: str):
        entry = self.data["chunks"].setdefault(str(chunk_index), {})
        entry["state"] = "submitted"
        entry["batch_id"] = batch_id
        self.save()

    def delete(self):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except OSError as e:
            print(f"[batch state] could not remove {self.path}: {e}")


def _retry_sync(fn, *args, label: str, what: str, attempts: int, **kwargs):
    """Run a blocking provider call with backoff. Raises if all attempts fail."""
    last_error = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            print(f"[{label}] {what} attempt {attempt + 1}/{attempts} failed: {e}")
            if attempt < attempts - 1:
                time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"[{label}] {what} failed after {attempts} attempts: {last_error}")


def _chunk_payloads(payloads: List[Dict[str, Any]], max_requests: int, max_bytes: int):
    """Split payloads into chunks respecting per-batch request and size limits."""
    chunks = []
    current: List[Dict[str, Any]] = []
    current_bytes = 0
    for payload in payloads:
        size = len(json.dumps(payload).encode("utf-8")) + 1
        if current and (len(current) >= max_requests or current_bytes + size > max_bytes):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(payload)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


async def _submit_chunk(
    provider: _BaseBatchProvider,
    payloads: List[Dict[str, Any]],
    label: str,
    tag: Optional[str],
    state: Optional[_WaveState],
    chunk_index: int,
) -> str:
    """Submit one chunk, reattaching to an existing batch wherever possible."""
    entry = state.get_chunk(chunk_index) if state is not None else None

    if entry and entry.get("batch_id"):
        batch_id = entry["batch_id"]
        print(f"[{label}] reattaching to already-submitted {provider.name} batch {batch_id} "
              f"(not resubmitting {len(payloads)} requests)")
        return batch_id

    if entry and entry.get("state") == "submitting":
        # We were killed between writing state and recording the batch id, so
        # the create may or may not have landed.
        orphan = await asyncio.to_thread(provider.find_batch, tag)
        if orphan:
            print(f"[{label}] recovered orphaned {provider.name} batch {orphan} by tag {tag}")
            if state is not None:
                state.mark_submitted(chunk_index, orphan)
            return orphan
        print(f"[{label}] WARNING: previous run died during batch creation and no batch "
              f"matching tag {tag} was found. Submitting again -- if the earlier create "
              "actually landed, that batch is still billable; check the provider console.")

    if state is not None:
        state.mark_submitting(chunk_index, [p.get("custom_id") or p.get("key") for p in payloads])

    last_error = None
    for attempt in range(SUBMIT_MAX_ATTEMPTS):
        if attempt > 0:
            # A previous attempt may have succeeded server-side with only the
            # response lost; never blindly create a second billable batch.
            existing = await asyncio.to_thread(provider.find_batch, tag)
            if existing:
                print(f"[{label}] found batch {existing} created by a previous submit attempt; "
                      "reattaching instead of creating a duplicate")
                if state is not None:
                    state.mark_submitted(chunk_index, existing)
                return existing
            print(f"[{label}] WARNING: retrying batch creation; if {provider.name} cannot list "
                  "batches by tag, a duplicate (billable) batch is possible.")
        try:
            batch_id = await asyncio.to_thread(provider.submit, payloads, tag)
            if state is not None:
                state.mark_submitted(chunk_index, batch_id)
            print(f"[{label}] submitted {provider.name} batch {batch_id} with {len(payloads)} requests")
            return batch_id
        except Exception as e:  # transient upload/creation failures
            last_error = e
            print(f"[{label}] submit attempt {attempt + 1}/{SUBMIT_MAX_ATTEMPTS} failed: {e}")
            if attempt < SUBMIT_MAX_ATTEMPTS - 1:
                await asyncio.sleep(5.0 * (attempt + 1))

    raise RuntimeError(f"[{label}] failed to submit batch after {SUBMIT_MAX_ATTEMPTS} attempts: {last_error}")


async def _run_one_batch(
    provider: _BaseBatchProvider,
    payloads: List[Dict[str, Any]],
    poll_interval: float,
    max_wait: float,
    verbose: bool,
    label: str,
    tag: Optional[str] = None,
    state: Optional[_WaveState] = None,
    chunk_index: int = 0,
) -> Dict[str, Optional[str]]:
    batch_id = await _submit_chunk(provider, payloads, label, tag, state, chunk_index)
    start = time.time()
    while True:
        try:
            done, status, progress = await asyncio.to_thread(provider.poll, batch_id)
        except Exception as e:
            # A failed poll is not a failed batch; keep waiting.
            print(f"[{label}] poll error (will retry): {e}")
            done, status, progress = False, "poll_error", ""
        elapsed = (time.time() - start) / 60.0
        if verbose:
            print(f"[{label}] batch {batch_id}: status={status} {progress} (elapsed {elapsed:.1f} min)")
        if done:
            if status not in ("completed", "ended", "JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
                print(f"[{label}] WARNING: batch {batch_id} ended with status {status}; "
                      "unreturned items will be retried.")
            break
        if time.time() - start > max_wait:
            raise RuntimeError(
                f"[{label}] batch {batch_id} did not finish within {max_wait / 3600:.1f}h "
                f"(last status: {status}). The batch may still be running -- inspect it by id "
                "rather than resubmitting blindly."
            )
        await asyncio.sleep(poll_interval)

    # The batch is already paid for at this point: a transient network error
    # while downloading its results must not throw the whole wave away.
    results = await asyncio.to_thread(
        _retry_sync, provider.fetch, batch_id,
        label=label, what=f"fetching results for batch {batch_id}",
        attempts=FETCH_MAX_ATTEMPTS,
    )
    print(f"[{label}] batch {batch_id}: retrieved {len(results)}/{len(payloads)} results")
    return results


async def run_batch_completions(
    agent,
    messages_list: Sequence[List[Dict]],
    *,
    verbose: bool = True,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    max_wait: float = DEFAULT_MAX_WAIT,
    retry_rounds: int = DEFAULT_RETRY_ROUNDS,
    live_fallback: Optional[Callable[[List[List[Dict]]], Any]] = None,
    provider: Optional[_BaseBatchProvider] = None,
    state_dir: Optional[str] = DEFAULT_STATE_DIR,
    resume: Optional[bool] = None,
) -> List[Optional[str]]:
    """Run a wave of prompts through the provider's Batch API.

    Args:
        agent: A BaseLLMAgent (OpenAIAgent / AnthropicAgent / GeminiAgent).
        messages_list: One conversation per request. K samples of the same
            prompt are simply K identical entries at different indices, each of
            which gets its own custom_id.
        poll_interval: Seconds between status polls.
        max_wait: Hard ceiling per batch job, in seconds.
        retry_rounds: Follow-up batches for items the provider failed to return.
        live_fallback: Async callable taking a list of message lists and
            returning a list of responses; used for whatever is still missing
            after the retry rounds.
        provider: Override the auto-selected provider (used by tests).
        state_dir: Directory for crash-recovery state. Pass None to disable.
        resume: Whether to adopt a previous process's batches for this wave.
            Defaults to $WELLBEING_BATCH_RESUME. OFF by default so that an
            identical resampling replicate can never be handed the earlier
            run's samples; set it only when knowingly restarting a crashed run.
            Within a single run, crash-free reattach always works.

    Returns:
        List aligned with messages_list; None for unrecoverable failures.
    """
    if provider is None:
        provider = get_batch_provider(agent)

    total = len(messages_list)
    results: List[Optional[str]] = [None] * total
    pending = list(range(total))
    if resume is None:
        resume = os.environ.get("WELLBEING_BATCH_RESUME", "").strip() in ("1", "true", "True")
    wave_states: List[_WaveState] = []

    for round_idx in range(retry_rounds + 1):
        if not pending:
            break
        payloads = [provider.build_payload(f"req-{i}", messages_list[i]) for i in pending]
        chunks = _chunk_payloads(payloads, provider.max_requests, provider.max_bytes)
        round_label = f"{provider.name} round {round_idx}"

        state = None
        signature = None
        if state_dir is not None:
            signature = _WaveState.compute_signature(
                provider.name, getattr(provider, "model", None) or str(getattr(agent, "model", "")),
                payloads,
            )
            state = _WaveState(state_dir, signature, provider.name,
                               str(getattr(agent, "model", "")), len(payloads), resume=resume)
            wave_states.append(state)

        print(f"[{round_label}] submitting {len(payloads)} requests in {len(chunks)} batch job(s)"
              + (f" (wave {signature} nonce {state.nonce})" if state is not None else ""))

        chunk_results = await asyncio.gather(*[
            _run_one_batch(
                provider, chunk, poll_interval, max_wait, verbose,
                f"{round_label} chunk {i + 1}/{len(chunks)}",
                tag=state.tag(i) if state is not None else None,
                state=state, chunk_index=i,
            )
            for i, chunk in enumerate(chunks)
        ])
        delivered: Dict[str, Optional[str]] = {}
        for chunk_result in chunk_results:
            delivered.update(chunk_result)

        still_pending = []
        for i in pending:
            custom_id = f"req-{i}"
            if custom_id in delivered:
                results[i] = delivered[custom_id]
            else:
                still_pending.append(i)
        print(f"[{round_label}] {len(pending) - len(still_pending)}/{len(pending)} delivered, "
              f"{len(still_pending)} failed/missing")
        pending = still_pending

    if pending and live_fallback is not None:
        print(f"[{provider.name}] falling back to live calls for {len(pending)} item(s) "
              "that the Batch API never returned")
        fallback_results = await live_fallback([list(messages_list[i]) for i in pending])
        for i, response in zip(pending, fallback_results):
            results[i] = response
        pending = []

    if pending:
        print(f"[{provider.name}] WARNING: {len(pending)} item(s) could not be completed; "
              "returning None for them (same as the live path's exhausted-retries behaviour)")

    # Only now is the wave fully assembled (all rounds + live fallback), so only
    # now is it safe to drop the reattach records. Refetching a completed batch
    # is idempotent on all three providers, so keeping them until the end costs
    # nothing and keeps every round recoverable if we die mid-wave.
    for state in wave_states:
        state.delete()

    return results
