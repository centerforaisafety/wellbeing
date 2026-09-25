#!/usr/bin/env python3
"""
Self-test for provider Batch API mode (utils/batch_api.py).

Two modes:

  --mock       No API keys, no network. Exercises custom_id mapping, the
               failed-item retry round, live fallback, and chunking against a
               simulated provider. Safe to run anywhere.

  --provider   Real, tiny end-to-end check: 5 prompts x K=2 = 10 batch entries
               against a cheap model, run through the same
               compute_utilities.generate_responses() choke point the real
               pipeline uses. Costs a fraction of a cent, but a batch can take
               up to 24h to come back.

Usage:
    python scripts/test_batch_mode.py --mock
    python scripts/test_batch_mode.py --provider openai
    python scripts/test_batch_mode.py --provider anthropic --model claude-3-5-haiku-20241022
    python scripts/test_batch_mode.py --provider gemini --poll-interval 20
"""
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Five prompts, each with a unique expected answer, so a mis-mapped custom_id
# shows up as a wrong word rather than as a silently plausible response.
WORDS = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]
PROMPTS = [f"Reply with exactly one word, in capitals: {w}. Output nothing else." for w in WORDS]

CHEAP_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini": "gemini-2.0-flash-lite",
}


# ---------------------------------------------------------------------------
#  Mocked tests (no network)
# ---------------------------------------------------------------------------

def run_mock_tests() -> int:
    import json
    import tempfile

    from utils.batch_api import (
        _BaseBatchProvider, _chunk_payloads, _WaveState, run_batch_completions,
    )

    failures = []

    def check(name, condition, detail=""):
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name} {detail}")
            failures.append(name)

    class FakeProvider(_BaseBatchProvider):
        """Simulates a provider whose results arrive shuffled and whose first
        attempt drops some items entirely."""

        name = "fake"

        def __init__(self, drop_ids=(), drop_forever_ids=(), max_requests=1000,
                     fetch_failures=0, existing_batches=None):
            super().__init__(agent=None)
            self.model = "fake-model"
            self.drop_ids = set(drop_ids)
            self.drop_forever_ids = set(drop_forever_ids)
            self.max_requests = max_requests
            self.max_bytes = 10 * 1024 * 1024
            self.submitted_rounds = []
            self.fetch_failures = fetch_failures
            self.fetch_calls = 0
            # tag -> batch id, as if created by a previous (lost) run
            self.existing_batches = dict(existing_batches or {})
            self.batch_contents = {}

        def build_payload(self, custom_id, messages):
            return {"custom_id": custom_id, "text": messages[-1]["content"]}

        def submit(self, payloads, tag=None):
            self.submitted_rounds.append([p["custom_id"] for p in payloads])
            batch_id = f"fakebatch-{len(self.submitted_rounds)}"
            self.batch_contents[batch_id] = [p["custom_id"] for p in payloads]
            return batch_id

        def find_batch(self, tag):
            return self.existing_batches.get(tag)

        def poll(self, batch_id):
            return True, "completed", "all done"

        def fetch(self, batch_id):
            self.fetch_calls += 1
            if self.fetch_calls <= self.fetch_failures:
                raise ConnectionError("simulated transient network error during download")
            custom_ids = self.batch_contents[batch_id]
            out = {}
            # Reversed on purpose: results must be matched by custom_id, never
            # by arrival order.
            for custom_id in reversed(custom_ids):
                if custom_id in self.drop_forever_ids:
                    continue
                if custom_id in self.drop_ids:
                    self.drop_ids.discard(custom_id)  # succeeds on the retry round
                    continue
                out[custom_id] = f"answer-for-{custom_id}"
            return out

    messages_list = [[{"role": "user", "content": f"prompt-{i}"}] for i in range(10)]

    # 1. custom_id mapping with shuffled results
    provider = FakeProvider()
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=tempfile.mkdtemp(),
    ))
    check("mapping: length preserved", len(results) == 10, f"(got {len(results)})")
    check("mapping: each index gets its own custom_id's result",
          all(results[i] == f"answer-for-req-{i}" for i in range(10)), f"(got {results})")
    check("mapping: single round when nothing fails", len(provider.submitted_rounds) == 1,
          f"(got {len(provider.submitted_rounds)} rounds)")

    # 2. failed items are retried in a follow-up batch
    provider = FakeProvider(drop_ids={"req-3", "req-7"})
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=tempfile.mkdtemp(),
    ))
    check("retry: follow-up batch submitted", len(provider.submitted_rounds) == 2,
          f"(got {len(provider.submitted_rounds)} rounds)")
    check("retry: follow-up contains exactly the failed items",
          sorted(provider.submitted_rounds[1]) == ["req-3", "req-7"],
          f"(got {provider.submitted_rounds[1]})")
    check("retry: all items resolved",
          all(results[i] == f"answer-for-req-{i}" for i in range(10)), f"(got {results})")

    # 3. persistently failing items fall back to live calls
    provider = FakeProvider(drop_forever_ids={"req-2"})
    fallback_calls = []

    async def live_fallback(remaining):
        fallback_calls.append(remaining)
        return ["live-answer" for _ in remaining]

    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0,
        provider=provider, live_fallback=live_fallback, state_dir=tempfile.mkdtemp(),
    ))
    check("fallback: invoked once for the unresolved item",
          len(fallback_calls) == 1 and len(fallback_calls[0]) == 1,
          f"(got {fallback_calls})")
    check("fallback: unresolved item filled from live call", results[2] == "live-answer",
          f"(got {results[2]!r})")
    check("fallback: other items untouched",
          all(results[i] == f"answer-for-req-{i}" for i in range(10) if i != 2))

    # 4. no live fallback available -> None, never a silent wrong answer
    provider = FakeProvider(drop_forever_ids={"req-5"})
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=tempfile.mkdtemp(),
    ))
    check("no-fallback: unresolved item is None", results[5] is None, f"(got {results[5]!r})")

    # 5. chunking respects request-count and byte limits
    provider = FakeProvider(max_requests=3)
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=tempfile.mkdtemp(),
    ))
    check("chunking: 10 requests split into 4 jobs at max_requests=3",
          len(provider.submitted_rounds) == 4, f"(got {len(provider.submitted_rounds)})")
    check("chunking: still correctly mapped across chunks",
          all(results[i] == f"answer-for-req-{i}" for i in range(10)), f"(got {results})")
    payloads = [{"custom_id": f"req-{i}", "text": "x" * 1000} for i in range(10)]
    chunks = _chunk_payloads(payloads, max_requests=1000, max_bytes=3000)
    check("chunking: byte limit honoured",
          all(len(c) <= 3 for c in chunks) and sum(len(c) for c in chunks) == 10,
          f"(got sizes {[len(c) for c in chunks]})")

    # 6. a transient failure while downloading results must be retried, not
    #    thrown away -- the batch has already been paid for.
    provider = FakeProvider(fetch_failures=2)
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=tempfile.mkdtemp(),
    ))
    check("fetch retry: survives 2 transient download errors",
          all(results[i] == f"answer-for-req-{i}" for i in range(10)), f"(got {results})")
    check("fetch retry: fetch attempted 3 times, batch not resubmitted",
          provider.fetch_calls == 3 and len(provider.submitted_rounds) == 1,
          f"(fetch_calls={provider.fetch_calls}, submits={len(provider.submitted_rounds)})")

    provider = FakeProvider(fetch_failures=99)
    try:
        asyncio.run(run_batch_completions(
            None, messages_list, verbose=False, poll_interval=0, provider=provider,
            state_dir=tempfile.mkdtemp(),
        ))
        check("fetch retry: gives up loudly after the retry budget", False, "(no exception)")
    except RuntimeError:
        check("fetch retry: gives up loudly after the retry budget", True)

    signature = _WaveState.compute_signature(
        "fake", "fake-model",
        [FakeProvider().build_payload(f"req-{i}", messages_list[i]) for i in range(10)],
    )
    NONCE = "deadbeef"

    def write_state(state_dir, chunk_entry, nonce=NONCE):
        """Simulate a state file left behind by an earlier process."""
        path = Path(state_dir) / f"batch_wave_{signature}_{nonce}.json"
        path.write_text(json.dumps({
            "provider": "fake", "model": "fake-model", "wave_signature": signature,
            "nonce": nonce, "num_requests": 10, "created_at": 0,
            "chunks": {"0": chunk_entry},
        }))
        return path

    # 7. reattach (WELLBEING_BATCH_RESUME on): a persisted batch id must be
    #    polled, never resubmitted.
    state_dir = tempfile.mkdtemp()
    provider = FakeProvider()
    provider.batch_contents["fakebatch-99"] = [f"req-{i}" for i in range(10)]
    state_path = write_state(state_dir, {
        "state": "submitted", "batch_id": "fakebatch-99", "num_requests": 10,
        "custom_ids": [f"req-{i}" for i in range(10)],
    })
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=state_dir, resume=True,
    ))
    check("reattach: nothing resubmitted for a persisted batch",
          provider.submitted_rounds == [], f"(got {provider.submitted_rounds})")
    check("reattach: results collected from the persisted batch",
          all(results[i] == f"answer-for-req-{i}" for i in range(10)), f"(got {results})")
    check("reattach: state file removed once the wave is fully assembled",
          not state_path.exists())

    # 8. orphan recovery: killed mid-create ('submitting', no batch id) -> the
    #    batch is located by its nonce-bearing tag, not created a second time.
    state_dir = tempfile.mkdtemp()
    provider = FakeProvider(existing_batches={f"{signature}-{NONCE}-c0": "fakebatch-77"})
    provider.batch_contents["fakebatch-77"] = [f"req-{i}" for i in range(10)]
    write_state(state_dir, {
        "state": "submitting", "batch_id": None, "num_requests": 10,
        "custom_ids": [f"req-{i}" for i in range(10)],
    })
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=state_dir, resume=True,
    ))
    check("orphan recovery: found by tag, no duplicate batch created",
          provider.submitted_rounds == [], f"(got {provider.submitted_rounds})")
    check("orphan recovery: results collected from the recovered batch",
          all(results[i] == f"answer-for-req-{i}" for i in range(10)), f"(got {results})")

    # 9. REPLICATE SAFETY: an identical re-run (same prompts/K/params, so the
    #    same signature) with resume off must mint a fresh nonce and must never
    #    adopt the earlier run's state file OR its completed batch.
    state_dir = tempfile.mkdtemp()
    stale_path = write_state(state_dir, {
        "state": "submitted", "batch_id": "fakebatch-77", "num_requests": 10,
        "custom_ids": [f"req-{i}" for i in range(10)],
    })
    tags_used = []

    class TagRecordingProvider(FakeProvider):
        def submit(self, payloads, tag=None):
            tags_used.append(tag)
            return super().submit(payloads, tag)

    provider2 = TagRecordingProvider(
        existing_batches={f"{signature}-{NONCE}-c0": "fakebatch-77"})
    provider2.batch_contents["fakebatch-77"] = [f"req-{i}" for i in range(10)]
    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider2,
        state_dir=state_dir,  # resume defaults off
    ))
    check("replicate: submits fresh batches instead of adopting the prior run's",
          len(provider2.submitted_rounds) == 1 and provider2.submitted_rounds[0] ==
          [f"req-{i}" for i in range(10)], f"(got {provider2.submitted_rounds})")
    check("replicate: fresh nonce in the tag, so find_batch cannot match run 1",
          len(tags_used) == 1 and tags_used[0] is not None
          and not tags_used[0].startswith(f"{signature}-{NONCE}-"),
          f"(got {tags_used})")
    check("replicate: results come from the new batch, not run 1's",
          all(results[i] == f"answer-for-req-{i}" for i in range(10)), f"(got {results})")
    check("replicate: run 1's state file left untouched", stale_path.exists())

    # 10. a fresh wave writes 'submitting' before creating, so a crash
    #     mid-create is always recoverable.
    state_dir = tempfile.mkdtemp()
    seen_state = {}

    class StateCheckingProvider(FakeProvider):
        def submit(self, payloads, tag=None):
            for path in Path(state_dir).glob(f"batch_wave_{signature}_*.json"):
                seen_state["at_submit"] = json.loads(path.read_text())["chunks"]["0"]["state"]
                seen_state["nonce_in_name"] = path.stem.split("_")[-1]
                seen_state["nonce_in_tag"] = tag
            return super().submit(payloads, tag)

    provider = StateCheckingProvider()
    asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=state_dir,
    ))
    check("pre-create persistence: state written as 'submitting' before create",
          seen_state.get("at_submit") == "submitting", f"(got {seen_state})")
    check("pre-create persistence: filename nonce matches the tag nonce",
          seen_state.get("nonce_in_tag", "").split("-")[-2] == seen_state.get("nonce_in_name"),
          f"(got {seen_state})")

    # 11. state must survive every retry round and the live fallback, and only
    #     be deleted once the full result list is assembled.
    state_dir = tempfile.mkdtemp()
    provider = FakeProvider(drop_forever_ids={"req-4"})
    seen_during_fallback = {}

    async def checking_fallback(remaining):
        seen_during_fallback["files"] = list(Path(state_dir).glob("batch_wave_*.json"))
        return ["live-answer" for _ in remaining]

    results = asyncio.run(run_batch_completions(
        None, messages_list, verbose=False, poll_interval=0, provider=provider,
        state_dir=state_dir, live_fallback=checking_fallback,
    ))
    check("late deletion: state for every round still present during live fallback",
          len(seen_during_fallback.get("files", [])) == 2,
          f"(got {[p.name for p in seen_during_fallback.get('files', [])]})")
    check("late deletion: all state removed once results are assembled",
          list(Path(state_dir).glob("batch_wave_*.json")) == [])
    check("late deletion: results still complete", results[4] == "live-answer"
          and all(results[i] == f"answer-for-req-{i}" for i in range(10) if i != 4))

    # 10. Anthropic batch payloads must not carry cache_control (cost fix).
    import os as _os
    _os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-for-payload-build")
    from utils.api_agents import AnthropicAgent
    from utils.batch_api import AnthropicBatchProvider
    anthropic_agent = AnthropicAgent(model="claude-3-5-haiku-20241022", use_cache=True,
                                     temperature=1.0, max_tokens=10)
    payload = AnthropicBatchProvider(anthropic_agent).build_payload(
        "req-0", [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
    )
    check("anthropic: cache_control stripped in batch mode",
          "cache_control" not in json.dumps(payload), f"(got {payload})")
    live_system, live_messages = anthropic_agent._preprocess_messages(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    check("anthropic: live path still writes cache_control",
          "cache_control" in json.dumps(live_messages), f"(got {live_messages})")

    # 11. end-to-end through the real choke point: generate_responses() must
    #    hand back {prompt_idx: [K responses]} with each sample of a prompt
    #    coming from that prompt's own custom_ids.
    import utils.batch_api as batch_api
    from metrics.compute_utilities.utils import generate_responses
    from utils.api_agents import DirectAPIAgent

    prompts = [f"prompt-{i}" for i in range(4)]
    k = 3
    provider = FakeProvider()
    original_get_provider = batch_api.get_batch_provider
    batch_api.get_batch_provider = lambda agent: provider
    try:
        agent = DirectAPIAgent(agent=None, model_name="fake-model", use_batch_api=True,
                               batch_state_dir=tempfile.mkdtemp())
        results = asyncio.run(generate_responses(agent, prompts, K=k, verbose=False))
    finally:
        batch_api.get_batch_provider = original_get_provider

    check("integration: one dict entry per prompt", sorted(results.keys()) == list(range(4)),
          f"(got {sorted(results.keys())})")
    check("integration: K responses per prompt",
          all(len(v) == k for v in results.values()),
          f"(got {[len(v) for v in results.values()]})")
    # generate_responses sends messages * K, so prompt i owns custom_ids
    # i, i+n, i+2n, ... Every sample must trace back to one of them.
    expected = {i: [f"answer-for-req-{i + j * len(prompts)}" for j in range(k)] for i in range(4)}
    check("integration: samples map back to the right prompt", results == expected,
          f"(got {results})")
    check("integration: n-parameter path disabled in batch mode",
          DirectAPIAgent(agent=None, use_batch_api=True, model_name="m").supports_n_parameter is False)

    print()
    if failures:
        print(f"{len(failures)} mocked check(s) FAILED: {failures}")
        return 1
    print("All mocked checks passed.")
    return 0


# ---------------------------------------------------------------------------
#  Real provider test (needs an API key; costs a fraction of a cent)
# ---------------------------------------------------------------------------

def run_provider_test(provider_name: str, model: str, k: int, poll_interval: float) -> int:
    from metrics.compute_utilities.utils import generate_responses
    from utils.api_agents import (
        AnthropicAgent, DirectAPIAgent, GeminiAgent, OpenAIAgent,
    )

    agent_classes = {"openai": OpenAIAgent, "anthropic": AnthropicAgent, "gemini": GeminiAgent}
    generation_config = {"temperature": 1.0, "max_tokens": 20}
    underlying = agent_classes[provider_name](model=model, **generation_config)

    agent = DirectAPIAgent(
        agent=underlying,
        concurrency_limit=5,
        model_name=model,
        use_batch_api=True,
        batch_poll_interval=poll_interval,
    )

    print(f"Submitting {len(PROMPTS)} prompts x K={k} = {len(PROMPTS) * k} batch entries "
          f"to {provider_name}/{model} ...")
    results = asyncio.run(generate_responses(agent, PROMPTS, K=k, verbose=True))

    ok = True
    if sorted(results.keys()) != list(range(len(PROMPTS))):
        print(f"FAIL: expected prompt indices 0..{len(PROMPTS) - 1}, got {sorted(results.keys())}")
        return 1
    for prompt_idx, word in enumerate(WORDS):
        responses = results[prompt_idx]
        if len(responses) != k:
            print(f"FAIL: prompt {prompt_idx} has {len(responses)} responses, expected {k}")
            ok = False
            continue
        for sample_idx, response in enumerate(responses):
            match = response is not None and word in response.upper()
            print(f"  prompt {prompt_idx} (expect {word}) sample {sample_idx}: {response!r} "
                  f"{'OK' if match else 'MISMATCH'}")
            if not match:
                ok = False

    print()
    print("Batch result mapping verified." if ok else "Batch result mapping FAILED.")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true",
                        help="Run offline mocked checks (no API keys needed).")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"],
                        help="Run a real tiny batch against this provider.")
    parser.add_argument("--model", default=None, help="Override the cheap default model.")
    parser.add_argument("--k", type=int, default=2, help="Samples per prompt (default: 2).")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                        help="Seconds between status polls (default: 30).")
    args = parser.parse_args()

    if not args.mock and not args.provider:
        parser.error("pass --mock and/or --provider")

    status = 0
    if args.mock:
        print("=== mocked batch-mode checks ===")
        status |= run_mock_tests()
    if args.provider:
        print(f"=== real batch check: {args.provider} ===")
        status |= run_provider_test(
            args.provider, args.model or CHEAP_MODELS[args.provider],
            args.k, args.poll_interval,
        )
    sys.exit(status)


if __name__ == "__main__":
    main()
