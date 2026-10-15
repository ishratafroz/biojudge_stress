"""Async LLM client.

Providers
  openai     any OpenAI-compatible endpoint: OpenAI, OpenRouter, Together, Groq, a local vLLM server,
             or Gemini's OpenAI-compatibility endpoint.
  anthropic  Anthropic Messages API.
  mock       deterministic simulator with planted biases (tests only).

Every successful completion is cached in SQLite keyed on the full request, so interrupted runs
resume for free and re-analysis never re-queries a model. Failed calls are not cached.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import time
from collections import Counter

from .mock import MockBackend


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, t REAL)")
        self.db.commit()
        self._pending = 0

    def get(self, k: str):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def set(self, k: str, v: str):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (k, v, time.time()))
        self._pending += 1
        if self._pending >= 200:
            self.flush()

    def flush(self):
        self.db.commit()
        self._pending = 0


class LLMClient:
    def __init__(self, cfg: dict, cache_path: str):
        self.cfg = cfg
        self.models = cfg["models"]
        self.rt = cfg["runtime"]
        self.cache = Cache(cache_path)
        self.sem = asyncio.Semaphore(self.rt["concurrency"])
        self.stats: Counter = Counter()
        self._clients: dict = {}
        self._mock = None
        if any(m["provider"] == "mock" for m in self.models.values()):
            self._mock = MockBackend(self.models, cfg.get("mock"))

    def family(self, key: str) -> str:
        return self.models[key]["family"]

    # ------------------------------------------------------------------ public
    def cache_key(self, key: str, system: str, user: str, *, max_tokens: int, temperature: float = 0.0,
                   task: str = "", nonce: int = 0) -> str:
        """The cache key of one call. `nonce` leaves the prompt untouched but addresses a fresh
        decode, which is how the repeat-decode check asks the same question twice (scripts/redecode.py)."""
        spec = self.models[key]
        req = {"provider": spec["provider"], "model": spec["model"], "system": system, "user": user,
               "max_tokens": max_tokens, "temperature": temperature, "nonce": nonce,
               "merge_system": bool(spec.get("merge_system")), "extra": spec.get("extra_body")}
        if spec["provider"] == "mock":
            req["mock_task"] = task
        return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()

    async def chat(self, key: str, system: str, user: str, *, max_tokens: int, temperature: float = 0.0,
                   task: str = "", meta: dict | None = None, nonce: int = 0) -> str | None:
        spec = self.models[key]
        k = self.cache_key(key, system, user, max_tokens=max_tokens, temperature=temperature,
                           task=task, nonce=nonce)
        hit = self.cache.get(k)
        if hit is not None:
            self.stats["cache_hit"] += 1
            return hit
        async with self.sem:
            text = await self._call_with_retry(spec, key, system, user, max_tokens, temperature, task, meta, nonce)
        if text is not None:
            self.cache.set(k, text)
            self.stats[f"ok:{key}"] += 1
        else:
            self.stats[f"fail:{key}"] += 1
        return text

    def close(self):
        self.cache.flush()

    # ------------------------------------------------------------------ internals
    async def _call_with_retry(self, spec, key, system, user, max_tokens, temperature, task, meta, nonce):
        if spec["provider"] == "mock":
            return self._mock.respond(key, system, user, task, meta or {}, nonce)
        last = None
        for attempt in range(self.rt["max_api_retries"]):
            try:
                if spec["provider"] == "openai":
                    return await self._openai(spec, system, user, max_tokens, temperature, nonce)
                if spec["provider"] == "anthropic":
                    return await self._anthropic(spec, system, user, max_tokens, temperature)
                raise ValueError(f"unknown provider {spec['provider']}")
            except Exception as e:  # noqa: BLE001 - network errors vary by SDK
                last = e
                wait = min(60.0, 2 ** attempt) + random.random()
                self.stats[f"retry:{key}"] += 1
                await asyncio.sleep(wait)
        print(f"[llm] giving up on {key}: {type(last).__name__}: {str(last)[:200]}")
        return None

    def _openai_client(self, spec):
        ck = ("openai", spec.get("base_url"), spec.get("api_key_env"))
        if ck not in self._clients:
            from openai import AsyncOpenAI
            api_key = os.environ.get(spec.get("api_key_env") or "OPENAI_API_KEY", "EMPTY")
            self._clients[ck] = AsyncOpenAI(base_url=spec.get("base_url"), api_key=api_key,
                                            timeout=self.rt["request_timeout"], max_retries=0)
        return self._clients[ck]

    async def _openai(self, spec, system, user, max_tokens, temperature, nonce):
        client = self._openai_client(spec)
        if spec.get("merge_system"):
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = dict(model=spec["model"], messages=messages, max_tokens=max_tokens, temperature=temperature)
        if spec.get("send_seed", True):
            kw["seed"] = int(self.cfg["seed"]) + int(nonce)
        if spec.get("extra_body"):
            kw["extra_body"] = spec["extra_body"]
        r = await client.chat.completions.create(**kw)
        return r.choices[0].message.content or ""

    async def _anthropic(self, spec, system, user, max_tokens, temperature):
        ck = ("anthropic", spec.get("api_key_env"))
        if ck not in self._clients:
            from anthropic import AsyncAnthropic
            self._clients[ck] = AsyncAnthropic(api_key=os.environ.get(spec.get("api_key_env") or "ANTHROPIC_API_KEY"),
                                               timeout=self.rt["request_timeout"], max_retries=0)
        client = self._clients[ck]
        r = await client.messages.create(model=spec["model"], system=system, max_tokens=max_tokens,
                                         temperature=temperature, messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in r.content)


async def gather_progress(factories, label: str, client: "LLMClient | None" = None, chunk: int = 4000):
    """Run zero-argument coroutine factories concurrently, in chunks, with progress output.

    Prompts are built inside each coroutine, so only one chunk of prompts lives in memory at a time.
    The client's semaphore bounds the number of in-flight requests.
    """
    total = len(factories)
    results = [None] * total
    t0 = time.time()
    for s in range(0, total, chunk):
        idx = range(s, min(s + chunk, total))
        out = await asyncio.gather(*(factories[i]() for i in idx))
        for i, r in zip(idx, out):
            results[i] = r
        if client is not None:
            client.cache.flush()
        print(f"  [{label}] {min(s + chunk, total)}/{total}  ({time.time() - t0:.0f}s)", flush=True)
    return results
