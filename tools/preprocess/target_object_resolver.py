"""LLM-based resolver mapping (instruction, candidates) → target object id.

Used by dataset preprocessors that need to figure out which scene object a
natural-language instruction is talking about. Calls DeepSeek (OpenAI-API
compatible) with retries and an on-disk cache, and returns None on any
failure so the caller can decide the fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TargetResolveError(Exception):
    """Raised by callers when no resolution strategy produces a target."""


class LLMTargetResolver:
    """Resolve target objects via an OpenAI-compatible chat API.

    Stateless from the caller's perspective: construct once, call
    `resolve(instruction, candidates)` many times. All caching, retrying,
    and validation lives inside this class.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        cache_path: str | Path | None = None,
        max_attempts: int = 3,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError("LLMTargetResolver requires a non-empty api_key")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._cache_path = Path(cache_path) if cache_path else None
        self._max_attempts = max_attempts
        self._timeout = timeout
        self._cache: dict[str, dict[str, Any]] = {}
        self._client = None  # lazy-init in _get_client
        self._load_cache()

    def resolve(
        self,
        instruction: str,
        candidates: list[dict],
    ) -> str | None:
        if not candidates:
            return None

        candidate_ids = [c["id"] for c in candidates]
        valid_ids = set(candidate_ids)

        key = self._cache_key(instruction, candidates)
        cached = self._cache.get(key)
        if cached is not None and cached.get("resolved_id") in valid_ids:
            return cached["resolved_id"]

        resolved = self._call_api(instruction, candidates, valid_ids)
        if resolved is None:
            return None

        self._cache[key] = {
            "instruction": instruction,
            "candidate_ids": sorted(candidate_ids),
            "resolved_id": resolved,
            "model": self._model,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self._save_cache()
        return resolved

    @staticmethod
    def _cache_key(instruction: str, candidates: list[dict]) -> str:
        ids = sorted(c["id"] for c in candidates)
        payload = json.dumps([instruction, ids], ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    SYSTEM_PROMPT = (
        "You are a robot task planner. Given a natural-language instruction "
        "and a list of candidate objects in the scene, pick the SINGLE object "
        "the robot is supposed to manipulate. You must return strictly valid "
        "JSON of the form {\"object_id\": \"<id>\"} where <id> is exactly one "
        "of the provided candidate ids. No prose, no markdown, no code fences."
    )

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "LLMTargetResolver requires the 'openai' package. "
                    "Install it in your conda env: pip install 'openai>=1.0'"
                ) from e
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    def _build_user_message(self, instruction: str, candidates: list[dict]) -> str:
        lines = [f"Instruction: {instruction}", "", "Candidates:"]
        for c in candidates:
            lines.append(f"- id: {c['id']}, name: {c.get('name', c['id'])}")
        return "\n".join(lines)

    def _parse_response(self, content: str, valid_ids: set[str]) -> str | None:
        payload = self._loads_lenient(content)
        if not isinstance(payload, dict):
            return None
        obj_id = payload.get("object_id")
        if not isinstance(obj_id, str) or obj_id not in valid_ids:
            return None
        return obj_id

    @staticmethod
    def _loads_lenient(content: str) -> Any:
        """Parse JSON, tolerating a single ```...``` markdown code fence.

        DeepSeek-V3/R1 (and other chat models) often wrap JSON in a fenced
        block despite being told not to. We try strict parse first, then
        strip an outer fence and retry once.
        """
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if not text.startswith("```"):
            return None
        lines = text.splitlines()
        # Drop the opening fence line (e.g. ``` or ```json).
        lines = lines[1:]
        # Drop a trailing fence line if present.
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        try:
            return json.loads("\n".join(lines).strip())
        except json.JSONDecodeError:
            return None

    def _call_api(
        self,
        instruction: str,
        candidates: list[dict],
        valid_ids: set[str],
    ) -> str | None:
        """Call the LLM with retries.

        Strategy:
        - Up to `max_attempts` total calls.
        - Network errors → exponential backoff (1s, 2s, 4s, ...) between attempts.
        - JSON parse failure or invalid object_id → immediate next attempt
          without sleeping (let the model self-correct), counted against
          `max_attempts`.
        - 4xx-style errors are treated as fatal and return None immediately.
        """
        client = self._get_client()
        user_msg = self._build_user_message(instruction, candidates)

        last_was_network_error = False
        for attempt in range(self._max_attempts):
            if attempt > 0 and last_was_network_error:
                time.sleep(2 ** (attempt - 1))

            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                )
            except Exception as e:
                if self._is_fatal_error(e):
                    logger.error(f"LLM call fatal error (no retry): {e}")
                    return None
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{self._max_attempts} "
                    f"network error: {e}"
                )
                last_was_network_error = True
                continue

            content = resp.choices[0].message.content or ""
            parsed = self._parse_response(content, valid_ids)
            if parsed is not None:
                return parsed

            logger.warning(
                f"LLM attempt {attempt + 1}/{self._max_attempts} "
                f"returned invalid response: {content!r}"
            )
            last_was_network_error = False

        return None

    @staticmethod
    def _is_fatal_error(exc: Exception) -> bool:
        """Return True for errors we should not retry (auth, quota, bad request)."""
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(exc, "http_status", None)
        if isinstance(status, int) and 400 <= status < 500:
            return True
        return False

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            self._cache = {}
            return
        try:
            self._cache = json.loads(self._cache_path.read_text())
            if not isinstance(self._cache, dict):
                raise ValueError("cache root is not a dict")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Target resolver cache at {self._cache_path} is corrupt "
                f"({e}); starting with empty cache."
            )
            self._cache = {}

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, indent=2))
        os.replace(tmp, self._cache_path)
