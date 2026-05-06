# Copyright (c) 2026 Nima Abaeian
#
# Author: Nima Abaeian
# Organization: Istituto Italiano di Tecnologia
# Lab: Cognitive Architecture for Collaborative Technologies
# License: GNU GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"executiveControl.py - iCub social interaction controller"

from __future__ import annotations

import concurrent.futures
import fcntl
import json
import os
import queue
import random
import re
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
import uuid

sys.setswitchinterval(0.05)
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_ALWAYSON_DIR = os.path.dirname(_MODULE_DIR)

load_dotenv()
load_dotenv(os.path.join(_MODULE_DIR, "llm.env"), override=False)

import httpx                   
from openai import AzureOpenAI 
import yarp                     

# ──────────────────────────────────────────────────────────────────────────────
# Orexigenic drive model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HungerSnapshot:
    level:             float
    state:             str           # HS1 | HS2 | HS3
    last_feed_ts:      float
    last_feed_payload: Optional[str]


class HungerModel:
    """Thread-safe stomach level that drains over time and persists to disk."""

    def __init__(
        self,
        drain_hours:        float = 5.0,
        hungry_threshold:   float = 60.0,
        starving_threshold: float = 25.0,
        persist_file:       Optional[str] = None,
        log_cb=None,
    ):
        self.drain_hours        = drain_hours
        self.hungry_threshold   = hungry_threshold
        self.starving_threshold = starving_threshold
        self.persist_file       = persist_file
        self._log               = log_cb or (lambda lvl, msg: None)

        self._lock              = threading.Lock()
        self.level:             float          = 100.0
        self.last_update_ts:    float          = time.time()
        self.last_feed_ts:      float          = 0.0
        self.last_feed_payload: Optional[str]  = None
        self._last_logged_pct:  int            = 100

        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.persist_file or not os.path.isfile(self.persist_file):
            return
        try:
            with open(self.persist_file, encoding="utf-8") as fh:
                d = json.load(fh)
            now = time.time()
            lvl = max(0.0, min(100.0, float(d.get("level", 100.0))))
            ts  = float(d.get("last_update_ts", now))
            if ts <= 0 or ts > now:
                ts = now
            self.level             = lvl
            self._last_logged_pct  = int(lvl)
            self.last_update_ts    = ts
            self.last_feed_ts      = float(d.get("last_feed_ts", 0.0) or 0.0)
            payload                = d.get("last_feed_payload")
            self.last_feed_payload = payload if isinstance(payload, str) else None
        except Exception as e:
            self._log("WARNING", f"HungerModel load failed: {e}")

    def _save(self) -> None:
        """Atomic write (must be called with lock held)."""
        if not self.persist_file:
            return
        try:
            directory = os.path.dirname(self.persist_file) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "level":             self.level,
                            "last_update_ts":    self.last_update_ts,
                            "last_feed_ts":      self.last_feed_ts,
                            "last_feed_payload": self.last_feed_payload,
                        },
                        fh,
                    )
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.persist_file)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            self._log("WARNING", f"HungerModel save failed: {e}")

    # ── public API ────────────────────────────────────────────────────────────

    def _drain(self, now: float) -> int:
        """Apply elapsed drain; return old percentage (must hold lock)."""
        old_pct     = int(self.level)
        rate        = (100.0 / (self.drain_hours * 3600.0)) if self.drain_hours > 0 else 0.0
        elapsed     = now - self.last_update_ts
        if elapsed > 0:
            self.level = max(0.0, min(100.0, self.level - elapsed * rate))
        self.last_update_ts = now
        return old_pct

    def update(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        with self._lock:
            old = self._drain(now)
            cur = int(self.level)
            if cur < self._last_logged_pct:
                self._last_logged_pct = cur
                self._log("DEBUG", f"Hunger: {cur}%")
            if cur != old:
                self._save()

    def feed(self, delta: float, payload: str, now: Optional[float] = None) -> None:
        now = now or time.time()
        with self._lock:
            self._drain(now)
            self.level             = min(100.0, self.level + delta)
            self.last_feed_ts      = now
            self.last_feed_payload = payload
            self._last_logged_pct  = int(self.level)
            self._save()

    def exert(self, energy_cost: float, now: Optional[float] = None) -> None:
        """
        Apply passive Orexigenic drive drain first, then subtract an active energy cost.
        Clamp the level to [0, 100] and persist atomically.
        """
        try:
            cost = max(0.0, float(energy_cost))
        except (TypeError, ValueError):
            return
        if cost <= 0.0:
            return
        now = now or time.time()
        with self._lock:
            self._drain(now)
            self.level = max(0.0, min(100.0, self.level - cost))
            self._last_logged_pct = int(self.level)
            self._save()

    def set_level(self, level: float, now: Optional[float] = None) -> None:
        now = now or time.time()
        with self._lock:
            self.level            = max(0.0, min(100.0, float(level)))
            self.last_update_ts   = now
            self._last_logged_pct = int(self.level)
            self._save()

    def snapshot(self, now: Optional[float] = None) -> HungerSnapshot:
        self.update(now)
        with self._lock:
            if self.level >= self.hungry_threshold:
                state = "HS1"
            elif self.level >= self.starving_threshold:
                state = "HS2"
            else:
                state = "HS3"
            return HungerSnapshot(
                level             = self.level,
                state             = state,
                last_feed_ts      = self.last_feed_ts,
                last_feed_payload = self.last_feed_payload,
            )

    def state(self) -> str:
        return self.snapshot().state


# ──────────────────────────────────────────────────────────────────────────────
# Interaction result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractionResult:
    """Mutable state bag threaded through every tree step."""
    success:              bool            = False
    initial_state:        str             = ""
    final_state:          str             = ""
    greeted:              bool            = False
    talked:               bool            = False
    replied_any:          bool            = False
    extracted_name:       Optional[str]   = None
    abort_reason:         Optional[str]   = None
    target_stayed_biggest: bool           = True
    resolved_face_id:     str             = ""
    interaction_tag:      str             = ""
    hunger_state_start:   str             = ""
    hunger_state_end:     str             = ""
    hunger_drive_enabled: bool            = True
    stomach_level_start:  float           = 100.0
    stomach_level_end:    float           = 100.0
    meals_eaten_count:    int             = 0
    last_meal_payload:    Optional[str]   = None
    active_energy_cost:   float           = 0.0
    homeostatic_reward:   float           = 0.0
    n_turns:              int             = 0
    trigger_mode:         str             = "proactive"   # "proactive" | "reactive"
    logs:                 List[Dict]      = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def aborted(self) -> bool:
        return bool(self.abort_reason)


@dataclass
class SpeechDispatch:
    speech_id:           int
    text:                str
    label:               str
    dispatch_mono:       float
    estimated_done_mono: float
    estimated_wait_sec:  float
    request_id:          Optional[int] = None
    done_logged:         bool          = False
    trace_label:         str           = ""
    trace_turn_index:    int           = 0
    trace_started_mono:  float         = 0.0
    trace_request_id:    Optional[int] = None


@dataclass(frozen=True)
class LlmTurnEvent:
    kind:          str
    request_id:    int
    turn_index:    int
    interaction_id: Optional[str]
    at_mono:       float
    text:          str           = ""
    error:         Optional[str] = None


@dataclass(frozen=True)
class LlmTurnRequest:
    prompt:          str
    system:          str
    max_tokens:      int
    timeout:         float
    max_len:         int
    turn_index:      int
    interaction_id:  Optional[str]
    request_id:      int           = 0
    stream:          bool          = True
    history:         Tuple[Tuple[str, str], ...] = ()


class LatencyTrace:
    """Per-turn timing trace with consistent low-overhead logging."""

    def __init__(
        self,
        module: "ExecutiveControlModule",
        *,
        label: str,
        turn_index: int,
        utterance: str = "",
        request_id: Optional[int] = None,
        started_mono: Optional[float] = None,
    ):
        self.module        = module
        self.label         = label
        self.turn_index    = turn_index
        self.request_id    = request_id
        self.started_mono  = started_mono or time.monotonic()
        self._events: Dict[str, float] = {}
        if utterance:
            self.mark_at("turn_start", self.started_mono, utterance_chars=len(utterance))
        else:
            self.mark_at("turn_start", self.started_mono)

    def has(self, event: str) -> bool:
        return event in self._events

    def get(self, event: str) -> Optional[float]:
        return self._events.get(event)

    def mark(self, event: str, **fields: Any) -> float:
        return self.mark_at(event, time.monotonic(), **fields)

    def mark_at(self, event: str, at_mono: float, **fields: Any) -> float:
        self._events[event] = at_mono
        self.log_detached(
            self.module,
            label=self.label,
            turn_index=self.turn_index,
            event=event,
            at_mono=at_mono,
            started_mono=self.started_mono,
            request_id=self.request_id,
            **fields,
        )
        return at_mono

    @staticmethod
    def log_detached(
        module: "ExecutiveControlModule",
        *,
        label: str,
        turn_index: int,
        event: str,
        at_mono: float,
        started_mono: float,
        request_id: Optional[int] = None,
        **fields: Any,
    ) -> None:
        parts = [
            f"label={label}",
            f"turn={turn_index}",
            f"event={event}",
            f"t={at_mono - started_mono:.3f}s",
        ]
        if request_id is not None:
            parts.append(f"req={request_id}")
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
        module._log("INFO", "LATENCY " + " ".join(parts))


class SpeechCoordinator:
    """Tracks TTS dispatch timing without forcing SS3 into stop-and-wait."""

    def __init__(self, module: "ExecutiveControlModule"):
        self.module = module
        self._lock  = threading.Lock()
        self._next_speech_id = 0
        self._current: Optional[SpeechDispatch] = None

    def dispatch(
        self,
        text: str,
        *,
        label: str,
        trace: Optional[LatencyTrace] = None,
        request_id: Optional[int] = None,
    ) -> Optional[SpeechDispatch]:
        if not text:
            return None
        ok = self.module._speak(text)
        if not ok:
            return None

        now  = time.monotonic()
        wait = self.module._estimate_speech_wait(text)
        with self._lock:
            self._next_speech_id += 1
            dispatch = SpeechDispatch(
                speech_id=self._next_speech_id,
                text=text,
                label=label,
                dispatch_mono=now,
                estimated_done_mono=now + wait,
                estimated_wait_sec=wait,
                request_id=request_id,
                trace_label=trace.label if trace is not None else "",
                trace_turn_index=trace.turn_index if trace is not None else 0,
                trace_started_mono=trace.started_mono if trace is not None else now,
                trace_request_id=trace.request_id if trace is not None else request_id,
            )
            self._current = dispatch

        if trace is not None:
            trace.mark_at(
                "tts_dispatch",
                now,
                speech_id=dispatch.speech_id,
                tts_label=label,
                tts_est_sec=wait,
                text_chars=len(text),
                time_to_first_speech_sec=now - trace.started_mono,
            )
        return dispatch

    def current(self) -> Optional[SpeechDispatch]:
        with self._lock:
            return self._current

    def remaining_sec(self, dispatch: Optional[SpeechDispatch] = None) -> float:
        with self._lock:
            cur = dispatch or self._current
            if cur is None:
                return 0.0
            return max(0.0, cur.estimated_done_mono - time.monotonic())

    def maybe_mark_done(
        self,
        trace: Optional[LatencyTrace] = None,
        dispatch: Optional[SpeechDispatch] = None,
    ) -> bool:
        emit: Optional[SpeechDispatch] = None
        with self._lock:
            cur = dispatch or self._current
            if cur and not cur.done_logged and time.monotonic() >= cur.estimated_done_mono:
                cur.done_logged = True
                emit = cur
                if self._current and self._current.speech_id == cur.speech_id:
                    self._current = None
        if emit is not None:
            at_mono = time.monotonic()
            if trace is not None and trace.label == emit.trace_label and trace.turn_index == emit.trace_turn_index:
                trace.mark_at(
                    "tts_done",
                    at_mono,
                    speech_id=emit.speech_id,
                    tts_label=emit.label,
                    turn_total_sec=at_mono - trace.started_mono,
                )
            elif emit.trace_label:
                LatencyTrace.log_detached(
                    self.module,
                    label=emit.trace_label,
                    turn_index=emit.trace_turn_index,
                    event="tts_done",
                    at_mono=at_mono,
                    started_mono=emit.trace_started_mono,
                    request_id=emit.trace_request_id,
                    speech_id=emit.speech_id,
                    tts_label=emit.label,
                    turn_total_sec=at_mono - emit.trace_started_mono,
                )
        return emit is not None

    def log_interruption(
        self,
        *,
        reason: str,
        trace: Optional[LatencyTrace] = None,
    ) -> bool:
        cur = self.current()
        if cur is None:
            return False
        remaining = self.remaining_sec(cur)
        if remaining <= 0:
            return False
        if trace is not None:
            trace.mark(
                "interruption",
                reason=reason,
                speech_id=cur.speech_id,
                tts_remaining_sec=remaining,
            )
        return True

    def wait_until_idle(
        self,
        *,
        trace: Optional[LatencyTrace] = None,
        poll_sec: float = 0.05,
    ) -> None:
        logged_wait = False
        while True:
            remaining = self.remaining_sec()
            if remaining <= 0:
                self.maybe_mark_done(trace=trace)
                return
            if not logged_wait and trace is not None:
                trace.mark("tts_wait_before_dispatch", tts_remaining_sec=remaining)
                logged_wait = True
            if self.module._abort_requested():
                return
            time.sleep(min(poll_sec, remaining))


class LatestOnlyLlmWorker:
    """Bounded latest-request-wins LLM execution for conversational turns."""

    def __init__(
        self,
        module: "ExecutiveControlModule",
        *,
        max_parallel: int = 3,
    ):
        self.module        = module
        self._max_parallel = max(1, max_parallel)
        self._lock         = threading.Lock()
        self._events: "queue.Queue[LlmTurnEvent]" = queue.Queue()
        self._active_ids: set[int] = set()
        self._next_request_id      = 0
        self._latest_request_id    = 0
        self._pending_request: Optional[LlmTurnRequest] = None
        self._stop_event = threading.Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_parallel,
            thread_name_prefix="llm-turn",
        )

    def close(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._pending_request = None
        self._executor.shutdown(wait=False)

    def poll_event(self, timeout: float) -> Optional[LlmTurnEvent]:
        try:
            return self._events.get(timeout=max(0.01, timeout))
        except queue.Empty:
            return None

    def is_latest(self, request_id: int) -> bool:
        with self._lock:
            return request_id == self._latest_request_id

    def submit(self, request: LlmTurnRequest) -> int:
        to_start: Optional[LlmTurnRequest] = None
        pending_replaced = False
        active_count = 0
        with self._lock:
            self._next_request_id += 1
            assigned = replace(request, request_id=self._next_request_id)
            self._latest_request_id = assigned.request_id
            if self._stop_event.is_set():
                self._pending_request = None
            elif len(self._active_ids) < self._max_parallel:
                self._active_ids.add(assigned.request_id)
                to_start = assigned
            else:
                pending_replaced = self._pending_request is not None
                self._pending_request = assigned
            active_count = len(self._active_ids)
        if to_start is not None:
            self._start_request(to_start)
        else:
            self.module._log(
                "INFO",
                f"llm_worker: request {assigned.request_id} pending "
                f"(active={active_count} replaced_pending={pending_replaced})",
            )
        return assigned.request_id

    def _start_request(self, request: LlmTurnRequest) -> None:
        self._executor.submit(self._run_request, request)

    def _finish_request(self, request_id: int) -> None:
        to_start: Optional[LlmTurnRequest] = None
        with self._lock:
            self._active_ids.discard(request_id)
            if not self._stop_event.is_set() and self._pending_request is not None and len(self._active_ids) < self._max_parallel:
                to_start = self._pending_request
                self._pending_request = None
                self._active_ids.add(to_start.request_id)
        if to_start is not None:
            self._start_request(to_start)

    def _emit(self, event: LlmTurnEvent) -> None:
        try:
            self._events.put_nowait(event)
        except queue.Full:
            self.module._log("WARNING", f"llm_worker: dropping event {event.kind} req={event.request_id}")

    @staticmethod
    def _chunk_text(choice: Any) -> str:
        delta = getattr(choice, "delta", None)
        if delta is None:
            return ""
        content = getattr(delta, "content", "") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item, dict):
                    inner = item.get("text")
                    if isinstance(inner, str):
                        parts.append(inner)
            return "".join(parts)
        return ""

    def _run_request(self, request: LlmTurnRequest) -> None:
        prev_iid = self.module._get_iid()
        self.module._set_iid(request.interaction_id)
        try:
            if self._stop_event.is_set():
                return
            if self.module.llm_client is None:
                self._emit(LlmTurnEvent(
                    kind="error",
                    request_id=request.request_id,
                    turn_index=request.turn_index,
                    interaction_id=request.interaction_id,
                    at_mono=time.monotonic(),
                    error="client_not_initialized",
                ))
                return

            messages: List[Dict[str, str]] = [{"role": "system", "content": request.system}]
            for role, content in request.history:
                if content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": request.prompt})

            kwargs: Dict[str, Any] = {
                "model": self.module._llm_deployment,
                "messages": messages,
                "max_completion_tokens": request.max_tokens,
                "timeout": request.timeout,
            }

            if request.stream:
                self._run_streaming_request(request, kwargs)
            else:
                self._run_sync_request(request, kwargs)
        finally:
            self.module._set_iid(prev_iid)
            self._finish_request(request.request_id)

    def _run_sync_request(self, request: LlmTurnRequest, kwargs: Dict[str, Any]) -> None:
        started = time.monotonic()
        try:
            resp = self.module.llm_client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
            self.module._log(
                "INFO",
                f"llm_worker[{request.request_id}]: sync create() returned in {time.monotonic()-started:.2f}s model={getattr(resp, 'model', '?')}",
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            if not self.is_latest(request.request_id):
                self._emit(LlmTurnEvent(
                    kind="cancelled",
                    request_id=request.request_id,
                    turn_index=request.turn_index,
                    interaction_id=request.interaction_id,
                    at_mono=time.monotonic(),
                ))
                return
            if text and len(text) < request.max_len:
                now = time.monotonic()
                self._emit(LlmTurnEvent("first_token", request.request_id, request.turn_index, request.interaction_id, now))
                self._emit(LlmTurnEvent("final", request.request_id, request.turn_index, request.interaction_id, now, text=text))
                return
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            self.module._log(
                "WARNING",
                f"llm_worker[{request.request_id}]: rejected response"
                f" finish_reason={finish_reason} text_len={len(text)} max_len={request.max_len}",
            )
            err = "empty_response_length" if finish_reason == "length" else "empty_response"
            self._emit(LlmTurnEvent(
                kind="error",
                request_id=request.request_id,
                turn_index=request.turn_index,
                interaction_id=request.interaction_id,
                at_mono=time.monotonic(),
                error=err,
            ))
        except Exception as e:
            self._emit(LlmTurnEvent(
                kind="error",
                request_id=request.request_id,
                turn_index=request.turn_index,
                interaction_id=request.interaction_id,
                at_mono=time.monotonic(),
                error=str(e),
            ))

    def _run_streaming_request(self, request: LlmTurnRequest, kwargs: Dict[str, Any]) -> None:
        kwargs["stream"] = True
        started = time.monotonic()
        stream = None
        first_token_mono: Optional[float] = None
        finish_reason = ""
        parts: List[str] = []
        chunk_count = 0

        try:
            stream = self.module.llm_client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
            self.module._log(
                "INFO",
                f"llm_worker[{request.request_id}]: stream opened in {time.monotonic()-started:.2f}s",
            )
            for chunk in stream:
                if self._stop_event.is_set():
                    self._emit(LlmTurnEvent(
                        kind="cancelled",
                        request_id=request.request_id,
                        turn_index=request.turn_index,
                        interaction_id=request.interaction_id,
                        at_mono=time.monotonic(),
                    ))
                    return
                if not self.is_latest(request.request_id):
                    self._emit(LlmTurnEvent(
                        kind="cancelled",
                        request_id=request.request_id,
                        turn_index=request.turn_index,
                        interaction_id=request.interaction_id,
                        at_mono=time.monotonic(),
                    ))
                    return

                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta_text = self._chunk_text(choice)
                if delta_text:
                    chunk_count += 1
                    parts.append(delta_text)
                    if first_token_mono is None:
                        first_token_mono = time.monotonic()
                        self._emit(LlmTurnEvent(
                            kind="first_token",
                            request_id=request.request_id,
                            turn_index=request.turn_index,
                            interaction_id=request.interaction_id,
                            at_mono=first_token_mono,
                        ))
                if getattr(choice, "finish_reason", None):
                    finish_reason = str(choice.finish_reason)

            text = "".join(parts).strip()
            if not self.is_latest(request.request_id):
                self._emit(LlmTurnEvent(
                    kind="cancelled",
                    request_id=request.request_id,
                    turn_index=request.turn_index,
                    interaction_id=request.interaction_id,
                    at_mono=time.monotonic(),
                ))
                return

            if text and len(text) < request.max_len:
                end_mono = time.monotonic()
                if first_token_mono is None:
                    self._emit(LlmTurnEvent(
                        kind="first_token",
                        request_id=request.request_id,
                        turn_index=request.turn_index,
                        interaction_id=request.interaction_id,
                        at_mono=end_mono,
                    ))
                self._emit(LlmTurnEvent(
                    kind="final",
                    request_id=request.request_id,
                    turn_index=request.turn_index,
                    interaction_id=request.interaction_id,
                    at_mono=end_mono,
                    text=text,
                ))
                return

            err = "empty_response_length" if finish_reason == "length" else "empty_response"
            self._emit(LlmTurnEvent(
                kind="error",
                request_id=request.request_id,
                turn_index=request.turn_index,
                interaction_id=request.interaction_id,
                at_mono=time.monotonic(),
                error=err,
            ))
        except Exception as e:
            self._emit(LlmTurnEvent(
                kind="error",
                request_id=request.request_id,
                turn_index=request.turn_index,
                interaction_id=request.interaction_id,
                at_mono=time.monotonic(),
                error=str(e),
            ))
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# Executive control module
# ──────────────────────────────────────────────────────────────────────────────

class ExecutiveControlModule(yarp.RFModule):
    """Proactive + reactive social interaction controller for iCub.

    Social states
    ─────────────
    ss1  unknown person    → greet + extract name → ss3
    ss2  known, not greeted → say hello → ss3
    ss3  known, greeted     → short LLM conversation → ss4
    ss4  no-op (already talked)
    """

    # ── nested types ─────────────────────────────────────────────────────────

    @dataclass(frozen=True)
    class LlmResult:
        ok:       bool
        text:     str           = ""
        error:    Optional[str] = None
        timed_out: bool         = False
        empty:    bool          = False
        invalid:  bool          = False

    @dataclass(frozen=True)
    class InteractionAttempt:
        interaction_id: str
        track_id:       int
        face_id:        str
        initial_state:  str
        result:         Dict[str, Any]

    # ── constants ─────────────────────────────────────────────────────────────

    # Paths
    _PROMPTS_CANDIDATES = [
        os.path.join(_MODULE_DIR,   "prompts.json"),
        os.path.join(_ALWAYSON_DIR, "prompts.json"),
    ]
    DB_FILE             = os.path.join(_MODULE_DIR, "data_collection", "executive_control.db")
    LAST_GREETED_FILE   = os.path.join(_MODULE_DIR, "memory", "last_greeted.json")
    GREETED_TODAY_FILE  = os.path.join(_MODULE_DIR, "memory", "greeted_today.json")
    HUNGER_STATE_FILE   = os.path.join(_MODULE_DIR, "memory", "hunger_state.json")

    # Timeouts (seconds)
    SS1_STT_TIMEOUT          = 18.0
    SS2_GREET_TIMEOUT        = 18.0
    SS3_STT_TIMEOUT          = 18.0
    LLM_TIMEOUT              = 8.0
    SS3_MAX_TURNS            = 3
    STT_POLL_INTERVAL_SEC    = 0.05

    # TTS timing
    TTS_WORDS_PER_SECOND = 3.0
    TTS_END_MARGIN       = 0.5
    TTS_MIN_WAIT         = 1.0
    TTS_MAX_WAIT         = 8.0
    TTS_POLL_INTERVAL_SEC = 0.05

    # Conversation latency tuning
    SS3_STARTER_MAX_TOKENS     = 40
    SS3_FOLLOWUP_MAX_TOKENS    = 56
    SS3_CLOSING_MAX_TOKENS     = 18
    SS3_TURN_MAX_LEN           = 140
    SS3_CLOSING_MAX_LEN        = 72
    SS3_LLM_STREAMING_ENABLED  = False
    SS3_LLM_WORKER_PARALLELISM = 3

    # Active metabolic costs for meaningful robot actions.
    CONVERSATION_TURN_ENERGY_COST = 3.6
    GREETING_ENERGY_COST          = 0.8
    NAME_QUESTION_ENERGY_COST     = 1.0
    HUNGER_PROMPT_ENERGY_COST     = 1.0
    STARTER_PROMPT_ENERGY_COST    = 1.2
    FEED_ACK_ENERGY_COST          = 0.8
    HUNGER_LEVEL_LOG_PERIOD_SEC   = 1.0

    # Target monitor
    MONITOR_HZ               = 15.0
    TARGET_LOST_TIMEOUT      = 12.0
    MONITOR_PACKET_STALE_SEC = 5.0
    MONITOR_WARMUP_SEC       = 2.0

    # Reactive path
    REACTIVE_GREET_REGEX    = re.compile(r"\b(hello|hi|hey|ciao|buongiorno|good\s+morning)\b")
    REACTIVE_GREET_COOLDOWN = 10.0
    _REACTIVE_GAZE_STATES   = frozenset({"MUTUAL_GAZE", "NEAR_GAZE"})

    VALID_STATES    = {"ss1", "ss2", "ss3", "ss4"}
    HUNGER_OFF_STATE = "HS0"
    DB_QUEUE_MAX    = 512
    TIMEZONE        = ZoneInfo("Europe/Rome")

    # Prompts (loaded from prompts.json at configure time)
    _P:             Dict[str, Any] = {}
    LLM_SYS_DEFAULT: str = ""
    LLM_SYS_JSON:    str = ""
    LLM_SYS_FAST:    str = ""

    # ── class-level prompt loader ─────────────────────────────────────────────

    @classmethod
    def _load_prompts(cls) -> None:
        for path in cls._PROMPTS_CANDIDATES:
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    cls._P = data.get("executiveControl", {})
                    cls.LLM_SYS_DEFAULT = cls._P.get("system_default", "")
                    cls.LLM_SYS_JSON    = cls._P.get("system_json",    "")
                    cls.LLM_SYS_FAST    = cls._P.get(
                        "system_fast",
                        "You are iCub speaking face to face. Output only one short natural spoken sentence. No markdown, no emojis, no explanations."
                    )
                    if not cls.LLM_SYS_DEFAULT:
                        print("[ERROR] Missing executiveControl.system_default in prompts.json")
                    return
                except Exception as e:
                    print(f"[ERROR] Failed to load prompts.json ({path}): {e}")
        print("[ERROR] prompts.json not found. Tried: " + ", ".join(cls._PROMPTS_CANDIDATES))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self.module_name = "executiveControl"
        self.period      = 0.02
        self._running    = True

        # YARP ports
        self.handle_port:   yarp.Port                   = yarp.Port()
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.stt_port:       Optional[yarp.BufferedPortBottle] = None
        self.speech_port:    Optional[yarp.BufferedPortBottle] = None
        self.hunger_port:    Optional[yarp.BufferedPortBottle] = None
        self.qr_port:        Optional[yarp.BufferedPortBottle] = None
        self._selector_rpc:  Optional[yarp.RpcClient]   = None
        self._vision_rpc:    Optional[yarp.RpcClient]   = None
        self._emotions_rpc:  Optional[yarp.RpcClient]   = None
        self.attach(self.handle_port)

        # Orexigenic drive
        self.hunger         = HungerModel(persist_file=self.HUNGER_STATE_FILE)
        self.hunger_enabled = True
        self._meal_mapping  = {"SMALL_MEAL": 10.0, "MEDIUM_MEAL": 25.0, "LARGE_MEAL": 45.0}
        self._qr_cooldown_sec       = 3.0
        self._feed_wait_timeout_sec = 8.0
        self._last_scan_ts_mono     = 0.0
        self._feed_condition        = threading.Condition()
        self._hunger_tree_active    = threading.Event()
        self._last_hunger_level_log_mono = 0.0
        self._last_hunger_level_logged: Optional[float] = None

        # Landmark cache
        self._faces_lock:               threading.Lock      = threading.Lock()
        self._latest_faces:             List[Dict]          = []
        self._latest_faces_ts:          float               = 0.0
        self._latest_landmarks_packet_ts: float             = 0.0

        # STT ownership
        self._stt_lock = threading.Lock()

        # Abort / interaction ownership
        self.abort_event              = threading.Event()
        self._interaction_abort_event = threading.Event()
        self._monitor_thread:   Optional[threading.Thread] = None
        self._current_track_id: Optional[int]              = None

        self._interaction_state_lock    = threading.Lock()
        self._interaction_mode:  str    = "idle"   # idle | proactive | reactive
        self._interaction_reason: str   = ""
        self._interaction_started_mono: float = 0.0

        # Azure LLM
        self.llm_client:              Optional[AzureOpenAI] = None
        self._llm_deployment:         str = ""
        self._llm_max_tokens:         int = 2000
        self.llm_retry_attempts       = 1
        self.llm_retry_delay          = 0.0

        # Reactive cooldowns
        self._reactive_greet_cooldown: Dict[str, float] = {}

        # Logging
        self.log_buffer:        List[Dict]           = []
        self._log_lock          = threading.Lock()
        self._interaction_logs: Dict[str, List[Dict]] = {}
        self._interaction_logs_lock = threading.Lock()
        self._thread_ctx        = threading.local()
        self._log_throttle_lock = threading.Lock()
        self._log_throttle_last: Dict[str, float] = {}
        self._llm_diagnostics_enabled = os.getenv("EXEC_CTRL_LLM_DIAGNOSTICS", "").strip().lower() in {
            "1", "true", "yes", "on",
        }

        # DB queue
        self._db_queue: queue.Queue = queue.Queue(maxsize=self.DB_QUEUE_MAX)

        # Stop events
        self._landmarks_stop = threading.Event()
        self._qr_stop        = threading.Event()
        self._reactive_stop  = threading.Event()

        # Thread handles
        self._landmarks_thread: Optional[threading.Thread] = None
        self._db_thread:        Optional[threading.Thread] = None
        self._qr_thread:        Optional[threading.Thread] = None
        self._reactive_thread:  Optional[threading.Thread] = None

        # Real-time conversation helpers
        self._speech = SpeechCoordinator(self)
        self._llm_turn_worker = LatestOnlyLlmWorker(
            self,
            max_parallel=self.SS3_LLM_WORKER_PARALLELISM,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            ExecutiveControlModule._load_prompts()

            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            self.setName(self.module_name)

            self.hunger = HungerModel(
                drain_hours        = rf.find("drain_hours").asFloat64()        if rf.check("drain_hours")        else 5.0,
                hungry_threshold   = rf.find("hungry_threshold").asFloat64()   if rf.check("hungry_threshold")   else 60.0,
                starving_threshold = rf.find("starving_threshold").asFloat64() if rf.check("starving_threshold") else 25.0,
                persist_file       = self.HUNGER_STATE_FILE,
                log_cb             = self._log,
            )
            if rf.check("qr_cooldown_sec"):
                self._qr_cooldown_sec = rf.find("qr_cooldown_sec").asFloat64()
            if rf.check("feed_wait_timeout_sec"):
                self._feed_wait_timeout_sec = rf.find("feed_wait_timeout_sec").asFloat64()
            if rf.check("llm_timeout"):
                self.LLM_TIMEOUT = max(0.5, rf.find("llm_timeout").asFloat64())
            if rf.check("llm_diagnostics"):
                self._llm_diagnostics_enabled = rf.find("llm_diagnostics").toString().strip().lower() in {
                    "1", "true", "yes", "on",
                }
            self.hunger_enabled = self._rf_hunger_enabled(rf)
            if not self.hunger_enabled:
                self.hunger.set_level(100.0)

            snap = self.hunger.snapshot()
            self._log(
                "INFO",
                f"Hunger drive {'enabled' if self.hunger_enabled else 'disabled'}; "
                f"stored stomach={snap.level:.1f}% ({snap.state}), effective={self._effective_hunger_state()}",
            )
            self._log_hunger_level_event(
                "configure",
                stimulus_type="mode",
                stimulus_label="enabled" if self.hunger_enabled else "disabled",
                level_after=snap.level,
                state_after=self._effective_hunger_state(snap),
                force=True,
            )
            self._set_face_emotion(self._effective_hunger_state(snap))

            # Ports
            self.handle_port.open("/" + self.module_name)
            base = f"/alwayson/{self.module_name}"

            self.landmarks_port = self._open_port(yarp.BufferedPortBottle(), f"{base}/landmarks:i")
            self.stt_port       = self._open_port(yarp.BufferedPortBottle(), f"{base}/stt:i")
            self.speech_port    = self._open_port(yarp.BufferedPortBottle(), f"{base}/speech:o")
            self.qr_port        = self._open_port(yarp.BufferedPortBottle(), f"{base}/qr:i",     optional=True)
            self.hunger_port    = self._open_port(yarp.BufferedPortBottle(), f"{base}/hunger:o", optional=True)

            if self.landmarks_port is None or self.stt_port is None or self.speech_port is None:
                return False

            self._ensure_json_file(self.LAST_GREETED_FILE,  {})
            self._ensure_json_file(self.GREETED_TODAY_FILE, {})
            self._init_db()
            self._setup_llm()

            try:
                self._get_selector_rpc()
                self._get_vision_rpc()
            except Exception as e:
                self._log("WARNING", f"Failed to pre-open RPC clients: {e}")

            # Background threads
            for attr, target, stop_ev in [
                ("_landmarks_thread", self._landmarks_reader_loop, self._landmarks_stop),
                ("_db_thread",        self._db_worker,             None),
                ("_qr_thread",        self._qr_reader_loop,        self._qr_stop),
                ("_reactive_thread",  self._reactive_loop,         self._reactive_stop),
            ]:
                if stop_ev is not None:
                    stop_ev.clear()
                t = threading.Thread(target=target, daemon=True)
                setattr(self, attr, t)
                t.start()

            # Second warm-up AFTER background loops start: tells us if
            # background thread activity slows down LLM calls.
            try:
                time.sleep(0.5)
                t0 = time.monotonic()
                self.llm_client.chat.completions.create(  # type: ignore[union-attr]
                    model=self._llm_deployment,
                    messages=[{"role": "user", "content": "ok"}],
                    max_completion_tokens=16,
                    timeout=20.0,
                )
                self._log("INFO", f"LLM warm-up (post-threads) ok in {time.monotonic()-t0:.2f}s")
            except Exception as e:
                self._log("WARNING", f"LLM warm-up (post-threads) failed: {type(e).__name__}: {e}")

            if self._llm_diagnostics_enabled:
                # Production diagnostics are opt-in only.
                try:
                    sys_msg = self._system_for_hs("HS1")
                    user_msg = self._prompt_for_hs("convo_starter_prompt", "HS1", "Say hi briefly.")
                    t0 = time.monotonic()
                    self.llm_client.chat.completions.create(  # type: ignore[union-attr]
                        model=self._llm_deployment,
                        messages=[{"role": "system", "content": sys_msg},
                                  {"role": "user", "content": user_msg}],
                        max_completion_tokens=self.SS3_STARTER_MAX_TOKENS,
                        timeout=20.0,
                    )
                    self._log("INFO", f"LLM diag (full-prompt, main thread) ok in {time.monotonic()-t0:.2f}s")
                except Exception as e:
                    self._log("WARNING", f"LLM diag (full-prompt, main thread) failed: {type(e).__name__}: {e}")

                try:
                    sys_msg = self._system_for_hs("HS1")
                    user_msg = self._prompt_for_hs("convo_starter_prompt", "HS1", "Say hi briefly.")
                    kwargs_diag: Dict[str, Any] = {
                        "model": self._llm_deployment,
                        "messages": [{"role": "system", "content": sys_msg},
                                     {"role": "user", "content": user_msg}],
                        "max_completion_tokens": self.SS3_STARTER_MAX_TOKENS,
                        "timeout": 20.0,
                    }
                    t0 = time.monotonic()
                    self._llm_create_with_watchdog(kwargs_diag, 22.0)
                    self._log("INFO", f"LLM diag (full-prompt, watchdog thread) ok in {time.monotonic()-t0:.2f}s")
                except Exception as e:
                    self._log("WARNING", f"LLM diag (full-prompt, watchdog thread) failed: {type(e).__name__}: {e}")

            self._log("INFO", "ExecutiveControlModule ready")
            return True
        except Exception as e:
            self._log("ERROR", f"configure() failed: {e}")
            traceback.print_exc()
            return False

    def _open_port(self, port, name: str, optional: bool = False) -> Optional[Any]:
        if not port.open(name):
            self._log("WARNING" if optional else "ERROR", f"Failed to open port: {name}")
            return None
        self._log("INFO", f"Port open: {name}")
        return port

    @staticmethod
    def _parse_boolish(value: str, default: bool = True) -> bool:
        v = (value or "").strip().lower()
        if v in {"1", "true", "yes", "on", "enabled", "enable"}:
            return True
        if v in {"0", "false", "no", "off", "disabled", "disable"}:
            return False
        return default

    def _rf_hunger_enabled(self, rf: yarp.ResourceFinder) -> bool:
        if rf.check("hunger_enabled"):
            return self._parse_boolish(rf.find("hunger_enabled").toString(), default=True)
        if rf.check("hunger_mode"):
            return self._parse_boolish(rf.find("hunger_mode").toString(), default=True)
        return True

    def _effective_hunger_state(self, snap: Optional[HungerSnapshot] = None) -> str:
        if not self.hunger_enabled:
            return self.HUNGER_OFF_STATE
        snap = snap or self.hunger.snapshot()
        return snap.state

    def _mark_homeostasis_start(self, result: InteractionResult, social_state: str) -> None:
        snap = self.hunger.snapshot() if self.hunger_enabled else None
        hs = self._effective_hunger_state(snap)
        result.hunger_state_start = hs
        result.hunger_drive_enabled = self.hunger_enabled
        result.stomach_level_start = snap.level if snap else 100.0
        result.interaction_tag = f"{social_state.upper()}{hs}"

    def _finalize_homeostasis_result(self, result: InteractionResult) -> None:
        snap = self.hunger.snapshot() if self.hunger_enabled else None
        result.hunger_state_end = self._effective_hunger_state(snap)
        result.stomach_level_end = snap.level if snap else 100.0
        result.homeostatic_reward = result.stomach_level_end - result.stomach_level_start

    def _log_hunger_level_event(
        self,
        event_type: str,
        *,
        stimulus_type: str = "sample",
        stimulus_label: Optional[str] = None,
        level_before: Optional[float] = None,
        level_after: Optional[float] = None,
        state_before: Optional[str] = None,
        state_after: Optional[str] = None,
        delta: Optional[float] = None,
        active_energy_cost: float = 0.0,
        meal_delta: float = 0.0,
        meal_payload: Optional[str] = None,
        result: Optional[InteractionResult] = None,
        reason: Optional[str] = None,
        force: bool = False,
    ) -> None:
        try:
            snap = self.hunger.snapshot()
            after = float(level_after if level_after is not None else snap.level)
            before = float(level_before if level_before is not None else after)
            state_to_log = state_after or self._effective_hunger_state(snap)
            state_from_log = state_before or state_to_log
            delta_to_log = float(delta if delta is not None else after - before)
            now_mono = time.monotonic()
            if not force and event_type == "sample":
                if now_mono - self._last_hunger_level_log_mono < self.HUNGER_LEVEL_LOG_PERIOD_SEC:
                    return
            self._last_hunger_level_log_mono = now_mono
            self._last_hunger_level_logged = after
            self._db_enqueue(("hunger_level_event", {
                "event_type": event_type,
                "stimulus_type": stimulus_type,
                "stimulus_label": stimulus_label,
                "reason": reason,
                "hunger_drive_enabled": int(bool(self.hunger_enabled)),
                "hunger_state_before": state_from_log,
                "hunger_state_after": state_to_log,
                "stomach_level_before": before,
                "stomach_level_after": after,
                "level_delta": delta_to_log,
                "active_energy_cost": float(active_energy_cost or 0.0),
                "meal_delta": float(meal_delta or 0.0),
                "meal_payload": meal_payload,
                "trigger_mode": getattr(result, "trigger_mode", None) if result else None,
                "social_state": getattr(result, "initial_state", None) if result else None,
                "interaction_tag": getattr(result, "interaction_tag", None) if result else None,
                "exec_interaction_id": self._get_iid(),
            }))
        except Exception as e:
            self._log("WARNING", f"hunger_level_event log failed: {e}")

    def _maybe_log_hunger_level_sample(self) -> None:
        snap = self.hunger.snapshot()
        previous = self._last_hunger_level_logged
        before = snap.level if previous is None else previous
        self._log_hunger_level_event(
            "sample",
            stimulus_type="passive_drain",
            level_before=before,
            level_after=snap.level,
            state_before=self._effective_hunger_state(snap),
            state_after=self._effective_hunger_state(snap),
            delta=snap.level - before,
        )

    def _charge_energy(
        self,
        cost: float,
        result: Optional[InteractionResult],
        reason: str,
    ) -> None:
        """
        Charge active metabolic energy for a meaningful robot action.
        Does nothing when Orexigenic drive is disabled.
        Records the cost in the interaction result when available.
        """
        if not self.hunger_enabled:
            return
        try:
            cost_f = float(cost)
        except (TypeError, ValueError):
            return
        if cost_f <= 0.0:
            return
        before = self.hunger.snapshot()
        self.hunger.exert(cost_f)
        after = self.hunger.snapshot()
        if result is not None:
            result.active_energy_cost += cost_f
        self._log_hunger_level_event(
            "active_cost",
            stimulus_type="interaction_cost",
            stimulus_label=reason,
            level_before=before.level,
            level_after=after.level,
            state_before=before.state,
            state_after=after.state,
            delta=after.level - before.level,
            active_energy_cost=cost_f,
            result=result,
            reason=reason,
            force=True,
        )
        self._log("DEBUG", f"energy: -{cost_f:.2f} reason={reason}")

    def interruptModule(self) -> bool:
        self._log("INFO", "Interrupting…")
        self._running = False
        self.abort_event.set()
        self._interaction_abort_event.set()
        self._llm_turn_worker.close()
        for ev in (self._landmarks_stop, self._qr_stop, self._reactive_stop):
            ev.set()
        for port in (self.handle_port, self.landmarks_port, self.stt_port,
                     self.speech_port, self.hunger_port, self.qr_port,
                     self._selector_rpc, self._vision_rpc):
            self._port_interrupt(port)
        return True

    def close(self) -> bool:
        self._log("INFO", "Closing…")
        for ev in (self._landmarks_stop, self._qr_stop, self._reactive_stop):
            ev.set()
        self._interaction_abort_event.set()
        self._llm_turn_worker.close()
        for port in (self._selector_rpc, self._vision_rpc):
            self._port_interrupt(port)
        for t in (self._landmarks_thread, self._qr_thread, self._reactive_thread):
            if t:
                t.join(timeout=2.0)
        self._db_enqueue(None)
        if self._db_thread:
            self._db_thread.join(timeout=3.0)
        for port in (self.handle_port, self.landmarks_port, self.stt_port,
                     self.speech_port, self.hunger_port, self.qr_port,
                     self._selector_rpc, self._vision_rpc):
            self._port_close(port)
        return True

    @staticmethod
    def _port_interrupt(port) -> None:
        try:
            if port:
                port.interrupt()
        except Exception:
            pass

    @staticmethod
    def _port_close(port) -> None:
        try:
            if port:
                port.close()
        except Exception:
            pass

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        self._maybe_log_hunger_level_sample()
        if self.hunger_port:
            b = self.hunger_port.prepare()
            b.clear()
            b.addString(self._effective_hunger_state())
            self.hunger_port.write()
        return self._running

    # ── RPC handler ──────────────────────────────────────────────────────────

    def respond(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        reply.clear()
        try:
            if cmd.size() < 1:
                return self._rpc_error(reply, "Empty command")
            command = cmd.get(0).asString()

            if command in ("status", "ping"):
                return self._cmd_status(reply)
            if command == "help":
                return self._cmd_help(reply)
            if command == "quit":
                return self._cmd_quit(reply)
            if command == "hunger_mode":
                return self._cmd_hunger_mode(cmd, reply)
            if command == "hunger":
                return self._cmd_hunger(cmd, reply)
            if command == "run":
                return self._cmd_run(cmd, reply)
            return self._rpc_error(reply, f"Unknown command: {command}")
        except Exception as e:
            self._log("ERROR", f"respond() exception: {e}")
            traceback.print_exc()
            return self._rpc_error(reply, str(e))

    def _cmd_status(self, reply: yarp.Bottle) -> bool:
        busy, mode, reason, for_sec = self._busy_snapshot()
        snap = self.hunger.snapshot() if self.hunger_enabled else None
        return self._rpc_ok(reply, {
            "success":         True,
            "status":          "ready",
            "module":          self.module_name,
            "busy":            busy,
            "busy_mode":       mode,
            "busy_reason":     reason,
            "busy_for_sec":    round(for_sec, 3),
            "hunger_enabled":  self.hunger_enabled,
            "hunger_state":    self._effective_hunger_state(snap),
            "hunger_level":    snap.level if snap else 100.0,
        })

    def _cmd_help(self, reply: yarp.Bottle) -> bool:
        reply.addString(
            "run <track_id> <face_id> <ss1|ss2|ss3|ss4>  -- start interaction\n"
            "hunger <hs0|hs1|hs2|hs3>                     -- set no-drive/full/hungry/starving\n"
            "hunger_mode <on|off>                         -- toggle hunger drive (off publishes HS0)\n"
            "status                                        -- check busy / hunger\n"
            "quit                                          -- shut down"
        )
        return True

    def _cmd_quit(self, reply: yarp.Bottle) -> bool:
        self._running = False
        self.stopModule()
        return self._rpc_ok(reply, {"success": True, "message": "Shutting down"})

    def _cmd_hunger_mode(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        if cmd.size() < 2:
            return self._rpc_error(reply, "Usage: hunger_mode <on|off>")
        arg = cmd.get(1).asString().strip().lower()
        if arg not in ("on", "off"):
            return self._rpc_error(reply, "Usage: hunger_mode <on|off>")
        before = self.hunger.snapshot()
        state_before = self._effective_hunger_state(before)
        self.hunger_enabled = arg == "on"
        self.hunger.set_level(100.0)
        if not self.hunger_enabled:
            with self._feed_condition:
                self._feed_condition.notify_all()
        snap = self.hunger.snapshot()
        self._log_hunger_level_event(
            "mode",
            stimulus_type="mode",
            stimulus_label=arg,
            level_before=before.level,
            level_after=snap.level,
            state_before=state_before,
            state_after=self._effective_hunger_state(snap),
            delta=snap.level - before.level,
            force=True,
        )
        self._log(
            "INFO",
            f"Hunger drive {'ON' if self.hunger_enabled else 'OFF'}; reset to 100%; "
            f"effective={self._effective_hunger_state(snap)}",
        )
        return self._rpc_ok(reply, {
            "success":        True,
            "hunger_enabled": self.hunger_enabled,
            "hunger_state":   self._effective_hunger_state(snap),
            "hunger_level":   snap.level,
        })

    def _cmd_hunger(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        if cmd.size() < 2:
            return self._rpc_error(reply, "Usage: hunger <hs0|hs1|hs2|hs3>")
        level_map = {"hs1": 100.0, "hs2": 59.0, "hs3": 24.0}
        arg = cmd.get(1).asString().lower()
        before = self.hunger.snapshot()
        state_before = self._effective_hunger_state(before)
        if arg == "hs0":
            self.hunger_enabled = False
            self.hunger.set_level(100.0)
            with self._feed_condition:
                self._feed_condition.notify_all()
            snap = self.hunger.snapshot()
            self._log_hunger_level_event(
                "manual_set",
                stimulus_type="mode",
                stimulus_label=arg,
                level_before=before.level,
                level_after=snap.level,
                state_before=state_before,
                state_after=self.HUNGER_OFF_STATE,
                delta=snap.level - before.level,
                force=True,
            )
            self._log("INFO", "Hunger drive manually disabled (HS0)")
            return self._rpc_ok(reply, {
                "success": True,
                "hunger_enabled": False,
                "hunger_level": snap.level,
                "hunger_state": self.HUNGER_OFF_STATE,
            })
        if arg not in level_map:
            return self._rpc_error(reply, "Usage: hunger <hs0|hs1|hs2|hs3>")
        self.hunger_enabled = True
        new_level = level_map[arg]
        self.hunger.set_level(new_level)
        snap = self.hunger.snapshot()
        self._log_hunger_level_event(
            "manual_set",
            stimulus_type="manual_level",
            stimulus_label=arg,
            level_before=before.level,
            level_after=snap.level,
            state_before=state_before,
            state_after=snap.state,
            delta=snap.level - before.level,
            force=True,
        )
        self._log("INFO", f"Hunger drive enabled; manually set to {new_level}% ({snap.state})")
        return self._rpc_ok(reply, {
            "success": True,
            "hunger_enabled": True,
            "hunger_level": snap.level,
            "hunger_state": snap.state,
        })

    def _cmd_run(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        if cmd.size() < 4:
            return self._rpc_error(reply, "Usage: run <track_id> <face_id> <ss1|ss2|ss3|ss4>")
        track_id     = cmd.get(1).asInt32()
        face_id      = cmd.get(2).asString()
        social_state = cmd.get(3).asString().lower()

        self._log("INFO", f"rpc: run track={track_id} face='{face_id}' state={social_state}")

        if social_state not in self.VALID_STATES:
            return self._rpc_error(reply, f"Invalid state: {social_state}")

        accepted, busy_mode, busy_reason = self._begin_interaction("proactive",
            f"rpc_run:{social_state}:track={track_id}:face={face_id}")
        if not accepted:
            self._log("INFO", f"rpc: run blocked – {busy_mode}: {busy_reason}")
            return self._rpc_ok(reply, {
                "success":     False,
                "error":       "another_interaction_running",
                "busy_mode":   busy_mode,
                "busy_reason": busy_reason,
            })

        interaction_id = uuid.uuid4().hex
        prev_iid = self._get_iid()
        try:
            self._init_ilog(interaction_id)
            self._set_iid(interaction_id)
            self._log("INFO",
                f"━━━ INTERACTION START type={social_state.upper()} "
                f"target='{face_id}' track={track_id} id={interaction_id[:8]} ━━━")

            result = self._execute_interaction(track_id, face_id, social_state, interaction_id)
            d      = result.to_dict()
            d["interaction_id"] = interaction_id
            resolved = result.resolved_face_id or face_id

            if result.abort_reason:
                self._log("WARNING",
                    f"━━━ INTERACTION END   ABORT reason={result.abort_reason} "
                    f"replied={result.replied_any} talked={result.talked} ━━━")
            else:
                self._log("INFO",
                    f"━━━ INTERACTION END   OK final={result.final_state} "
                    f"replied={result.replied_any} talked={result.talked} ━━━")

            d["logs"] = self._pop_ilog(interaction_id)
            self._db_enqueue(("interaction", asdict(self.InteractionAttempt(
                interaction_id = interaction_id,
                track_id       = track_id,
                face_id        = resolved,
                initial_state  = social_state,
                result         = d,
            ))))

            compact = self._build_compact_result(interaction_id, track_id, face_id, social_state, result)
            return self._rpc_ok(reply, compact)
        finally:
            self._set_iid(prev_iid)
            self._pop_ilog(interaction_id)
            self._end_interaction("proactive")

    def _build_compact_result(
        self, iid: str, track_id: int, face_id: str, social_state: str, r: InteractionResult
    ) -> Dict[str, Any]:
        c: Dict[str, Any] = {
            "interaction_id": iid,
            "success":        bool(r.success) or (r.talked and r.abort_reason is None),
            "track_id":       track_id,
            "face_id":        face_id,
            "resolved_face_id": r.resolved_face_id or face_id,
            "name":           None,
            "name_extracted": False,
            "abort_reason":   r.abort_reason,
            "initial_state":  social_state,
            "final_state":    r.final_state,
            "replied_any":    r.replied_any,
        }
        if r.extracted_name:
            c["name"]           = r.extracted_name
            c["name_extracted"] = True
        elif social_state in ("ss2", "ss3", "ss4"):
            c["name"] = r.resolved_face_id or face_id

        for key in (
            "interaction_tag",
            "hunger_state_start",
            "hunger_state_end",
            "hunger_drive_enabled",
            "stomach_level_start",
            "stomach_level_end",
            "active_energy_cost",
            "homeostatic_reward",
            "meals_eaten_count",
            "last_meal_payload",
            "n_turns",
            "trigger_mode",
        ):
            val = getattr(r, key, None)
            if val is not None:
                c[key] = val
        return c

    # ── interaction orchestration ─────────────────────────────────────────────

    def _execute_interaction(
        self,
        track_id:      int,
        face_id:       str,
        social_state:  str,
        interaction_id: Optional[str] = None,
    ) -> InteractionResult:
        self._interaction_abort_event.clear()
        result = InteractionResult(initial_state=social_state, final_state=social_state)
        result.resolved_face_id = face_id
        self._mark_homeostasis_start(result, social_state)

        if social_state == "ss4":
            result.success     = True
            result.final_state = "ss4"
            self._log("INFO", "ss4: no-op")
            self._finalize_homeostasis_result(result)
            return result

        # Resolve unconfirmed face IDs
        if not self._face_resolved(face_id):
            self._log("INFO", "face_id unresolved – waiting…")
            face_id = self._wait_face_resolve(track_id, face_id, 5.0)
            result.resolved_face_id = face_id
            if not self._face_resolved(face_id):
                result.abort_reason = "face_id_unresolved"
                self._finalize_homeostasis_result(result)
                return result

        self._start_monitor(track_id, face_id, result, interaction_id)
        self._selector_set_track(track_id)

        try:
            hs = result.hunger_state_start

            # Orexigenic drive overrides (HS3 only)
            if self.hunger_enabled and hs == "HS3":
                self._run_hunger_tree(social_state, hs, result)
                if social_state == "ss3" and not self._should_abort(result):
                    self._log("INFO", "Resuming SS3 after feeding")
                    self._run_ss3(face_id, result)
            else:
                if social_state == "ss1":
                    self._run_ss1(track_id, face_id, result)
                elif social_state == "ss2":
                    self._run_ss2(track_id, face_id, result)
                elif social_state == "ss3":
                    self._run_ss3(face_id, result)
        except Exception as e:
            self._log("ERROR", f"Tree execution error: {e}")
            result.abort_reason = f"exception: {e}"
        finally:
            self._stop_monitor()
            self._selector_set_track(-1)
            self._finalize_homeostasis_result(result)

        return result

    # ── social state trees ────────────────────────────────────────────────────

    def _run_ss1(self, track_id: int, face_id: str, result: InteractionResult) -> None:
        """Unknown person → greet → extract name → nice to meet you → ss3."""
        tag = "[SS1|unknown]"
        self._log("INFO", f"{tag} START")

        self._stt_clear()
        greet_trace = LatencyTrace(self, label=f"{tag}|greet", turn_index=0)
        if self._speech.dispatch(self._P.get("ss1_greeting", "Hi there!"), label="ss1_greeting", trace=greet_trace) is None:
            result.abort_reason = "tts_dispatch_failed"
            self._log("WARNING", f"{tag} ABORT: tts_dispatch_failed")
            return
        self._charge_energy(self.GREETING_ENERGY_COST, result, "ss1_greeting")
        greet_trace.mark("listen_open")
        result.greeted = True
        if self._should_abort(result):
            return

        response, _ = self._wait_for_user_utterance(self.SS1_STT_TIMEOUT, trace=greet_trace)
        if not response:
            result.abort_reason = result.abort_reason or "no_response_greeting"
            self._log("WARNING", f"{tag} ABORT: no_response_greeting")
            return
        result.replied_any = True

        if self._should_abort(result):
            return

        name, reason = self._ask_name(tag, result)
        if reason != "ok" or not name:
            result.abort_reason = result.abort_reason or reason
            self._log("WARNING", f"{tag} ABORT: {result.abort_reason}")
            return

        result.extracted_name = name
        self._submit_face_name(track_id, name)
        self._write_last_greeted(track_id, face_id=face_id, code=name, person_key=name)
        if self._speak_wait(self._P.get("ss1_nice_to_meet", "Nice to meet you")):
            self._charge_energy(self.GREETING_ENERGY_COST, result, "ss1_nice_to_meet")
        else:
            result.abort_reason = result.abort_reason or "tts_dispatch_failed"
            self._log("WARNING", f"{tag} ABORT: tts_dispatch_failed")
            return

        result.success     = True
        result.final_state = "ss3"
        self._log("INFO", f"[SS1|{name}] DONE")

    def _run_ss2(self, track_id: int, face_id: str, result: InteractionResult) -> None:
        """Known but not greeted → say hello → ss3 conversation."""
        tag = f"[SS2|{face_id}]"
        self._log("INFO", f"{tag} START")

        if face_id.lower() in ("unknown", "unmatched") or face_id.isdigit():
            result.abort_reason = "invalid_name"
            self._log("WARNING", f"{tag} ABORT: invalid_name")
            return

        try:
            greeted = self._load_json(self.GREETED_TODAY_FILE, {})
            if isinstance(greeted, dict) and face_id in greeted:
                self._log("INFO", f"{tag} '{face_id}' already greeted today -> SS3")
                result.success     = True
                result.final_state = "ss3"
                self._run_ss3(face_id, result)
                return
        except Exception as e:
            self._log("WARNING", f"Failed to read greeted_today: {e}")

        utterance = self._greet_known(face_id, timeout=self.SS2_GREET_TIMEOUT, attempts=2,
                                      tag=tag, result=result)
        if not utterance:
            result.abort_reason = result.abort_reason or "no_response_greeting"
            self._log("WARNING", f"{tag} ABORT: no_response_greeting")
            return

        self._log("INFO", f"{tag} response: '{utterance}'")
        self._write_last_greeted(track_id, face_id, face_id, face_id)
        threading.Thread(target=self._mark_greeted_today, args=(face_id,), daemon=True).start()
        result.success     = True
        result.final_state = "ss3"
        self._run_ss3(face_id, result)

    def _wait_for_user_utterance(
        self,
        timeout: float,
        *,
        trace: Optional[LatencyTrace] = None,
    ) -> Tuple[Optional[str], Optional[float]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._abort_requested():
                return None, None
            if trace is not None:
                self._speech.maybe_mark_done(trace=trace)
            text = self._stt_read_once()
            if text:
                received_mono = time.monotonic()
                if trace is not None:
                    self._speech.log_interruption(reason="user_barge_in", trace=trace)
                return text, received_mono
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(self.STT_POLL_INTERVAL_SEC, remaining))
        if trace is not None:
            self._speech.maybe_mark_done(trace=trace)
        return None, None

    def _run_ss3(self, face_id: str, result: InteractionResult) -> None:
        """Short proactive conversation (up to SS3_MAX_TURNS)."""
        tag = f"[SS3|{face_id}]"
        self._log("INFO", f"{tag} START")
        if self._should_abort(result):
            return

        hs = self._current_hs()
        self._log("INFO", f"{tag} hunger={hs}")

        starter_trace = LatencyTrace(self, label=f"{tag}|starter", turn_index=0)
        starter_trace.mark("llm_request_submitted", max_tokens=self.SS3_STARTER_MAX_TOKENS, stream=0)
        starter = self._llm_get_starter(hs)
        starter_ready_mono = time.monotonic()
        starter_trace.mark_at(
            "first_token_received",
            starter_ready_mono,
            time_to_first_response_sec=starter_ready_mono - starter_trace.started_mono,
        )
        starter_trace.mark_at("last_token_received", starter_ready_mono, text_chars=len(starter))
        if not starter:
            result.abort_reason = result.abort_reason or "llm_starter_failed"
            self._log("WARNING", f"{tag} ABORT: llm_starter_failed")
            return

        self._stt_clear()
        self._speech.wait_until_idle(trace=starter_trace, poll_sec=self.TTS_POLL_INTERVAL_SEC)
        if self._speech.dispatch(starter, label="ss3_starter", trace=starter_trace) is None:
            result.abort_reason = result.abort_reason or "tts_dispatch_failed"
            self._log("WARNING", f"{tag} ABORT: tts_dispatch_failed")
            return
        self._charge_energy(self.STARTER_PROMPT_ENERGY_COST, result, "ss3_starter")

        if self._should_abort(result):
            return

        starter_trace.mark("listen_open")
        self._log("INFO", f"{tag} listening…")
        utterance, utterance_mono = self._wait_for_user_utterance(self.SS3_STT_TIMEOUT, trace=starter_trace)
        if not utterance:
            result.abort_reason = result.abort_reason or "no_response_conversation"
            self._log("WARNING", f"{tag} ABORT: no_response_conversation")
            return

        turns = self._run_conversation(
            tag,
            utterance,
            face_id=face_id,
            first_utterance_mono=utterance_mono,
            result=result,
            prior_assistant=starter,
        )
        result.n_turns = turns
        if turns > 0:
            result.success     = True
            result.talked      = True
            result.final_state = "ss4"
            self._log("INFO", f"{tag} DONE turns={turns}")
        else:
            result.abort_reason = result.abort_reason or "no_response_conversation"
            self._log("WARNING", f"{tag} ABORT: {result.abort_reason}")

    def _run_conversation(
        self,
        tag: str,
        first_utterance: str,
        *,
        face_id: str = "",
        first_utterance_mono: Optional[float] = None,
        result: Optional[InteractionResult] = None,
        prior_assistant: Optional[str] = None,
    ) -> int:
        """Run up to SS3_MAX_TURNS follow-up turns with latest-utterance-wins semantics."""
        turns          = 0
        utterance      = first_utterance
        utterance_mono = first_utterance_mono or time.monotonic()
        interaction_id = self._get_iid()
        history: List[Tuple[str, str]] = []
        if prior_assistant:
            history.append(("assistant", prior_assistant))

        while utterance and turns < self.SS3_MAX_TURNS:
            if result is not None and self._should_abort(result):
                break
            if result is not None:
                result.replied_any = True

            next_turn = turns + 1
            is_last   = next_turn >= self.SS3_MAX_TURNS
            trace = LatencyTrace(
                self,
                label=tag,
                turn_index=next_turn,
                utterance=utterance,
                started_mono=utterance_mono,
            )
            trace.mark_at("stt_final_received", utterance_mono, utterance_chars=len(utterance))
            trace.mark_at("end_of_turn_detected", utterance_mono)
            self._log("INFO", f"{tag} turn {next_turn}/{self.SS3_MAX_TURNS}: '{utterance}'")

            reply: Optional[str] = None
            if self._is_greeting(utterance):
                reply = self._local_reply_fallback(utterance, is_last, face_id)
                now = time.monotonic()
                trace.mark_at(
                    "first_token_received",
                    now,
                    time_to_first_response_sec=now - trace.started_mono,
                    source="local",
                )
                trace.mark_at("last_token_received", now, text_chars=len(reply), source="local")
                trace.mark("local_reply_selected", reason="greeting_detected", text_chars=len(reply))
            else:
                hs = self._current_hs()
                req = self._build_reply_request(
                    utterance,
                    is_last=is_last,
                    hs=hs,
                    turn_index=next_turn,
                    interaction_id=interaction_id,
                    history=tuple(history),
                )
                request_id = self._llm_turn_worker.submit(req)
                trace.request_id = request_id
                trace.mark(
                    "llm_request_submitted",
                    max_tokens=req.max_tokens,
                    stream=int(req.stream),
                )

                superseding_utterance: Optional[str] = None
                superseding_mono: Optional[float] = None
                while True:
                    if result is not None and self._should_abort(result):
                        return turns

                    # Keep listening while generation is in flight so a newer
                    # utterance can supersede this request without blocking.
                    self._speech.maybe_mark_done(trace=trace)
                    event = self._llm_turn_worker.poll_event(self.STT_POLL_INTERVAL_SEC)
                    if event is not None:
                        if event.request_id != request_id:
                            if event.kind in ("final", "error", "cancelled") and event.request_id < request_id:
                                self._log("DEBUG", f"{tag} discard stale llm event kind={event.kind} req={event.request_id} current={request_id}")
                            continue
                        if event.kind == "first_token":
                            trace.mark_at(
                                "first_token_received",
                                event.at_mono,
                                time_to_first_response_sec=event.at_mono - trace.started_mono,
                            )
                            continue
                        if event.kind == "final":
                            if not trace.has("first_token_received"):
                                trace.mark_at(
                                    "first_token_received",
                                    event.at_mono,
                                    time_to_first_response_sec=event.at_mono - trace.started_mono,
                                )
                            trace.mark_at("last_token_received", event.at_mono, text_chars=len(event.text))
                            reply = event.text.strip()
                            break
                        if event.kind == "error":
                            trace.mark_at("llm_error", event.at_mono, error=event.error or "unknown")
                            reply = self._local_reply_fallback(utterance, is_last, face_id)
                            fallback_mono = time.monotonic()
                            if not trace.has("first_token_received"):
                                trace.mark_at(
                                    "first_token_received",
                                    fallback_mono,
                                    time_to_first_response_sec=fallback_mono - trace.started_mono,
                                    source="local_fallback",
                                )
                            trace.mark_at(
                                "last_token_received",
                                fallback_mono,
                                text_chars=len(reply),
                                source="local_fallback",
                            )
                            trace.mark("local_fallback", reason=event.error or "llm_error", text_chars=len(reply))
                            break
                        if event.kind == "cancelled":
                            trace.mark_at("llm_cancelled", event.at_mono, reason="superseded")
                            break

                    newer = self._stt_read_once()
                    if newer:
                        superseding_utterance = newer
                        superseding_mono = time.monotonic()
                        trace.mark("interruption", reason="new_user_utterance_before_reply", utterance_chars=len(newer))
                        break

                if superseding_utterance:
                    utterance = superseding_utterance
                    utterance_mono = superseding_mono or time.monotonic()
                    continue

            if not reply:
                if result is not None and not result.abort_reason:
                    result.abort_reason = "llm_reply_failed"
                self._log("WARNING", f"{tag} ABORT: llm_reply_failed")
                break

            if result is not None and self._should_abort(result):
                break

            reply = reply.replace("—", ", ")

            self._speech.wait_until_idle(trace=trace, poll_sec=self.TTS_POLL_INTERVAL_SEC)
            dispatch = self._speech.dispatch(
                reply,
                label="ss3_reply",
                trace=trace,
                request_id=trace.request_id,
            )
            if dispatch is None:
                if result is not None and not result.abort_reason:
                    result.abort_reason = "tts_dispatch_failed"
                self._log("WARNING", f"{tag} ABORT: tts_dispatch_failed")
                break

            self._charge_energy(
                self.CONVERSATION_TURN_ENERGY_COST,
                result,
                "ss3_conversation_turn",
            )
            history.append(("user", utterance))
            history.append(("assistant", reply))

            turns += 1
            if is_last:
                self._speech.wait_until_idle(trace=trace, poll_sec=self.TTS_POLL_INTERVAL_SEC)
                break

            trace.mark("listen_open")
            utterance, utterance_mono = self._wait_for_user_utterance(self.SS3_STT_TIMEOUT, trace=trace)
            if not utterance:
                self._log("INFO", f"{tag} no further response")
                break

        return turns

    # ── Orexigenic drive / QR feeding tree ───────────────────────────────────

    def _run_hunger_tree(self, social_state: str, hs: str, result: InteractionResult,
                         intro_text: Optional[str] = None) -> None:
        self._log("INFO", f"Hunger tree: {hs}")
        self._hunger_tree_active.set()
        try:
            self._stt_clear()
            ask = intro_text if intro_text is not None else self._P.get("hunger_ask_feed", "I'm so hungry, would you feed me please?")
            if self._speak_wait(ask):
                self._charge_energy(self.HUNGER_PROMPT_ENERGY_COST, result, "hunger_ask_feed")
            else:
                result.abort_reason = result.abort_reason or "tts_dispatch_failed"
                return
            result.talked = True

            meals, timeouts, max_timeouts = 0, 0, 2
            drive_disabled = False
            wait_since = time.time()

            while not self._should_abort(result):
                if not self.hunger_enabled:
                    drive_disabled = True
                    self._log("INFO", "Hunger tree stopped: drive disabled")
                    break
                hs_before     = self.hunger.snapshot().state
                fed, payload, new_ts = self._wait_feed_since(wait_since, self._feed_wait_timeout_sec)

                if fed:
                    result.replied_any = True
                    meals += 1
                    result.last_meal_payload = payload
                    snap = self.hunger.snapshot()
                    self._log("INFO", f"Feed #{meals}: {payload} → stomach {snap.level:.1f}")
                    if snap.state != hs_before:
                        self._set_face_emotion(snap.state)
                    if self._speak_wait(self._feed_ack(hs_before)):
                        self._charge_energy(self.FEED_ACK_ENERGY_COST, result, "feed_ack")
                    if snap.state == "HS1":
                        break
                    if self._speak_wait(self._P.get("hunger_still_hungry", "I'm still hungry. Give me more please.")):
                        self._charge_energy(self.HUNGER_PROMPT_ENERGY_COST, result, "hunger_still_hungry")
                    wait_since = new_ts
                    timeouts   = 0
                else:
                    if not self.hunger_enabled:
                        drive_disabled = True
                        self._log("INFO", "Hunger tree stopped: drive disabled")
                        break
                    if self._should_abort(result):
                        break
                    timeouts += 1
                    if timeouts >= max_timeouts:
                        if not result.abort_reason:
                            result.abort_reason = "no_food_qr"
                        break
                    if self._speak_wait(self._P.get("hunger_look_around", "Take a look around, you will find some food for me.")):
                        self._charge_energy(self.HUNGER_PROMPT_ENERGY_COST, result, "hunger_look_around")
                    wait_since = time.time()

            result.meals_eaten_count = meals
            if drive_disabled:
                result.success = True
            elif meals > 0:
                result.success = True
            elif not result.abort_reason:
                result.abort_reason = "no_food_qr"
            result.final_state = social_state
        finally:
            self._hunger_tree_active.clear()

    def _wait_feed_since(self, ts: float, timeout: float) -> Tuple[bool, Optional[str], float]:
        with self._feed_condition:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._abort_requested():
                    return False, None, time.time()
                if not self.hunger_enabled:
                    return False, None, time.time()
                with self.hunger._lock:
                    lfts    = self.hunger.last_feed_ts
                    payload = self.hunger.last_feed_payload
                if lfts > ts:
                    return True, payload, lfts
                wait_for = deadline - time.monotonic()
                if wait_for > 0:
                    self._feed_condition.wait(min(wait_for, 0.5))
            return False, None, time.time()

    def _feed_ack(self, hs_before: str) -> str:
        if hs_before == "HS1":
            return self._P.get("feed_ack_hs1", "Oh no, I'm already so full! One more bite and I might actually explode!")
        if hs_before == "HS3":
            return self._P.get("feed_ack_hs3", "Oh wow, thank you! You literally just saved my life!")
        return self._P.get("feed_ack_hs2", "yummy! Thank you so much!")

    # ── QR reader loop ────────────────────────────────────────────────────────

    def _qr_reader_loop(self) -> None:
        while self._running and not self._qr_stop.is_set():
            if not self.qr_port:
                self._qr_stop.wait(0.1)
                continue
            try:
                btl = self.qr_port.read(False)
                if not btl or btl.size() == 0:
                    self._qr_stop.wait(0.05)
                    continue

                raw = btl.get(0).asString()
                val = (raw or "").strip().upper()
                if val not in self._meal_mapping:
                    if raw:
                        self._log("DEBUG", f"Unknown QR payload: '{raw}'")
                    self._qr_stop.wait(0.02)
                    continue

                if not self.hunger_enabled:
                    self._log("DEBUG", f"Ignoring QR '{val}' (hunger OFF)")
                    self._qr_stop.wait(0.02)
                    continue

                now_mono = time.monotonic()
                if now_mono - self._last_scan_ts_mono < self._qr_cooldown_sec:
                    self._qr_stop.wait(0.02)
                    continue

                self._last_scan_ts_mono = now_mono
                now           = time.time()
                delta         = self._meal_mapping[val]
                snap_before   = self.hunger.snapshot()
                hs_before     = snap_before.state
                self.hunger.feed(delta, val, now)
                snap          = self.hunger.snapshot()
                self._log_hunger_level_event(
                    "feeding",
                    stimulus_type="feeding",
                    stimulus_label=val,
                    level_before=snap_before.level,
                    level_after=snap.level,
                    state_before=hs_before,
                    state_after=snap.state,
                    delta=snap.level - snap_before.level,
                    meal_delta=delta,
                    meal_payload=val,
                    reason="qr_feed",
                    force=True,
                )
                self._log("INFO", f"QR: {val} (+{delta}) → stomach {snap.level:.1f} ({snap.state})")
                if snap.state != hs_before:
                    self._set_face_emotion(snap.state)

                with self._feed_condition:
                    self._feed_condition.notify_all()

                if not self._hunger_tree_active.is_set():
                    if self._speak(self._feed_ack(hs_before)):
                        self._charge_energy(self.FEED_ACK_ENERGY_COST, None, "ambient_feed_ack")
                    self._db_enqueue(("reactive", {
                        "type":                 "qr_feed",
                        "track_id":             None,
                        "name":                 None,
                        "payload":              val,
                        "hunger_state_before":  hs_before,
                        "stomach_level_before": snap_before.level,
                        "hunger_state_after":   snap.state,
                        "stomach_level_after":  snap.level,
                    }))

                self._qr_stop.wait(0.02)
            except Exception as e:
                self._log("WARNING", f"QR loop error: {e}")
                self._qr_stop.wait(0.1)

    # ── reactive greeting loop ────────────────────────────────────────────────

    def _reactive_loop(self) -> None:
        self._log("INFO", "Reactive loop started")
        while self._running and not self._reactive_stop.is_set():
            busy, mode, _, _ = self._busy_snapshot()
            if busy and mode != "reactive":
                self._reactive_stop.wait(0.05)
                continue
            try:
                if not self.stt_port:
                    self._reactive_stop.wait(0.05)
                    continue

                utterance = self._stt_read_once()
                if not utterance:
                    self._reactive_stop.wait(0.05)
                    continue
                if not self._is_greeting(utterance):
                    self._log_throttled("DEBUG", "reactive_ngreet",
                                        f"Reactive: '{(utterance or '')[:40]}' – not a greeting", 1.5)
                    continue

                self._log("INFO", f"Reactive: greeting '{utterance}'")
                candidate = self._reactive_candidate()
                if not candidate:
                    self._log_throttled("DEBUG", "reactive_nocand", "Reactive: no candidate", 1.0)
                    continue

                track_id, face_id, is_known = candidate
                ck = face_id if is_known else f"unknown:{track_id}"
                if not self._reactive_cooldown_ok(ck):
                    self._log("DEBUG", f"Reactive: '{ck}' in cooldown")
                    continue

                if is_known:
                    self._run_reactive_greeting(track_id, face_id)
                else:
                    self._run_reactive_unknown_intro(track_id, face_id)
            except Exception as e:
                self._log("WARNING", f"Reactive loop error: {e}")
                self._reactive_stop.wait(0.05)
        self._log("INFO", "Reactive loop stopped")

    def _run_reactive_greeting(self, track_id: int, name: str) -> None:
        accepted, mode, reason = self._begin_interaction("reactive", f"reactive_greeting:{track_id}:{name}")
        if not accepted:
            self._log("DEBUG", f"Reactive greeting skipped – {mode}: {reason}")
            return
        interaction_id = uuid.uuid4().hex
        dummy = InteractionResult(initial_state="ss3", final_state="ss3")
        dummy.trigger_mode = "reactive"
        dummy.resolved_face_id = name
        self._mark_homeostasis_start(dummy, "ss3")
        prev_iid = self._get_iid()
        try:
            self._init_ilog(interaction_id)
            self._set_iid(interaction_id)
            self._interaction_abort_event.clear()
            self._selector_set_track(track_id)
            self._start_monitor(track_id, name, dummy)
            tag = f"[RGREET|{name}]"
            try:
                self._log("INFO", f"{tag} START (track={track_id})")
                hs = self._effective_hunger_state()
                if self.hunger_enabled and hs == "HS3":
                    hunger_ask = self._P.get("hunger_ask_feed", "I'm so hungry, would you feed me please?")
                    intro = f"Hello, {hunger_ask}"
                    self._run_hunger_tree("ss3", hs, dummy, intro_text=intro)
                    dummy.success = not dummy.aborted
                    dummy.final_state = "ss3"
                    return

                utterance = self._greet_known(
                    name,
                    timeout=self.SS3_STT_TIMEOUT,
                    attempts=1,
                    tag=tag,
                    result=dummy,
                )
                if not dummy.aborted:
                    dummy.success = True
                    dummy.final_state = "ss3"

                if utterance and not self._should_abort(dummy):
                    greeting = self._P.get("reactive_greeting", "Hi {name}").format(name=name)
                    turns    = self._run_conversation(
                        f"[RSS3|{name}]",
                        utterance,
                        face_id=name,
                        prior_assistant=greeting,
                        result=dummy,
                    )
                    dummy.n_turns = turns
                    if turns > 0:
                        dummy.success = True
                        dummy.talked = True
                        dummy.final_state = "ss4"
                    self._log("INFO", f"[RSS3|{name}] DONE turns={turns}")

                if not self._abort_requested() and not dummy.aborted:
                    self._write_last_greeted(track_id, face_id=name, code=name, person_key=name)
                    self._mark_greeted_today(name)
                    self._db_enqueue(("reactive", {
                        "interaction_id": interaction_id,
                        "type": "greeting",
                        "track_id": track_id,
                        "name": name,
                        "payload": None,
                    }))
            finally:
                self._stop_monitor()
                self._finalize_homeostasis_result(dummy)
                self._selector_submit_interaction_result(
                    dummy,
                    track_id=track_id,
                    face_id=name,
                    social_state="ss3",
                    person_id=name,
                    interaction_id=interaction_id,
                )
                self._selector_reset_cooldown(name, track_id)
                self._selector_set_track(-1)
        finally:
            self._set_iid(prev_iid)
            self._pop_ilog(interaction_id)
            self._end_interaction("reactive")

    def _run_reactive_unknown_intro(self, track_id: int, face_id: str) -> None:
        accepted, mode, reason = self._begin_interaction("reactive", f"reactive_unknown:{track_id}:{face_id}")
        if not accepted:
            self._log("DEBUG", f"Reactive unknown skipped – {mode}: {reason}")
            return
        interaction_id = uuid.uuid4().hex
        dummy = InteractionResult(initial_state="ss1", final_state="ss1")
        dummy.trigger_mode = "reactive"
        dummy.resolved_face_id = face_id
        self._mark_homeostasis_start(dummy, "ss1")
        person_id: Optional[str] = None
        prev_iid = self._get_iid()
        try:
            self._init_ilog(interaction_id)
            self._set_iid(interaction_id)
            self._interaction_abort_event.clear()
            self._selector_set_track(track_id)
            self._start_monitor(track_id, face_id, dummy)
            try:
                self._log("INFO", f"Reactive unknown intro: track={track_id} face='{face_id}'")

                # Greet without requiring a greeting response (they already said hi)
                self._stt_clear()
                if self._speak_wait(self._P.get("ss1_greeting", "Hi there!")):
                    self._charge_energy(self.GREETING_ENERGY_COST, dummy, "reactive_unknown_greeting")
                    dummy.greeted = True
                else:
                    dummy.abort_reason = dummy.abort_reason or "tts_dispatch_failed"
                    return
                if self._abort_requested() or dummy.aborted:
                    return

                hs = self._effective_hunger_state()
                if self.hunger_enabled and hs == "HS3" and not self._should_abort(dummy):
                    self._run_hunger_tree("ss1", hs, dummy)

                if self._abort_requested() or dummy.aborted:
                    return

                name, reason2 = self._ask_name("[SS1|unknown]", result=dummy)
                if reason2 != "ok" or not name:
                    dummy.abort_reason = dummy.abort_reason or reason2
                    self._log("INFO", f"Reactive unknown intro: {reason2}")
                    return
                if self._abort_requested() or dummy.aborted:
                    return

                self._log("INFO", f"Reactive unknown intro: name='{name}'")
                person_id = name
                dummy.extracted_name = name
                dummy.resolved_face_id = name
                self._submit_face_name(track_id, name)
                self._write_last_greeted(track_id, name, name, name)
                self._mark_greeted_today(name)
                if self._speak_wait(self._P.get("ss1_nice_to_meet", "Nice to meet you")):
                    self._charge_energy(self.GREETING_ENERGY_COST, dummy, "reactive_nice_to_meet")
                else:
                    dummy.abort_reason = dummy.abort_reason or "tts_dispatch_failed"
                    return
                dummy.success = True
                dummy.final_state = "ss3"
                self._db_enqueue(("reactive", {
                    "interaction_id": interaction_id,
                    "type": "unknown_intro",
                    "track_id": track_id,
                    "name": name,
                    "payload": face_id,
                }))
            finally:
                self._stop_monitor()
                self._finalize_homeostasis_result(dummy)
                self._selector_submit_interaction_result(
                    dummy,
                    track_id=track_id,
                    face_id=face_id,
                    social_state="ss1",
                    person_id=person_id,
                    interaction_id=interaction_id,
                )
                self._selector_reset_cooldown(face_id, track_id)
                self._selector_set_track(-1)
        finally:
            self._set_iid(prev_iid)
            self._pop_ilog(interaction_id)
            self._end_interaction("reactive")

    # ── shared interaction helpers ─────────────────────────────────────────────

    def _greet_known(
        self,
        name:     str,
        timeout:  float,
        attempts: int = 1,
        tag:      str = "[greet]",
        result:   Optional[InteractionResult] = None,
    ) -> Optional[str]:
        tpl = self._P.get("ss2_greeting", "Hello {name}")
        for attempt in range(max(1, attempts)):
            self._log("INFO", f"{tag} greeting attempt {attempt+1}/{attempts}")
            if result is not None and self._should_abort(result):
                return None
            self._stt_clear()
            greet_trace = LatencyTrace(self, label=f"{tag}|greet", turn_index=0)
            if self._speech.dispatch(tpl.format(name=name), label="known_greeting", trace=greet_trace) is None:
                if result is not None:
                    result.abort_reason = result.abort_reason or "tts_dispatch_failed"
                return None
            self._charge_energy(self.GREETING_ENERGY_COST, result, "known_greeting")
            greet_trace.mark("listen_open")
            if result is not None:
                result.greeted = True
            if result is not None and self._should_abort(result):
                return None
            utterance, _ = self._wait_for_user_utterance(timeout, trace=greet_trace)
            if utterance:
                if result is not None:
                    result.replied_any = True
                return utterance
            self._log("WARNING", f"{tag} no response (attempt {attempt+1}/{attempts})")
        return None

    def _ask_name(
        self, tag: str, result: Optional[InteractionResult]
    ) -> Tuple[Optional[str], str]:
        """Ask for name with one retry; return (name, reason)."""
        self._stt_clear()
        self._speech.wait_until_idle(poll_sec=self.TTS_POLL_INTERVAL_SEC)
        if self._speak(self._P.get("ss1_ask_name", "We have not met, what's your name?")):
            self._charge_energy(self.NAME_QUESTION_ENERGY_COST, result, "ss1_ask_name")
        else:
            return None, "tts_dispatch_failed"
        if result is not None and self._should_abort(result):
            return None, "aborted"

        utt = self._stt_wait(self.SS1_STT_TIMEOUT)
        if not utt:
            return None, "no_response_name"
        name = self._extract_name(utt)
        if name:
            return name, "ok"

        self._log("INFO", f"{tag} retrying name extraction")
        if result is not None and self._should_abort(result):
            return None, "aborted"
        self._stt_clear()
        self._speech.wait_until_idle(poll_sec=self.TTS_POLL_INTERVAL_SEC)
        if self._speak(self._P.get("ss1_ask_name_retry", "Sorry, I didn't catch that. What's your name?")):
            self._charge_energy(self.NAME_QUESTION_ENERGY_COST, result, "ss1_ask_name_retry")
        else:
            return None, "tts_dispatch_failed"
        utt2 = self._stt_wait(self.SS1_STT_TIMEOUT)
        if utt2:
            name = self._extract_name(utt2)
            if name:
                return name, "ok"
        return None, "name_extraction_failed"

    # ── abort helpers ─────────────────────────────────────────────────────────

    def _abort_requested(self) -> bool:
        return self.abort_event.is_set() or self._interaction_abort_event.is_set()

    def _should_abort(self, result: InteractionResult) -> bool:
        if self._abort_requested():
            if not result.abort_reason:
                result.abort_reason = "target_monitor_abort"
            return True
        return result.aborted

    # ── interaction ownership ─────────────────────────────────────────────────

    def _begin_interaction(self, mode: str, reason: str) -> Tuple[bool, str, str]:
        with self._interaction_state_lock:
            if self._interaction_mode != "idle":
                return False, self._interaction_mode, self._interaction_reason
            self._interaction_mode          = mode
            self._interaction_reason        = reason
            self._interaction_started_mono  = time.monotonic()
        return True, mode, reason

    def _end_interaction(self, mode: str) -> None:
        with self._interaction_state_lock:
            if self._interaction_mode == mode:
                self._interaction_mode         = "idle"
                self._interaction_reason       = ""
                self._interaction_started_mono = 0.0

    def _busy_snapshot(self) -> Tuple[bool, str, str, float]:
        with self._interaction_state_lock:
            mode    = self._interaction_mode
            reason  = self._interaction_reason
            started = self._interaction_started_mono
        busy     = mode != "idle"
        for_sec  = max(0.0, time.monotonic() - started) if busy and started > 0 else 0.0
        return busy, mode, reason, for_sec

    # ── target monitor ────────────────────────────────────────────────────────

    def _start_monitor(
        self,
        track_id:      int,
        face_id:       str,
        result:        InteractionResult,
        interaction_id: Optional[str] = None,
    ) -> None:
        self._current_track_id = track_id
        self._monitor_thread   = threading.Thread(
            target = self._monitor_loop,
            args   = (track_id, face_id, result, interaction_id),
            daemon = True,
        )
        self._monitor_thread.start()

    def _stop_monitor(self) -> None:
        self._interaction_abort_event.set()
        self._current_track_id = None

    def _monitor_loop(
        self,
        track_id:       int,
        expected_face:  str,
        result:         InteractionResult,
        interaction_id: Optional[str] = None,
    ) -> None:
        prev_iid = self._get_iid()
        if interaction_id:
            self._set_iid(interaction_id)
        expected_norm = ""
        if self._face_resolved(expected_face):
            expected_norm = self._norm_name(expected_face)

        last_seen   = time.monotonic()
        started_at  = time.monotonic()
        last_iter   = time.monotonic()
        interval    = 1.0 / self.MONITOR_HZ

        try:
            while not self._abort_requested():
                now = time.monotonic()
                # Thread starvation guard
                if now - last_iter > 1.5:
                    last_seen = now
                last_iter = now

                with self._faces_lock:
                    faces      = list(self._latest_faces)
                    faces_ts   = self._latest_faces_ts
                    packet_ts  = self._latest_landmarks_packet_ts

                packet_age = (time.time() - packet_ts) if packet_ts > 0 else float("inf")
                faces_age  = (time.time() - faces_ts)  if faces_ts  > 0 else float("inf")

                if packet_age > self.MONITOR_PACKET_STALE_SEC:
                    # Landmark stream is stale; do not count this as target absence.
                    last_seen = time.monotonic()
                    time.sleep(interval)
                    continue
                if faces_age > self.MONITOR_PACKET_STALE_SEC:
                    # Parsed face list is stale; pause absence countdown until fresh data arrives.
                    last_seen = time.monotonic()
                    faces = []

                found = any(int(f.get("track_id", -1)) == track_id for f in faces)
                if not found and expected_norm:
                    found = any(
                        self._norm_name(str(f.get("face_id", ""))) == expected_norm
                        for f in faces if self._face_resolved(str(f.get("face_id", "")))
                    )

                if found:
                    last_seen = time.monotonic()
                else:
                    # Grace period right after monitor start to avoid false aborts during selector settle.
                    if time.monotonic() - started_at < self.MONITOR_WARMUP_SEC:
                        time.sleep(interval)
                        continue
                    absent = time.monotonic() - last_seen
                    if absent > self.TARGET_LOST_TIMEOUT:
                        self._log("WARNING", f"Monitor: track {track_id} absent {absent:.1f}s")
                        result.target_stayed_biggest = False
                        result.abort_reason          = "target_lost"
                        self._interaction_abort_event.set()
                        return

                time.sleep(interval)
        finally:
            if interaction_id:
                self._set_iid(prev_iid)

    # ── face helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _face_resolved(face_id: str) -> bool:
        return face_id.lower() not in ("recognizing", "unmatched")

    def _wait_face_resolve(self, track_id: int, face_id: str, timeout: float) -> str:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._abort_requested():
                return face_id
            with self._faces_lock:
                faces = list(self._latest_faces)
            for f in faces:
                if f.get("track_id") == track_id:
                    fid = f.get("face_id", "recognizing")
                    if self._face_resolved(fid):
                        return fid
            time.sleep(0.2)
        return face_id

    def _latest_faces_snapshot(self, staleness: float = 2.0) -> List[Dict]:
        with self._faces_lock:
            if time.time() - self._latest_faces_ts > staleness:
                return []
            return list(self._latest_faces)

    def _reactive_candidate(self) -> Optional[Tuple[int, str, bool]]:
        faces = self._latest_faces_snapshot(30.0)
        if not faces:
            return None

        candidates: List[Tuple[int, str, float, bool]] = []
        for f in faces:
            fid      = str(f.get("face_id", "")).strip()
            tid      = f.get("track_id")
            bbox     = f.get("bbox", [0, 0, 0, 0])
            area     = (bbox[2] * bbox[3]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 0
            if not isinstance(tid, int) or tid < 0:
                continue
            is_known = bool(fid and fid.lower() not in ("unknown", "unmatched", "recognizing") and not fid.isdigit())
            candidates.append((tid, fid, area, is_known))

        if not candidates:
            return None
        best = max(candidates, key=lambda c: c[2])
        self._log("DEBUG", f"Reactive: selected '{best[1]}' ({'known' if best[3] else 'unknown'}, track={best[0]}, area={best[2]:.0f})")
        return (best[0], best[1], best[3])

    def _reactive_cooldown_ok(self, key: str) -> bool:
        k   = key.lower()
        now = time.monotonic()
        if now - self._reactive_greet_cooldown.get(k, 0.0) < self.REACTIVE_GREET_COOLDOWN:
            return False
        self._reactive_greet_cooldown[k] = now
        return True

    # ── landmarks reader ──────────────────────────────────────────────────────

    def _landmarks_reader_loop(self) -> None:
        while self._running and not self._landmarks_stop.is_set():
            if not self.landmarks_port:
                self._landmarks_stop.wait(0.01)
                continue
            try:
                bottle = self.landmarks_port.read(False)
                if bottle is not None:
                    now     = time.time()
                    faces   = []
                    with self._faces_lock:
                        self._latest_landmarks_packet_ts = now
                    for i in range(bottle.size()):
                        face_bottle = bottle.get(i).asList()
                        if face_bottle:
                            d = self._parse_face_bottle(face_bottle)
                            if d:
                                faces.append(d)
                    if faces:
                        with self._faces_lock:
                            self._latest_faces    = faces
                            self._latest_faces_ts = now
                    elif bottle.size() > 0:
                        self._log("DEBUG", f"Landmarks: {bottle.size()} sub-bottles, 0 parsed")
                else:
                    self._landmarks_stop.wait(0.01)
            except Exception as e:
                self._log("WARNING", f"Landmarks loop error: {e}")
                self._landmarks_stop.wait(0.05)

    def _parse_face_bottle(self, bottle: yarp.Bottle) -> Optional[Dict]:
        data = {}
        try:
            i = 0
            while i < bottle.size():
                item = bottle.get(i)
                if item.isList():
                    lst = item.asList()
                    if lst and lst.size() >= 2:
                        key = lst.get(0).asString()
                        if key in ("bbox", "gaze_direction"):
                            data[key] = [lst.get(j).asFloat64() for j in range(1, lst.size())]
                    i += 1
                else:
                    if i + 1 >= bottle.size():
                        break
                    key = item.asString()
                    val = bottle.get(i + 1)
                    if   key in ("face_id", "distance", "attention", "zone"):
                        data[key] = val.asString()
                    elif key in ("track_id", "is_talking"):
                        data[key] = val.asInt32()
                    elif key in ("time_in_view", "pitch", "yaw", "roll", "cos_angle"):
                        data[key] = val.asFloat64()
                    i += 2
            return data if data else None
        except Exception as e:
            self._log("WARNING", f"Face bottle parse error: {e}")
            return None

    # ── speech I/O ────────────────────────────────────────────────────────────

    def _speak(self, text: str) -> bool:
        if not self.speech_port:
            return False
        try:
            b = self.speech_port.prepare()
            b.clear()
            b.addString(text)
            self.speech_port.write()
            return True
        except Exception as e:
            self._log("ERROR", f"speak() failed: {e}")
            return False

    def _estimate_speech_wait(self, text: str) -> float:
        words = len((text or "").split())
        wait = words / self.TTS_WORDS_PER_SECOND + self.TTS_END_MARGIN
        return max(self.TTS_MIN_WAIT, min(self.TTS_MAX_WAIT, wait))

    def _speak_wait(self, text: str) -> bool:
        self._speech.wait_until_idle(poll_sec=self.TTS_POLL_INTERVAL_SEC)
        dispatch = self._speech.dispatch(text, label="blocking_tts")
        if dispatch is None:
            return False
        self._speech.wait_until_idle(poll_sec=self.TTS_POLL_INTERVAL_SEC)
        return True

    def _stt_clear(self) -> None:
        cleared = 0
        with self._stt_lock:
            while self.stt_port and self.stt_port.read(False):
                cleared += 1
        if cleared:
            self._log("DEBUG", f"Cleared {cleared} STT messages")

    def _stt_read_once(self) -> Optional[str]:
        if not self.stt_port:
            return None
        acquired = self._stt_lock.acquire(timeout=self.STT_POLL_INTERVAL_SEC)
        if not acquired:
            return None
        try:
            bottle = self.stt_port.read(False)
        except Exception as e:
            self._log("WARNING", f"STT read error: {e}")
            return None
        finally:
            self._stt_lock.release()

        if bottle and bottle.size() > 0:
            text = self._parse_stt(bottle)
            if text:
                self._log("DEBUG", f"STT: '{text}'")
                return text
        return None

    def _stt_wait(self, timeout: float) -> Optional[str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._abort_requested():
                return None
            text = self._stt_read_once()
            if text:
                return text
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(self.STT_POLL_INTERVAL_SEC, remaining))
        return None

    def _parse_stt(self, bottle: yarp.Bottle) -> Optional[str]:
        try:
            if bottle.size() >= 1:
                raw = bottle.get(0).toString()
                if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
                    t = raw[1:-1]
                else:
                    t = raw
                return t.strip() or None
        except Exception as e:
            self._log("WARNING", f"STT parse error: {e}")
        return None

    # ── text utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _norm_name(name: str) -> str:
        """Lowercase, strip diacritics for robust matching."""
        s = (name or "").strip().lower()
        return "".join(
            c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
        )

    @staticmethod
    def _norm_text(text: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())

    def _is_greeting(self, utterance: str) -> bool:
        return bool(self.REACTIVE_GREET_REGEX.search(self._norm_text(utterance)))

    def _extract_name(self, utterance: str) -> Optional[str]:
        # Fast regex
        m = re.search(
            r"(?i)(?:my name is|my name's|i am|i'm|im|call me)\s+([a-z][a-z'\-]+)",
            utterance,
        )
        if m:
            return m.group(1).title()
        # LLM fallback
        try:
            result = self._llm_extract_name(utterance)
        except Exception as e:
            self._log("WARNING", f"Name extraction error: {e}")
            return None
        if isinstance(result, dict) and result.get("answered") and result.get("name"):
            return result["name"]
        return None

    # ── YARP RPC clients ──────────────────────────────────────────────────────

    def _get_selector_rpc(self) -> yarp.RpcClient:
        if self._selector_rpc is None:
            client = yarp.RpcClient()
            lp     = f"/{self.module_name}/salienceNetwork/rpc"
            if not client.open(lp):
                raise RuntimeError(f"Failed to open {lp}")
            if not client.addOutput("/salienceNetwork"):
                client.close()
                raise RuntimeError("Failed to connect to /salienceNetwork")
            self._selector_rpc = client
        return self._selector_rpc

    def _get_vision_rpc(self) -> yarp.RpcClient:
        if self._vision_rpc is None:
            client = yarp.RpcClient()
            lp     = f"/{self.module_name}/vision/rpc"
            if not client.open(lp):
                raise RuntimeError(f"Failed to open {lp}")
            if not client.addOutput("/alwayson/vision/rpc"):
                client.close()
                raise RuntimeError("Failed to connect to /alwayson/vision/rpc")
            self._vision_rpc = client
        return self._vision_rpc

    def _selector_set_track(self, track_id: int) -> None:
        try:
            cmd = yarp.Bottle()
            cmd.addString("set_track_id")
            cmd.addInt32(track_id)
            reply = yarp.Bottle()
            self._get_selector_rpc().write(cmd, reply)
            self._log("INFO", f"Selector override → track_id={track_id}")
        except Exception as e:
            self._log("WARNING", f"selector_set_track failed: {e}")
            if self._selector_rpc:
                self._selector_rpc.close()
                self._selector_rpc = None

    def _selector_reset_cooldown(self, face_id: str, track_id: int) -> None:
        try:
            cmd = yarp.Bottle()
            cmd.addString("reset_cooldown")
            cmd.addString(face_id)
            cmd.addInt32(track_id)
            reply = yarp.Bottle()
            self._get_selector_rpc().write(cmd, reply)
        except Exception as e:
            self._log("WARNING", f"selector_reset_cooldown failed: {e}")

    def _selector_submit_interaction_result(
        self,
        result: InteractionResult,
        track_id: int,
        face_id: str,
        social_state: str,
        person_id: Optional[str] = None,
        interaction_id: Optional[str] = None,
    ) -> None:
        """
        Send completed reactive interaction results to salienceNetwork
        so homeostatic learning is applied consistently.
        """
        try:
            iid = interaction_id or self._get_iid() or uuid.uuid4().hex
            payload = self._build_compact_result(iid, track_id, face_id, social_state, result)
            if person_id:
                payload["person_id"] = person_id
            cmd = yarp.Bottle()
            cmd.addString("interaction_result")
            cmd.addString(json.dumps(payload, ensure_ascii=False))
            reply = yarp.Bottle()
            self._get_selector_rpc().write(cmd, reply)
            if reply.size() > 0 and reply.get(0).asString() == "ok":
                self._log("DEBUG", f"selector interaction_result submitted id={iid[:8]}")
            else:
                self._log("WARNING", f"selector interaction_result unexpected reply: {reply.toString()}")
        except Exception as e:
            self._log("WARNING", f"selector_submit_interaction_result failed: {e}")
            if self._selector_rpc:
                self._selector_rpc.close()
                self._selector_rpc = None

    def _get_emotions_rpc(self) -> yarp.RpcClient:
        if self._emotions_rpc is None:
            client = yarp.RpcClient()
            lp     = f"/{self.module_name}/emotions/rpc"
            if not client.open(lp):
                raise RuntimeError(f"Failed to open {lp}")
            if not client.addOutput("/icub/face/emotions/in"):
                client.close()
                raise RuntimeError("Failed to connect to /icub/face/emotions/in")
            self._emotions_rpc = client
        return self._emotions_rpc

    def _set_face_emotion(self, hs: str) -> None:
        _HS_EMOTIONS = {
            "HS3": [("mou", "sad"), ("leb", "sad"), ("reb", "sad")],
            "HS2": [("mou", "sad"), ("leb", "neu"), ("reb", "neu")],
            "HS1": [("all", "hap")],
        }
        parts = _HS_EMOTIONS.get(hs)
        if parts is None:
            return

        def _send():
            try:
                rpc = self._get_emotions_rpc()
                for part, expr in parts:
                    cmd = yarp.Bottle()
                    cmd.addString("set")
                    cmd.addString(part)
                    cmd.addString(expr)
                    rpc.write(cmd)
                self._log("INFO", f"Face emotion set for {hs}")
            except Exception as e:
                self._log("WARNING", f"_set_face_emotion({hs}) failed: {e}")
                if self._emotions_rpc:
                    self._emotions_rpc.close()
                    self._emotions_rpc = None

        threading.Thread(target=_send, daemon=True).start()

    def _submit_face_name(self, track_id: int, name: str) -> bool:
        try:
            cmd = yarp.Bottle()
            cmd.addString("name")
            cmd.addString(name)
            cmd.addString("id")
            cmd.addInt32(track_id)
            reply = yarp.Bottle()
            self._get_vision_rpc().write(cmd, reply)
            resp = reply.toString()
            self._log("INFO", f"Submitted name '{name}' for track {track_id}: {resp}")
            return "ok" in resp.lower()
        except Exception as e:
            self._log("ERROR", f"submit_face_name failed: {e}")
            if self._vision_rpc:
                self._vision_rpc.close()
                self._vision_rpc = None
            return False

    # ── LLM integration ───────────────────────────────────────────────────────

    def _setup_llm(self) -> None:
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().strip('"').strip("'")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY",  "").strip().strip('"').strip("'")
        api_ver    = (os.getenv("OPENAI_API_VERSION", "") or os.getenv("AZURE_OPENAI_API_VERSION", "")).strip().strip('"').strip("'")
        deployment = os.getenv("AZURE_DEPLOYMENT_GPT41_MINI", "contact-Yogaexperiment_gpt41mini")

        if not all([endpoint, api_key, api_ver, deployment]):
            raise RuntimeError("Missing one or more Azure OpenAI env vars")

        if "/openai/" in endpoint:
            endpoint = endpoint.split("/openai/")[0]
        endpoint = endpoint.rstrip("/")

        # Disable HTTP connection pooling. In this network environment the
        # upstream (NAT / hotspot) silently drops idle TCP flows after a few
        # seconds; when httpx tries to reuse a dead socket it waits ~10s for
        # TCP timeout before giving up and reconnecting — adding ~10s to the
        # first LLM call after any idle gap. A fresh TLS handshake is only
        # ~500ms, so forcing a new connection every call is strictly faster
        # for this flow. Tested configurations (SO_KEEPALIVE + long
        # keepalive_expiry) did not prevent the drops.
        http_client = httpx.Client(
            timeout=self.LLM_TIMEOUT,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=0,
                keepalive_expiry=0.0,
            ),
        )
        self.llm_client       = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key,
                                             api_version=api_ver, timeout=self.LLM_TIMEOUT,
                                             max_retries=0, http_client=http_client)
        self._llm_deployment  = deployment
        self._log("INFO", f"Azure LLM ready – deployment={self._llm_deployment}")

        # Pre-warm HTTP/TLS connection so the first real interaction call
        # doesn't pay DNS + TLS handshake latency inside a 12s budget.
        try:
            t0 = time.monotonic()
            self.llm_client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "ok"}],
                max_completion_tokens=16,
                timeout=20.0,
            )
            self._log("INFO", f"LLM warm-up ok in {time.monotonic()-t0:.2f}s")
        except Exception as e:
            self._log("WARNING", f"LLM warm-up failed: {type(e).__name__}: {e}")

    def _llm_call(
        self,
        prompt:          str,
        system:          str,
        max_tokens:      int,
        timeout:         Optional[float] = None,
        response_format: Optional[Any]   = None,
        deployment:      Optional[str]   = None,
    ) -> LlmResult:
        if self.llm_client is None:
            return self.LlmResult(ok=False, error="client_not_initialized")

        model_name = (deployment or self._llm_deployment).strip() or self._llm_deployment
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "model":                model_name,
            "messages":             messages,
            "max_completion_tokens": max_tokens,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_err: str = "no_attempts"
        last_tmo: bool = False
        for attempt in range(self.llm_retry_attempts):
            if self._abort_requested():
                return self.LlmResult(ok=False, error="interaction_aborted")
            try:
                req_timeout = timeout if timeout is not None else self.LLM_TIMEOUT
                kwargs["timeout"] = req_timeout
                resp = self.llm_client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                if text:
                    return self.LlmResult(ok=True, text=text)
                finish_reason = str(getattr(choice, "finish_reason", "") or "")
                if finish_reason == "length":
                    return self.LlmResult(ok=False, error="empty_response_length", empty=True)
                return self.LlmResult(ok=False, error="empty_response", empty=True)
            except Exception as e:
                last_err = str(e)
                last_tmo = "timeout" in last_err.lower() or "timed out" in last_err.lower()
                self._log("WARNING", f"LLM attempt {attempt+1}/{self.llm_retry_attempts}: {e}")
                if attempt < self.llm_retry_attempts - 1:
                    time.sleep(self.llm_retry_delay)

        return self.LlmResult(ok=False, error=last_err, timed_out=last_tmo)

    def _llm_create_with_watchdog(self, kwargs: Dict[str, Any], timeout_sec: float):
        """Run Azure call in a daemon worker and enforce a hard wall-clock timeout."""
        result_q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1)
        worker_started = [False]
        worker_entered_create = [False]

        def _worker() -> None:
            worker_started[0] = True
            try:
                worker_entered_create[0] = True
                t_start = time.monotonic()
                resp = self.llm_client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
                self._log("INFO", f"llm_worker: create() returned in {time.monotonic()-t_start:.2f}s")
                result_q.put(("ok", resp))
            except Exception as e:
                self._log("WARNING", f"llm_worker: exception {type(e).__name__}: {e}")
                result_q.put(("err", e))

        t = threading.Thread(target=_worker, daemon=True, name="llm-call")
        t.start()
        self._log("INFO", f"llm_watchdog: thread.start() issued, timeout={timeout_sec:.1f}s")

        try:
            status, payload = result_q.get(timeout=max(0.1, timeout_sec))
        except queue.Empty as e:
            self._log(
                "WARNING",
                f"llm_watchdog: hard_timeout started={worker_started[0]} "
                f"entered_create={worker_entered_create[0]} "
                f"thread_alive={t.is_alive()} timeout_sec={timeout_sec:.1f}s",
            )
            raise TimeoutError(f"LLM hard timeout after {timeout_sec:.1f}s") from e

        if status == "err":
            raise payload
        return payload

    def _current_hs(self) -> str:
        if not self.hunger_enabled:
            return self.HUNGER_OFF_STATE
        try:
            return self.hunger.snapshot().state
        except Exception as e:
            self._log("WARNING", f"hs_snapshot_failed: {e}")
            return "HS1"

    def _prompt_for_hs(self, base_key: str, hs: str, default_value: str) -> str:
        hs_key = f"{base_key}_{hs.lower()}"
        return self._P.get(hs_key) or self._P.get(base_key, default_value)

    def _system_for_hs(self, hs: str) -> str:
        base = self.LLM_SYS_DEFAULT or self.LLM_SYS_FAST
        overlay_key = {
            self.HUNGER_OFF_STATE: "system_overlay_hs0",
            "HS1": "system_overlay_hs1",
            "HS2": "system_overlay_hs2",
            "HS3": "system_overlay_hs3",
        }.get(hs, "system_overlay_hs1")
        overlay = self._P.get(overlay_key, "")
        return f"{base}\n\n{overlay}".strip() if overlay else base

    def _local_starter_fallback(self, hs: str) -> str:
        if hs == "HS2":
            return random.choice([
                "how's your day going?",
                "what have you been up to?",
                "how are you doing today?",
            ])
        return random.choice([
            "what are you up to today?",
            "how's your day going so far?",
            "anything interesting going on?",
            "how are you doing?",
        ])

    def _local_reply_fallback(self, utterance: str, is_last: bool, face_id: str = "") -> str:
        if self._is_greeting(utterance):
            if face_id and face_id.lower() not in ("unknown", "unmatched", "recognizing") and not face_id.isdigit():
                return self._P.get("reactive_greeting", "Hi {name}").format(name=face_id)
            return self._P.get("ss1_greeting", "Hi there!")
        if is_last:
            return "okay, thanks for talking with me"
        return "mm tell me a little more"

    def _build_reply_request(
        self,
        utterance: str,
        *,
        is_last: bool,
        hs: str,
        turn_index: int,
        interaction_id: Optional[str],
        history: Tuple[Tuple[str, str], ...] = (),
    ) -> LlmTurnRequest:
        if is_last:
            tmpl = self._prompt_for_hs(
                "closing_ack_prompt",
                hs,
                "Person said: '{last_utterance}'\nWarm acknowledgment, 4 to 8 words. No question mark. Output only.",
            )
            max_tokens = self.SS3_CLOSING_MAX_TOKENS
            max_len    = self.SS3_CLOSING_MAX_LEN
        else:
            tmpl = self._prompt_for_hs(
                "followup_prompt",
                hs,
                "User said: '{last_utterance}'\nRespond in 1 sentence, 22 words or fewer. At most one short follow-up question. Output only.",
            )
            max_tokens = self.SS3_FOLLOWUP_MAX_TOKENS
            max_len    = self.SS3_TURN_MAX_LEN

        return LlmTurnRequest(
            prompt=tmpl.format(last_utterance=utterance),
            system=self._system_for_hs(hs),
            max_tokens=max_tokens,
            timeout=self.LLM_TIMEOUT,
            max_len=max_len,
            turn_index=turn_index,
            interaction_id=interaction_id,
            stream=self.SS3_LLM_STREAMING_ENABLED,
            history=history,
        )

    def _llm_text_with_fallback(
        self,
        *,
        prompt: str,
        system: str,
        max_tokens: int,
        timeout: float,
        max_len: int,
        log_tag: str,
        retry_on_empty: bool = True,
    ) -> Optional[str]:
        if self._abort_requested():
            return None
        res = self._llm_call(
            prompt,
            system=system,
            max_tokens=max_tokens,
            timeout=timeout,
            deployment=self._llm_deployment,
        )

        clean = res.text.strip("\"'").strip() if res.ok else ""
        if clean and len(clean) < max_len:
            return clean

        if res.empty and retry_on_empty:
            retry_tokens = min(max_tokens * 2, 512)
            if retry_tokens > max_tokens and not self._abort_requested():
                self._log("WARNING", f"{log_tag}: empty response, retrying with {retry_tokens} tokens")
                res_retry = self._llm_call(
                    prompt,
                    system=system,
                    max_tokens=retry_tokens,
                    timeout=timeout,
                    deployment=self._llm_deployment,
                )
                clean_retry = res_retry.text.strip("\"'").strip() if res_retry.ok else ""
                if clean_retry and len(clean_retry) < max_len:
                    return clean_retry
                if not res_retry.ok:
                    self._log("WARNING", f"{log_tag}: retry failed: {res_retry.error}")
            return None

        if not res.ok:
            self._log("WARNING", f"{log_tag}: failed: {res.error}")
        return None

    @staticmethod
    def _strip_json(text: str) -> str:
        text = text.strip()
        for prefix in ("```json", "```"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _llm_extract_name(self, utterance: str) -> Dict:
        failure = {"answered": False, "name": None, "confidence": 0.0}
        tmpl    = self._P.get("extract_name_prompt",
            'Utterance: "{utterance}"\nExtract the speaker\'s own name.\n'
            '- Patterns: "my name is", "I\'m", "call me", "I am".\n'
            "- Title Case, single token. If unclear: answered=false, name=null, confidence=0.0.")
        prompt  = tmpl.format(utterance=utterance)
        schema  = {
            "type": "object",
            "properties": {
                "answered":   {"type": "boolean"},
                "name":       {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["answered", "name", "confidence"],
            "additionalProperties": False,
        }
        res = self._llm_call(prompt, system=self.LLM_SYS_JSON, max_tokens=64,
                             timeout=self.LLM_TIMEOUT,
                             response_format={"type": "json_schema", "json_schema": {
                                 "name": "name_extraction", "strict": True, "schema": schema}})
        if not res.ok:
            # Plain JSON fallback
            res = self._llm_call(
                prompt + "\nReturn ONLY compact JSON: {answered,name,confidence}.",
                system=self.LLM_SYS_JSON, max_tokens=64, timeout=self.LLM_TIMEOUT)

        if not res.ok:
            self._log("WARNING", f"name_extract failed: {res.error}")
            return failure
        try:
            parsed = json.loads(self._strip_json(res.text))
        except json.JSONDecodeError:
            self._log("WARNING", "name_extract: invalid JSON")
            return failure

        answered = parsed.get("answered")
        name_raw = parsed.get("name")
        if not isinstance(answered, bool):
            return failure
        if answered and not (isinstance(name_raw, str) and name_raw.strip()):
            return failure
        try:
            conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        except (TypeError, ValueError):
            return failure
        return {"answered": answered, "name": name_raw.strip() if name_raw else None, "confidence": conf}

    def _llm_get_starter(self, hs: str) -> str:
        prompt = self._prompt_for_hs(
            "convo_starter_prompt",
            hs,
            "Ask ONE short friendly question about the person's day (6–12 words). No greeting. Output only the sentence.",
        )
        system = self._system_for_hs(hs)

        req = LlmTurnRequest(
            prompt=prompt,
            system=system,
            max_tokens=self.SS3_STARTER_MAX_TOKENS,
            timeout=self.LLM_TIMEOUT,
            max_len=96,
            turn_index=0,
            interaction_id=self._get_iid(),
            stream=self.SS3_LLM_STREAMING_ENABLED,
        )
        request_id = self._llm_turn_worker.submit(req)
        deadline = time.monotonic() + min(self.LLM_TIMEOUT, 2.0)
        text: Optional[str] = None
        while time.monotonic() < deadline:
            if self._abort_requested():
                break
            event = self._llm_turn_worker.poll_event(0.1)
            if event is None:
                continue
            if event.request_id != request_id:
                continue
            if event.kind == "final":
                text = event.text.strip()
                break
            if event.kind in ("error", "cancelled"):
                break
        if text:
            return text
        self._log("WARNING", f"starter_llm_empty hs={hs}; using local fallback")
        return self._local_starter_fallback(hs)

    # ── JSON persistence ──────────────────────────────────────────────────────

    @staticmethod
    def _ensure_json_file(path: str, default: Any) -> None:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(default, fh)

    @staticmethod
    def _load_json(path: str, default: Any) -> Any:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return default

    @staticmethod
    def _save_json_atomic(path: str, data: Any) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write_last_greeted(self, track_id: int, face_id: str, code: str, person_key: Optional[str] = None) -> None:
        try:
            path      = self.LAST_GREETED_FILE
            lock_path = path + ".lock"
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    raw = self._load_json(path, {})
                    entries: Dict[str, Any] = {}
                    if isinstance(raw, dict):
                        entries = {str(k): v for k, v in raw.items() if isinstance(v, dict)}
                    elif isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, dict):
                                k = item.get("assigned_code_or_name") or item.get("face_id")
                                if k:
                                    entries[str(k)] = item

                    key = (person_key or "").strip()
                    if not key:
                        key = face_id if (face_id and face_id.lower() not in ("unknown", "unmatched", "recognizing")) \
                              else f"unknown:{track_id}"

                    entries[key] = {
                        "timestamp":             datetime.now().isoformat(),
                        "track_id":              track_id,
                        "face_id":               face_id,
                        "assigned_code_or_name": code,
                    }
                    self._save_json_atomic(path, entries)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception as e:
            self._log("ERROR", f"write_last_greeted failed: {e}")

    def _mark_greeted_today(self, name: str) -> None:
        key = (name or "").strip()
        if not key:
            return
        try:
            path      = self.GREETED_TODAY_FILE
            lock_path = path + ".lock"
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    raw     = self._load_json(path, {})
                    entries = raw if isinstance(raw, dict) else {}
                    entries[key] = datetime.now().astimezone().isoformat()
                    self._save_json_atomic(path, entries)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception as e:
            self._log("WARNING", f"mark_greeted_today failed: {e}")

    # ── SQLite ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.DB_FILE), exist_ok=True)
            conn = sqlite3.connect(self.DB_FILE)
            c    = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT,
                timestamp TEXT, track_id INTEGER, face_id TEXT,
                initial_state TEXT, final_state TEXT,
                success INTEGER, abort_reason TEXT,
                greeted INTEGER, talked INTEGER, replied_any INTEGER,
                extracted_name TEXT, target_stayed_biggest INTEGER,
                interaction_tag TEXT,
                hunger_state_start TEXT, hunger_state_end TEXT,
                hunger_drive_enabled INTEGER,
                stomach_level_start REAL, stomach_level_end REAL,
                meals_eaten_count INTEGER, last_meal_payload TEXT,
                active_energy_cost REAL NOT NULL DEFAULT 0.0,
                homeostatic_reward REAL NOT NULL DEFAULT 0.0,
                n_turns INTEGER NOT NULL DEFAULT 0,
                trigger_mode TEXT NOT NULL DEFAULT 'proactive',
                day_rome TEXT,
                transcript TEXT, full_result TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS reactive_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                track_id INTEGER, name TEXT, payload TEXT,
                hunger_state_before TEXT,
                stomach_level_before REAL,
                hunger_state_after   TEXT,
                stomach_level_after  REAL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS hunger_level_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                stimulus_type TEXT,
                stimulus_label TEXT,
                reason TEXT,
                hunger_drive_enabled INTEGER,
                hunger_state_before TEXT,
                hunger_state_after TEXT,
                stomach_level_before REAL,
                stomach_level_after REAL,
                level_delta REAL,
                active_energy_cost REAL NOT NULL DEFAULT 0.0,
                meal_delta REAL NOT NULL DEFAULT 0.0,
                meal_payload TEXT,
                trigger_mode TEXT,
                social_state TEXT,
                interaction_tag TEXT,
                exec_interaction_id TEXT
            )""")
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_i_time    ON interactions(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_i_track   ON interactions(track_id)",
                "CREATE INDEX IF NOT EXISTS idx_i_iid     ON interactions(interaction_id)",
                "CREATE INDEX IF NOT EXISTS idx_i_state   ON interactions(initial_state, final_state, success)",
                "CREATE INDEX IF NOT EXISTS idx_i_hs      ON interactions(hunger_state_start)",
                "CREATE INDEX IF NOT EXISTS idx_i_day_rome ON interactions(day_rome)",
                "CREATE INDEX IF NOT EXISTS idx_i_trigger  ON interactions(trigger_mode, initial_state)",
                "CREATE INDEX IF NOT EXISTS idx_r_time    ON reactive_interactions(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_r_type    ON reactive_interactions(type)",
                "CREATE INDEX IF NOT EXISTS idx_hle_time  ON hunger_level_events(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_hle_event ON hunger_level_events(event_type, stimulus_type)",
                "CREATE INDEX IF NOT EXISTS idx_hle_iid   ON hunger_level_events(exec_interaction_id)",
            ]:
                c.execute(idx_sql)
            self._create_analytics_views(conn)
            conn.commit()
            conn.close()
            self._log("INFO", f"DB ready: {self.DB_FILE}")
        except Exception as e:
            self._log("ERROR", f"DB init failed: {e}")

    def _create_analytics_views(self, conn: sqlite3.Connection) -> None:
        c = conn.cursor()
        views = {
            "v_proactive_interactions": """
                SELECT interaction_id, timestamp,
                       COALESCE(day_rome, substr(timestamp,1,10)) AS day_rome,
                       track_id, face_id,
                       COALESCE(NULLIF(extracted_name,''),face_id) AS user_key,
                       initial_state, final_state,
                       CAST(success AS INTEGER) AS success,
                       CAST(greeted AS INTEGER) AS greeted,
                       CAST(talked AS INTEGER) AS talked,
                       CAST(COALESCE(replied_any,0) AS INTEGER) AS replied_any,
                       abort_reason, interaction_tag,
                       hunger_state_start, hunger_state_end,
                       CAST(COALESCE(hunger_drive_enabled, CASE WHEN hunger_state_start='HS0' THEN 0 ELSE 1 END) AS INTEGER) AS hunger_drive_enabled,
                       stomach_level_start, stomach_level_end,
                       meals_eaten_count, last_meal_payload,
                       COALESCE(active_energy_cost, 0.0) AS active_energy_cost,
                       COALESCE(homeostatic_reward, stomach_level_end - stomach_level_start) AS homeostatic_reward,
                       COALESCE(n_turns, 0) AS n_turns,
                       COALESCE(trigger_mode, 'proactive') AS trigger_mode
                FROM interactions WHERE initial_state IN ('ss1','ss2','ss3')""",
            "v_metric_ss3_daily": """
                SELECT day_rome, hunger_state_start,
                       COUNT(*) AS launched,
                       SUM(CASE WHEN success=1 AND final_state='ss4' THEN 1 ELSE 0 END) AS reached_ss4,
                       1.0*SUM(CASE WHEN success=1 AND final_state='ss4' THEN 1 ELSE 0 END)/MAX(COUNT(*),1) AS rate
                FROM v_proactive_interactions WHERE initial_state='ss3'
                GROUP BY day_rome, hunger_state_start""",
            "v_metric_response_rate_daily": """
                SELECT day_rome, hunger_state_start,
                       COUNT(*) AS launched,
                       SUM(CASE WHEN replied_any=1 THEN 1 ELSE 0 END) AS replied,
                       1.0*SUM(CASE WHEN replied_any=1 THEN 1 ELSE 0 END)/MAX(COUNT(*),1) AS rate
                FROM v_proactive_interactions GROUP BY day_rome, hunger_state_start""",
            "v_metric_repeat_users_daily": """
                SELECT
                    day_rome,
                    hunger_state_start,
                    COUNT(*)                                                          AS total_visits,
                    COUNT(DISTINCT user_key)                                          AS unique_users,
                    SUM(CASE WHEN visit_count > 1 THEN 1 ELSE 0 END)                 AS repeat_visit_count,
                    1.0 * SUM(CASE WHEN visit_count > 1 THEN 1 ELSE 0 END)
                          / MAX(COUNT(DISTINCT user_key), 1)                          AS repeat_user_rate
                FROM (
                    SELECT
                        day_rome,
                        hunger_state_start,
                        COALESCE(NULLIF(extracted_name, ''), face_id)                 AS user_key,
                        COUNT(*) OVER (
                            PARTITION BY day_rome,
                                         COALESCE(NULLIF(extracted_name, ''), face_id)
                        )                                                             AS visit_count
                    FROM interactions
                    WHERE trigger_mode = 'proactive'
                      AND initial_state IN ('ss1', 'ss2', 'ss3')
                )
                GROUP BY day_rome, hunger_state_start""",
            "v_metric_depth_progression": """
                SELECT
                    COALESCE(day_rome, substr(timestamp,1,10))                               AS day_rome,
                    initial_state,
                    hunger_state_start,
                    COUNT(*)                                                                  AS launched,
                    SUM(CASE WHEN final_state = 'ss4' THEN 1 ELSE 0 END)                     AS reached_ss4,
                    1.0 * SUM(CASE WHEN final_state = 'ss4' THEN 1 ELSE 0 END)
                          / MAX(COUNT(*), 1)                                                  AS completion_rate,
                    AVG(CAST(n_turns AS REAL))                                               AS avg_turns,
                    MAX(n_turns)                                                             AS max_turns,
                    SUM(CASE WHEN n_turns >= 3 THEN 1 ELSE 0 END)                           AS deep_interactions,
                    SUM(CASE WHEN replied_any = 1 AND final_state != 'ss4' THEN 1 ELSE 0 END) AS replied_but_no_ss4
                FROM interactions
                WHERE initial_state IN ('ss1', 'ss2', 'ss3')
                GROUP BY day_rome, initial_state, hunger_state_start""",
            "v_hunger_level_timeline": """
                SELECT timestamp,
                       event_type,
                       stimulus_type,
                       stimulus_label,
                       reason,
                       CAST(hunger_drive_enabled AS INTEGER) AS hunger_drive_enabled,
                       hunger_state_before,
                       hunger_state_after,
                       stomach_level_before,
                       stomach_level_after,
                       level_delta,
                       active_energy_cost,
                       meal_delta,
                       meal_payload,
                       trigger_mode,
                       social_state,
                       interaction_tag,
                       exec_interaction_id
                FROM hunger_level_events""",
        }
        for name, body in views.items():
            c.execute(f"DROP VIEW IF EXISTS {name}")
            c.execute(f"CREATE VIEW {name} AS {body}")

    def _db_enqueue(self, item: Any) -> None:
        try:
            self._db_queue.put_nowait(item)
        except queue.Full:
            try:
                self._db_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._db_queue.put_nowait(item)
            except queue.Full:
                self._log("WARNING", "DB queue full, dropping item")

    def _db_worker(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        while True:
            try:
                item = self._db_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue
            if item is None:
                break
            table, data = item
            try:
                if conn is None:
                    conn = sqlite3.connect(self.DB_FILE, timeout=10.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA temp_store=MEMORY")
                if table == "interaction":
                    self._db_save_interaction(conn, data)
                elif table == "reactive":
                    self._db_save_reactive(conn, data)
                elif table == "hunger_level_event":
                    self._db_save_hunger_level_event(conn, data)
            except Exception as e:
                self._log("ERROR", f"DB write failed: {e}")
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    def _db_save_interaction(self, conn: sqlite3.Connection, data: Dict) -> None:
        try:
            r    = data["result"]
            logs = r.get("logs", [])
            transcript = json.dumps(
                [l["message"] for l in logs
                 if any(kw in l.get("message", "") for kw in ("User:", "Robot:", "Asking", "Response"))],
                ensure_ascii=False,
            )
            r_compact = {k: v for k, v in r.items() if k != "logs"}
            day_rome  = datetime.now(self.TIMEZONE).date().isoformat()
            conn.cursor().execute(
                """INSERT INTO interactions
                (interaction_id,timestamp,track_id,face_id,initial_state,final_state,
                 success,abort_reason,greeted,talked,replied_any,extracted_name,
                 target_stayed_biggest,interaction_tag,hunger_state_start,hunger_state_end,
                 hunger_drive_enabled,stomach_level_start,stomach_level_end,meals_eaten_count,last_meal_payload,
                 active_energy_cost,homeostatic_reward,n_turns,trigger_mode,day_rome,
                 transcript,full_result)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data.get("interaction_id"),
                    datetime.now(self.TIMEZONE).isoformat(),
                    data["track_id"], data["face_id"], data["initial_state"],
                    r.get("final_state",""),
                    int(r.get("success", False)),      r.get("abort_reason"),
                    int(r.get("greeted", False)),       int(r.get("talked", False)),
                    int(r.get("replied_any", False)),   r.get("extracted_name"),
                    int(r.get("target_stayed_biggest", True)),
                    r.get("interaction_tag"),
                    r.get("hunger_state_start"),        r.get("hunger_state_end"),
                    int(bool(r.get("hunger_drive_enabled", r.get("hunger_state_start") != self.HUNGER_OFF_STATE))),
                    r.get("stomach_level_start"),       r.get("stomach_level_end"),
                    r.get("meals_eaten_count"),         r.get("last_meal_payload"),
                    r.get("active_energy_cost", 0.0),
                    r.get("homeostatic_reward", 0.0),
                    r.get("n_turns", 0),
                    r.get("trigger_mode", "proactive"),
                    day_rome,
                    transcript,
                    json.dumps(r_compact, ensure_ascii=False),
                ),
            )
            conn.commit()
        except Exception as e:
            self._log("ERROR", f"DB save_interaction failed: {e}")

    def _db_save_reactive(self, conn: sqlite3.Connection, data: Dict) -> None:
        try:
            conn.cursor().execute(
                """INSERT INTO reactive_interactions
                (interaction_id,timestamp,type,track_id,name,payload,
                 hunger_state_before,stomach_level_before,hunger_state_after,stomach_level_after)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    data.get("interaction_id"),
                    datetime.now().astimezone().isoformat(),
                    data.get("type"),
                    data.get("track_id"),
                    data.get("name"),
                    data.get("payload"),
                    data.get("hunger_state_before"),
                    data.get("stomach_level_before"),
                    data.get("hunger_state_after"),
                    data.get("stomach_level_after"),
                ),
            )
            conn.commit()
        except Exception as e:
            self._log("ERROR", f"DB save_reactive failed: {e}")

    def _db_save_hunger_level_event(self, conn: sqlite3.Connection, data: Dict) -> None:
        try:
            conn.cursor().execute(
                """INSERT INTO hunger_level_events
                (timestamp,event_type,stimulus_type,stimulus_label,reason,
                 hunger_drive_enabled,hunger_state_before,hunger_state_after,
                 stomach_level_before,stomach_level_after,level_delta,
                 active_energy_cost,meal_delta,meal_payload,trigger_mode,
                 social_state,interaction_tag,exec_interaction_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(self.TIMEZONE).isoformat(),
                    data.get("event_type"),
                    data.get("stimulus_type"),
                    data.get("stimulus_label"),
                    data.get("reason"),
                    int(data.get("hunger_drive_enabled", 0)),
                    data.get("hunger_state_before"),
                    data.get("hunger_state_after"),
                    data.get("stomach_level_before"),
                    data.get("stomach_level_after"),
                    data.get("level_delta"),
                    data.get("active_energy_cost", 0.0),
                    data.get("meal_delta", 0.0),
                    data.get("meal_payload"),
                    data.get("trigger_mode"),
                    data.get("social_state"),
                    data.get("interaction_tag"),
                    data.get("exec_interaction_id"),
                ),
            )
            conn.commit()
        except Exception as e:
            self._log("ERROR", f"DB save_hunger_level_event failed: {e}")

    # ── per-interaction log tracking ──────────────────────────────────────────

    def _init_ilog(self, iid: str) -> None:
        with self._interaction_logs_lock:
            self._interaction_logs[iid] = []

    def _pop_ilog(self, iid: str) -> List[Dict]:
        with self._interaction_logs_lock:
            return self._interaction_logs.pop(iid, [])

    def _set_iid(self, iid: Optional[str]) -> None:
        self._thread_ctx.interaction_id = iid

    def _get_iid(self) -> Optional[str]:
        return getattr(self._thread_ctx, "interaction_id", None)

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, level: str, message: str) -> None:
        ts    = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{level}] {message}")
        entry = {"timestamp": ts, "level": level, "message": message}
        with self._log_lock:
            self.log_buffer.append(entry)
            if len(self.log_buffer) > 2000:
                self.log_buffer = self.log_buffer[-2000:]
        iid = self._get_iid()
        if iid:
            with self._interaction_logs_lock:
                buf = self._interaction_logs.get(iid)
                if buf is not None:
                    buf.append(entry)

    def _log_throttled(self, level: str, key: str, message: str, interval: float = 1.0) -> None:
        now = time.monotonic()
        with self._log_throttle_lock:
            if now - self._log_throttle_last.get(key, 0.0) < interval:
                return
            self._log_throttle_last[key] = now
        self._log(level, message)

    def _recent_logs(self, limit: int = 200) -> List[Dict]:
        with self._log_lock:
            return list(self.log_buffer[-limit:]) if limit > 0 else list(self.log_buffer)

    # ── RPC reply helpers ─────────────────────────────────────────────────────

    def _rpc_ok(self, reply: yarp.Bottle, data: Dict) -> bool:
        reply.addString("ok")
        reply.addString(json.dumps(data, ensure_ascii=False))
        return True

    def _rpc_error(self, reply: yarp.Bottle, error: str) -> bool:
        reply.addString("ok")
        reply.addString(json.dumps({"success": False, "error": error, "logs": self._recent_logs()}, ensure_ascii=False))
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _run_with_signal_handling(module: ExecutiveControlModule, rf: yarp.ResourceFinder) -> bool:
    stop_requested = threading.Event()
    run_done       = threading.Event()
    run_state: Dict[str, Any] = {"result": False, "error": None}

    def _runner():
        try:
            run_state["result"] = bool(module.runModule(rf))
        except BaseException as e:
            run_state["error"] = e
        finally:
            run_done.set()

    def _on_signal(signum, _frame):
        if not stop_requested.is_set():
            stop_requested.set()
            try:
                os.write(2, f"\n[INFO] Signal {signum} received – shutting down.\n".encode("utf-8", "replace"))
            except Exception:
                pass

    prev_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _on_signal)
        except Exception:
            pass

    run_thread = threading.Thread(target=_runner, name=f"{module.module_name}-run", daemon=True)
    run_thread.start()

    interrupted    = False
    shutdown_after: Optional[float] = None
    try:
        while not run_done.wait(0.2):
            if stop_requested.is_set() and not interrupted:
                interrupted    = True
                shutdown_after = time.monotonic() + 2.0
                module.interruptModule()
            elif interrupted and shutdown_after and time.monotonic() >= shutdown_after:
                break
    finally:
        for sig, handler in prev_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        module.interruptModule()
        module.close()

    run_thread.join(timeout=0.5)
    if run_state["error"] is not None:
        raise run_state["error"]
    return bool(run_state["result"])


if __name__ == "__main__":
    import sys

    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("[ERROR] YARP network unavailable – start yarpserver first.")
        sys.exit(1)

    module = ExecutiveControlModule()
    rf     = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.configure(sys.argv)

    print("=" * 60)
    print(" ExecutiveControlModule")
    print(" ss1=unknown  ss2=known/not-greeted  ss3=greeted  ss4=no-op")
    print()
    print(" yarp connect /alwayson/vision/landmarks:o /alwayson/executiveControl/landmarks:i")
    print(" yarp connect /speech2text/text:o          /alwayson/executiveControl/stt:i")
    print(" yarp connect /alwayson/executiveControl/speech:o /acapelaSpeak/speech:i")
    print()
    print(" RPC: echo 'run <track_id> <face_id> <ss1|ss2|ss3|ss4>' | yarp rpc /executiveControl")
    print("=" * 60)

    try:
        _run_with_signal_handling(module, rf)
    finally:
        yarp.Network.fini()
