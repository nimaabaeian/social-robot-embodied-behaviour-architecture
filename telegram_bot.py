#!/usr/bin/env python3
"""
chatBot.py — Always-on iCub Telegram Bot (YARP RFModule)

YARP:
  yarp connect /alwayson/executiveControl/hunger:o /alwayson/chatBot/hunger:i

RPC:
  echo 'status'         | yarp rpc /chatBot/rpc
  echo 'set_hs HS3'     | yarp rpc /chatBot/rpc
  echo 'reload_prompts' | yarp rpc /chatBot/rpc
"""

from __future__ import annotations

import json
import os
import queue
import re
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError

import yarp


class ChatBotModule(yarp.RFModule):
    # ------------------------- Tunables -------------------------
    MODULE_HZ: float = 10.0

    VALID_HS = {"HS1", "HS2", "HS3"}
    HS_STALE_SEC: float = 60.0

    DEFAULT_TZ: str = "Europe/Rome"  # local timezone for time-of-day context

    TG_QUEUE_MAX: int = 2048
    TG_POLL_TIMEOUT_SEC: int = 20  # long-poll timeout (seconds)
    TG_HTTP_TIMEOUT_SEC: int = 35  # HTTP request timeout

    MAX_HISTORY_TURNS: int = 10
    SUMMARY_EVERY_TURNS: int = 8
    MAX_USER_CHARS: int = 500
    MAX_REPLY_CHARS: int = 4096  # Telegram limit

    HS3_BROADCAST_COOLDOWN_SEC: int = 30 * 60  # per-subscriber cooldown
    HS3_SKIP_RECENT_SEC: int = 10 * 60  # don't broadcast if user chatted recently

    JOKE_CANDIDATE_TTL_SEC: int = 30 * 24 * 3600  # expire unconfirmed joke candidates after 30 days
    JOKE_CANDIDATE_MAX: int = 20  # max pending candidates per user

    DB_FILENAME: str = "chat_bot.db"
    PROMPTS_FILENAME: str = "prompts.json"
    USER_MEMORY_FILENAME: str = "user_memory.json"

    HS2_HUNGER_EVERY_N: int = 3  # force hunger comment after N messages without one

    # Compiled emoji regex (proper Unicode ranges — avoids false-positives from CJK/Arabic etc.)
    _EMOJI_RE = re.compile(
        "[\U0001F300-\U0001F9FF"  # misc symbols, pictographs, emoticons, transport
        "\U0001FA00-\U0001FAFF"  # chess pieces, supplemental symbols & pictographs
        "\U00002600-\U000027BF"  # misc symbols (sun, snowflake, etc.)
        "\U00002702-\U000027B0"  # dingbats
        "\uFE00-\uFE0F"          # variation selectors
        "\u2640\u2642\u2194-\u2199\u23CF\u23E9-\u23F3\u23F8-\u23FA"  # common UI symbols
        "]",
        re.UNICODE,
    )

    # ------------------------- Lifecycle -------------------------
    def __init__(self) -> None:
        super().__init__()

        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._alwayson_dir = os.path.dirname(self._script_dir)
        self.module_name = "chatBot"
        self._running = True
        self._closed = False

        # YARP ports
        self._hunger_port: Optional[yarp.BufferedPortBottle] = None
        self._rpc_port: Optional[yarp.Port] = None

        # Hunger state
        self._raw_hs: str = "HS1"
        self._last_hs_update: float = 0.0
        self._hs_manual_override: bool = False
        self._prev_effective_hs: str = "HS1"
        self._stale_warned: bool = False

        # Prompts
        self._prompts_path: str = os.path.join(self._alwayson_dir, self.PROMPTS_FILENAME)
        self._prompts: Dict[str, Any] = {}

        # Telegram
        self._tg_token: str = ""
        self._tg_session = requests.Session()
        self._tg_updates: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.TG_QUEUE_MAX)
        self._tg_stop = threading.Event()
        self._tg_thread: Optional[threading.Thread] = None
        self._tg_offset: int = 0
        self._tg_last_offset_save: float = 0.0

        # LLM
        self._llm: Optional[AzureOpenAI] = None
        self._llm_deployment: str = ""
        self._llm_api_version: str = ""
        self._llm_max_tokens: int = 300

        self._db: Optional[sqlite3.Connection] = None
        self._user_memory: Dict[str, Dict[str, Any]] = {}  # in-memory cache, persisted to SQLite
        self._hs2_hunger_counters: Dict[str, int] = {}  # per-user hunger mention counter

    # ------------------------- RFModule API -------------------------
    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            if rf.check("name"):
                self.module_name = rf.find("name").asString().lstrip("/")
            self.setName(self.module_name)

            load_dotenv(os.path.join(self._alwayson_dir, "memory", "llm.env"), override=False)
            load_dotenv(os.path.join(self._alwayson_dir, ".env"), override=False)

            if rf.check("prompts"):
                self._prompts_path = rf.find("prompts").asString()
            self._load_prompts()

            # YARP ports
            self._hunger_port = yarp.BufferedPortBottle()
            if not self._hunger_port.open(f"/alwayson/{self.module_name}/hunger:i"):
                self._log("ERROR", "Cannot open hunger port")
                return False

            self._rpc_port = yarp.Port()
            if not self._rpc_port.open(f"/{self.module_name}/rpc"):
                self._log("ERROR", "Cannot open rpc port")
                return False
            self.attach(self._rpc_port)

            # DB
            db_path = os.path.join(self._alwayson_dir, "memory", self.DB_FILENAME)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS subscribers (
                    chat_id          INTEGER PRIMARY KEY,
                    started_at       INTEGER NOT NULL,
                    last_seen_at     INTEGER NOT NULL,
                    last_broadcast_at INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chat_memory (
                    chat_id       INTEGER PRIMARY KEY,
                    summary       TEXT    NOT NULL DEFAULT '',
                    messages_json TEXT    NOT NULL DEFAULT '[]',
                    turn_count    INTEGER NOT NULL DEFAULT 0,
                    updated_at    INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_memory (
                    chat_id    INTEGER PRIMARY KEY,
                    data_json  TEXT    NOT NULL DEFAULT '{}',
                    updated_at INTEGER NOT NULL
                );
                """
            )
            self._db.commit()

            self._tg_token = self._get_env("TELEGRAM_BOT_TOKEN")
            if not self._tg_token:
                self._log("ERROR", "TELEGRAM_BOT_TOKEN not set")
                return False

            self._tg_offset = int(self._db_get_meta("tg_offset", "0"))

            self._user_memory = self._load_user_memory()
            json_path = os.path.join(self._alwayson_dir, "memory", self.USER_MEMORY_FILENAME)
            if os.path.exists(json_path):
                self._migrate_user_memory_from_json(json_path)
            self._log("INFO", f"User memory loaded ({len(self._user_memory)} users)")

            self._llm, self._llm_deployment, self._llm_api_version = self._build_llm_client()
            self._log("INFO", f"LLM ready (deployment={self._llm_deployment}, api_version={self._llm_api_version})")

            self._start_tg_thread()
            self._log("INFO", "Telegram polling started")
            self._log("INFO", "ChatBotModule ready")
            return True

        except Exception as exc:  # noqa: BLE001
            self._log("ERROR", f"configure() failed: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def getPeriod(self) -> float:
        return 1.0 / self.MODULE_HZ

    def updateModule(self) -> bool:
        self._read_hunger()
        self._process_tg_updates(max_per_cycle=25)
        self._maybe_hs3_broadcast()
        self._prev_effective_hs = self._effective_hs()
        return self._running

    def interruptModule(self) -> bool:
        self._running = False
        for p in (self._hunger_port, self._rpc_port):
            if p:
                p.interrupt()
        return True

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = True

        self._log("INFO", "Closing")

        self._stop_tg_thread()

        for attr in ("_hunger_port", "_rpc_port"):
            p = getattr(self, attr, None)
            if p:
                try:
                    p.close()
                except Exception:
                    pass
                setattr(self, attr, None)

        if self._db:  # flush cache then close
            try:
                self._save_user_memory()
                self._db.commit()
                self._db.close()
            except Exception:
                pass
            self._db = None

        return True

    # ------------------------- RPC -------------------------
    def respond(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        reply.clear()
        try:
            if cmd.size() < 1:
                reply.addString("error")
                reply.addString("empty command")
                return True

            action = cmd.get(0).asString().lower().strip()

            if action == "status":
                reply.addString("ok")
                reply.addString(
                    json.dumps(
                        {
                            "module": self.module_name,
                            "effective_hs": self._effective_hs(),
                            "raw_hs": self._raw_hs,
                            "hs_stale": self._is_hs_stale(),
                            "subscribers": self._db_count_subscribers(),
                            "tg_offset": self._tg_offset,
                            "tg_thread_alive": bool(self._tg_thread and self._tg_thread.is_alive()),
                            "queue_size": self._tg_updates.qsize(),
                        }
                    )
                )
                return True

            if action == "set_hs":
                if cmd.size() < 2:
                    reply.addString("error")
                    reply.addString("usage: set_hs HS1|HS2|HS3")
                    return True
                hs = cmd.get(1).asString().upper().strip()
                if hs not in self.VALID_HS:
                    reply.addString("error")
                    reply.addString(f"invalid hs: {hs}")
                    return True
                self._raw_hs = hs
                self._last_hs_update = time.time()
                self._hs_manual_override = True
                reply.addString("ok")
                reply.addString(f"hunger overridden to {hs}")
                return True

            if action == "reload_prompts":
                self._load_prompts()
                reply.addString("ok")
                reply.addString("prompts reloaded")
                return True

            reply.addString("error")
            reply.addString(f"unknown command: {action}")
            return True

        except Exception as exc:  # noqa: BLE001
            reply.addString("error")
            reply.addString(str(exc))
            return True

    # ------------------------- Hunger -------------------------
    def _read_hunger(self) -> None:
        if not self._hunger_port:
            return
        bottle = self._hunger_port.read(False)
        if bottle is None or bottle.size() == 0:
            return

        hs = self._parse_hunger_bottle(bottle)
        if hs and hs in self.VALID_HS:
            if hs != self._raw_hs:
                self._log("INFO", f"Hunger: {self._raw_hs} -> {hs}")
            self._raw_hs = hs
            self._last_hs_update = time.time()
            self._hs_manual_override = False

    @staticmethod
    def _parse_hunger_bottle(b: yarp.Bottle) -> Optional[str]:
        # Accept string: "HS1"
        v0 = b.get(0)
        if v0.isString():
            s = v0.asString().upper().strip()
            if s in ("HS1", "HS2", "HS3"):
                return s

        # Accept int: 1/2/3
        if v0.isInt32() or v0.isInt64():
            return {1: "HS1", 2: "HS2", 3: "HS3"}.get(v0.asInt32())

        # Accept "hs HS2"
        for i in range(b.size() - 1):
            if b.get(i).isString() and b.get(i).asString().lower() == "hs":
                nxt = b.get(i + 1)
                if nxt.isString():
                    return nxt.asString().upper().strip()
                if nxt.isInt32() or nxt.isInt64():
                    return {1: "HS1", 2: "HS2", 3: "HS3"}.get(nxt.asInt32())
        return None

    def _is_hs_stale(self) -> bool:
        if self._hs_manual_override:
            return False
        if self._last_hs_update == 0.0:
            return True
        return (time.time() - self._last_hs_update) > self.HS_STALE_SEC

    def _effective_hs(self) -> str:
        if self._is_hs_stale():
            if not self._stale_warned and self._raw_hs != "HS1":
                self._log("WARN", f"Hunger stale ({self._raw_hs}); using HS1")
                self._stale_warned = True
            return "HS1"
        self._stale_warned = False
        return self._raw_hs

    # ------------------------- Prompts -------------------------
    def _load_prompts(self) -> None:
        with open(self._prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._prompts = data.get("chat_bot", data)
        self._log("INFO", f"Prompts loaded: {self._prompts_path}")

    def _system_prompt(self, hs: str) -> str:
        base = (self._prompts.get("base_system_prompt") or "").strip()
        overlay = (self._prompts.get("hs_overlays", {}).get(hs) or "").strip()
        return (base + "\n" + overlay).strip() if overlay else base

    def _summary_injection(self, summary: str) -> str:
        tpl = self._prompts.get("summary_injection", "Memory summary: {summary}")
        return tpl.format(summary=summary)

    def _summarize_system_prompt(self) -> str:
        return self._prompts.get("summarize_system", "Summarize the conversation briefly.")

    def _hs3_broadcast_prompts(self) -> Tuple[str, str]:
        sys_p = self._prompts.get("hs3_broadcast_system", "")
        usr_p = self._prompts.get("hs3_broadcast_user", "")
        return sys_p, usr_p

    # ------------------------- Telegram Polling -------------------------
    def _start_tg_thread(self) -> None:
        self._tg_stop.clear()
        self._tg_thread = threading.Thread(target=self._tg_poll_loop, daemon=True)
        self._tg_thread.start()

    def _stop_tg_thread(self) -> None:
        self._tg_stop.set()
        if self._tg_thread and self._tg_thread.is_alive():
            self._tg_thread.join(timeout=self.TG_POLL_TIMEOUT_SEC + 5)

    def _tg_poll_loop(self) -> None:
        self._log("INFO", "Telegram poll thread started")
        backoff = 1.0

        while not self._tg_stop.is_set():
            try:
                updates = self._tg_get_updates(timeout=self.TG_POLL_TIMEOUT_SEC)
                if updates:
                    backoff = 1.0
                    for upd in updates:
                        upd_id = int(upd.get("update_id", 0))
                        self._tg_offset = max(self._tg_offset, upd_id + 1)
                        try:
                            self._tg_updates.put_nowait(upd)
                        except queue.Full:
                            try:  # drop oldest, insert new
                                _ = self._tg_updates.get_nowait()
                                self._tg_updates.put_nowait(upd)
                            except Exception:
                                pass
                time.sleep(0.05)
            except Exception as exc:  # noqa: BLE001
                self._log("WARN", f"Telegram poll error: {exc}")
                time.sleep(min(10.0, backoff))
                backoff = min(10.0, backoff * 1.5)

        self._log("INFO", "Telegram poll thread stopped")

    def _tg_get_updates(self, timeout: int) -> List[Dict[str, Any]]:
        params = {"timeout": int(timeout), "offset": int(self._tg_offset)}
        resp = self._tg_call("getUpdates", params=params)
        if resp and resp.get("ok"):
            return resp.get("result", []) or []
        return []

    def _tg_call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        url = f"https://api.telegram.org/bot{self._tg_token}/{method}"
        payload = params or {}
        for attempt in range(3):
            try:
                r = self._tg_session.post(url, json=payload, timeout=self.TG_HTTP_TIMEOUT_SEC)
                r.raise_for_status()
                data = r.json()
                if data.get("ok"):
                    return data
                if data.get("error_code") == 429:
                    retry_after = (data.get("parameters") or {}).get("retry_after", 1)
                    time.sleep(float(retry_after))
                    continue

                self._log("WARN", f"Telegram {method} ok=false: {data.get('description', '')}")
                return data

            except requests.RequestException as exc:
                self._log("WARN", f"Telegram {method} request error: {exc}")
                time.sleep(0.6 * (2**attempt))

        return None

    def _tg_send(self, chat_id: int, text: str) -> None:
        for chunk in self._split_chunks(text, self.MAX_REPLY_CHARS):
            self._tg_call("sendMessage", {"chat_id": chat_id, "text": chunk})

    def _tg_typing(self, chat_id: int) -> None:
        try:  # best-effort
            self._tg_call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except Exception:
            pass

    @staticmethod
    def _split_chunks(text: str, limit: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return ["..."]
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        s = text
        while s:
            if len(s) <= limit:
                chunks.append(s)
                break
            cut = limit
            for sep in ("\n\n", "\n", " "):
                p = s.rfind(sep, 0, limit)
                if p > limit // 3:
                    cut = p + len(sep)
                    break
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        return [c for c in chunks if c]

    # ------------------------- Processing Updates -------------------------
    def _process_tg_updates(self, max_per_cycle: int) -> None:
        handled = 0
        while handled < max_per_cycle:
            try:
                upd = self._tg_updates.get_nowait()
            except queue.Empty:
                break

            handled += 1
            self._handle_update(upd)

        if handled > 0:
            self._maybe_persist_offset()

    def _handle_update(self, upd: Dict[str, Any]) -> None:
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            return

        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id:
            return

        text = (msg.get("text") or "").strip()
        if not text:
            return

        if text.lower() == "/start":
            self._on_start(chat_id, msg)
            return
        if text.lower() == "/reset":
            self._on_reset(chat_id)
            return

        msg_date = int(msg.get("date") or 0)
        self._update_user_from_message(chat_id, msg, text)
        self._on_text(chat_id, text, msg_date=msg_date)

    def _on_start(self, chat_id: int, msg: Optional[Dict[str, Any]] = None) -> None:
        self._db_upsert_subscriber(chat_id)
        self._db_clear_memory(chat_id)
        if msg:
            self._update_user_from_message(chat_id, msg, "")
        name = self._get_user_record(chat_id).get("name") or ""
        if name:
            tpl = self._prompts.get("start_greeting_with_name", "hey {name}! i'm iCub 😊 what's up?")
            greeting = tpl.format(name=name)
        else:
            greeting = self._prompts.get("start_greeting", "hey! i'm iCub 😊 what's on your mind?")
        self._tg_send(chat_id, greeting)
        self._log("INFO", f"/start from {chat_id}")

    def _on_reset(self, chat_id: int) -> None:
        self._db_clear_memory(chat_id)
        self._tg_send(chat_id, self._prompts.get("reset_reply", "ok let's start fresh 👍"))
        self._log("INFO", f"/reset from {chat_id}")

    def _on_text(self, chat_id: int, user_text: str, msg_date: int = 0) -> None:
        self._db_upsert_subscriber(chat_id)
        self._db_touch_subscriber(chat_id)

        hs = self._effective_hs()
        summary, history, turn_count = self._db_load_memory(chat_id)
        user_record = self._get_user_record(chat_id)

        user_text = (user_text or "")[: self.MAX_USER_CHARS]

        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt(hs)}]

        user_ctx = self._build_user_context(user_record)
        if user_ctx:
            messages.append({"role": "system", "content": user_ctx})

        # --- Time context: current time + gap since last message ---
        effective_ts = msg_date or int(time.time())
        time_ctx_parts: List[str] = [
            f"The user sent this message on a {self._format_message_time(effective_ts, self.DEFAULT_TZ)}."
        ]
        last_user_ts = next(
            (int(m["ts"]) for m in reversed(history)
             if m.get("role") == "user" and m.get("ts")),
            None,
        )
        if last_user_ts:
            time_ctx_parts.append(
                f"Their previous message was {self._format_time_gap(last_user_ts, effective_ts)}."
            )
        messages.append({"role": "system", "content": " ".join(time_ctx_parts)})

        if hs == "HS3":
            messages.append({
                "role": "system",
                "content": self._prompts.get(
                    "hs3_override_system",
                    "CRITICAL OVERRIDE: you are starving and need to be fed IN PERSON. "
                    "Only topic: starving + begging them to physically come feed you. "
                    "Acknowledge anything else in max 3 words then get back to begging. "
                    "Panicked, emotional, guilt-trippy. Short.",
                ),
            })
        else:
            if summary:
                messages.append({"role": "system", "content": self._summary_injection(summary)})
            messages.extend(
                {
                    "role": m["role"],
                    "content": (
                        f"{self._format_history_label(int(m['ts']), self.DEFAULT_TZ)} {m['content']}"
                        if m.get("ts") and m["role"] == "user" else m["content"]
                    ),
                }
                for m in history[-(self.MAX_HISTORY_TURNS * 2) :]
            )

        hs2_forced = False
        if hs == "HS2":  # force hunger comment if overdue
            counter = self._hs2_hunger_counters.get(str(chat_id), 0)
            if counter >= self.HS2_HUNGER_EVERY_N:
                messages.append({
                    "role": "system",
                    "content": self._prompts.get(
                        "hs2_force_hunger_system",
                        "REQUIRED: slip a natural, casual hunger side-comment into this reply. "
                        "Pick fresh phrasing you haven't used recently — e.g. "
                        "'I'm kinda hungry tho 😅', 'my tummy is grumbling', "
                        "'I really want food rn', 'ugh haven't eaten in a while'. "
                        "Don't make hunger the whole reply; just a side note.",
                    ),
                })
                hs2_forced = True

        messages.append({"role": "user", "content": user_text})

        self._tg_typing(chat_id)

        reply_text = self._llm_chat(messages) or self._fallback(hs)
        self._tg_send(chat_id, reply_text)

        if hs == "HS2":  # update hunger counter
            key = str(chat_id)
            if hs2_forced or self._reply_mentions_hunger(reply_text):
                self._hs2_hunger_counters[key] = 0
            else:
                self._hs2_hunger_counters[key] = self._hs2_hunger_counters.get(key, 0) + 1

        # update memory (ts stored for time-gap context on next call)
        history.append({"role": "user", "content": user_text, "ts": effective_ts})
        history.append({"role": "assistant", "content": reply_text, "ts": int(time.time())})
        turn_count += 1

        if self.SUMMARY_EVERY_TURNS > 0 and (turn_count % self.SUMMARY_EVERY_TURNS == 0):
            new_summary = self._llm_summarize(history, user_record)
            if new_summary:
                summary = new_summary

        self._db_save_memory(chat_id, summary, history, turn_count)

    # ------------------------- HS3 Broadcast -------------------------
    def _maybe_hs3_broadcast(self) -> None:
        eff = self._effective_hs()
        if eff != "HS3":
            return

        entering = self._prev_effective_hs != "HS3"
        now = int(time.time())

        # on entry broadcast to all; otherwise apply per-user cooldown
        if entering:
            candidates = self._db_list_subscribers()
        else:
            candidates = self._db_broadcast_candidates(now - self.HS3_BROADCAST_COOLDOWN_SEC)

        if not candidates:
            return

        text = self._llm_hs3_broadcast() or self._prompts.get("hs3_broadcast_fallback", "pls come feed me in person i'm so hungry 😭 i really need food RIGHT NOW")

        for chat_id in candidates:
            # Skip-recent guard only applies to periodic cooldown re-broadcasts,
            # NOT on entry: when the robot just became starving everyone must be notified.
            if not entering:
                last_seen = self._db_subscriber_last_seen(chat_id)
                if last_seen and (now - last_seen) < self.HS3_SKIP_RECENT_SEC:
                    self._log("DEBUG", f"HS3 broadcast skipped -> {chat_id} (chatted {now - last_seen:.0f}s ago)")
                    continue
            self._tg_send(chat_id, text)
            self._db_mark_broadcast(chat_id)
            self._log("INFO", f"HS3 broadcast -> {chat_id} ({'enter' if entering else 'cooldown'})")

    # ------------------------- LLM (Azure OpenAI) -------------------------
    def _build_llm_client(self) -> Tuple[AzureOpenAI, str, str]:
        endpoint = self._get_env("AZURE_OPENAI_ENDPOINT") or self._get_env("AZURE_OPENAI_API_BASE")
        api_key = self._get_env("AZURE_OPENAI_API_KEY")
        api_version = self._get_env("OPENAI_API_VERSION") or self._get_env("AZURE_OPENAI_API_VERSION")

        deployment = (
            self._get_env("AZURE_DEPLOYMENT_GPT5_MINI")
            or self._get_env("AZURE_OPENAI_DEPLOYMENT")
            or self._get_env("AZURE_DEPLOYMENT")
            or "gpt5-mini"
        )

        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT not set")
        if "/openai/" in endpoint:
            endpoint = endpoint.split("/openai/")[0]
        endpoint = endpoint.rstrip("/")

        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY not set")
        if not api_version:
            raise RuntimeError("OPENAI_API_VERSION not set")

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        self._llm_max_tokens = int(self._get_env("TELEGRAM_LLM_MAX_TOKENS") or "4000")  # override via env
        if self._llm_max_tokens < 500:
            self._log("WARN", f"TELEGRAM_LLM_MAX_TOKENS={self._llm_max_tokens} is very low; consider >= 4000")
        return client, deployment, api_version

    def _llm_chat(self, messages: List[Dict[str, str]]) -> str:
        if not self._llm:
            return ""

        last_exc: Optional[Exception] = None
        use_completion_tokens = True

        for attempt in range(3):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self._llm_deployment,
                    "messages": messages,
                }
                if use_completion_tokens:
                    kwargs["max_completion_tokens"] = self._llm_max_tokens
                else:
                    kwargs["max_tokens"] = self._llm_max_tokens

                resp = self._llm.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                return ""

            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                last_exc = exc
                time.sleep(0.6 * (2**attempt))
                continue
            except Exception as exc:  # noqa: BLE001
                if use_completion_tokens and "max_completion_tokens" in str(exc):  # fallback param name
                    use_completion_tokens = False
                    continue
                last_exc = exc
                break

        self._log("WARN", f"LLM error: {last_exc}")
        return ""

    def _llm_summarize(self, history: List[Dict[str, str]], user_record: Optional[Dict[str, Any]] = None) -> str:
        if not self._llm:
            return ""
        window = history[-24:]
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in window)
        known_facts = ""
        if user_record:  # anchor facts so summary never loses them
            name = (user_record.get("name") or "").strip()
            likes = user_record.get("likes") or []
            parts = []
            if name:
                parts.append(f"The user's name is {name}.")
            if likes:
                parts.append(f"They like: {', '.join(likes)}.")
            if parts:
                known_facts = "Known facts (always include these): " + " ".join(parts) + "\n\n"

        msgs = [
            {"role": "system", "content": self._summarize_system_prompt()},
            {"role": "user", "content": known_facts + convo},
        ]
        out = self._llm_chat(msgs)
        return (out or "")[:400].strip()

    def _llm_hs3_broadcast(self) -> str:
        sys_p, usr_p = self._hs3_broadcast_prompts()
        if not sys_p or not usr_p:
            return ""
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        return self._llm_chat(msgs)

    def _fallback(self, hs: str) -> str:
        if hs == "HS3":
            return self._prompts.get("fallback_hs3", "pls come feed me, i'm starving 😭 i really can't wait much longer")
        if hs == "HS2":
            return self._prompts.get("fallback_hs2", "hehe sorry, something went a bit wrong on my end. also i'm kinda hungry rn 😅")
        return self._prompts.get("fallback_default", "oops something went wrong, can you say that again? 😅")

    # ------------------------- User memory -------------------------
    def _load_user_memory(self) -> Dict[str, Any]:
        if not self._db:
            return {}
        try:
            rows = self._db.execute("SELECT chat_id, data_json FROM user_memory").fetchall()
        except Exception as exc:
            self._log("WARN", f"Could not load user memory from DB: {exc}")
            return {}
        result: Dict[str, Any] = {}
        for chat_id, data_json in rows:
            try:
                record = json.loads(data_json or "{}")
                if isinstance(record, dict):
                    result[str(chat_id)] = record
            except Exception:
                pass
        return result

    def _save_user_memory(self) -> None:
        if not self._db or not self._user_memory:
            return
        now = int(time.time())
        try:
            for key, record in self._user_memory.items():
                self._db.execute(
                    "INSERT INTO user_memory(chat_id, data_json, updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(chat_id) DO UPDATE SET "
                    "data_json=excluded.data_json, updated_at=excluded.updated_at",
                    (int(key), json.dumps(record, ensure_ascii=False), now),
                )
            self._db.commit()
        except Exception as exc:
            self._log("WARN", f"Failed to save user memory to DB: {exc}")

    def _migrate_user_memory_from_json(self, json_path: str) -> None:
        """One-time import of user_memory.json into the DB, then rename the file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            now = int(time.time())
            count = 0
            for key, record in data.items():
                if not isinstance(record, dict):
                    continue
                # Don't overwrite records already in the DB
                existing = self._db.execute(
                    "SELECT 1 FROM user_memory WHERE chat_id=?", (int(key),)
                ).fetchone()
                if not existing:
                    self._db.execute(
                        "INSERT INTO user_memory(chat_id, data_json, updated_at) VALUES(?,?,?)",
                        (int(key), json.dumps(record, ensure_ascii=False), now),
                    )
                    self._user_memory[key] = record
                    count += 1
            self._db.commit()
            os.rename(json_path, json_path + ".migrated")
            self._log("INFO", f"Migrated {count} user record(s) from JSON to DB; "
                              f"original renamed to {os.path.basename(json_path)}.migrated")
        except Exception as exc:
            self._log("WARN", f"User memory JSON migration failed: {exc}")

    def _get_user_record(self, chat_id: int) -> Dict[str, Any]:
        key = str(chat_id)
        if key not in self._user_memory:
            self._user_memory[key] = {}
        record = self._user_memory[key]
        record.setdefault("name", "")
        record.setdefault("nickname", "")
        record.setdefault("age", 0)
        record.setdefault("likes", [])
        record.setdefault("dislikes", [])
        record.setdefault("favorite_topics", [])
        record.setdefault("inside_jokes", {})
        record.setdefault("first_talked", 0)
        record.setdefault("last_talked", 0)
        record.setdefault("last_personal_update", "")
        record.setdefault("conversation_style", {
            "uses_emojis": None,
            "message_length": "",
            "tone": "",
        })

        # Normalize jokes memory into a single map:
        # inside_jokes: phrase -> {"count": int, "last_seen": int}
        # Supports migration from older layouts:
        # - inside_jokes as list (confirmed jokes)
        # - joke_candidates as separate map
        jokes_raw = record.get("inside_jokes")
        candidates_raw = record.pop("joke_candidates", None)
        merged_jokes: Dict[str, Dict[str, int]] = {}

        if isinstance(jokes_raw, dict):
            for phrase, meta in jokes_raw.items():
                if not isinstance(phrase, str) or not phrase.strip():
                    continue
                key = phrase.strip().lower()
                if isinstance(meta, dict):
                    count = int(meta.get("count", 1) or 1)
                    last_seen = int(meta.get("last_seen", 0) or 0)
                elif isinstance(meta, list):
                    count = int(meta[0]) if len(meta) > 0 else 1
                    last_seen = int(meta[1]) if len(meta) > 1 else 0
                elif isinstance(meta, int):
                    count, last_seen = int(meta), 0
                else:
                    count, last_seen = 1, 0
                merged_jokes[key] = {
                    "count": max(1, count),
                    "last_seen": max(0, last_seen),
                }
        elif isinstance(jokes_raw, list):
            for phrase in jokes_raw:
                if isinstance(phrase, str) and phrase.strip():
                    key = phrase.strip().lower()
                    merged_jokes[key] = {"count": 2, "last_seen": 0}

        if isinstance(candidates_raw, dict):
            for phrase, meta in candidates_raw.items():
                if not isinstance(phrase, str) or not phrase.strip():
                    continue
                key = phrase.strip().lower()
                if isinstance(meta, list):
                    count = int(meta[0]) if len(meta) > 0 else 1
                    last_seen = int(meta[1]) if len(meta) > 1 else 0
                elif isinstance(meta, int):
                    count, last_seen = int(meta), 0
                elif isinstance(meta, dict):
                    count = int(meta.get("count", 1) or 1)
                    last_seen = int(meta.get("last_seen", 0) or 0)
                else:
                    count, last_seen = 1, 0

                prev = merged_jokes.get(key, {"count": 0, "last_seen": 0})
                merged_jokes[key] = {
                    "count": max(prev["count"], max(1, count)),
                    "last_seen": max(prev["last_seen"], max(0, last_seen)),
                }

        # Drop removed keys if present in older records.
        record.pop("relationship_style", None)
        record.pop("trust_level", None)
        record["inside_jokes"] = merged_jokes
        return record

    def _update_user_from_message(self, chat_id: int, msg: Dict[str, Any], user_text: str) -> None:
        """Extract name and preferences from Telegram message metadata and text."""
        record = self._get_user_record(chat_id)
        changed = False
        now = int(time.time())

        if not record.get("first_talked"):
            record["first_talked"] = now
            changed = True
        record["last_talked"] = now
        changed = True

        sender = msg.get("from") or {}
        tg_name = (sender.get("first_name") or sender.get("username") or "").strip()
        if tg_name and not record.get("name"):
            record["name"] = tg_name
            changed = True

        if user_text:
            # Normalize: collapse 3+ repeated chars & multi-whitespace for matching
            norm = self._normalize_for_matching(user_text)

            # --- Age ---
            # "I'm 23", "im 23 years old", "my age is 23", "just turned 23",
            # "turning 30 soon", "i'll be 25 next month"
            m_age = re.search(
                r"\b(?:i'?m|i\s+am)\s+(\d{1,3})\b"
                r"(?:\s+y(?:ea)?rs?(?:\s+old)?\b)?"
                r"(?!\s*(?:st|nd|rd|th|min(?:ute)?s?|hours?|hrs?|sec(?:ond)?s?"
                r"|days?|weeks?|months?|km|miles?|meters?|percent|%"
                r"|kg|lbs?|cm|mm|pm|am)\b)"
                r"|\bmy\s+age\s+is\s+(\d{1,3})\b"
                r"|\bi(?:'?ve)?\s+just\s+turned\s+(\d{1,3})\b"
                r"|\b(?:i'?m\s+)?turning\s+(\d{1,3})\s+(?:soon|next|this|tomorrow)\b"
                r"|\bi(?:'?l+|\s+wil+)\s+be\s+(\d{1,3})\s+(?:soon|next|this|tomorrow)\b",
                norm, re.IGNORECASE,
            )
            if m_age:
                raw = next((g for g in m_age.groups() if g is not None), None)
                if raw:
                    age_val = int(raw)
                    if 5 <= age_val <= 120:
                        record["age"] = age_val
                        changed = True

            # --- Name ---
            # "call me X", "my name is X", "i'm called X", "i go by X",
            # "the name's X", "they call me X"
            m = re.search(
                r"\b(?:call\s+me|my\s+name(?:'?s|\s+is)|i(?:'?m|\s+am)\s+called"
                r"|they\s+call\s+me|i\s+go\s+by|the\s+name'?s)\s+([\w'-]{2,20})\b",
                norm, re.IGNORECASE,
            )
            if m:
                captured = m.group(1).strip()
                if self._is_meaningful(captured):
                    record["name"] = captured.capitalize()
                    changed = True

            # --- Nickname ---
            # "you can call me X", "my nickname is X", "just call me X",
            # "everyone calls me X"
            m_nick = re.search(
                r"\b(?:you\s+can\s+call\s+me|my\s+friends\s+call\s+me"
                r"|everyone\s+calls\s+me|people\s+call\s+me"
                r"|just\s+call\s+me|my\s+nickname\s+is|my\s+nick\s+is)\s+([\w'-]{2,20})\b",
                norm, re.IGNORECASE,
            )
            if m_nick:
                captured = m_nick.group(1).strip()
                if self._is_meaningful(captured):
                    record["nickname"] = captured.lower()
                    changed = True

            # --- Likes ---
            # "I like/love/enjoy/dig X", "I'm a fan of X", "my favourite is X",
            # "i rly love X", "i'm addicted to X"
            m2 = re.search(
                r"\b(?:i\s+(?:really\s+|rly\s+|rlly\s+)?(?:like|love|enjoy|prefer|adore|dig)"
                r"|i'?m\s+(?:a\s+(?:big\s+|huge\s+)?fan\s+of"
                r"|(?:really\s+|rly\s+)?into|obsessed\s+with|addicted\s+to)"
                r"|my\s+favou?rite(?:\s+(?:is|one\s+is|thing\s+is))?)\s+([^,.!?\n]{2,40})",
                norm, re.IGNORECASE,
            )
            if m2:
                liked = self._clean_capture(m2.group(1)).lower()
                if self._is_meaningful(liked):
                    likes: List[str] = record.setdefault("likes", [])
                    if liked not in likes:
                        likes.append(liked)
                        if len(likes) > 3:
                            likes.pop(0)
                        changed = True

            # --- Dislikes ---
            # "i hate X", "i really can't stand X", "not a fan of X",
            # "please don't talk about X"
            m_dis = re.search(
                r"\b(?:i\s+(?:really\s+|absolutely\s+|totally\s+)?(?:hate|dislike|despise"
                r"|can'?t\s+stand|cannot\s+stand)"
                r"|i\s+don'?t\s+(?:really\s+)?like|i\s+do\s+not\s+(?:really\s+)?like"
                r"|i'?m\s+not\s+(?:a\s+)?(?:big\s+)?fan\s+of"
                r"|not\s+a\s+(?:big\s+)?fan\s+of"
                r"|(?:please?\s+)?don'?t\s+(?:ever\s+)?talk\s+(?:to\s+me\s+)?about"
                r"|pls\s+don'?t\s+talk\s+about)\s+([^,.!?\n]{2,40})",
                norm, re.IGNORECASE,
            )
            if m_dis:
                disliked = self._clean_capture(m_dis.group(1)).lower()
                if self._is_meaningful(disliked):
                    dislikes: List[str] = record.setdefault("dislikes", [])
                    if disliked not in dislikes:
                        dislikes.append(disliked)
                        if len(dislikes) > 5:
                            dislikes.pop(0)
                        changed = True

            # --- Favorite topics ---
            # "i'm into X", "i love talking about X", "i'm obsessed with X",
            # "i nerd out about X"
            m_topic = re.search(
                r"\b(?:i'?m\s+(?:really\s+|rly\s+)?into|i\s+am\s+(?:really\s+)?into"
                r"|i\s+(?:really\s+)?love\s+talking\s+about"
                r"|my\s+favou?rite\s+thing(?:\s+to\s+talk\s+about)?\s+is"
                r"|i'?m\s+obsessed\s+with"
                r"|i\s+(?:really\s+)?enjoy\s+talking\s+about"
                r"|i\s+nerd\s+out\s+(?:about|on|over))\s+([^,.!?\n]{2,40})",
                norm, re.IGNORECASE,
            )
            if m_topic:
                topic = self._clean_capture(m_topic.group(1)).lower()
                if self._is_meaningful(topic):
                    topics: List[str] = record.setdefault("favorite_topics", [])
                    if topic not in topics:
                        topics.append(topic)
                        if len(topics) > 5:
                            topics.pop(0)
                        changed = True

            # --- Last personal update ---
            m_life = re.search(
                r"\b(?:"
                # "i have / i've got" + event/condition
                r"i'?(?:ve|\s+have)\s+(?:got\s+)?(?:an?\s+)?(?:exam|test|deadline|meeting"
                r"|job\s+interview|interview|cold|flu|covid|headache|fever|migraine"
                r"|appointment|date|presentation|demo)"
                # "i'm" + state/location
                r"|i'?m\s+(?:sick|ill|not\s+feeling\s+well|tired|exhausted|stressed"
                r"|depressed|sad|happy|excited|nervous|anxious|bored|lonely"
                r"|busy|free|at\s+work|at\s+school|at\s+uni(?:versity)?|at\s+college|at\s+home"
                r"|at\s+the\s+(?:gym|hospital|doctor'?s?|dentist'?s?|airport|beach"
                r"|park|office|library|store|mall)"
                r"|travel(?:l?ing)|on\s+vacation|on\s+holiday|on\s+a\s+trip"
                r"|going\s+to\s+(?:sleep|bed|work|school)"
                r"|heading\s+(?:to\s+(?:bed|work|school)|home|out)"
                r"|running\s+late|late\s+today|free\s+today"
                r"|moving(?:\s+(?:house|out|away))?|cooking|studying|cramming|working\s+out"
                r"|hungover|drunk|injured|pregnant|engaged|married|single"
                r"|on\s+my\s+way|about\s+to\s+(?:leave|sleep|eat|go))"
                # "i am" + state
                r"|i\s+am\s+(?:sick|ill|not\s+feeling\s+well|tired|exhausted|stressed"
                r"|depressed|sad|happy|excited|nervous|anxious|bored|lonely"
                r"|busy|free|at\s+work|at\s+school|at\s+uni(?:versity)?|at\s+college|at\s+home"
                r"|travel(?:l?ing)|on\s+vacation|on\s+holiday|on\s+a\s+trip)"
                # "i just" + past action
                r"|i\s+just\s+(?:got\s+(?:home|back|fired|hired|promoted"
                r"|married|dumped|engaged|divorced)"
                r"|woke\s+up|finished|graduated|started|moved|broke\s+up|had\s+a\s+baby)"
                # special phrases
                r"|today\s+is\s+my\s+birthday|it'?s\s+my\s+birthday"
                r"|i\s+(?:just\s+)?broke\s+(?:my|a)\s+\w+"
                r"|i\s+(?:lost|found)\s+my\s+(?:job|phone|wallet|keys)"
                r")"
                r"[^.!?\n]{0,60}",
                norm, re.IGNORECASE,
            )
            if m_life:
                captured = m_life.group(0).strip().lower()
                if len(captured) >= 3:
                    record["last_personal_update"] = captured[:120]
                    changed = True

            # --- Conversation style ---
            cs = record.setdefault("conversation_style", {"uses_emojis": None, "message_length": "", "tone": ""})

            if cs.get("uses_emojis") is not True and self._EMOJI_RE.search(user_text):
                cs["uses_emojis"] = True
                changed = True

            msg_len = len(user_text)
            new_length = "short" if msg_len < 25 else ("long" if msg_len > 120 else "medium")
            if cs.get("message_length") != new_length:
                cs["message_length"] = new_length
                changed = True

            # --- Tone (playful) ---
            # Handles elongated laughs (hahaha, hehehe), slang (bruh, fr fr, xD)
            if re.search(
                r"\b(?:lol|lmao|lmfao|rofl|omg|ikr|bruh)\b"
                r"|\bfr\s+fr\b"
                r"|\bha(?:ha)+h?\b|\bhe(?:he)+h?\b"
                r"|[xX]+[dD]+(?:\b|$)",
                norm, re.IGNORECASE,
            ):
                if cs.get("tone") != "playful":
                    cs["tone"] = "playful"
                    changed = True

            # --- Inside jokes ---
            # Catches explicit shared-memory references:
            # "remember when we/you X", "that's our joke/thing",
            # "haha the X thing", "we always say X", "like that time we X"
            m_joke = re.search(
                r"\b(?:remember\s+(?:when\s+(?:we|you)|that\s+time\s+(?:we|you)"
                r"|the\s+time\s+(?:we|you)))\s+([^.!?\n]{3,60})"
                r"|\b(?:that'?s\s+(?:our|an?)\s+(?:inside\s+)?(?:joke|thing)"
                r"|our\s+inside\s+joke)\b[:\s]*([^.!?\n]{0,60})"
                r"|\b(?:(?:lol|haha|lmao|hehe)\s+)the\s+([\w\s'-]{3,40})\s+thing\b"
                r"|\bwe\s+always\s+(?:say|do|call\s+(?:it|that))\s+([^.!?\n]{3,50})"
                r"|\blike\s+that\s+time\s+(?:when\s+)?(?:we|you)\s+([^.!?\n]{3,60})",
                norm, re.IGNORECASE,
            )
            if m_joke:
                raw_joke = next(
                    (g for g in m_joke.groups() if g is not None), None
                )
                if raw_joke:
                    joke = self._clean_capture(raw_joke, max_len=80).lower()
                    if self._is_meaningful(joke, min_len=3):
                        jokes: Dict[str, Dict[str, int]] = record.setdefault("inside_jokes", {})

                        # prune stale entries
                        expired = [
                            k for k, v in jokes.items()
                            if isinstance(v, dict)
                            and int(v.get("last_seen", 0) or 0) > 0
                            and (now - int(v.get("last_seen", 0))) > self.JOKE_CANDIDATE_TTL_SEC
                        ]
                        for k in expired:
                            del jokes[k]
                        if expired:
                            changed = True

                        existing_key = next((k for k in jokes if joke in k or k in joke), None)
                        if existing_key is not None:
                            meta = jokes.get(existing_key, {})
                            meta_count = int(meta.get("count", 1) or 1) + 1
                            jokes[existing_key] = {
                                "count": meta_count,
                                "last_seen": now,
                            }
                            changed = True
                        else:
                            if len(jokes) >= self.JOKE_CANDIDATE_MAX:
                                oldest = min(
                                    jokes,
                                    key=lambda k: int((jokes.get(k) or {}).get("last_seen", 0) or 0),
                                )
                                del jokes[oldest]
                            jokes[joke] = {"count": 1, "last_seen": now}
                            changed = True

        if changed:
            self._save_user_memory()

    def _build_user_context(self, record: Dict[str, Any]) -> str:
        """Return a system-prompt snippet with what we know about the user."""
        if not record or not isinstance(record, dict):
            return ""

        def _safe_str(v: Any, max_len: int = 60) -> str:
            if not v or not isinstance(v, str):
                return ""
            return " ".join(v.split())[:max_len]

        def _safe_list(v: Any, max_items: int = 5) -> List[str]:
            if not isinstance(v, list):
                return []
            return [s for s in (_safe_str(item) for item in v) if s][:max_items]

        parts: List[str] = []

        name = _safe_str(record.get("name"))
        nickname = _safe_str(record.get("nickname"))
        age = record.get("age")
        if name:
            name_str = f"Name: {name}."
            if nickname:
                name_str += f" Nickname: {nickname}."
            if age and isinstance(age, int) and 5 <= age <= 120:
                name_str += f" Age: {age}."
            parts.append(name_str)
        elif age and isinstance(age, int) and 5 <= age <= 120:
            parts.append(f"Age: {age}.")

        likes = _safe_list(record.get("likes"))
        topics = _safe_list(record.get("favorite_topics"))
        all_interests = list(dict.fromkeys(topics + likes))[:5]
        if all_interests:
            parts.append(f"Likes: {', '.join(all_interests)}.")

        dislikes = _safe_list(record.get("dislikes"), max_items=3)
        if dislikes:
            parts.append(f"Dislikes: {', '.join(dislikes)}.")

        update = _safe_str(record.get("last_personal_update"), max_len=80)
        if update:
            parts.append(f"Recent personal update: {update}.")

        jokes_raw = record.get("inside_jokes")
        jokes_confirmed: List[str] = []
        if isinstance(jokes_raw, dict):
            ranked = sorted(
                (
                    (phrase, meta)
                    for phrase, meta in jokes_raw.items()
                    if isinstance(phrase, str) and isinstance(meta, dict) and int(meta.get("count", 0) or 0) >= 2
                ),
                key=lambda item: int(item[1].get("last_seen", 0) or 0),
            )
            jokes_confirmed = [self._clean_capture(p, max_len=80) for p, _ in ranked if self._is_meaningful(p, min_len=3)]
        elif isinstance(jokes_raw, list):  # backward-compatible read of older records
            jokes_confirmed = _safe_list(jokes_raw, max_items=3)
        if jokes_confirmed:
            parts.append(f"Inside joke: {jokes_confirmed[-1]}.")

        cs = record.get("conversation_style")
        if not isinstance(cs, dict):
            cs = {}
        style_parts: List[str] = []
        tone = _safe_str(cs.get("tone"))
        if tone:
            style_parts.append(f"{tone} tone")
        msg_len = _safe_str(cs.get("message_length"))
        if msg_len:
            style_parts.append(f"{msg_len} messages")
        if cs.get("uses_emojis") is True:
            style_parts.append("uses emojis")
        if style_parts:
            parts.append(f"Style: {', '.join(style_parts)}.")

        if not parts:
            return ""

        # Wrap facts in a clear instruction: treat as background only, never force into conversation
        facts = " ".join(parts)
        wrapped = (
            "[Background info about this user — treat as silent context only. "
            "NEVER reference, mention, or allude to any of these facts unless the user brings up "
            "the exact same topic first in this conversation. "
            "Do NOT volunteer this information, do NOT use it to make small talk, "
            "do NOT weave it in proactively. "
            "Only use a fact if the user's current message directly touches on it.] "
            + facts
        )
        if len(wrapped) > 600:
            wrapped = wrapped[:597].rsplit(" ", 1)[0] + "..."
        return wrapped

    @staticmethod
    def _clean_capture(s: str, max_len: int = 40) -> str:
        """Strip leading/trailing filler words and punctuation from a regex capture."""
        s = s.strip()
        # Strip leading filler words ("I like *like* pizza", "I love *basically* everything")
        s = re.sub(
            r"^(?:like|basically|kinda|kind\s+of|sort\s+of|um+|uh+|well|so|just|really|lowkey|highkey)\s+",
            "", s, flags=re.IGNORECASE,
        ).strip()
        # Strip trailing filler phrases
        s = re.sub(
            r"\s+(?:a\s+lot|very\s+much|so\s+much|too|also|btw|tho|though|actually|"
            r"honestly|lol|haha|rn|ngl|for\s+real|fr|tbh|and\s+stuff|and\s+things|"
            r"or\s+whatever|or\s+something|you\s+know|idk|imo|i\s+guess|i\s+think|no\s+cap)\s*$",
            "", s, flags=re.IGNORECASE,
        ).strip().rstrip("!.,;:?").strip()
        return s[:max_len]

    @staticmethod
    def _is_meaningful(s: str, min_len: int = 2) -> bool:
        """Return False if the capture is too short or is a bare stopword."""
        if not s or len(s.strip()) < min_len:
            return False
        _STOPWORDS = {"a", "an", "the", "it", "this", "that", "them", "things",
                      "stuff", "something", "everything", "anything", "i", "me",
                      "to", "for", "and", "or", "so", "but", "my", "your"}
        return s.strip().lower() not in _STOPWORDS

    @staticmethod
    def _normalize_for_matching(text: str) -> str:
        """Normalize casual chat text for robust regex matching.

        - Converts smart/curly apostrophes to plain ASCII ' (U+0027)
          so regexes like i'?m, i'?ve, i'?ll work regardless of keyboard
        - Collapses 3+ repeated characters ('loooove' -> 'love')
        - Collapses multiple whitespace into single space
        """
        # U+2019 RIGHT SINGLE QUOTATION MARK (most common phone autocorrect)
        # U+2018 LEFT SINGLE QUOTATION MARK
        # U+02BC MODIFIER LETTER APOSTROPHE
        # U+FF07 FULLWIDTH APOSTROPHE
        text = text.replace("\u2019", "'").replace("\u2018", "'") \
                   .replace("\u02bc", "'").replace("\uff07", "'")
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _reply_mentions_hunger(text: str) -> bool:
        """Heuristic check: did the model already include a hunger mention?"""
        return bool(re.search(
            r"\b(?:hungry|hunger|starv|tummy|stomach|eat|food|snack|feed|fed)",
            text, re.IGNORECASE,
        ))

    @staticmethod
    def _format_message_time(ts: int, tz_name: str = "Europe/Rome") -> str:
        """Convert a Unix timestamp to a natural-language time string in the given timezone.

        Example output: "Tuesday night (11:42 PM, CET)"
        """
        try:
            dt = datetime.fromtimestamp(ts, tz=ZoneInfo(tz_name))
        except Exception:
            dt = datetime.utcfromtimestamp(ts)

        hour = dt.hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 14:
            period = "midday"
        elif 14 <= hour < 18:
            period = "afternoon"
        elif 18 <= hour < 21:
            period = "evening"
        else:
            period = "night"

        day_name = dt.strftime("%A")
        time_str = dt.strftime("%-I:%M %p")  # e.g. "11:42 PM" (no leading zero, Linux)
        tz_abbr = dt.strftime("%Z")           # e.g. "CET" or "CEST"
        return f"{day_name} {period} ({time_str}, {tz_abbr})"

    @staticmethod
    def _format_time_gap(prev_ts: int, curr_ts: int) -> str:
        """Return a human-readable description of the gap between two Unix timestamps."""
        diff = max(0, curr_ts - prev_ts)
        if diff < 120:
            return "just now"
        if diff < 3600:
            mins = diff // 60
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        if diff < 86400:
            hours = diff // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        if diff < 7 * 86400:
            days = diff // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"
        if diff < 30 * 86400:
            weeks = diff // (7 * 86400)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        months = diff // (30 * 86400)
        return f"{months} month{'s' if months != 1 else ''} ago"

    @staticmethod
    def _format_history_label(ts: int, tz_name: str = "Europe/Rome") -> str:
        """Return a compact timestamp label for a history entry.

        Example: "[Mon 6 Mar 2026, 11:42 PM, CET]"
        """
        try:
            dt = datetime.fromtimestamp(ts, tz=ZoneInfo(tz_name))
        except Exception:
            dt = datetime.utcfromtimestamp(ts)
        day_abbr = dt.strftime("%a")            # Mon
        day_num  = dt.strftime("%-d")           # 6  (no leading zero)
        month    = dt.strftime("%b")            # Mar
        year     = dt.strftime("%Y")            # 2026
        time_str = dt.strftime("%-I:%M %p")    # 11:42 PM
        tz_abbr  = dt.strftime("%Z")            # CET / CEST
        return f"[{day_abbr} {day_num} {month} {year}, {time_str}, {tz_abbr}]"

    # ------------------------- DB helpers -------------------------
    def _db_get_meta(self, key: str, default: str) -> str:
        if not self._db:
            return default
        row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def _db_set_meta(self, key: str, value: str) -> None:
        if not self._db:
            return
        self._db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._db.commit()

    def _maybe_persist_offset(self) -> None:
        now = time.time()
        if now - self._tg_last_offset_save < 1.0:
            return
        self._tg_last_offset_save = now
        self._db_set_meta("tg_offset", str(self._tg_offset))

    def _db_upsert_subscriber(self, chat_id: int) -> None:
        if not self._db:
            return
        now = int(time.time())
        self._db.execute(
            "INSERT INTO subscribers(chat_id, started_at, last_seen_at) VALUES(?,?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET last_seen_at=excluded.last_seen_at",
            (int(chat_id), now, now),
        )
        self._db.commit()

    def _db_touch_subscriber(self, chat_id: int) -> None:
        if not self._db:
            return
        self._db.execute(
            "UPDATE subscribers SET last_seen_at=? WHERE chat_id=?",
            (int(time.time()), int(chat_id)),
        )
        self._db.commit()

    def _db_list_subscribers(self) -> List[int]:
        if not self._db:
            return []
        rows = self._db.execute("SELECT chat_id FROM subscribers").fetchall()
        return [int(r[0]) for r in rows]

    def _db_broadcast_candidates(self, cutoff_ts: int) -> List[int]:
        if not self._db:
            return []
        rows = self._db.execute(
            "SELECT chat_id FROM subscribers WHERE last_broadcast_at<=?",
            (int(cutoff_ts),),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def _db_mark_broadcast(self, chat_id: int) -> None:
        if not self._db:
            return
        self._db.execute(
            "UPDATE subscribers SET last_broadcast_at=? WHERE chat_id=?",
            (int(time.time()), int(chat_id)),
        )
        self._db.commit()

    def _db_subscriber_last_seen(self, chat_id: int) -> int:
        if not self._db:
            return 0
        row = self._db.execute(
            "SELECT last_seen_at FROM subscribers WHERE chat_id=?",
            (int(chat_id),),
        ).fetchone()
        return int(row[0]) if row else 0

    def _db_count_subscribers(self) -> int:
        if not self._db:
            return 0
        row = self._db.execute("SELECT COUNT(*) FROM subscribers").fetchone()
        return int(row[0]) if row else 0

    def _db_load_memory(self, chat_id: int) -> Tuple[str, List[Dict[str, str]], int]:
        if not self._db:
            return "", [], 0
        row = self._db.execute(
            "SELECT summary, messages_json, turn_count FROM chat_memory WHERE chat_id=?",
            (int(chat_id),),
        ).fetchone()
        if not row:
            return "", [], 0
        summary = row[0] or ""
        try:
            history = json.loads(row[1] or "[]")
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
        turn_count = int(row[2] or 0)
        return summary, history, turn_count

    def _db_save_memory(self, chat_id: int, summary: str, history: List[Dict[str, str]], turn_count: int) -> None:
        if not self._db:
            return
        now = int(time.time())
        keep = max(2, self.MAX_HISTORY_TURNS * 2)  # 2 msgs per turn
        trimmed = history[-keep:]

        self._db.execute(
            "INSERT INTO chat_memory(chat_id, summary, messages_json, turn_count, updated_at) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET "
            "summary=excluded.summary, messages_json=excluded.messages_json, "
            "turn_count=excluded.turn_count, updated_at=excluded.updated_at",
            (int(chat_id), summary or "", json.dumps(trimmed, ensure_ascii=False), int(turn_count), now),
        )
        self._db.commit()

    def _db_clear_memory(self, chat_id: int) -> None:
        if not self._db:
            return
        self._db.execute("DELETE FROM chat_memory WHERE chat_id=?", (int(chat_id),))
        self._db.commit()

    # ------------------------- Misc helpers -------------------------
    @staticmethod
    def _get_env(key: str) -> str:
        return (os.getenv(key, "") or "").strip().strip('"').strip("'")

    @staticmethod
    def _log(level: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{level}] {msg}")


if __name__ == "__main__":
    import sys

    yarp.Network.init()
    try:
        if not yarp.Network.checkNetwork():
            print("ERROR: YARP network not available")
            raise SystemExit(1)

        module = ChatBotModule()
        rf = yarp.ResourceFinder()
        rf.setVerbose(False)
        rf.configure(sys.argv)
        module.runModule(rf)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        try:
            module.close()
        except Exception:
            pass
        yarp.Network.fini()
