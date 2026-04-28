# Always-on Cognitive Architecture — Embodied Behaviour (iCub)

Robot: iCub  ·  Platform: YARP

This repository contains a compact, always-on cognitive architecture for short, face-to-face social interactions and a parallel remote Telegram social channel. The design separates perception, attention/arbitration, and interaction execution into modular YARP modules to meet real-time and persistence requirements.

Contents
- `vision.py`         — perception front-end (camera → landmarks, QR, targetBox)
- `salienceNetwork.py` — attention selection and proactive gating (IPS)
# Always-on Cognitive Architecture — Embodied Behaviour (iCub)

Robot: iCub  ·  Platform: YARP

This repository contains a compact, always-on cognitive architecture for short, face-to-face social interactions and a parallel remote Telegram social channel. The design separates perception, attention/arbitration, and interaction execution into modular YARP modules to meet real-time and persistence requirements.

Contents
- `vision.py`         — perception front-end (camera → landmarks, QR, targetBox)
- `salienceNetwork.py` — attention selection and proactive gating (IPS)
- `executiveControl.py` — interaction execution (social-state trees, hunger, STT/TTS, LLM)
- `chatBot.py`        — Telegram companion and per-user memory
- `prompts.json`      — shared prompt configuration for behavior and LLMs

Principles
- Clear separation of concerns: perception → arbitration → execution
- Real-time front-end with backlog draining and lightweight bottles
- Stateful arbitration with adaptive per-person weights and habituation
- Interaction executor with guarded monitors, timeouts, and persistent memory
- Async LLM usage with fallbacks and latency-conscious orchestration

High-level dataflow
1) Perception: `vision.py` reads the camera, detects and tracks faces, computes landmarks (pose, gaze, mouth motion), runs recognition against `faces/`, and emits per-face landmark bottles and QR events.
2) Arbitration: `salienceNetwork.py` consumes landmarks and optional STM context, computes an Interest Priority Score (IPS) per face, applies hysteresis and habituation, assigns social states (ss1..ss4) from local JSON memory, and selects an attention target. When gating passes, it issues an RPC `run(track_id, face_id, ss)` to `executiveControl.py` and streams target command(s) back to `vision.py`.
3) Execution: `executiveControl.py` locks the target, starts a monitor, chooses hunger vs social trees (HS0..HS3 influence), performs STT/TTS and short LLM turns for conversational flows, updates persisted memory (greeted/talked), and returns a compact interaction result so the selector can update learning weights.
4) Remote channel: `executiveControl.py` publishes hunger state to `chatBot.py` over a YARP hunger port. `chatBot.py` runs independently, manages Telegram long-polling, per-user memory and summaries, HS3 broadcasts, and analytic logging to SQLite.

Core concepts
- Interest Priority Score (IPS): normalized combination of proximity, centrality, approach velocity, and gaze, modulated by per-person learned weights. IPS drives attention ranking and proactive gating.
- Social states: `ss1` (unknown), `ss2` (known, not greeted), `ss3` (known, greeted, not talked), `ss4` (already talked today). Social state determines proactive thresholds and which interaction tree to run.
- Hunger model: persistent, time-draining stomach level in `executiveControl.py` influencing behavior and broadcasting to `chatBot.py` (HS0..HS3 semantics).
- Adaptive learning: per-person weights shift after interactions (success/failure) and are persisted to JSON to tune future IPS calculations.
- LLM usage: short assistant turns (name extraction, convo starter, follow-ups, closings) with latency-aware orchestration and fallbacks; prompts are centralized in `prompts.json`.

Ports and RPC (overview)
- Vision ports: `/alwayson/vision/landmarks:o`, `/alwayson/vision/qr:o`, `/alwayson/vision/targetCmd:i` (selector → vision)
- Salience ports: `/alwayson/salienceNetwork/landmarks:i`, `/alwayson/salienceNetwork/targetCmd:o`, `/alwayson/salienceNetwork/debug:o` and RPC server `/salienceNetwork` (set_track_id, reset_cooldown)
- Executive RPC: `/executiveControl` (status/run), hunger publish `/alwayson/executiveControl/hunger:o`
- ChatBot ports: RPC `/chatBot/rpc`, hunger input `/alwayson/chatBot/hunger:i`

Persistence and analytics
- Short-term social state: `memory/greeted_today.json`, `memory/talked_today.json`, `memory/last_greeted.json`, `memory/hunger_state.json`
- Learning weights: `memory/learning.json`
- Interaction analytics and Telegram memory: per-module SQLite DBs in `data_collection/` for event logging and analysis.

Operational notes
- The perception loop drains stale frames and only processes the freshest frame each cycle to reduce latency.
- `salienceNetwork.py` runs a high-frequency attention loop and lower-frequency gating for proactive interactions; it includes IO and DB worker threads to offload disk writes.
- `executiveControl.py` protects interactions with a target monitor (timeout if face lost), has STT/TTS coordination (speech dispatch + timing heuristics), and uses a latest-only LLM worker to prioritize the most recent user input.
- `chatBot.py` is independent and tolerant of intermittent network/API errors; it summarizes chats periodically and stores compact summaries for fast context building.

Development
- Use YARP to connect ports and RPCs between modules during local integration testing.
- Shared prompts are in `prompts.json`; both `executiveControl.py` and `chatBot.py` load overlays from this file.

Contact
For questions about design or instrumentation, check the top of each module (`vision.py`, `salienceNetwork.py`, `executiveControl.py`, `chatBot.py`) for implementation notes and tuning constants.

---
This README is an architecture summary rather than a usage tutorial. If you want, I can add a concise run / quickstart section next (YARP init, recommended RF config, and example yarp connect commands).

### Social interaction trees

**SS1 — Unknown person**
```text
1. Greet
2. Wait for response  →  no response  →  abort
3. Ask name           →  no response  →  abort
4. Extract name via LLM (retry once)  →  fail  →  abort
5. Register face identity via vision RPC
6. Update last_greeted.json + greeted_today.json
7. "Nice to meet you" → success, advance to ss3 path
```

**SS2 — Known, not greeted today**
```text
1. "Hi <name>"
2. Wait for response  →  responded  →  enter SS3 tree
3. No response        →  "Hi <name>" again (attempt 2)
4. Responded          →  enter SS3 tree
   No response        →  abort
```

**SS3 — Known, greeted, not talked today**
```text
1. Look up face name in Telegram DB for personalized context
   (likes, dislikes, topics, inside jokes, age)
2. Generate conversation starter via direct LLM call
  (at turn start, with local fallback on failure)
3. Speak starter
4. Up to 3 turns:
  wait STT utterance (15 s timeout)
     → submit follow-up or closing LLM request (async, fallback)
     → speak reply
5. If user responded at least once → talked=True, success, advance to ss4
   Otherwise                       → abort no_response_conversation
```

**SS4 — Already fully interacted today**
```text
No operation. Returns success immediately.
```

### Hunger subsystem

`HungerModel` maintains a continuous stomach level (0–100%) that drains over time.

```text
Drain rate:  100% over 5 hours (~0.0055%/s)

**Recent Changes (last 4 commits)**

- **executiveControl.py:** Added a thread-safe `HungerModel` with persistent hunger state and thresholds (HS0..HS3); introduced `LatencyTrace`, `SpeechCoordinator`, and a `LatestOnlyLlmWorker` to improve LLM turn latency and allow latest-request-wins execution; refined SS3 conversation flow, TTS timing heuristics, QR feeding and reactive greeting loops, and expanded interaction logging and persistence (last_greeted / greeted_today / hunger_state files).
- **chatBot.py:** Telegram companion improvements: integrated Azure OpenAI client with warmup/chat/summarize flows, stronger per-user memory and summary injection, HS3 broadcast mechanics, session tracking, SQLite analytics views, and more robust long-polling / typing indications and error handling.
- **prompts.json:** Expanded and reorganized prompts used by both modules — new hunger overlays, HS3 broadcast templates, system overrides, and specialized prompts for name extraction, conversation starters, follow-ups, and closings.
- **salienceNetwork.py:** IPS and selection refinements: per-person adaptive weights, habituation decay, hysteresis bonuses, context-aware cooldowns, improved eligibility gating, and background IO/DB workers for learning and analytics persistence.

These four commits collectively tighten end-to-end behaviour: attention scoring and gating are more adaptive, interactions (both face-to-face and Telegram) are more context-aware and hunger-sensitive, and LLM usage is optimized for responsiveness and safe fallbacks.
Persists:    hunger_state.json (survives restarts)

HS1: level >= 60%  (normal)
HS2: level >= 25%  (hungry)
HS3: level <  25%  (starving)
```

`hunger_mode` controls whether hunger logic is active:

```text
hunger_mode on  -> normal draining + HS transitions + hunger tree routing
hunger_mode off -> hunger logic disabled, treated as HS1/100%, QR meal scans ignored

mode switch always resets stored level to 100%
```

**Hunger feeding tree**

```text
1. Ask for food ("I'm hungry, would you feed me?")
2. Wait for a QR scan (up to `feed_wait_timeout_sec`, default 8 s)
3. QR received  → apply meal delta, then acknowledge feed
                → if still hungry → ask for more
                → if HS1 reached  → done
4. No QR        → prompt user to look for food (prompt #1)
5. Second miss  → abort (no_food_qr)
```

Feed deltas from QR codes:

| QR payload | Stomach delta |
|---|---|
| `SMALL_MEAL` | +10% |
| `MEDIUM_MEAL` | +25% |
| `LARGE_MEAL` | +45% |

QR detection is throttled to avoid duplicate feeds (3 s cooldown per scan).

### Responsive interactions

A separate thread monitors STT and QR events outside of proactive interactions.

```text
Event arrives
  |
  +--> proactive running?  yes → drop immediately (never deferred)
  |
  +--> event type:
        greeting in STT ("hello", "hi", "hey", "ciao", "buongiorno")
          + person must be in MUTUAL_GAZE or NEAR_GAZE attention state
          + if multiple qualifying faces, largest bounding-box face is selected
          + 10 s per-face cooldown
          → greet/intro response to selected face

        QR feed outside proactive
          → short acknowledgment
```

### LLM execution model

SS3 replies use a latest-only asynchronous worker so new user speech can supersede older in-flight LLM requests.

```text
interaction thread
  -> submit request to LatestOnlyLlmWorker
  -> keep polling worker events + keep listening to STT
  -> if newer user utterance arrives first: supersede current request
  -> if final reply arrives: speak it
  -> if LLM error: use local fallback (never stall conversation)

background:
  -> bounded parallel worker threads run Azure calls
  -> emits first_token / final / error / cancelled events
  -> stale or superseded results are discarded
```

The SS3 conversation starter is generated at turn start (synchronous call with local fallback).

### Ports and RPC

| Direction | Port | Content |
|---|---|---|
| Input | `/alwayson/executiveControl/landmarks:i` | Face landmarks (target monitor) |
| Input | `/alwayson/executiveControl/stt:i` | Speech-to-text transcripts |
| Input | `/alwayson/executiveControl/qr:i` | QR payload strings from vision |
| Output | `/alwayson/executiveControl/speech:o` | TTS text to speech synthesizer |
| Output | `/alwayson/executiveControl/hunger:o` | Current hunger state string |
| RPC server | `/executiveControl` | `status`, `ping`, `run`, `hunger`, `hunger_mode`, `help`, `quit` |

**RPC `hunger` command level mapping:**

```text
hs1 → set level to 100%
hs2 → set level to  59%
hs3 → set level to  24%
```

---

## 3.4 `chatBot.py` — Telegram Relationship Layer

### What it does

Maintains a persistent social relationship with each user over Telegram, independent of physical proximity. The robot's hunger state shapes how it communicates.

- Long-poll Telegram updates in a background thread
- Generates responses with Azure OpenAI, using per-user memory and chat history
- Injects time context (time of day, gap since last message) into every reply
- Extracts user profile from message text via regex (no extra LLM call needed)
- Periodically summarizes conversation history to keep context compact
- Broadcasts starvation alerts (HS3) to all Telegram subscribers
- Tracks inactivity-based chat sessions (new session after 30 min idle)
- Logs per-message/per-broadcast analytics with `session_id` and `turn_count_at_event`, then builds daily/user/session SQL views

### Hunger-driven persona

```text
HS1 (normal)   → standard friendly social texting
HS2 (hungry)   → normal + hunger leakage every 3 messages without a mention
HS3 (starving) → strict override: acknowledge everything else in ≤3 words,
                  then beg the person to come feed the robot in person
                  (panicked, emotional, guilt-trippy, short)
```

### HS3 broadcast logic

```text
HS3 entered → immediately broadcast to ALL subscribers

While remaining HS3:
  → periodic re-broadcast per subscriber (cooldown: 30 min each)
  → skip users who chatted within the last 10 min (don't spam active users)

Broadcast message is LLM-generated, with a prompt-template fallback on failure.
```

### Conversation memory model

Each chat has two layers of memory:

```text
rolling history  → last 10 turns stored verbatim (with timestamps)
summary          → regenerated every 8 turns via LLM
                   anchored to known facts (name, likes) so they survive compression
```

Both are stored in `chat_bot.db` and survive restarts.

### User profile extraction

User facts are extracted passively from message text using regex patterns. No extra LLM call.

| Field | Example triggers |
|---|---|
| `name` | "my name is X", "call me X", "I'm called X", "they call me X" |
| `nickname` | "just call me X", "my nickname is X", "everyone calls me X" |
| `age` | "I'm 23", "I just turned 25", "turning 30 soon" |
| `likes` | "I love X", "I'm a fan of X", "my favourite is X" |
| `dislikes` | "I hate X", "can't stand X", "not a fan of X" |
| `favorite_topics` | "I'm really into X", "I love talking about X" |
| `last_personal_update` | "I just moved to X", "I started a new job" |
| `conversation_style` | emoji usage, message length, tone |

**Inside jokes** are tracked with a count and timestamp. A joke must be referenced at least 2 times before it is considered confirmed and used in conversation context.

Additionally, Telegram metadata (sender name from `from.first_name`) is used to pre-populate `name` before any text extraction.

### RPC

| Command | Effect |
|---|---|
| `status` | Returns effective hunger state (`HS1/HS2/HS3` or empty when unavailable), raw state, source (`port/rpc/none`), stale flag, subscriber count, queue size, thread health |
| `help` | Returns supported RPC commands |
| `set_hs HS1\|HS2\|HS3` | Sets hunger state source to RPC and forces effective hunger to that value |
| `clear_hs` | Clears hunger state (`source=none`, effective hunger becomes empty) |
| `reload_prompts` | Reload `prompts.json` without restart |

**Stale hunger protection:** if hunger source is `port` and no update arrives for 60 s, effective hunger becomes empty (no hunger overlay), not `HS0`.

---

## 3.5 `prompts.json` — Prompt Surface

Central prompt file loaded by both `executiveControl` and `chatBot` at startup, with runtime reload support.

**`executiveControl` section:**
- `system_default` — base spoken-conversation system prompt
- `system_overlay_hs1`, `system_overlay_hs2` — hunger overlays for normal and hungry states
- `system_fast` — short fallback for a single spoken sentence
- `system_json` — strict JSON prompt for name extraction
- `extract_name_prompt` — extracts a speaker's self-stated name from an utterance
- `convo_starter_prompt`, `convo_starter_prompt_hs1`, `convo_starter_prompt_hs2`
- `followup_prompt`, `followup_prompt_hs1`, `followup_prompt_hs2`
- `closing_ack_prompt`, `closing_ack_prompt_hs1`, `closing_ack_prompt_hs2`
- `ss1_greeting`, `ss1_ask_name`, `ss1_ask_name_retry`, `ss1_nice_to_meet`
- `ss2_greeting`, `reactive_greeting`
- `hunger_ask_feed`, `hunger_still_hungry`, `hunger_look_around`
- `feed_ack_hs1`, `feed_ack_hs2`, `feed_ack_hs3`

**`chat_bot` section:**
- `base_system_prompt`
- `base_system_prompt` biases Telegram replies toward natural rhythm, reduced repetition, and context-specific emotional responses
- `hs_overlays.HS1`, `hs_overlays.HS2`, `hs_overlays.HS3` — layered on top of base
- `hs3_override_system` — strict starvation override injected at the top of the message list
- `hs2_force_hunger_system` — forced hunger comment directive
- `hs3_broadcast_system`, `hs3_broadcast_user` — LLM prompts for HS3 broadcasts
- `hs3_broadcast_fallback` — hard-coded fallback if LLM fails
- `summarize_system` — prompt for periodic history summarization
- `summary_injection` — template to inject summary back into context
- `hs3_recovery_trigger`, `hs3_recovery_fallback` — used when recovering from HS3 starvation mode
- `reset_reply` — short reset message
- `fallback_default`, `fallback_hs2`, `fallback_hs3`

---

## 4) State Machines

### 4.1 Social state (assigned in `salienceNetwork.py`)

```text
        face_id unknown
              |
              v
             ss1  (Unknown — ask name)
              |
        name registered
              |
              v
             ss2  (Known, not yet greeted today)
              |
        greeted_today = yes
              |
              v
             ss3  (Greeted, not yet talked today)
              |
        talked_today = yes
              |
              v
             ss4  (Fully interacted — no proactive re-approach today)
```

Social state resets daily. `greeted_today.json` and `talked_today.json` hold the current day's flags.

### 4.2 Hunger state (managed by `HungerModel` in `executiveControl.py`)

```text
level >= 60%          → HS1  (normal)
25% <= level < 60%    → HS2  (hungry)
level < 25%           → HS3  (starving)
```

### 4.3 Chatbot effective hunger

```text
source = rpc                          → use RPC hunger state (HS1/HS2/HS3)
source = port and fresh (<= 60 s)     → use streamed hunger state (HS1/HS2/HS3)
source = port and stale (> 60 s)      → effective hunger = empty
source = none                         → effective hunger = empty
```

---

## 5) Failure and Abort Paths

### 5.1 Target monitor (proactive interaction abort)

```text
monitor thread checks landmarks every ~67 ms
  → face present           → last_seen reset
  → face absent > 12 s     → interaction abort flag set
                              → stops all waits (STT / LLM / speech)
                              → returns result with abort_reason = "target_lost"
```

Thread starvation guard: if the monitor thread stalls for > 1.5 s (e.g. system load), `last_seen` is advanced to prevent false aborts.

### 5.2 No-response paths per social tree

```text
SS1:
  greeting → no response                   → abort no_response_greeting
  ask name → no response                   → abort no_response_name
  name extraction fails (incl. retry)      → abort name_extraction_failed

SS2:
  attempt 1 no response → attempt 2
  attempt 2 no response                    → abort no_response_greeting

SS3:
  starter sent, no utterance in 15 s       → abort no_response_conversation
```

### 5.3 Responsive event gating

```text
STT greeting or QR arrives
  |
  +--> proactive running?  yes → drop immediately (never deferred or queued)
  +--> proactive running?  no  → execute now
```

### 5.4 Salience → executive gate

```text
candidate selected
  |
  +--> cooldown elapsed?
  +--> IPS >= SS threshold?
  +--> executiveControl RPC reachable?
  +--> executiveControl not busy?

any gate fails → skip, keep tracking, retry next cycle
all pass       → spawn interaction thread
```

### 5.5 Queue pressure

```text
any internal queue full (DB / IO / Telegram)
  → drop oldest entry
  → enqueue newest
  → main loop continues without blocking
```

### 5.6 Stale hunger (chatBot)

```text
no fresh hunger update for 60 s while source=port
  → effective hunger becomes empty
  → hunger-drive overlay is disabled until a fresh port update or RPC set_hs arrives
```

### 5.7 LLM unavailability

```text
LLM call fails or times out
  → executiveControl: returns fallback string from prompts.json (interaction continues)
  → chatBot: returns hunger-appropriate fallback string (message still sent)
```

---

## 6) YARP Interfaces

### 6.1 Core stream wiring

```bash
# Perception → Selection
yarp connect /alwayson/vision/landmarks:o /alwayson/salienceNetwork/landmarks:i

# Selection → Vision (target command loop)
yarp connect /alwayson/salienceNetwork/targetCmd:o /alwayson/vision/targetCmd:i

# Vision → FaceTracker (hardware gaze control)
yarp connect /alwayson/vision/targetBox:o /faceTracker/target:i

# Vision → Executive (QR codes)
yarp connect /alwayson/vision/qr:o /alwayson/executiveControl/qr:i

# Speech pipeline
yarp connect /speech2text/text:o /alwayson/executiveControl/stt:i
yarp connect /alwayson/executiveControl/speech:o /acapelaSpeak/speech:i

# Hunger stream
yarp connect /alwayson/executiveControl/hunger:o /alwayson/chatBot/hunger:i
```

Optional:

```bash
# STM context for adaptive cooldown
yarp connect /alwayson/stm/context:o /alwayson/salienceNetwork/context:i
```

### 6.2 RPC endpoints

| Module | Command | Description |
|---|---|---|
| `vision` | `name <name> id <track_id>` | Enroll face identity at runtime |
| `vision` | `help`, `process on/off`, `quit` | Module control |
| `salienceNetwork` | `set_track_id <int>` | Force attention target (override IPS) |
| `salienceNetwork` | `reset_cooldown <face_id> <track_id>` | Clear interaction cooldown for a person |
| `executiveControl` | `run <track_id> <face_id> <ss>` | Manually trigger an interaction |
| `executiveControl` | `hunger <hs1\|hs2\|hs3>` | Manually set stomach level |
| `executiveControl` | `hunger_mode <on\|off>` | Enable/disable hunger logic globally (resets level to 100%) |
| `executiveControl` | `status`, `ping`, `help`, `quit` | Module control |
| `chatBot` | `status` | Returns hunger state, subscriber count, and thread health |
| `chatBot` | `help` | Returns available chatbot RPC commands |
| `chatBot` | `set_hs HS1\|HS2\|HS3` | Override effective hunger persona |
| `chatBot` | `clear_hs` | Clear hunger source/state so effective hunger is empty |
| `chatBot` | `reload_prompts` | Reload prompts.json without restart |

---

## 7) Persistence and Memory

### 7.1 Memory architecture

Four layers of memory across increasing time scales:

```text
Frame scale (milliseconds)
  vision.py → face geometry, gaze, talking signal (ephemeral, not stored)

Interaction scale (seconds / minutes)
  executiveControl.db → proactive and responsive interaction records

Day scale
  greeted_today.json  → who was greeted today (drives ss2 → ss3)
  talked_today.json   → who was talked to today (drives ss3 → ss4)
  last_greeted.json   → timestamp of last greet per person

Long-term person scale (across sessions)
  learning.json       → per-person IPS weights (adapts selection behavior)

Relationship scale (permanent)
  chat_bot.db         → Telegram chat history, user profiles, and message/event analytics
```

### 7.2 Who writes what

| Module | Writes |
|---|---|
| `salienceNetwork.py` | `greeted_today.json`, `talked_today.json`, `learning.json`, `salience_network.db` (+ analytics SQL views) |
| `executiveControl.py` | `last_greeted.json`, `greeted_today.json`, `hunger_state.json`, `executive_control.db` (expanded interaction analytics fields + SQL views) |
| `chatBot.py` | `chat_bot.db` (`meta`, `subscribers`, `chat_memory`, `user_memory`, `chat_events` with `session_id` + `turn_count_at_event`) + analytics SQL views |

### 7.3 Write safety model

All writes are non-blocking from the main loop:

```text
module main loop
  → enqueue write/log event
  → background worker drains queue
  → atomic file replace (tempfile + os.replace) or SQLite WAL commit
  → main loop never stalls on I/O
```

### 7.4 SQLite databases

| Database | Tables | Contents |
|---|---|---|
| `data_collection/executive_control.db` | `interactions`, `reactive_interactions` | Proactive + reactive records, including replied-any, turn-depth (`n_turns`), trigger mode, and hunger transition fields |
| `data_collection/salience_network.db` | `target_selections`, `ss_changes`, `learning_changes`, `interaction_attempts` | Selection events, learning deltas, and interaction attempts enriched with `hunger_state` + proactive flag |
| `memory/chat_bot.db` | `meta`, `subscribers`, `chat_memory`, `user_memory`, `chat_events` | Telegram state, per-chat history, per-user profiles, and per-event analytics including `session_id` + turn-count snapshots |

### 7.4.1 Analytics SQL views (auto-created at startup)

- `executiveControl.py` creates: `v_proactive_interactions`, `v_metric_ss3_daily`, `v_metric_response_rate_daily`, `v_metric_repeat_users_daily`, `v_metric_depth_progression`
- `salienceNetwork.py` creates: `v_interaction_attempts_clean`, `v_interaction_attempts_daily` (grouped by `day_rome`, `hunger_state`, `start_ss`)
- `chatBot.py` creates: `v_chat_events_clean`, `v_chat_daily_metrics`, `v_chat_user_daily`, `v_chat_session_metrics`

### 7.5 JSON files

```text
memory/greeted_today.json     → {face_id: date_string}
memory/talked_today.json      → {face_id: date_string}
memory/last_greeted.json      → {face_id: {name, timestamp, ...}}
memory/learning.json          → {person_id: {w_prox, w_cent, w_vel, w_gaze}}
memory/hunger_state.json      → {level, last_update_ts, last_feed_ts, last_feed_payload}
```

---

## 8) Concurrency Model

```text
vision.py
  - main RFModule loop (detection, landmarks, target delegation)
  - _face_identity_lock  (guards identity maps)

salienceNetwork.py
  - main loop (IPS scoring, arbitration, target cmd)
  - interaction thread (runs RPC call + waits for result)
  - IO worker thread (JSON reads/writes)
  - DB worker thread (SQLite inserts)
  - last_greeted refresh thread (periodic background refresh)
  - RPC prewarm thread (background connection attempt on startup)

executiveControl.py
  - main loop (hunger drain + publish)
  - target monitor thread (per-interaction)
  - landmarks reader thread (continuous)
  - QR reader thread (continuous)
  - responsive loop thread (continuous)
  - latest-only LLM worker manager (spawns bounded per-request worker threads)
  - DB worker thread (SQLite inserts)

chatBot.py
  - main loop (hunger read + update drain + HS3 broadcast + HS3-exit recovery broadcast)
  - Telegram polling thread (long-poll + update queue)
```

---

## 9) Startup Order

```text
1. yarpserver
2. vision.py
3. salienceNetwork.py
4. executiveControl.py
5. chatBot.py
6. Connect YARP ports (see §6.1)
```

---

## 10) Operational Guarantees

- Proactive and responsive execution paths are mutually exclusive — responsive events are dropped (not deferred) while a proactive interaction runs.
- `salienceNetwork` starts and runs without STM context connected — the context port is optional and can be connected at any time.
- `chatBot` remains resilient if `executiveControl` disconnects — stale hunger becomes empty after 60 s (hunger overlay disabled until fresh input or RPC set).
- `executiveControl` can run with hunger mode OFF — interactions proceed as hunger-neutral (`HS1`/100%), and QR feeding events are ignored.
- All LLM calls have fallback strings — no interaction or Telegram reply ever blocks indefinitely on LLM availability.
- All file I/O and DB writes are off the main loop — latency spikes in storage never stall perception or interaction.
- The hunger level survives restarts — `HungerModel` loads `hunger_state.json` on startup and continues draining from the saved level.
