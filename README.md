# Embodied Behaviour

> **Robot:** iCub · **Platform:** YARP · **Author:** Nima Abaeian

---

## Table of Contents

1. [Big-Picture Overview](#1-big-picture-overview)
2. [End-to-End System Flow](#2-end-to-end-system-flow)
3. [Perception Layer (`perception.py`)](#3-perception-layer-perceptionpy)
4. [Face Selection (`faceSelector.py`)](#4-face-selection-faceselectorpy)
   - [SS — Social States](#41-ss--social-states)
   - [LS — Learning States](#42-ls--learning-states)
   - [Target Selection & Cooldown](#43-target-selection--cooldown)
   - [Triggering an Interaction](#44-triggering-an-interaction)
   - [Reward & Learning Updates](#45-reward--learning-updates)
5. [Interaction Manager (`interactionManager.py`)](#5-interaction-manager-interactionmanagerpy)
   - [SS1 — Unknown Person](#51-ss1--unknown-person)
   - [SS2 — Known, Not Greeted](#52-ss2--known-not-greeted)
   - [SS3 — Known, Greeted, Not Talked](#53-ss3--known-greeted-not-talked)
   - [HS — Hunger State & QR Feeding](#54-hs--hunger-state--qr-feeding)
   - [Target Monitor](#55-target-monitor)
   - [Responsive Interactions](#56-responsive-interactions)
   - [LLM Integration](#57-llm-integration)
   - [RPC Interface](#58-rpc-interface)
6. [Telegram Bot (`telegram_bot.py`)](#6-telegram-bot-telegram_botpy)
   - [Hunger-Aware Personality](#61-hunger-aware-personality)
   - [HS3 Broadcast Alerts](#62-hs3-broadcast-alerts)
   - [Long-Term User Memory](#63-long-term-user-memory)
   - [Conversation Memory](#64-conversation-memory)
7. [State Machines — Quick Reference](#7-state-machines--quick-reference)
8. [Shared Memory & Databases](#8-shared-memory--databases)
9. [Threading Architecture](#9-threading-architecture)
10. [Key Constants](#10-key-constants)
11. [YARP Connection Commands](#11-yarp-connection-commands)

---

## 1. Big-Picture Overview

iCub continuously watches the people in front of it through a camera. A perception pipeline detects faces, resolves their identity, and streams rich per-face observations — distance, gaze, bounding box, head pose — over YARP every frame. Three tightly coupled modules act on that stream:

| Module | Role |
|---|---|
| **`faceSelector`** | Interprets raw observations into social/learning states, picks one target, gates and fires interactions |
| **`interactionManager`** | Executes conversation and behavior trees (greet → ask name → chat), calls Azure LLM, drives TTS/STT |
| **`telegram_bot`** | Extends the relationship over Telegram; adapts personality to hunger level; persists a rich user profile that `interactionManager` reads back during face-to-face chats |

All three modules share a small folder of JSON and SQLite files (`memory/`) that forms the robot's persistent social memory.

---

## 2. End-to-End System Flow

```
Camera frames
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│  perception.py  (VisionAnalyzer @ 20 Hz)                           │
│  · MediaPipe face mesh → head pose, gaze direction                 │
│  · Object recognition → bounding box, track_id, face_id            │
│  · Derives: distance class · attention class · time_in_view        │
│  · Publishes one YARP Bottle per face → /alwayson/vision/          │
│    landmarks:o                                                     │
└────────────────────────────────────────────────────────────────────┘
     │  per-face Bottle  (every frame)
     ▼
┌────────────────────────────────────────────────────────────────────┐
│  faceSelector.py  (@ 20 Hz)                                        │
│  · Assigns SS (social state) and LS (learning state) to each face  │
│  · Picks the face with the biggest bounding box as the target      │
│  · Waits until the target's identity is resolved                   │
│  · Checks cooldown + LS eligibility gates                          │
│  · Fires: RPC  run(track_id, face_id, ss) → interactionManager     │
└────────────────────────────────────────────────────────────────────┘
     │  RPC: run(track_id, face_id, ss)
     ▼
┌────────────────────────────────────────────────────────────────────┐
│  interactionManager.py                                             │
│  · Executes SS1/SS2/SS3 conversation tree (or hunger-feed tree)    │
│  · Monitors that the target stays visible throughout               │
│  · Speaks (TTS), listens (STT), generates replies (Azure LLM)      │
│  · Registers new faces with the object-recognition system          │
│  · Returns compact JSON result → faceSelector                      │
│  · Broadcasts hunger state (HS1/HS2/HS3) at 1 Hz                   │
└────────────────────────────────────────────────────────────────────┘
     │  compact JSON  (success, final_state, abort_reason, …)
     ▼
faceSelector updates memory (greeted, talked, LS) + logs to SQLite

     │  /interactionManager/hunger:o  (1 Hz)
     ▼
┌────────────────────────────────────────────────────────────────────┐
│  telegram_bot.py  (@ 10 Hz)                                        │
│  · LLM-powered Telegram chatbot for registered users               │
│  · Personality shifts with hunger: friendly → desperate            │
│  · Broadcasts starvation alerts in HS3                             │
│  · Passively builds a long-term user profile (name, likes, jokes)  │
│  · Writes user_memory → memory/telegram_bot.db                     │
│    (interactionManager reads this in SS3 for personalized chat)    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Perception Layer (`perception.py`)

`perception.py` is the **vision front-end**. It runs as a YARP `RFModule` at 20 Hz and wraps two sub-systems:

- **MediaPipe Face Landmarker** — detects 478 face landmarks per frame and fits a 6-point 3-D face model to compute head-pose (pitch, yaw, roll) and gaze direction.
- **Object Recognition** — tracks face identities across frames, assigning each face a stable `track_id` and eventually a resolved `face_id` (person name).

**What gets published** per detected face on `/alwayson/vision/landmarks:o`:

| Field | Description |
|---|---|
| `track_id` | Stable integer ID for the physical face in the frame |
| `face_id` | Resolved identity (`"Alice"`, `"unknown"`, `"recognizing"`) |
| `bbox` | Bounding box `(x, y, w, h)` in pixels |
| `distance` | `SO_CLOSE` / `CLOSE` / `FAR` / `VERY_FAR` (derived from normalised bbox height) |
| `attention` | `MUTUAL_GAZE` / `NEAR_GAZE` / `SIDE_GAZE` / `UNKNOWN` (from head-pose yaw/pitch) |
| `time_in_view` | Seconds the `track_id` has been continuously visible |
| `talking` | Whether the person's lips are moving (mouth-motion std buffer) |
| `head_pose` | Yaw, pitch, roll in degrees |

**Distance thresholds** (normalised bbox height `h_norm = h / frame_height`):

| Class | Condition |
|---|---|
| `SO_CLOSE` | `h_norm > 0.40` |
| `CLOSE` | `0.20 < h_norm ≤ 0.40` |
| `FAR` | `0.10 < h_norm ≤ 0.20` |
| `VERY_FAR` | `h_norm ≤ 0.10` |

Faces detected by object recognition but not matched by MediaPipe (e.g., too far for landmarks) are still published with neutral pose values and `attention = "UNKNOWN"`, so `faceSelector` can still track their `time_in_view` and bbox size.

---

## 4. Face Selection (`faceSelector.py`)

`faceSelector` reads the per-face stream, adds social and learning meaning to each observation, and decides when and who to interact with.

**Main loop at 20 Hz:**

1. Read STM context label from `/alwayson/stm/context:o` (non-blocking — module runs fine without it).
2. Day-change check → reload memory JSON; prune today's `greeted` / `talked` records if it is a new day.
3. Parse all face Bottles from the landmarks port.
4. Compute SS and LS for every face.
5. Identify the face with the **largest bounding box** as the candidate.
6. If the candidate's `face_id` is still `"recognizing"` → **wait**; never fall back to a smaller resolved face.
7. If identity resolved, not in cooldown, eligible (LS gates pass), and `interactionManager` is idle → spawn interaction thread.
8. Annotate and publish camera image; publish debug status.

### 4.1 SS — Social States

SS is computed fresh every cycle from memory files. It encodes today's history with a specific person.

```
Known?    NO  ─────────────────────────────► ss1  Unknown
          YES
           ├─ Greeted today?   NO ──────────► ss2  Known, Not Greeted
          YES
           ├─ Talked today?    NO ──────────► ss3  Known, Greeted, Not Talked
          YES  ─────────────────────────────► ss4  Known, Greeted, Talked  (no-op)
```

| SS | Meaning | What happens |
|---|---|---|
| `ss1` | Stranger | Greet, ask name, register identity |
| `ss2` | Known, not greeted today | Say hi by name; transition to SS3 |
| `ss3` | Known, greeted but not conversed today | Initiate multi-turn LLM-powered chat |
| `ss4` | Fully interacted today | No-op; resets on day change |

### 4.2 LS — Learning States

LS is **per-person** and persists in `memory/learning.json`. It controls how demanding the eligibility gate is — iCub is cautious with strangers and relaxes as it learns each person.

| LS | Label | Distance allowed | Attention required | Min `time_in_view` |
|---|---|---|---|---|
| `LS1` | Early / strict | `SO_CLOSE`, `CLOSE` only | `MUTUAL_GAZE` only | ≥ 3.0 s |
| `LS2` | Developing | `SO_CLOSE`, `CLOSE`, `FAR` | `MUTUAL_GAZE`, `NEAR_GAZE` | ≥ 1.0 s |
| `LS3` | Advanced | Any | Any | ≥ 0.0 s |

A face is **eligible** for interaction if its SS is not `ss4` and its current LS constraints are all satisfied (LS3 has no constraints).

### 4.3 Target Selection & Cooldown

**Always pick the biggest bounding box.** If that face is still being identified, wait — never fall back to a smaller resolved alternative.

Cooldown duration adapts to the scene cluster received from the STM module:

| Scene label | Interpretation | Cooldown |
|---|---|---|
| `1` | Active / lively | 3 s |
| `0` | Calm / quiet | 15 s |
| `−1` | Unknown or STM absent | 5 s |

Known people use a `person_id` cooldown key. Unknown people use `"unknown:<track_id>"`.

### 4.4 Triggering an Interaction

```
updateModule() detects eligible target
  └─ Pre-check: interactionManager status → skip if busy or unreachable
  └─ Spawn _run_interaction_thread(target)

_run_interaction_thread
  1. Re-check status (skip if another interaction started)
  2. Signal robot body: ao_start
  3. RPC → interactionManager:  "run <track_id> <face_id> <ss>"
  4. Wait for compact JSON result
  5. Process result (update memory, compute reward, update LS)
  6. Signal robot body: ao_stop
  [finally] Mark interaction_busy = False; record cooldown timestamp
```

**YARP ports used by faceSelector:**

| Port | Direction | Purpose |
|---|---|---|
| `/faceSelector/landmarks:i` | IN | Per-face observations from perception |
| `/faceSelector/img:i` | IN | Camera frame for annotation |
| `/faceSelector/img:o` | OUT | Annotated image |
| `/faceSelector/debug:o` | OUT | Debug status Bottle |
| `/faceSelector/interactionManager:rpc` | OUT | `run` / `status` RPC |
| `/faceSelector/interactionInterface:rpc` | OUT | `ao_start` / `ao_stop` to robot body |
| `/faceSelector/context:i` | IN | STM scene-context label (optional) |

**Image annotation colour guide:**

| Colour | Meaning |
|---|---|
| Green | Active interaction target |
| Yellow | Eligible and ready |
| White | Present but not eligible |

Label above each box: `Alice (T:3) · ss2 | LS2 | LG:09:30 · CLOSE/MUT · area=12400`

### 4.5 Reward & Learning Updates

After every interaction `faceSelector` receives the compact JSON result and adjusts the person's LS:

| Outcome | Reward |
|---|---|
| Success + name extracted | `+2` |
| Success (no name) | `+1` |
| `not_responded` | `−1` |
| `face_disappeared` (first time in 30 s window) | `−1` |
| `face_disappeared` (≥ 2 times in 30 s window) | `−2` |
| Other failure | `−1` |

LS update:

```
reward > 0  →  LS = min(3, LS + 1)   advance
reward < 0  →  LS = max(1, LS − 1)   regress
reward = 0  →  no change
```

Changes are logged to `data_collection/face_selector.db` and atomically persisted to `memory/learning.json`.

---

## 5. Interaction Manager (`interactionManager.py`)

`interactionManager` receives the RPC call from `faceSelector` and executes the appropriate conversation tree. It owns TTS output, STT input, the LLM thread pool, the target monitor, and the hunger model.

**YARP ports:**

| Port | Direction | Purpose |
|---|---|---|
| `/interactionManager` | IN (RPC) | Main command handle |
| `/interactionManager/landmarks:i` | IN | Face stream for target monitoring |
| `/interactionManager/stt:i` | IN | Speech-to-text transcripts |
| `/interactionManager/speech:o` | OUT | TTS text to Acapela speaker |
| `/interactionManager/camLeft:i` | IN | Camera feed for QR code reading |
| `/interactionManager/hunger:o` | OUT | Hunger state broadcast at 1 Hz |

### 5.1 SS1 — Unknown Person

```
① Say greeting via TTS (`ss1_greeting` prompt key, default: "Hi there!")
② Wait for any response (10 s)  →  no response: abort
③ Say "We haven't met — what's your name?"
④ Wait for name response (10 s)  →  no response: abort
⑤ Extract name:
     fast regex  ("My name is X", "I'm X", "Call me X", …)
     → LLM fallback (GPT-5 nano, strict JSON schema, confidence clamped 0–1)
   Retry once on failure  →  still fails: abort name_extraction_failed
⑥ Register name with /objectRecognition RPC
⑦ Write memory/last_greeted.json
⑧ Say "Nice to meet you"

Result: success=True, final_state=ss3
```

### 5.2 SS2 — Known, Not Greeted

```
① Say "Hello <name>"
② Wait for response (10 s)
   Responded   →  write last_greeted  →  chain into SS3 tree
   No response →  retry once
③ Say "Hello <name>" (attempt 2)
④ Wait for response (10 s)
   Responded   →  chain into SS3
   No response →  abort: no_response_greeting
```

### 5.3 SS3 — Known, Greeted, Not Talked

Before speaking, iCub looks up whether this person has a Telegram profile and personalises the LLM prompts if so.

```
① Telegram user lookup (memory/telegram_bot.db, accent-insensitive name match)
   Match found  →  build user_context (name, age, likes, recent update, jokes…)
   No match     →  user_context = ""

② Generate opening question (LLM, async + abort-aware):
   user_context set  →  personalised starter prompt
   no context        →  pre-fetched generic starter from background cache

③ Say the opening question
④ Kick off next background starter pre-fetch

Conversation loop  (up to 3 turns):
  ├─ Wait for user response (12 s)
  │    No response  →  end loop
  ├─ Turn 1 or 2:  LLM follow-up (≤ 22 words, sentiment-aware; personalised if context)
  ├─ Turn 3 (last): LLM warm closing acknowledgment (4–8 words, no question; personalised)
  └─ Say the robot's reply

≥ 1 user turn  →  talked=True, final_state=ss4
0 user turns   →  abort: no_response_conversation
```

`telegram_user_matched = true` is recorded in the result when a Telegram profile was successfully used for personalisation.

### 5.4 HS — Hunger State & QR Feeding

The robot has an internal energy level (0–100 %) that drains to zero over 5 hours. Current level maps to a hunger state:

| HS | Level | Meaning |
|---|---|---|
| `HS1` | ≥ 60 % | Satisfied |
| `HS2` | 25–59 % | Hungry |
| `HS3` | < 25 % | Starving |

The hunger-feed tree **overrides** the social tree when: HS3 is active (any SS), or HS2 + ss3.

```
① Say "I'm so hungry, would you feed me please?"
Loop:
  ├─ Wait for QR scan (8 s)
  ├─ QR detected  →  feed(+delta) → say "Yummy, thank you!"
  │    Still hungry  →  say "I'm still hungry. Give me more."
  │    HS1 reached  →  break (satisfied)
  └─ No QR within 8 s:
       1st timeout  →  say "Take a look around, you'll find some food."
       2nd consecutive timeout  →  abort: no_food_qr

QR → hunger delta:
  SMALL_MEAL   →  +10 %
  MEDIUM_MEAL  →  +25 %
  LARGE_MEAL   →  +45 %
```

The QR reader runs in a daemon thread scanning via `cv2.QRCodeDetector` at ~50 fps.

### 5.5 Target Monitor

A dedicated thread runs **in parallel** with every active interaction (at 15 Hz). It checks that the `track_id` is still present in the landmarks stream. If the face is continuously absent for more than 12 seconds, it fires `abort_event` — which every STT wait, LLM poll, and speech wait responds to immediately.

```
_target_monitor_loop  (15 Hz, staleness tolerance 5 s)
  ├─ track_id found   →  reset last-seen timer
  └─ track_id absent  >  12 s  →  set abort_event: target_lost
```

### 5.6 Responsive Interactions

These handle user-initiated events that arise while no proactive interaction is running. Events are **dropped immediately** (not queued) if an interaction is in progress.

**Responsive Greeting**
- Trigger: STT matches `"hello"`, `"hi"`, `"ciao"`, `"good morning"` (word-boundary regex)
- Candidate selection: pick the **biggest-bbox face among all visible faces**; the utterance itself is the attention signal — no gaze check needed
- Cooldown: 10 s per candidate key (`<known_name>` for known users, `unknown:<track_id>` for unresolved users)
- Action (known face): say `"Hi <name>"` + write `last_greeted`; wait 12 s for a follow-up; if received, enter a full SS3-style conversation loop (up to 3 turns, optional Telegram personalisation)
- Action (unknown face): run a lightweight SS1-style intro in-place (`ss1_greeting` → ask name → one retry for name extraction), register the extracted name asynchronously, mark greeted-today, and say `"Nice to meet you"`

**Responsive QR Acknowledgment**
- Trigger: QR scan detected outside of any active interaction
- Action: say `"Yummy, thank you!"`

### 5.7 LLM Integration

**Backend:** Azure OpenAI via the direct OpenAI SDK (`openai.AzureOpenAI`)  
**Deployment:** `gpt5-nano` for all tasks (name extraction + conversation)  
**Timeout:** 60 s · **Retries:** 3 × 1 s delay  
**Concurrency:** Single-worker `ThreadPoolExecutor`; futures polled with abort-awareness (100 ms checks)

`setup_azure_llms()` creates one shared Azure client and keeps the logical split
between `llm_extract` and `llm_chat` only for routing/forward-compatibility.

All prompt templates and fixed speech strings live in `prompts.json` under `"interactionManager"`. Hardcoded fallbacks apply if the file or key is absent.

| LLM call | Purpose |
|---|---|
| `_llm_generate_convo_starter` | Short wellbeing/day question; personalised when user_context is set |
| `_llm_generate_followup` | Sentiment-aware follow-up ≤ 22 words; personalised when user_context is set |
| `_llm_generate_closing_acknowledgment` | Warm 4–8 word closing, no question; personalised when user_context is set |
| `_llm_extract_name` | JSON name extraction with confidence schema validation |

All calls use `max_completion_tokens = 2000`.

**Speech timing** (`_speak_and_wait`):
```
wait = word_count / 3.0 + 0.5   clamped to [1.0, 8.0] seconds
```
Abort event checked every 100 ms so the robot can be interrupted mid-sentence.

**Required env vars:** `AZURE_OPENAI_ENDPOINT` · `AZURE_OPENAI_API_KEY` · `OPENAI_API_VERSION` · `AZURE_DEPLOYMENT_GPT5_NANO`

### 5.8 RPC Interface

| Command | Arguments | Returns |
|---|---|---|
| `run` | `<track_id> <face_id> <ss1\|ss2\|ss3\|ss4>` | Compact JSON result |
| `hunger` | `<hs1\|hs2\|hs3>` | Force hunger level (100 % / 59 % / 24 %) |
| `status` / `ping` | — | `{"success":true, "busy":<bool>}` |
| `help` | — | Plain-text command list |
| `quit` | — | Shutdown |

**Compact result format:**

```json
{
  "success": true,
  "track_id": 3,
  "name": "Alice",
  "name_extracted": true,
  "abort_reason": null,
  "initial_state": "ss1",
  "final_state": "ss3",
  "interaction_tag": "SS1HS1",
  "hunger_state_start": "HS1",
  "hunger_state_end": "HS1",
  "stomach_level_start": 85.2,
  "stomach_level_end": 85.1,
  "telegram_user_matched": true
}
```

Abort reasons are compacted before returning: `target_lost` / `target_monitor_abort` → `"face_disappeared"` (only when user never spoke); everything else without speech → `"not_responded"`. If the user spoke, no negative reason is recorded.

---

## 6. Telegram Bot (`telegram_bot.py`)

`telegram_bot` is the **remote companion channel**. It runs as a YARP `RFModule` (10 Hz main loop), connecting registered users to iCub over Telegram at any time.

**YARP ports:**

| Port | Direction | Purpose |
|---|---|---|
| `/{module}/hunger:i` | IN | Hunger state from `interactionManager` (1 Hz) |
| `/{module}/rpc` | IN (RPC) | `status` · `set_hs` · `reload_prompts` |

**Per-message pipeline:**

1. Long-poll daemon thread receives Telegram update (20 s timeout) → queue.
2. Main thread drains queue (max 25 messages per cycle).
3. Upsert subscriber; call `_effective_hs()` → pick hunger personality overlay (staleness guard: fall back to HS1 if no update for > 60 s).
4. Load conversation memory (summary + rolling history) and user profile from DB.
5. Inject time context: day/time + gap since last message.
6. Build LLM message list with appropriate hunger overlay.
7. Call Azure LLM (synchronous) → send reply to Telegram.
8. Append turn to history; regenerate summary every 8 turns.
9. Persist everything to `memory/telegram_bot.db`.

### 6.1 Hunger-Aware Personality

| HS | Persona |
|---|---|
| `HS1` (≥ 60 %) | Normal, warm, friendly chat |
| `HS2` (25–59 %) | Normal chat, but slips in a casual hunger comment every 3 messages |
| `HS3` (< 25 %) | Full system-prompt override: every reply pivots back to begging the user to physically come feed the robot — urgent, emotional, a little dramatic |

### 6.2 HS3 Broadcast Alerts

When the bot **enters** HS3, every subscriber receives an LLM-generated starvation alert immediately (no cooldown). During sustained HS3, re-broadcasts are sent per-subscriber with:

- Per-subscriber cooldown: **30 minutes**
- Skip if the subscriber chatted in the last **10 minutes**
- Falls back to a fixed `hs3_broadcast_fallback` string if the LLM call fails

### 6.3 Long-Term User Memory

The bot passively builds a profile from natural conversation — no forms required:

| Field | How extracted |
|---|---|
| `name` | Telegram `first_name` at first contact; regex: `"my name is X"`, `"call me X"` |
| `nickname` | Regex: `"you can call me X"`, `"everyone calls me X"` |
| `age` | Regex: `"I'm 23"`, `"just turned 23"`, `"turning 25 next month"` |
| `likes` | Regex: `"I love/enjoy/adore X"`, `"my favourite is X"` (FIFO, max 3) |
| `dislikes` | Regex: `"I hate X"`, `"can't stand X"` (max 5) |
| `favorite_topics` | Regex: `"I'm into X"`, `"I nerd out about X"` (max 5) |
| `last_personal_update` | Life-event regex (`"I just got promoted"`, `"my exam is today"`) max 120 chars |
| `conversation_style` | Derived from message length, emoji presence, playful slang |
| `relationship_style` | `"protective"` on empathy signals (`"poor iCub"`, `"are you ok"`) |
| `inside_jokes` | References appearing ≥ 2× across separate messages (confirmed; max 5; candidates expire in 30 days) |
| `trust_level` | `"friend"` → `"close_friend"` on explicit signals (`"I've never told anyone"`, `"you really get me"`) |

This profile is written to `memory/telegram_bot.db` and read by `interactionManager` during SS3.  
Name matching is accent-insensitive: `André` matches `andre` (NFD Unicode decomposition).

### 6.4 Conversation Memory

Stored in the `chat_memory` table per user:

| Field | Description |
|---|---|
| `summary` | LLM summary of past exchanges (max 400 chars); regenerated every 8 turns |
| `messages_json` | Rolling window of the last 20 messages (10 turn-pairs), each timestamped |
| `turn_count` | Total turns since last `/reset` |

Every user message in the LLM context is prefixed with a timestamp label (`[Mon 6 Mar 2026, 11:42 PM, CET]`), and a time-gap note (`"Their previous message was 3 days ago"`) is injected as a system message for temporal awareness.

---

## 7. State Machines — Quick Reference

### Social State Machine (SS)

```
[ss1] Unknown
  │  Greet + ask name + extract name + register
  ▼
[ss3] Known, Greeted, Not Talked
  │  Conversation loop (≥ 1 user turn)
  ▼
[ss4] Known, Greeted, Talked  ◄── terminal for the day
  │  (day change resets → ss2)
  ▼
[ss2] Known, Not Greeted
  │  "Hi <name>" + response received
  └────────────────────────────────► [ss3]
```

Computed each cycle in `faceSelector` from memory files; executed in `interactionManager` trees.

### Learning State Machine (LS)

```
[LS1] Strict               SO_CLOSE or CLOSE · MUTUAL_GAZE only · time_in_view ≥ 3 s
  │   reward +1 or +2
  ▼
[LS2] Relaxed              + FAR allowed · + NEAR_GAZE allowed · time_in_view ≥ 1 s
  │   reward +1 or +2
  ▼
[LS3] Advanced             no constraints at all

Any level:  reward −1 or −2  →  regress one step  (floor: LS1)
Any level:  reward +1 or +2  →  advance one step  (ceiling: LS3)
```

Persisted per-person in `memory/learning.json`.

### Hunger State Machine (HS)

```
HS1 (≥60%)  ──drain──►  HS2 (25–59%)  ──drain──►  HS3 (<25%)
     ◄──── QR feed (+10 / +25 / +45 %) ──────────────────────
```

Drains to 0 over 5 hours. HS string broadcast at 1 Hz to all connected subscribers.

---

## 8. Shared Memory & Databases

All persistent state lives in `memory/` and `data_collection/`.

### JSON Memory Files

Written atomically via `os.replace()` on a temp file. `memory/greeted_today.json`
also uses a shared companion lock file (`greeted_today.json.lock`) so
cross-process read-modify-write updates from `faceSelector` and
`interactionManager` serialize correctly.

| File | Owner | Content |
|---|---|---|
| `memory/learning.json` | `faceSelector` R+W | Per-person LS + `updated_at` |
| `memory/greeted_today.json` | Both R+W | ISO timestamp of each person's last greeting today |
| `memory/talked_today.json` | `faceSelector` R+W | ISO timestamp of each person's last conversation today |
| `memory/last_greeted.json` | `interactionManager` writes · `faceSelector` reads | Latest greeting record per person |

### SQLite Databases

| File | Owner | Tables |
|---|---|---|
| `data_collection/face_selector.db` | `faceSelector` | `target_selections` · `ss_changes` · `ls_changes` |
| `data_collection/interaction_manager.db` | `interactionManager` | `interactions` · `responsive_interactions` |
| `memory/telegram_bot.db` | `telegram_bot` writes · `interactionManager` reads | `subscribers` · `chat_memory` · `user_memory` · `meta` |

### Prompts File

`prompts.json` (workspace root) stores all LLM prompt templates and fixed speech strings. Loaded at startup by `interactionManager` (`"interactionManager"` key) and `telegram_bot` (`"telegram_bot"` key). Hot-reloadable in `telegram_bot` via `reload_prompts` RPC without restarting.

---

## 9. Threading Architecture

### `faceSelector`

```
Main thread  (updateModule @ 20 Hz)
├── _io_thread            →  async JSON writes (queue-driven)
├── _db_thread            →  async SQLite writes (queue-driven)
├── _lg_refresh_thread    →  re-reads last_greeted.json at 5 Hz
├── _prewarm_thread       →  pre-warms RPC connections at startup (one-shot)
└── interaction_thread    →  spawned per interaction; RPC call + result processing
```

**Locks:** `state_lock` · `_interaction_lock` · `_memory_lock` · `_last_greeted_lock`

### `interactionManager`

```
Main thread  (updateModule @ 1 Hz)   →  hunger.update() + hunger broadcast
RPC handle thread                    →  respond() (YARP-managed)
├── _landmarks_reader_thread    →  continuously parses /landmarks:i
├── _db_thread                  →  async SQLite writes
├── _qr_reader_thread           →  QR scanning @ ~50 fps
├── _responsive_thread          →  watches STT for user-initiated greetings
├── _prewarm_thread             →  pre-warms RPC connections (one-shot)
└── [per interaction]:
      ├── _monitor_thread       →  target monitor @ 15 Hz
      └── LLM future            →  single-slot ThreadPoolExecutor
```

**Primitives:** `run_lock` · `abort_event` · `_responsive_active` · `_feed_condition` · `_faces_lock`

### `telegram_bot`

```
Main thread  (updateModule @ 10 Hz)
├── _read_hunger()             →  drains /hunger:i (non-blocking)
├── _process_tg_updates()      →  drains update queue (max 25/cycle); LLM calls synchronous
└── _maybe_hs3_broadcast()     →  fires alerts when HS3 is active

RPC handle thread              →  respond() (YARP-managed)
_tg_poll_loop (daemon)         →  long-polls Telegram getUpdates every 20 s
```

---

## 10. Key Constants

### `faceSelector`

| Constant | Value | Purpose |
|---|---|---|
| `period` | 0.05 s | Main loop (20 Hz) |
| `cooldown_lively` | 3.0 s | Cooldown for STM label = 1 (active scene) |
| `cooldown_calm` | 15.0 s | Cooldown for STM label = 0 (quiet scene) |
| `cooldown_default` | 5.0 s | Cooldown for STM label = −1 (unknown) |
| `DISAPPEAR_WINDOW_SEC` | 30 s | Window for counting face_disappeared events |
| `DISAPPEAR_THRESHOLD` | 2 | Events before harsh −2 penalty |
| `LS_MIN_TIME_IN_VIEW (LS1)` | 2.0 s | Min dwell time for LS1 gate |
| `LS_MIN_TIME_IN_VIEW (LS2)` | 1.0 s | Min dwell time for LS2 gate |

### `interactionManager`

| Constant | Value | Purpose |
|---|---|---|
| `period` | 1.0 s | Main loop (hunger update) |
| `SS1_STT_TIMEOUT` | 10 s | Wait for greeting response in SS1 |
| `SS2_GREET_TIMEOUT` | 10 s | Wait for greeting response in SS2 |
| `SS3_STT_TIMEOUT` | 12 s | Wait per conversation turn |
| `SS3_MAX_TURNS` | 3 | Max conversation turns |
| `LLM_TIMEOUT` | 60 s | Max LLM wait |
| `TARGET_LOST_TIMEOUT` | 12 s | Absence before declaring target lost |
| `MONITOR_HZ` | 15 Hz | Target monitor polling rate |
| `RESPONSIVE_GREET_COOLDOWN_SEC` | 10 s | Per-name cooldown for reactive greetings |
| `TTS_WORDS_PER_SECOND` | 3.0 | Used to estimate speech duration |

### `telegram_bot`

| Constant | Value | Purpose |
|---|---|---|
| `MODULE_HZ` | 10 Hz | Main loop rate |
| `HS_STALE_SEC` | 60 s | Fall back to HS1 if hunger not updated within window |
| `MAX_HISTORY_TURNS` | 10 | Rolling message window (20 messages total) |
| `SUMMARY_EVERY_TURNS` | 8 | Regenerate summary every N turns |
| `HS3_BROADCAST_COOLDOWN_SEC` | 1800 s | Per-subscriber HS3 re-broadcast cooldown |
| `HS3_SKIP_RECENT_SEC` | 600 s | Skip re-broadcast if subscriber chatted recently |
| `HS2_HUNGER_EVERY_N` | 3 | Force hunger comment after N messages (HS2) |

---

## 11. YARP Connection Commands

```bash
# ── Perception → modules ─────────────────────────────────────────────
yarp connect /alwayson/vision/landmarks:o  /faceSelector/landmarks:i
yarp connect /icub/camcalib/left/out       /faceSelector/img:i

yarp connect /alwayson/vision/landmarks:o  /interactionManager/landmarks:i
yarp connect /speech2text/text:o           /interactionManager/stt:i
yarp connect /icub/cam/left                /interactionManager/camLeft:i
yarp connect /interactionManager/speech:o  /acapelaSpeak/speech:i

# ── Hunger broadcast ─────────────────────────────────────────────────
yarp connect /interactionManager/hunger:o  /telegramBot/hunger:i

# ── RPC testing ──────────────────────────────────────────────────────
echo "status"              | yarp rpc /interactionManager
echo "run 3 Alice ss2"     | yarp rpc /interactionManager
echo "hunger hs3"          | yarp rpc /interactionManager   # force starving

echo "status"              | yarp rpc /telegramBot/rpc
echo "set_hs HS3"          | yarp rpc /telegramBot/rpc
echo "reload_prompts"      | yarp rpc /telegramBot/rpc
```

> `faceSelector` auto-connects its own RPC ports to `interactionManager` at startup, and auto-wires `/faceSelector/context:i` ← `/alwayson/stm/context:o`. No manual wiring needed for those.
