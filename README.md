# Always On Cognitive Architecture — `Embodied Behaviour` 
> **Modules:** `faceSelector.py` · `interactionManager.py`
> **Platform:** YARP (Yet Another Robot Platform)  
> **Robot:** iCub
> **Author:** Nima Abaeian
--

## Table of Contents

1. [Overview](#1-overview)
2. [System Block Diagram](#2-system-block-diagram)
3. [Module: `faceSelector`](#3-module-faceselector)
   - [Purpose](#31-purpose)
   - [YARP Ports](#32-yarp-ports)
   - [State Model](#33-state-model)
   - [Main Loop — updateModule()](#34-main-loop--updatemodule)
   - [Face Selection Policy](#35-face-selection-policy)
   - [Interaction Trigger Flow](#36-interaction-trigger-flow)
   - [Reward & Learning State Updates](#37-reward--learning-state-updates)
   - [Background Threads](#38-background-threads)
   - [Persistent Data Files](#39-persistent-data-files)
   - [SQLite Logging](#310-sqlite-logging)
   - [Image Annotation & Visualization](#311-image-annotation--visualization)
4. [Module: `interactionManager`](#4-module-interactionmanager)
   - [Purpose](#41-purpose)
   - [YARP Ports](#42-yarp-ports)
   - [State Trees (Interaction Flows)](#43-state-trees-interaction-flows)
   - [SS1 — Unknown Person](#44-ss1--unknown-person)
   - [SS2 — Known, Not Greeted](#45-ss2--known-not-greeted)
   - [SS3 — Known, Greeted, Not Talked](#46-ss3--known-greeted-not-talked)
   - [Hunger / QR Feeding Tree](#47-hunger--qr-feeding-tree)
   - [Target Monitor](#48-target-monitor)
   - [Responsive Interaction Path](#49-responsive-interaction-path)
   - [LLM Integration (Azure OpenAI)](#410-llm-integration-azure-openai)
   - [Speech Output (TTS)](#411-speech-output-tts)
   - [STT (Speech-to-Text) Input](#412-stt-speech-to-text-input)
   - [HungerModel](#413-hungermodel)
   - [RPC Interface](#414-rpc-interface)
   - [Database](#415-database)
5. [Cross-Module Data Flow](#5-cross-module-data-flow)
6. [State Transition Diagrams](#6-state-transition-diagrams)
   - [Social State Machine](#61-social-state-machine)
   - [Learning State Machine](#62-learning-state-machine)
7. [Threading Architecture](#7-threading-architecture)
8. [Memory Files Reference](#8-memory-files-reference)
9. [Key Constants Reference](#9-key-constants-reference)
10. [YARP Connection Commands](#10-yarp-connection-commands)

---

## 1. Overview

The`Embodied Behaviour` Module in the `AlwaysOn Cognitive Architecture` system implements a **adaptive social interaction architecture** for the iCub robot. 

The Embodied Behaviour has two tightly coupled modules:

| **`faceSelector`** |
| **`interactionManager`** |
---

## 2. System Block Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                      alwaysOn Architecture                           │
└──────────────────────────────────────────────────────────────────────┘

      Perception (faces, image)                     User inputs
                 │                                  (speech, QR)
                 │                                        │
                 v                                        │
┌──────────────────────────────────────────────────────────────────────┐
│ faceSelector                                                         │
│ - Parse face observations (id, bbox, attention, distance)            │
│ - Compute Social State (SS) and Learning State (LS)                  │
│ - Apply eligibility + cooldown rules                                 │
│ - Select biggest-bbox face as the active target                      │
│ - Trigger proactive interaction and update learning/memory           │
└──────────────────────────────────────────────────────────────────────┘
                 │
                 │ RPC call: run(track_id, face_id, ss)
                 v
┌──────────────────────────────────────────────────────────────────────┐
│ interactionManager                                                   │
│ - Run interaction trees (SS1/SS2/SS3; SS4 no-op)                     │
│ - Run hunger feed tree (HS override) with QR feeding events          │
│ - Monitor target continuity (still visible + still biggest)          │
│ - Use LLM for name extraction and conversation generation            │
│ - Coordinate STT/TTS + behavior execution                            │
│ - Handle responsive greetings/QR acknowledgments when proactive idle │
│ - Drive robot actions (speech + behaviors)                           │
└──────────────────────────────────────────────────────────────────────┘
                 │
                 │ compact JSON result (success, final_state, abort...)
                 └──────────────────────────────► back to faceSelector

┌──────────────────────────────────────────────────────────────────────┐
│ State Machines                                                       │
│ - SS (Social): ss1 / ss2 / ss3 / ss4                                 │
│   computed in faceSelector, executed in interactionManager trees     │
│ - LS (Learning): LS1 <-> LS2 <-> LS3                                 │
│   maintained in faceSelector via reward updates                      │
│ - HS (Hunger): HS1 <-> HS2 <-> HS3                                   │
│   maintained in interactionManager (can override social tree)        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ Shared Memory + Logs                                                 │
│ - JSON memory (learning, greeted/talked, last_greeted)               │
│ - SQLite logs (faceSelector + interactionManager records)            │
└──────────────────────────────────────────────────────────────────────┘
faceSelector ─────────────── read/write ───────────────┐
                                                        ├── memory
interactionManager ───────── read/write ───────────────┘

```
---

## 3. Module: `faceSelector`

### 3.1 Purpose

`faceSelector` is the **perception and selection layer**. It:
- Reads raw face landmark data from the vision system
- Parses Social State (SS), Learning State (LS), distance, attention, and bounding box per face
- Selects **one target**: always the face with the **biggest bounding box**
- Waits for that face's identity (`face_id`) to be resolved before proceeding
- Checks eligibility gates (cooldown, LS constraints, SS state)
- Fires off an interaction thread that calls `interactionManager` via RPC
- Processes the interaction result to update learning states and social memory

### 3.2 YARP Ports

| Port | Type | Direction | Purpose |
|---|---|---|---|
| `/faceSelector/landmarks:i` | BufferedPortBottle | IN | Face landmark data from vision |
| `/faceSelector/img:i` | BufferedPortImageRgb | IN | Camera image for annotation |
| `/faceSelector/img:o` | BufferedPortImageRgb | OUT | Annotated camera image |
| `/faceSelector/debug:o` | Port | OUT | Debug status bottle |
| `/faceSelector/interactionManager:rpc` | RpcClient | OUT | Trigger interactions |
| `/faceSelector/interactionInterface:rpc` | RpcClient | OUT | Send `ao_start`/`ao_stop` signals |

### 3.3 State Model

#### Social States (SS)

| State | Meaning | Condition |
|---|---|---|
| `ss1` | **Unknown** | `face_id` not recognized |
| `ss2` | **Known, Not Greeted** | Known person, not greeted today |
| `ss3` | **Known, Greeted, No Talk** | Known, greeted today, no conversation yet |
| `ss4` | **Known, Greeted, Talked** | Full interaction completed today (no further action) |

```
is_known? ─── NO ──────────────────► ss1
              │
             YES
              ├── greeted_today? ─── NO ──► ss2
              │
             YES
              ├── talked_today? ─── NO ──► ss3
              │
             YES ────────────────────────► ss4  (no-op)
```

#### Learning States (LS)

| State | Description | Distance allowed | Attention required |
|---|---|---|---|
| `LS1` | Early — strict constraints | `SO_CLOSE`, `CLOSE`, `FAR` | `MUTUAL_GAZE` only |
| `LS2` | Developing — relaxed | `SO_CLOSE`, `CLOSE`, `FAR`, `VERY_FAR` | `MUTUAL_GAZE`, `NEAR_GAZE` |
| `LS3` | Advanced — no constraints | Any | Any |

LS values are **per-person** and stored in `learning.json`. They evolve via **reward shaping** after each interaction.

#### Eligibility Check (`_is_eligible`)

A face is eligible for interaction if:
- It is **not** `ss4`
- If `LS3` → always eligible
- If `LS1` or `LS2` → must satisfy distance AND attention constraints for that LS level

### 3.4 Main Loop — `updateModule()`

Runs at **20 Hz** (period = 0.05 s). Steps each cycle:

```
1. Check ports connected (landmarks + img). Wait if not.
2. Day-change check → reload memory JSON from disk, then prune greeted_today / talked_today if new day
3. _read_landmarks()     → parse face bottles from YARP
4. _read_image()         → get camera frame (with frame skip)
5. _compute_face_states()→ enrich each face with SS, LS, eligibility, last_greeted_ts
6. _select_biggest_face()→ find face with max bbox area

   IF face_id NOT resolved (still "recognizing"/"unmatched") → WAIT, do not fall back

   IF resolved AND not in cooldown:
     └─ IF eligible AND ss != ss4 AND interactionManager status is available+idle:
          → set interaction_busy = True
          → start _run_interaction_thread(candidate)
        ELSE:
          → skip proactive spawn this cycle

7. Annotate & publish image
8. Publish debug bottle
```

### 3.5 Face Selection Policy

> **Rule:** Always pick the biggest bbox. Never fall back to a smaller resolved face.

```python
biggest = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])

if not _is_face_id_resolved(biggest['face_id']):
    # wait — do not switch to a different face
    return  

# Proceed with the biggest resolved face
```

**Cooldown key logic:**
- Known person → key = `person_id` (e.g., `"Alice"`)
- Unknown person → key = `f"unknown:{track_id}"`
- Cooldown duration: **5.0 seconds** between interactions with the same person

### 3.6 Interaction Trigger Flow

```
updateModule proactive trigger
  ├─ Select biggest resolved face (cooldown + eligibility checks)
  ├─ Pre-spawn check interactionManager status
  │     └─ If unavailable or busy → skip spawn
  └─ Spawn _run_interaction_thread(target)

_run_interaction_thread(target)
  ├─ Re-check interactionManager status → skip if unavailable/busy
  ├─ Skip if ss4
  ├─ _execute_interaction_interface("ao_start") → signal the robot body
  ├─ _run_interaction_manager(track_id, face_id, ss)
  │     └─ RPC: "run <track_id> <face_id> <ss>"
  │     └─ Returns compact JSON result
  ├─ _process_interaction_result(result, target)
  │     ├─ Update greeted_today / talked_today
  │     ├─ Update track_to_person mapping
  │     ├─ Compute reward delta
  │     └─ Update learning state (LS)
  └─ _execute_interaction_interface("ao_stop")
  
  [finally]
  └─ interaction_busy = False
  └─ Clear selected target/bbox and update cooldown timestamp
```

### 3.7 Reward & Learning State Updates

#### Reward Computation (`_compute_reward`)

| Scenario | Reward |
|---|---|
| Success + name extracted | `+2` |
| Success (no name) | `+1` |
| Failure: `not_responded` | `-1` |
| Failure: `face_disappeared` (first time in 30s window) | `-1` |
| Failure: `face_disappeared` (repeated ≥ 2 times in 30s) | `-2` |
| Other failure | `-1` |

#### Learning State Update (`_update_learning_state`)

```
delta > 0  →  new_ls = min(3, current_ls + 1)   (advance)
delta < 0  →  new_ls = max(1, current_ls - 1)   (regress)
delta = 0  →  no change
```

Changes are logged to SQLite (`ls_changes` table) and persisted to `learning.json`.

#### Face Disappear Penalty Tracking

Uses a **sliding window** per person:
- Window: 30 seconds
- Threshold: 2 events
- Below threshold → mild penalty (`-1`)
- At/above threshold → harsh penalty (`-2`)

### 3.8 Background Threads

| Thread | Function | Purpose |
|---|---|---|
| `_io_thread` | `_io_worker()` | Drains `_io_queue` → saves JSON files asynchronously |
| `_db_thread` | `_db_worker()` | Drains `_db_queue` → writes SQLite records |
| `_lg_refresh_thread` | `_last_greeted_refresh_loop()` | Re-reads `last_greeted.json` every 0.2s (5 Hz) |
| `_prewarm_thread` | `_prewarm_rpc_connections()` | Pre-warms RPC connections at startup (one-time, daemon) |
| `interaction_thread` | `_run_interaction_thread()` | Spawned per interaction, runs the RPC call |

### 3.9 Persistent Data Files

| File | Content | Format |
|---|---|---|
| `memory/learning.json` | Per-person LS values + updated_at | `{"people": {"Alice": {"ls": 2, "updated_at": "..."}}}` |
| `memory/greeted_today.json` | ISO timestamps of today's greetings | `{"Alice": "2026-02-26T09:30:00+01:00"}` |
| `memory/talked_today.json` | ISO timestamps of today's conversations | `{"Alice": "2026-02-26T09:35:00+01:00"}` |
| `memory/last_greeted.json` | Latest greeted entry per person | `{"Alice": {"timestamp": "...", "track_id": 3, ...}}` |

All writes are **atomic** (written to a temp file then `os.replace()`).

### 3.10 SQLite Logging

**Database:** `data_collection/face_selector.db`

| Table | Logged event | Key columns |
|---|---|---|
| `target_selections` | Biggest-bbox candidate after face_id resolves and cooldown passes (logged before eligibility/ss4 gate) | `track_id`, `face_id`, `person_id`, `bbox_area`, `ss`, `ls`, `eligible` |
| `ss_changes` | Social state transitions | `person_id`, `old_ss`, `new_ss` |
| `ls_changes` | Learning state transitions | `person_id`, `old_ls`, `new_ls`, `reward_delta` |

### 3.11 Image Annotation & Visualization

Every face is drawn with:
- **Green box** → currently active interaction target
- **Yellow box** → eligible (ready for interaction)
- **White box** → present but not eligible

Labels drawn above each box:
```
Alice (T:3)                    ← person_id + track_id
ss2 | LS2 | LG:09:30           ← social state, learning state, last greeted time
CLOSE/MUT                      ← distance / attention (3-char)
area=12400                     ← bbox area in pixels²
```

Status overlay (top-left): `Status: BUSY | Faces: 2`

---

## 4. Module: `interactionManager`

### 4.1 Purpose

`interactionManager` is the **dialogue and behavior execution layer**. It:
- Receives RPC commands from `faceSelector` (`run <track_id> <face_id> <state>`)
- Executes the appropriate **social state tree** (SS1/SS2/SS3)
- Uses **Azure OpenAI (via LangChain)** for natural language generation and name extraction
- Listens to **STT** for user responses
- Sends **TTS speech** through YARP
- Continuously monitors if the target face is still the biggest (abort if not)
- Handles **responsive interactions** (user-initiated greetings, QR feeding)
- Manages the robot's **hunger model**

### 4.2 YARP Ports

| Port | Type | Direction | Purpose |
|---|---|---|---|
| `/interactionManager` | Port (RPC) | IN | Main RPC handle (run / status / quit) |
| `/interactionManager/landmarks:i` | BufferedPortBottle | IN | Face data for target monitoring |
| `/interactionManager/stt:i` | BufferedPortBottle | IN | Speech-to-text transcripts |
| `/interactionManager/speech:o` | Port | OUT | TTS text → Acapela speaker |
| `/interactionManager/camLeft:i` | BufferedPortImageRgb | IN | Camera for QR code reading |

**Lazy RPC Clients (created on first use):**

| Client connects to | Purpose |
|---|---|
| `/interactionInterface` | Send `exe <behaviour>` commands (ao_start, ao_stop, ao_hi, ...) |
| `/objectRecognition` | Submit `name <name> id <track_id>` for face labeling |

### 4.3 State Trees (Interaction Flows)

The module supports 4 social states. Only SS1–SS3 have active trees:

| State | Tree | What happens |
|---|---|---|
| `ss1` | `_run_ss1_tree` | Greet unknown → ask name → extract name → register |
| `ss2` | `_run_ss2_tree` | Greet by name → wait response → chain to SS3 |
| `ss3` | `_run_ss3_tree` | Proactive conversation starter → up to 3 turns |
| `ss4` | no-op | Immediately returns success |

**Hunger override:** If hunger state is `HS3` (starving), or `HS2` + `ss3`, the **hunger feed tree** runs instead of the social tree.

### 4.4 SS1 — Unknown Person

```
┌─────────────────────────────────────────────────────┐
│ SS1: Unknown Person                                 │
│                                                     │
│  ① Run behaviour: ao_hi                            │
│  ② Wait STT response (10s)                         │
│      └─ No response → ABORT: no_response_greeting   │
│  ③ Say "We have not met, what's your name?"        │
│  ④ Wait STT response (10s)                         │
│      └─ No response → ABORT: no_response_name       │
│  ⑤ Extract name (regex + LLM fallback)             │
│      └─ Fail → Say "Sorry, I didn't catch that"     │
│            └─ Retry once                            │
│            └─ Fail → ABORT: name_extraction_failed  │
│  ⑥ Register name via /objectRecognition RPC        │
│  ⑦ Write last_greeted.json                         │
│  ⑧ Say "Nice to meet you"                          │
│                                                     │
│  Result: success=True, final_state=ss3              │
└─────────────────────────────────────────────────────┘
```

**Name extraction pipeline:**
1. **Fast regex:** patterns like `"My name is X"`, `"I'm X"`, `"Call me X"` (supports apostrophes/hyphens in names)
2. **LLM fallback (Azure GPT-5 nano):** strict JSON extraction with schema validation and confidence clamped to `[0.0, 1.0]`

### 4.5 SS2 — Known, Not Greeted

```
┌─────────────────────────────────────────────────────┐
│ SS2: Known, Not Greeted                             │
│                                                     │
│  ① Say "Hello <name>"    (attempt 1)               │
│  ② Wait STT response (10s)                         │
│      └─ Responded → write last_greeted              │
│              → final_state=ss3                      │
│              → chain to _run_ss3_tree()             │
│      └─ No response → retry once                    │
│  ③ Say "Hello <name>"    (attempt 2)               │
│  ④ Wait STT response (10s)                         │
│      └─ Responded → chain to ss3                    │
│      └─ No response → ABORT: no_response_greeting   │
└─────────────────────────────────────────────────────┘
```

Validates `face_id` is a real name (not `"unknown"`, `"unmatched"`, or a digit).

### 4.6 SS3 — Known, Greeted, Not Talked

```
┌─────────────────────────────────────────────────────┐
│ SS3: Short Conversation (max 3 turns)               │
│                                                     │
│  ① Use cached LLM-generated starter question       │
│     (pre-fetched in background, e.g. "How's your    │
│      day going?")                                   │
│  ② Say the starter                                 │
│  ③ Schedule next background starter prefetch       │
│                                                     │
│  Loop (up to 3 turns):                              │
│    ├─ Wait STT response (12s)                       │
│    │    └─ No response → end loop                   │
│    ├─ Turn 1 or 2: LLM generate follow-up           │
│    ├─ Turn 3 (last): LLM generate closing ack       │
│    └─ Say robot's reply                             │
│                                                     │
│  ≥1 response → talked=True, final_state=ss4         │
│  0 responses → ABORT: no_response_conversation      │
└─────────────────────────────────────────────────────┘
```

Note: `SS3_MAX_TIME = 120.0s` is defined in code but currently not enforced in the SS3 loop.

### 4.7 Hunger / QR Feeding Tree

Triggered when `HungerModel` reports the robot is hungry/starving. Overrides the social tree.

```
Hunger States:
  HS1: level ≥ 60%   (satisfied)
  HS2: 25% ≤ level < 60%  (hungry)
  HS3: level < 25%   (starving)

Trigger conditions:
  HS3 → always replaces SS1/SS2/SS3
  HS2 + ss3 → replaces SS3

Flow:
  ① Say "I'm so hungry, would you feed me please?"
  Loop:
    ├─ Wait for QR scan event (8s timeout)
    ├─ Fed → say "Yummy, thank you so much."
    │        → if HS1 → break (satisfied)
    │        → else → say "I'm still hungry. Give me more."
    └─ Timeout handling → prompt once, then abort on next consecutive timeout
  
  QR Mapping:
    SMALL_MEAL  → +10 hunger
    MEDIUM_MEAL → +25 hunger
    LARGE_MEAL  → +45 hunger

  Timeout policy:
    1st timeout → say "Take a look around, you will find some food for me."
    2nd consecutive timeout → ABORT: no_food_qr
    (timeout counter resets after any successful feed)

  Result: success if ≥1 meal eaten; final_state unchanged (no ss promotion)
```

The **QR reader** runs in its own daemon thread (`_qr_reader_loop`), reading from `/interactionManager/camLeft:i` at ~50fps using `cv2.QRCodeDetector`.

### 4.8 Target Monitor

A dedicated thread runs **alongside every interaction** (at 15 Hz) checking that the interaction target remains:
1. **Still visible** in the landmarks stream
2. **Still the biggest-bbox face**

```
_target_monitor_loop(track_id, result):
  Loop at 15 Hz:
    ├─ Parse latest landmarks
    ├─ Find face with track_id
    ├─ If found:
    │    └─ If another face is now biggest → ABORT: target_not_biggest
    └─ If not found:
         └─ Wait TARGET_LOST_TIMEOUT (3.0s)
         └─ Still missing → ABORT: target_lost
```

**Starvation guard:** If the monitor thread was blocked for >1.5s (GIL starvation), it resets its lost-timer silently to avoid false aborts.

Abort reasons cascade: the monitor sets `abort_event`, which every STT wait loop, speak-and-wait, and LLM future poll checks.

### 4.9 Responsive Interaction Path

The responsive path handles **user-initiated events** that arise independently of the proactive cycle:

#### Responsive Greeting
- **Trigger:** User says `"hello"`, `"hi"`, `"ciao"`, `"good morning"` (matched by regex word boundary search)
- **Condition:** Biggest-bbox known face (no gaze requirement—utterance itself is sufficient signal)
- **Cooldown:** 10 seconds per name
- **Action:** Say `"Hi <name>"` + write `last_greeted`

#### Responsive QR Acknowledgment
- **Trigger:** QR scan detected (outside of a proactive interaction)
- **Action:** Say `"yummy, thank you"`

**Safety:** Responsive interactions are **dropped** (not deferred) if a proactive interaction is running. The `run_lock` and `_responsive_active` event prevent any concurrency conflicts.

### 4.10 LLM Integration (Azure OpenAI)

**Backend:** Azure OpenAI via `langchain_openai.AzureChatOpenAI`  
**Env loading order:** `load_dotenv()` then `memory/llm.env` (`override=False`)  
**Request timeout:** `LLM_TIMEOUT = 60.0s`  
**Retries:** 3 attempts with 1s delay

**Required environment variables (validated at startup):**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `OPENAI_API_VERSION` (or `AZURE_OPENAI_API_VERSION`)

**Deployments (with defaults):**
- `AZURE_DEPLOYMENT_GPT5_NANO` → `contact-Yogaexperiment_gpt5nano` (JSON/name extraction)
- `AZURE_DEPLOYMENT_GPT5_MINI` → `contact-Yogaexperiment_gpt5mini` (conversation text generation)

**Routing and options:**
- `_llm_json(...)` requests route to the `gpt5-nano` client.
- Conversational generation routes to the `gpt5-mini` client.
- `options["num_predict"]` is mapped to `max_completion_tokens`.
- Temperature is not passed (GPT-5 deployment constraint in code comments).

| LLM Function | Purpose | Key params |
|---|---|---|
| `_llm_generate_convo_starter` | One short wellbeing/day question | `num_predict`→`max_completion_tokens` (2000) |
| `_llm_generate_followup` | Short sentiment-aware follow-up (≤22 words) | `num_predict`→`max_completion_tokens` (2000) |
| `_llm_generate_closing_acknowledgment` | Warm short closing (4–8 words, no question) | `num_predict`→`max_completion_tokens` (2000) |
| `_llm_extract_name` | JSON name extraction with schema | `num_predict`→`max_completion_tokens` (2000), strict JSON system prompt |

**LLM thread pool:** Single-worker `ThreadPoolExecutor` — one LLM call at a time.  
Futures are polled with `_await_future_abortable()` which checks `abort_event` every 100ms and can cancel the future early.

**Startup:** On `configure()`, the module:
1. Creates Azure clients via `setup_azure_llms()` (nano + mini)
2. Pre-fetches a conversation starter in the background
3. Pre-warms RPC connections (background thread sends `status` pings to avoid TCP setup latency on first interaction)

### 4.11 Speech Output (TTS)

```python
_speak(text)          → writes Bottle to /interactionManager/speech:o
_speak_and_wait(text) → speak() + estimated wait based on word count
```

**Wait estimation:**
```
wait = word_count / 3.0 + 0.5   (words_per_second=3.0, end_margin=0.5)
wait = clamp(wait, 1.0, 8.0)
```

During `speak_and_wait`, the abort event is checked every 100ms so the robot can be interrupted mid-speech.

### 4.12 STT (Speech-to-Text) Input

Reads from `/interactionManager/stt:i` (connected to `/speech2text/text:o`).

```python
_wait_user_utterance_abortable(timeout):
  Loop until timeout:
    ├─ Check abort_event → return None if set
    ├─ Read stt_port (non-blocking)
    ├─ If text → return stripped text
    └─ sleep 0.1s
    
    GIL compensation: if loop body took >0.5s, extend timeout by that amount
```

**Buffer clearing** (`_clear_stt_buffer`): Done before each expected utterance to discard stale transcripts.

### 4.13 HungerModel

Simulates the robot's "hunger" as a level from 0–100:

```python
HungerModel(drain_hours=6.0, hungry_threshold=60.0, starving_threshold=25.0)

update()  → decrements level based on elapsed time (drains to 0 in drain_hours)
feed(delta, payload) → increments level (capped at 100)
get_state() → "HS1" (≥60), "HS2" (≥25), "HS3" (<25)
```

Thread-safe via internal `_lock`. Updated every `updateModule()` cycle (1 Hz).

### 4.14 RPC Interface

The module exposes an RPC handle at `/interactionManager`.

**Supported commands:**

| Command | Arguments | Returns |
|---|---|---|
| `run` | `<track_id> <face_id> <ss1\|ss2\|ss3\|ss4>` | Compact JSON result |
| `status` / `ping` | — | `{"success":true, "busy":<bool>, ...}` |
| `help` | — | Command list |
| `quit` | — | Shutdown |

**Concurrency:** `run_lock` (non-blocking acquire) ensures only one interaction runs at a time. If busy, returns `{"error": "Another action is running"}`.

**Compact result format (returned to faceSelector):**
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
  "stomach_level_end": 85.1
}
```

**Abort reason compaction:**
- `target_lost` / `target_not_biggest` / `target_monitor_abort` → `"face_disappeared"`
- Anything else → `"not_responded"`

### 4.15 Database

**File:** `data_collection/interaction_manager.db`

| Table | What it stores |
|---|---|
| `interactions` | Full record of every proactive interaction: states, success, abort, transcript |
| `responsive_interactions` | Responsive greeting and QR feed events |

A background `_db_thread` drains the `_db_queue` to avoid blocking the interaction thread.

---

## 5. Cross-Module Data Flow

```
┌───────────────────────────────────────────────────────────────────────┐
│              Complete Interaction Cycle (Proactive + Responsive)      │
│                                                                       │
│  A) Proactive Path (faceSelector-driven)                              │
│  1. Vision landmarks/image → faceSelector                             │
│  2. faceSelector parses faces + computes SS/LS/eligibility            │
│  3. faceSelector selects biggest resolved face (cooldown-aware)       │
│  4. faceSelector → RPC run(track_id, face_id, ss) → interactionMgr    │
│  5. interactionManager executes tree (SS or hunger-feed override),    │
│     with monitor + STT/TTS + behaviors + optional name registration   │
│  6. interactionManager returns compact JSON result                    │
│  7. faceSelector updates memory/LS/cooldown + DB logs                 │
│                                                                       │
│  B) Responsive Path (interactionManager internal, event-driven)       │
│  R1. STT greeting or QR feed event detected                           │
│  R2. If proactive interaction is running (run_lock busy) → DROP event │
│  R3. If idle → run responsive greeting or responsive QR acknowledgment│
│  R4. Execute behaviors + speech, update greeting memory when needed   │
│  R5. Log event in responsive_interactions DB table                    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 6. State Transition Diagrams

### 6.1 Social State Machine

```
         [ss1: Unknown]
              │
      Greet + Ask Name + Extract Name + Register
              │   Success
              ▼
         [ss3: Known, Greeted, No Talk]
              │
         Conversation (≥1 user turn)
              │   Success
              ▼
         [ss4: Known, Greeted, Talked]    ← TERMINAL (today)
              │
         (next day reset → back to ss2)
              │
              ▼
         [ss2: Known, Not Greeted]
              │
         Say "Hi <name>" + Response received
              │   Success
              └────────────────────────────► [ss3]
```

### 6.2 Learning State Machine

```
         [LS1: Strict]
         ·  SO_CLOSE or CLOSE
         ·  MUTUAL_GAZE only
              │ reward +1 or +2
              ▼
         [LS2: Relaxed]
         ·  Any distance except VERY_FAR
         ·  MUTUAL_GAZE or NEAR_GAZE
              │ reward +1 or +2
              ▼
         [LS3: Advanced]
         ·  No constraints
         ·  Always eligible

         Any state: reward -1 or -2 → regress one level (min LS1)
         Any state: reward +1 or +2 → advance one level (max LS3)
```

---

## 7. Threading Architecture

### `faceSelector` Threads

```
Main thread (updateModule @ 20Hz)
├── _io_thread           → JSON file saves (queue-driven)
├── _db_thread           → SQLite writes (queue-driven)
├── _lg_refresh_thread   → last_greeted.json re-read (5Hz loop)
├── _prewarm_thread      → pre-warm RPC connections at startup (one-time)
└── interaction_thread   → spawned per interaction
      └── (wait for interactionManager RPC, then process result)
```

### `interactionManager` Threads

```
Main thread (updateModule @ 1Hz) → hunger.update()
RPC handle thread                → respond() (YARP managed)
├── _landmarks_reader_thread     → continuously parses /landmarks:i
├── _db_thread                   → async SQLite writes
├── _qr_reader_thread            → camera QR scanning (50fps)
├── _responsive_thread           → watches STT for user-initiated greetings
├── _prewarm_thread              → pre-warm RPC connections at startup (one-time)
└── [per interaction]:
      ├── _monitor_thread        → target monitor (15Hz)
      ├── LLM future             → single-slot ThreadPoolExecutor
      └── behaviour thread       → ao_hi in SS1 (fire-and-forget)
```

**Locks & Events:**

| Primitive | Purpose |
|---|---|
| `state_lock` (faceSelector) | Protect shared runtime state snapshots (`current_faces`, target metadata, cooldown map, etc.) |
| `_interaction_lock` (faceSelector) | Serialize `interaction_busy` transitions and spawn/finalize interaction decisions |
| `_memory_lock` (faceSelector) | Protect memory dicts (`greeted_today`, `talked_today`, `learning_data`) and JSON I/O snapshots |
| `_last_greeted_lock` (faceSelector) | Protect `_last_greeted_snapshot` for background refresh |
| `run_lock` (interactionManager) | Mutual exclusion for interaction execution |
| `abort_event` (interactionManager) | Signal abort to all interaction sub-steps |
| `_responsive_active` (interactionManager) | Prevent overlap of responsive + proactive |
| `_feed_condition` (interactionManager) | Condition variable for QR feed notification |
| `_faces_lock` (interactionManager) | Protect `_latest_faces` shared with monitor |

---

## 8. Memory Files Reference

All files under `modules/alwaysOn/memory/`:

| File | R/W | Owner | Description |
|---|---|---|---|
| `learning.json` | R+W | faceSelector | LS per person, `{"people": {"Alice": {"ls": 2, "updated_at": "..."}}}` |
| `greeted_today.json` | R+W | faceSelector + interactionManager | ISO timestamps of today's greetings |
| `talked_today.json` | R+W | faceSelector | ISO timestamps of today's talks |
| `last_greeted.json` | R+W | interactionManager (write) / faceSelector (read) | Last greeting record per person |

---

## 9. Key Constants Reference

### `faceSelector`

| Constant | Value | Purpose |
|---|---|---|
| `period` | 0.05s | Main loop rate (20 Hz) |
| `interaction_cooldown` | 5.0s | Minimum time between interactions with same person |
| `DISAPPEAR_WINDOW_SEC` | 30.0s | Window for counting face_disappeared events |
| `DISAPPEAR_THRESHOLD` | 2 | Events before harsh penalty kicks in |
| `frame_skip_rate` | 0 | Process every frame (0 = no skip) |

### `interactionManager`

| Constant | Value | Purpose |
|---|---|---|
| `period` | 1.0s | Main loop (hunger update) |
| `SS1_STT_TIMEOUT` | 10.0s | Wait for greeting response (SS1) |
| `SS2_STT_TIMEOUT` | 10.0s | Wait for name response (SS1) |
| `SS2_GREET_TIMEOUT` | 10.0s | Wait for greeting response (SS2) |
| `SS3_STT_TIMEOUT` | 12.0s | Wait per conversation turn (SS3) |
| `SS3_MAX_TURNS` | 3 | Maximum conversation turns |
| `SS3_MAX_TIME` | 120.0s | Defined SS3 total-time cap (currently not enforced in loop) |
| `LLM_TIMEOUT` | 60.0s | Maximum LLM wait |
| `MONITOR_HZ` | 15.0 | Target monitor polling rate |
| `TARGET_LOST_TIMEOUT` | 8.0s | Grace period before declaring target lost |
| `RESPONSIVE_GREET_COOLDOWN_SEC` | 10.0s | Per-name cooldown for reactive greetings |
| `TTS_WORDS_PER_SECOND` | 3.0 | Used to estimate speech duration |

---

## 10. YARP Connection Commands

```bash
# faceSelector
yarp connect /alwayson/vision/landmarks:o  /faceSelector/landmarks:i
yarp connect /icub/camcalib/left/out       /faceSelector/img:i
# (faceSelector auto-connects its RPC ports to interactionManager)

# interactionManager
yarp connect /alwayson/vision/landmarks:o  /interactionManager/landmarks:i
yarp connect /speech2text/text:o           /interactionManager/stt:i
yarp connect /icub/cam/left                /interactionManager/camLeft:i
yarp connect /interactionManager/speech:o  /acapelaSpeak/speech:i

# RPC test
echo "status" | yarp rpc /interactionManager
echo "run 3 Alice ss2" | yarp rpc /interactionManager
```
