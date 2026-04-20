# Embodied Behaviour

> **Robot:** iCub
> **Platform:** YARP
> **Author:** Nima Abaeian

Technical reference for the always-on proactive social robot Cognitive Architecture.
Covers perception, selection, interaction, and long-term relationship layers end-to-end.

---

## 1) Big Picture

Four continuously running layers, each with a distinct responsibility:

| Layer | Module | Role |
|---|---|---|
| See | `vision.py` | Perceive faces, pose, gaze, talking, QR |
| Choose | `salienceNetwork.py` | Decide who to look at and who to approach |
| Act | `executiveControl.py` | Run interaction trees, manage hunger |
| Remember | `chatBot.py` | Long-term relationship via Telegram |

### System map

```text
           CAMERA
             |
             v
 +------------------------------+
 | vision.py                    |
 | YOLO + ByteTrack + MediaPipe |
 | gaze / pose / talking / QR   |
 +------------------------------+
             |  /alwayson/vision/landmarks:o
             v
 +------------------------------+
 | salienceNetwork.py           |
 | IPS scoring + SS assignment  |
 | two-layer arbitration        |
 +------------------------------+
             |  RPC run(track_id, face_id, ss)
             v
 +------------------------------+
 | executiveControl.py          |
 | SS trees + hunger tree       |
 | TTS / STT / QR feed / LLM    |
 +------------------------------+
             |  /alwayson/executiveControl/hunger:o
             v
 +------------------------------+
 | chatBot.py                   |
 | Telegram LLM + user memory   |
 | hunger-aware persona         |
 +------------------------------+
```

---

## 2) End-to-End Dataflow

```text
[Frame arrives]
  -> vision.py drains backlog, keeps freshest frame
  -> YOLO detects faces, ByteTrack assigns stable track_id
  -> MediaPipe computes head pose, gaze direction
  -> lip motion analysis decides is_talking
  -> face_recognition matches known identities (with sticky retry)
  -> QR detector (throttled, once per 10 frames) emits new codes
  -> landmarks bottle published per face

[salienceNetwork loop @ 20 Hz]
  -> reads landmark stream
  -> computes IPS score per face
  -> assigns social state (ss1..ss4) from memory files
  -> selects attention target (gaze) continuously
  -> when gates pass: spawns interaction thread → RPC run()

[executiveControl on run()]
  -> starts target monitor thread (abort if face gone > 12s)
  -> chooses behavior path: hunger tree or social SS tree
  -> TTS/STT exchanges, synchronous starter + latest-only async LLM follow-up replies
  -> writes replied/greeted/talked + turn-depth/trigger/hunger analytics to SQLite
  -> publishes hunger state continuously

[chatBot main loop @ 10 Hz]
  -> reads hunger port
  -> drains Telegram update queue (up to 25/cycle)
  -> generates hunger-aware LLM replies
  -> groups messages into inactivity-based sessions (30 min gap)
  -> extracts user profile from message text
  -> broadcasts HS3 starvation alerts to subscribers
```

### Conceptual model

```text
Perception (high rate) → Selection gate → Interaction transaction → Memory update
    continuous              opportunistic        bounded/abortable       persistent
```

---

## 3) Module Details

## 3.1 `vision.py` — Perception Front-End

### What it does

Processes the camera stream and publishes high-fidelity per-face descriptors for downstream modules.

- **YOLO face detection** — detects faces with confidence threshold, expands bounding boxes by 10% for better face crops
- **ByteTrack** — maintains stable `track_id` across frames, even through brief occlusions
- **face_recognition matching** — compares embeddings against a `faces/` image database; uses min+median distance for robustness; unknown faces are retried up to 3 times with configurable interval; identity is kept sticky for 1.5 s after a tracker drop
- **MediaPipe Face Landmarker** — computes 3D head pose (pitch/yaw/roll), gaze direction vector, and cosine angle to camera for up to 10 faces
- **Talking detection** — tracks normalized mouth-open values in a 10-frame sliding window; face is flagged `is_talking` if the standard deviation exceeds a threshold
- **QR detection** — throttled to 1 in 10 frames; emits each new QR value exactly once per appearance (hysteresis window suppresses repeat triggers)
- **Target delegation** — receives a lightweight `[track_id, ips_score]` command from `salienceNetwork`, finds the matching bounding box, and forwards a FaceTracker-compatible full bbox bottle
- **Backlog draining** — reads the camera port until no new frames are queued, then processes only the freshest frame to minimize display lag
- **Runtime naming** — `name <person_name> id <track_id>` RPC command enrolls the visible face into the identity database on the fly

### Per-face landmark output

```text
face_id, track_id,
bbox(x,y,w,h), zone, distance,
gaze_direction(x,y,z), pitch, yaw, roll, cos_angle,
attention, is_talking, time_in_view
```

### Perception pipeline

```text
RGB frame (freshest, backlog drained)
  |
  +--> YOLO face detector (conf > 0.7, bbox +10%)
  |       |
  |       +--> ByteTrack → stable track_id
  |       |
  |       +--> face_recognition → face_id
  |              (sticky 1.5s, retry up to 3x)
  |
  +--> MediaPipe Face Landmarker → pose / gaze
  |
  +--> Lip motion std-dev → is_talking flag
  |
  +--> QR detector (1/10 frames) → /alwayson/vision/qr:o
  |
  +--> fused per-face landmark bottle → /alwayson/vision/landmarks:o
```

### Ports and RPC

| Direction | Port | Content |
|---|---|---|
| Input | `/alwayson/vision/img:i` | Raw RGB camera stream |
| Input | `/alwayson/vision/targetCmd:i` | `[track_id, ips]` from salienceNetwork |
| Output | `/alwayson/vision/landmarks:o` | Per-face landmark bottles |
| Output | `/alwayson/vision/features:o` | Scene-level compact features |
| Output | `/alwayson/vision/targetBox:o` | FaceTracker-compatible bbox bottle |
| Output | `/alwayson/vision/faces_view:o` | Annotated debug image |
| Output | `/alwayson/vision/qr:o` | Detected QR payload strings |
| RPC | `/alwayson/vision/rpc` | `name`, `help`, `process`, `quit` |

---

## 3.2 `salienceNetwork.py` — Target Selection and Gating

### What it does

Consumes the face landmark stream and decides two things every loop cycle:

1. **Who to look at** (attention target — continuous)
2. **Who to approach** (dialogue trigger — opportunistic, gated)

It also updates per-person adaptive weights after each interaction outcome.

### Two-layer arbitration

```text
LAYER A — ATTENTION (head/eye gaze target)
  Priority order:
    1) RPC override track_id (set_track_id command)
       — set at start of every interaction (proactive or responsive),
         released (-1) when the interaction ends
    2) Active interaction lock (holds gaze on current partner)
    3) Highest IPS face

LAYER B — DIALOGUE (start proactive interaction)
  All of these must pass:
    - interaction_busy == false
    - no RPC override active
    - candidate social state eligible (IPS >= SS threshold)
    - cooldown passed for this face
    - executiveControl status != busy
```

### IPS score (Interest Priority Score)

The IPS ranks how interesting each face is right now.

**Step 1 — Normalize input signals**

```text
s_prox = clamp(bbox_height / image_height, 0, 1)     # closeness

cx, cy = bbox center
s_cent = clamp(1 - dist_to_image_center / max_dist, 0, 1)  # centrality

s_vel  = clamp((area_now - area_prev) / (W*H) * 10, 0, 1)  # approach speed

s_gaze = max(0, cos_angle)                           # looking toward camera
```

**Step 2 — Weighted sum (per-person or baseline)**

```text
baseline:  w_prox=0.5, w_cent=0.15, w_vel=0.3, w_gaze=0.5

base_ips = w_prox*s_prox + w_cent*s_cent + w_vel*s_vel + w_gaze*s_gaze
```

If a person has been interacted with before, their learned weights replace the baseline.

**Step 3 — Hysteresis bonus**

```text
if face.track_id == current_target:
    ips += 0.3    # stabilizes current target, prevents noisy switching
```

**Step 4 — Habituation decay (conditional)**

Applied only to the current tracked face and only when both are true:
- interaction is not running (stops immediately when any proactive or responsive interaction begins)
- more than one face is visible

```text
t_idle      = min(time_in_view, time_since_last_interaction)
habituation = exp(-0.10 * t_idle)
ips_current = ips_current * habituation
```

This gradually releases gaze lock toward a more novel face.

**Step 5 — Eligibility thresholds by social state**

```text
ss1 (unknown)            IPS >= 1.10  (stricter threshold)
ss2 (known, ungreeted)   IPS >= 0.90  (lowest proactive threshold)
ss3 (greeted, no talk)   IPS >= 1.00  (still proactive, less than ss2)
ss4 (fully talked)       IPS >= 99.0  (never proactively triggered)
```

**Tracking gate (look-only)**

```text
start/switch threshold     = min_track_ips (default 0.9)
stop threshold             = min_track_ips - hysteresis (default 0.8)
stop debounce              = 2.0 s
keepalive resend           = every 0.5 s
```

### Adaptive learning

After each interaction the weights of the involved person shift:

```text
success → increase prox/vel weights, decrease gaze weight
          (reward proximity and approach behavior)

failure → decrease prox/vel weights, increase gaze weight
          (require stronger gaze signal before re-approaching)

shift rate = ±0.15 per interaction
saved to learning.json
```

### Social state assignment

Social state determines which interaction tree runs and sets the IPS threshold.

```text
face_id == "unknown"                              → ss1
face_id known, not in greeted_today.json          → ss2
face_id known, greeted today, not in talked_today → ss3
face_id known, greeted and talked today           → ss4
```

### Context-aware cooldown

An optional STM context stream adjusts how long before re-approaching the same face:

```text
STM label = 1  (lively)   → short cooldown (3 s)
STM label = 0  (calm)     → long cooldown (15 s)
STM missing               → default cooldown (5 s)
```

### Ports and RPC

| Direction | Port | Content |
|---|---|---|
| Input | `/alwayson/salienceNetwork/landmarks:i` | Face landmark bottles from vision |
| Input | `/alwayson/salienceNetwork/context:i` | Optional STM context label |
| Output | `/alwayson/salienceNetwork/targetCmd:o` | `[track_id, ips]` to vision |
| Output | `/alwayson/salienceNetwork/debug:o` | Debug state bottle |
| RPC server | `/salienceNetwork` | `set_track_id`, `reset_cooldown` |
| RPC client | `/salienceNetwork/executiveControl:rpc` | `status`, `run` |
| RPC client | `/salienceNetwork/faceTracker:rpc` | `run` on startup, `sus` on close |

---

## 3.3 `executiveControl.py` — Interaction Engine

### What it does

- Executes social interaction trees for each social state (ss1–ss4)
- Runs a hunger feeding tree when the robot is hungry
- Handles responsive interactions (greeting detection from STT, opportunistic QR feeds)
- Manages the `HungerModel` — a time-draining stomach level that persists across restarts
- Supports `hunger_mode on|off` to enable/disable hunger-driven behavior globally
- Publishes current hunger state to `chatBot`
- Performs cross-channel personalization: looks up the face's name in the Telegram DB to personalize face-to-face conversation with learned preferences (likes, dislikes, topics, inside jokes)
- Uses Azure OpenAI for name extraction and short conversational turns
- Logs compact interaction analytics (e.g., replied-any, hunger context, turn depth, trigger mode) to SQLite

### Behavior routing

```text
run(track_id, face_id, ss)
  |
  +--> ss4?  → no-operation success, return immediately
  |
  +--> resolve face_id (wait up to 5 s if still "recognizing")
  |
  +--> start target monitor (abort if face absent > 12 s)
  |
  +--> check hunger state:
        HS3 (starving)          → hunger tree always
        HS2 (hungry) + ss3      → hunger tree
        otherwise               → social SS tree
  |
  +--> return compact result JSON to salienceNetwork
```

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
| `status` | Returns effective hunger state (`HS0/HS1/HS2/HS3`), subscriber count, queue size, thread health |
| `set_hs HS1\|HS2\|HS3` | Manual override of effective hunger state (bypasses stale protection) |
| `reload_prompts` | Reload `prompts.json` without restart |

**Stale hunger protection:** if no hunger update arrives for 60 s and no manual override is active, effective state falls back to HS0 (hunger drive unavailable) to prevent stale hunger behavior.

---

## 3.5 `prompts.json` — Prompt Surface

Central prompt file loaded by both `executiveControl` and `chatBot` at startup, with runtime reload support.

**`executiveControl` section:**
- `system_default`, `system_json` — base LLM system prompts
- `system_default` now emphasizes emotional mirroring, varied sentence openings, and natural kid-like hesitation phrases
- `ss1_greeting`, `ss1_ask_name`, `ss1_ask_name_retry`, `ss1_nice_to_meet`
- `ss2_greeting`
- `convo_starter_prompt[_hs1|_hs2]`, `followup_prompt[_hs1|_hs2]`, `closing_ack_prompt[_hs1|_hs2]`
- `hunger_ask_feed`, `hunger_still_hungry`, `hunger_look_around`, `feed_ack_hs1|feed_ack_hs2|feed_ack_hs3`
- `reactive_greeting`

**`chat_bot` section:**
- `base_system_prompt`
- `base_system_prompt` now further biases toward natural rhythm, reduced repetition, and context-specific emotional responses
- `hs_overlays.HS0`, `hs_overlays.HS1`, `hs_overlays.HS2`, `hs_overlays.HS3` — layered on top of base
- `hs3_override_system` — strict starvation override injected at the top of the message list
- `hs2_force_hunger_system` — forced hunger comment directive
- `hs3_broadcast_system`, `hs3_broadcast_user` — LLM prompts for HS3 broadcasts
- `hs3_broadcast_fallback` — hard-coded fallback if LLM fails
- `summarize_system` — prompt for periodic history summarization
- `summary_injection` — template to inject summary back into context
- `start_greeting`, `start_greeting_with_name`, `reset_reply`
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
hunger stream fresh and no override  → use raw hunger state from stream
hunger stale (> 60 s) and no override → force effective state to HS0
manual override active               → use overridden state regardless of staleness
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
no fresh hunger update for 60 s (and no manual override)
  → effective hunger forced to HS0
  → hunger-drive overlay is disabled until hunger stream becomes fresh again
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
| `chatBot` | `set_hs HS1\|HS2\|HS3` | Override effective hunger persona |
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
  - main loop (hunger read + update drain + HS3 broadcast)
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
- `chatBot` remains resilient if `executiveControl` disconnects — stale hunger falls back to HS0 after 60 s.
- `executiveControl` can run with hunger mode OFF — interactions proceed as hunger-neutral (`HS1`/100%), and QR feeding events are ignored.
- All LLM calls have fallback strings — no interaction or Telegram reply ever blocks indefinitely on LLM availability.
- All file I/O and DB writes are off the main loop — latency spikes in storage never stall perception or interaction.
- The hunger level survives restarts — `HungerModel` loads `hunger_state.json` on startup and continues draining from the saved level.
