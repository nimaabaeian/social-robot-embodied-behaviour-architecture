# Evaluation Logging

The evaluation databases include shared experiment metadata on the main event tables so analyses can filter runs consistently across executive control, salience, and Telegram logs.

## Metadata Fields

- `run_id`: stable identifier for one experimental run. Set `ALWAYSON_RUN_ID` for shared cross-module runs. If it is missing, each module generates one UUID at startup.
- `experiment_condition`: experimental condition. Allowed values are `drive_enabled`, `drive_disabled`, `ablation`, `pilot`, and `unknown`.
- `is_test_run`: `1` for pilot/debug/test runs, otherwise `0`.
- `valid_for_analysis`: `1` for rows intended for analysis. If unset, it defaults to `0` for test runs and `1` otherwise.

## Research Questions

**RQ1.** To what extent does the designed orexigenic drive fulfil the core functions of classical homeostasis, namely internal monitoring, deficit detection, deficit-to-action conversion, and behavioural prioritisation?

**RQ2.** Does the expression of an internal orexigenic deficit promote human caregiving behaviour and recovery-oriented engagement sufficient to support reliable energy recovery in an always-on social robot?

## Evaluation Method

### RQ1 — Homeostatic Control

RQ1 is evaluated as a homeostatic control question across four sub-functions:

1. **Internal monitoring.** Use `hunger_level_events` and `v_drain_segments` to verify continuous internal monitoring and the passive drain rate. The `v_hunger_level_timeline` view provides the full time-series.

2. **Deficit detection.** Use `v_hs_transitions` to check HS1→HS2→HS3 threshold consistency and confirm that state boundaries align with the configured `hungry_threshold` and `starving_threshold`.

3. **Deficit-to-action conversion.** Use `v_active_cost_breakdown` to verify action energy costs. Use feeding rows in `hunger_level_events` to verify QR meal recovery. Deficit-to-action conversion is evaluated from `v_interactions_clean` by conditioning `interaction_tag`, `hunger_state_start`, and recovery outcomes on `experiment_condition`.

4. **Behavioural prioritisation.** Evaluated at two levels. Verbal hunger expression is measured with `v_metric_hunger_mention_rate` and the chatbot `hunger_mentioned` fields. Adaptive prioritisation is measured with `v_homeostatic_learning_changes_clean` and `face_ips_events`, which expose reward-driven IPS weight changes and their downstream salience scores.

### RQ2 — Caregiving and Energy Recovery

RQ2 is evaluated through caregiving and energy recovery. The core measures are:

- **Feeding frequency:** `v_hs3_episodes.meals_received` and `v_interactions_clean.meals_eaten_count`.
- **Time-to-first-feed:** `v_hs3_episodes.time_to_first_feed_sec`.
- **HS3 resolution rate:** `v_hs3_episodes.resolved_by_feeding` and `exit_cause`.
- **Hunger-tree success:** `v_interactions_clean.abort_reason` conditioned on hunger state.

The main comparison is `drive_enabled` versus `drive_disabled`. Chat `hs3_proactive` and `hs3_recovery` events from the chatbot database are channel-specific evidence of expressed deficit and recovery signalling, not proof of cross-channel feeding.

## Run Data Collection

Use one terminal for setup and launch so the robot modules inherit the experiment metadata.

1. Open a terminal in the repository root:

```bash
cd /usr/local/src/robot/cognitiveInteraction/alwaysOn-embodiedBehaviour
```

2. Choose the run labels:

- `run_id`: unique label for this run.
- `experiment_condition`: `drive_enabled`, `drive_disabled`, `ablation`, `pilot`, or `unknown`.

3. Export the matching variables using one of the examples below.

4. Confirm the important values:

```bash
echo "$ALWAYSON_RUN_ID"
echo "$ALWAYSON_EXPERIMENT_CONDITION"
echo "$ALWAYSON_VALID_FOR_ANALYSIS"
```

5. Launch the YARP application from the same terminal:

```bash
yarpmanager --application app/alwaysOn-embodiedBehaviour/scripts/alwaysOn-embodiedBehaviour.xml
```

6. Run the interaction session, then stop the modules normally from YARP Manager.

7. Export clean CSV files:

```bash
python3 scripts/export_evaluation_data.py
```

The runtime SQLite databases are written automatically to:

```text
modules/data_collection/executive_control.db
modules/data_collection/salience_network.db
modules/data_collection/chat_bot.db
```

## Environment

Drive-enabled run:

```bash
export ALWAYSON_RUN_ID="exp01_drive_on_2026_05_11"
export ALWAYSON_EXPERIMENT_CONDITION="drive_enabled"
export ALWAYSON_IS_TEST_RUN="0"
export ALWAYSON_VALID_FOR_ANALYSIS="1"
```

Drive-disabled run:

```bash
export ALWAYSON_RUN_ID="exp01_drive_off_2026_05_11"
export ALWAYSON_EXPERIMENT_CONDITION="drive_disabled"
export ALWAYSON_IS_TEST_RUN="0"
export ALWAYSON_VALID_FOR_ANALYSIS="1"
```

Pilot/debug run:

```bash
export ALWAYSON_RUN_ID="pilot_debug_001"
export ALWAYSON_EXPERIMENT_CONDITION="pilot"
export ALWAYSON_IS_TEST_RUN="1"
export ALWAYSON_VALID_FOR_ANALYSIS="0"
```

## Clean Views

Clean views include only rows where `valid_for_analysis = 1` and expose the metadata fields:

- Executive: `v_interactions_clean`, `v_interaction_turns`, `v_hunger_level_timeline`, `v_hs_transitions`, `v_hs3_episodes`, `v_metric_hunger_mention_rate`, `v_active_cost_breakdown`, `v_drain_segments`
- Salience: `v_interaction_attempts_clean`, `v_target_selections_clean`, `v_homeostatic_learning_changes_clean`, `v_face_ips_timeline`
- ChatBot: `v_chat_events_clean`, `v_chat_messages_clean`, `v_chat_session_metrics`

Quality-control views are available in each database, including missing-metadata and condition-count summaries.

## CSV Export

The export helper does not require YARP:

```bash
python3 scripts/export_evaluation_data.py
```

By default it reads the SQLite files in `modules/data_collection/` and writes CSVs under `modules/data_collection/exports/<run_id-or-timestamp>/`.

The export includes executive homeostasis views, salience interaction/learning views, chat engagement views, and the IPS timeline.

## Quality Checks

After a pilot or real run, inspect the quality views before analysis. The missing-metadata views should return zero rows for final experiment data:

```bash
sqlite3 modules/data_collection/executive_control.db \
  "SELECT * FROM v_quality_interaction_missing_metadata LIMIT 20;"

sqlite3 modules/data_collection/salience_network.db \
  "SELECT * FROM v_quality_salience_missing_metadata LIMIT 20;"

sqlite3 modules/data_collection/chat_bot.db \
  "SELECT * FROM v_quality_chat_missing_metadata LIMIT 20;"
```

Use the condition-count views to confirm that rows are labelled with the expected run and condition:

```bash
sqlite3 modules/data_collection/executive_control.db \
  "SELECT * FROM v_quality_condition_counts;"

sqlite3 modules/data_collection/salience_network.db \
  "SELECT * FROM v_quality_attempt_counts;"

sqlite3 modules/data_collection/chat_bot.db \
  "SELECT * FROM v_quality_chat_condition_counts;"
```
