# Evaluation Logging

The evaluation databases include shared experiment metadata on the main event tables so analyses can filter runs consistently across executive control, salience, and Telegram logs.

## Metadata Fields

- `run_id`: stable identifier for one experimental run. Set `ALWAYSON_RUN_ID` for shared cross-module runs. If it is missing, each module generates one UUID at startup.
- `experiment_condition`: experimental condition. Allowed values are `drive_enabled`, `drive_disabled`, `ablation`, `pilot`, and `unknown`.
- `scenario_id`: scenario label such as `lab_hri_short_interaction`. Defaults to `unspecified`.
- `participant_id`: SHA-256 pseudonym. The raw participant label is not stored in this field.
- `is_test_run`: `1` for pilot/debug/test runs, otherwise `0`.
- `valid_for_analysis`: `1` for rows intended for analysis. If unset, it defaults to `0` for test runs and `1` otherwise.

`ALWAYSON_PARTICIPANT_SALT` should be set for real experiments. If it is missing, the modules use a deterministic local fallback salt and log a warning.

## Run Data Collection

Use one terminal for setup and launch so the robot modules inherit the experiment metadata.

1. Open a terminal in the repository root:

```bash
cd /usr/local/src/robot/cognitiveInteraction/alwaysOn-embodiedBehaviour
```

2. Choose the run labels:

- `run_id`: unique label for this run.
- `experiment_condition`: `drive_enabled`, `drive_disabled`, `ablation`, `pilot`, or `unknown`.
- `scenario_id`: short scenario name.
- `participant_id`: local participant code, for example `P001`.

3. Export the matching variables using one of the examples below.

4. Confirm the important values:

```bash
echo "$ALWAYSON_RUN_ID"
echo "$ALWAYSON_EXPERIMENT_CONDITION"
echo "$ALWAYSON_SCENARIO_ID"
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
export ALWAYSON_SCENARIO_ID="lab_hri_short_interaction"
export ALWAYSON_PARTICIPANT_ID="P001"
export ALWAYSON_PARTICIPANT_SALT="CHANGE_ME_FOR_REAL_STUDY"
export ALWAYSON_IS_TEST_RUN="0"
export ALWAYSON_VALID_FOR_ANALYSIS="1"
```

Drive-disabled run:

```bash
export ALWAYSON_RUN_ID="exp01_drive_off_2026_05_11"
export ALWAYSON_EXPERIMENT_CONDITION="drive_disabled"
export ALWAYSON_SCENARIO_ID="lab_hri_short_interaction"
export ALWAYSON_PARTICIPANT_ID="P001"
export ALWAYSON_PARTICIPANT_SALT="CHANGE_ME_FOR_REAL_STUDY"
export ALWAYSON_IS_TEST_RUN="0"
export ALWAYSON_VALID_FOR_ANALYSIS="1"
```

Pilot/debug run:

```bash
export ALWAYSON_RUN_ID="pilot_debug_001"
export ALWAYSON_EXPERIMENT_CONDITION="pilot"
export ALWAYSON_SCENARIO_ID="debug"
export ALWAYSON_IS_TEST_RUN="1"
export ALWAYSON_VALID_FOR_ANALYSIS="0"
```

## Clean Views

Clean views include only rows where `valid_for_analysis = 1` and expose the metadata fields:

- Executive: `v_interactions_clean`
- Salience: `v_interaction_attempts_clean`, `v_target_selections_clean`, `v_homeostatic_learning_changes_clean`
- ChatBot: `v_chat_events_clean`, `v_chat_messages_clean`

Quality-control views are available in each database, including missing-metadata and condition-count summaries.

## CSV Export

The export helper does not require YARP:

```bash
python3 scripts/export_evaluation_data.py
```

By default it reads the SQLite files in `modules/data_collection/` and writes CSVs under `modules/data_collection/exports/<run_id-or-timestamp>/`.

The salience export includes interaction attempts, target selections, homeostatic learning changes, and the IPS timeline so analyses can inspect deficit-driven prioritisation and adaptive weight changes.

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

Use the condition-count views to confirm that rows are labelled with the expected run, condition, and scenario:

```bash
sqlite3 modules/data_collection/executive_control.db \
  "SELECT * FROM v_quality_condition_counts;"

sqlite3 modules/data_collection/salience_network.db \
  "SELECT * FROM v_quality_attempt_counts;"

sqlite3 modules/data_collection/chat_bot.db \
  "SELECT * FROM v_quality_chat_condition_counts;"
```
