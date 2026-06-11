#!/usr/bin/env bash
# Source this file before starting any module or server:
#   source scripts/set_run_env.sh

# Unique identifier for this experimental run.
# Convention: <label>_<YYYY_MM_DD>  e.g. "exp01_2026_05_11", "pilot_debug_001"
# All four modules (vision, salienceNetwork, executiveControl, chatBot) must
# share the same value so their DB rows can be joined in analysis.
export ALWAYSON_RUN_ID="exp01_2026_05_11"

# Whether this is a test/pilot/debug run that should be excluded from analysis.
# Values: "0" (real run, include in analysis) | "1" (test/debug, exclude)
export ALWAYSON_IS_TEST_RUN="0"

# Whether rows written in this run should be included in analysis.
# Values: "1" (include) | "0" (exclude)
# Defaults to "0" automatically when ALWAYSON_IS_TEST_RUN=1.
export ALWAYSON_VALID_FOR_ANALYSIS="1"

# Free-form label for the experimental arm/condition of this run.
# Stored once per DB in schema_info (not per-row); one run = one condition.
# Leave empty if the run has no distinct condition to contrast.
# Example values:
#   ""              – no condition (single-arm study or pilot)
#   "baseline"      – control condition, hunger drive enabled, default behaviour
#   "drive_off"     – Orexigenic hunger drive disabled (hunger_drive_enabled=0)
#   "transparent"   – robot verbalises its hunger state to people
#   "opaque"        – robot does NOT verbalise hunger state
#   "priority"      – chatbot proactive mode: priority-ranked by affinity
#   "broadcast"     – chatbot proactive mode: sent to all subscribers
#   "pilot"         – informal pilot session, not part of the main study
export ALWAYSON_EXPERIMENT_CONDITION=""

echo "[alwayson] run_id=${ALWAYSON_RUN_ID}  test=${ALWAYSON_IS_TEST_RUN}  valid=${ALWAYSON_VALID_FOR_ANALYSIS}  condition=${ALWAYSON_EXPERIMENT_CONDITION:-<none>}"