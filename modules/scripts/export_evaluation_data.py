#!/usr/bin/env python3
"""Export clean evaluation tables/views from the always-on data logs."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_collection"


EXPORTS: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    # ── executive: core interaction records ──────────────────────────────────
    # v_interactions_clean filters valid_for_analysis=1 and adds derived cols
    # (user_key, abort_category, coalesced homeostatic_reward / hunger_drive).
    ("executive", "v_interactions_clean", "executive_interactions_clean.csv", None),
    # v_interaction_turns has no filter in its view definition — apply here.
    (
        "executive",
        "v_interaction_turns",
        "executive_interaction_turns.csv",
        "SELECT * FROM v_interaction_turns WHERE valid_for_analysis = 1",
    ),
    # Reactive events: ambient QR feeds (with feeder attribution), greetings,
    # unknown-person intros.
    (
        "executive",
        "reactive_interactions",
        "executive_reactive_interactions.csv",
        "SELECT * FROM reactive_interactions WHERE valid_for_analysis = 1",
    ),
    # ── executive: hunger / stomach ──────────────────────────────────────────
    # Raw hunger events (drain samples, active costs, feedings, mode changes).
    (
        "executive",
        "hunger_level_events",
        "executive_hunger_level_events_clean.csv",
        "SELECT * FROM hunger_level_events WHERE valid_for_analysis = 1",
    ),
    # State transitions only (before != after). v_hs_transitions has no
    # valid_for_analysis filter in its own definition — apply here.
    (
        "executive",
        "v_hs_transitions",
        "executive_hs_transitions.csv",
        "SELECT * FROM v_hs_transitions WHERE valid_for_analysis = 1",
    ),
    # HS3 (critical hunger) episode timeline with feeding resolution columns.
    ("executive", "v_hs3_episodes",          "executive_hs3_episodes.csv",          None),
    # Contiguous passive-drain segments with empirical drain-rate calculation.
    ("executive", "v_drain_segments",        "executive_drain_segments.csv",        None),
    # Per-interaction active energy cost breakdown by stimulus label.
    ("executive", "v_active_cost_breakdown", "executive_active_cost_breakdown.csv", None),
    # ── executive: aggregated metrics ────────────────────────────────────────
    # All four metric views build on v_interactions_clean (already filtered).
    ("executive", "v_metric_hunger_mention_rate",  "executive_metric_hunger_mention_rate.csv",  None),
    ("executive", "v_metric_ss3_daily",            "executive_metric_ss3_daily.csv",            None),
    ("executive", "v_metric_response_rate_daily",  "executive_metric_response_rate_daily.csv",  None),
    ("executive", "v_metric_repeat_users_daily",   "executive_metric_repeat_users_daily.csv",   None),
    ("executive", "v_metric_depth_progression",    "executive_metric_depth_progression.csv",    None),
    # ── salience: interaction attempts & daily rollups ────────────────────────
    # v_interaction_attempts_clean: filtered, adds abort_category + coalesced
    # hunger_state / is_proactive. Linkable to executive via exec_interaction_id.
    ("salience", "v_interaction_attempts_clean", "salience_interaction_attempts_clean.csv", None),
    # v_interaction_attempts_daily builds on _clean (already filtered).
    ("salience", "v_interaction_attempts_daily", "salience_interaction_attempts_daily.csv", None),
    # Per-interaction lifecycle events (start + end). View is unfiltered —
    # apply valid_for_analysis here.
    (
        "salience",
        "v_interaction_state_events",
        "salience_interaction_state_events.csv",
        "SELECT * FROM v_interaction_state_events WHERE valid_for_analysis = 1",
    ),
    # ── salience: social state & face attention ───────────────────────────────
    # Per-person social-state transitions (old_ss → new_ss). Table has no view.
    (
        "salience",
        "ss_changes",
        "salience_ss_changes.csv",
        "SELECT * FROM ss_changes WHERE valid_for_analysis = 1",
    ),
    # Frame-level face attention log with the full IPS score breakdown. The
    # logged weights are the fixed BASELINE_WEIGHTS (the adaptive mechanism is
    # the affinity-shifted eligibility threshold, captured in target_selections).
    # View itself is unfiltered — apply valid_for_analysis here.
    (
        "salience",
        "v_face_ips_timeline",
        "salience_face_ips_timeline.csv",
        "SELECT * FROM v_face_ips_timeline WHERE valid_for_analysis = 1",
    ),
    # Salience-engine target selections (one row per attention frame), including
    # the decision-time learned values (affinity, effective_threshold) that the
    # eligibility test used for that person at that moment.
    ("salience", "v_target_selections_clean", "salience_target_selections_clean.csv", None),
    # ── salience: homeostatic learning ───────────────────────────────────────
    # Weight-update log per person after each interaction (old_* → new_* weights).
    ("salience", "v_homeostatic_learning_changes_clean", "salience_homeostatic_learning_changes_clean.csv", None),
    # ── chatbot: messages & sessions ─────────────────────────────────────────
    # Full per-message text log (role, hs, text, latency, telegram timestamps).
    ("chatbot", "v_chat_messages_clean",      "chatbot_messages_clean.csv",      None),
    # High-level per-event log (user_message, assistant_reply, proactives, etc.).
    ("chatbot", "v_chat_events_clean",        "chatbot_events_clean.csv",        None),
    # Per-session rollup (depth, hunger peaks, fallback count, duration).
    ("chatbot", "v_chat_session_metrics",     "chatbot_session_metrics.csv",     None),
    # ── chatbot: aggregated metrics ───────────────────────────────────────────
    # All metric views build on v_chat_events_clean (already filtered).
    ("chatbot", "v_chat_daily_metrics",       "chatbot_daily_metrics.csv",       None),
    ("chatbot", "v_chat_user_daily",          "chatbot_user_daily.csv",          None),
    # ── chatbot: linking & proactive campaigns ────────────────────────────────
    # chat_id ↔ person_id confirmation lifecycle with JSON-extracted fields.
    ("chatbot", "v_chat_link_events",         "chatbot_link_events.csv",         None),
    # Proactive campaign decisions (priority vs. broadcast, counts).
    ("chatbot", "v_chat_proactive_selection", "chatbot_proactive_selection.csv", None),
    # ── vision: organised landmarks:o log ─────────────────────────────────────
    # Per-face frame-level landmark output (bbox, zone, distance, gaze, head
    # pose, attention, talking). All faces of a frame share frame_id + timestamp.
    # v_landmark_events_clean filters valid_for_analysis = 1.
    ("vision", "v_landmark_events_clean", "vision_landmark_events_clean.csv", None),
    # ── data-quality audits ───────────────────────────────────────────────────
    # Sanity tables to run before analysis: rows with NULL run_id, out-of-range
    # stomach levels, and per-condition row counts to spot imbalance. All are
    # intentionally unfiltered (they audit the raw tables, including invalid rows).
    ("executive", "v_quality_hunger_invalid_levels",      "quality_executive_hunger_invalid_levels.csv",     None),
    ("executive", "v_quality_interaction_missing_metadata", "quality_executive_interaction_missing_metadata.csv", None),
    ("executive", "v_quality_condition_counts",           "quality_executive_condition_counts.csv",          None),
    ("salience",  "v_quality_salience_missing_metadata",  "quality_salience_missing_metadata.csv",           None),
    ("salience",  "v_quality_attempt_counts",             "quality_salience_attempt_counts.csv",             None),
    ("chatbot",   "v_quality_chat_missing_metadata",      "quality_chatbot_missing_metadata.csv",            None),
    ("chatbot",   "v_quality_chat_condition_counts",      "quality_chatbot_condition_counts.csv",            None),
)


def _schema_value(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        row = conn.execute("SELECT value FROM schema_info WHERE key=?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    if row and row[0]:
        return str(row[0])
    return None


def _first_run_id(paths: Iterable[Path]) -> Optional[str]:
    for path in paths:
        if not path.exists():
            continue
        try:
            with sqlite3.connect(path) as conn:
                value = _schema_value(conn, "run_id")
                if value:
                    return value
        except sqlite3.Error:
            continue
    return None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def _export_query(conn: sqlite3.Connection, query: str, out_path: Path) -> int:
    cur = conn.execute(query)
    columns = [desc[0] for desc in cur.description or []]
    count = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in cur:
            writer.writerow(row)
            count += 1
    return count


def _export_db(db_name: str, db_path: Path, out_dir: Path) -> None:
    if not db_path.exists():
        print(f"[skip] {db_name}: missing DB {db_path}")
        return
    try:
        with sqlite3.connect(db_path) as conn:
            for export_db, source, filename, query in EXPORTS:
                if export_db != db_name:
                    continue
                sql = query or f"SELECT * FROM {source}"
                out_path = out_dir / filename
                try:
                    count = _export_query(conn, sql, out_path)
                    print(f"[ok] {filename}: {count} rows")
                except sqlite3.Error as exc:
                    print(f"[skip] {filename}: {exc}")
    except sqlite3.Error as exc:
        print(f"[skip] {db_name}: cannot open {db_path}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export clean always-on evaluation data to CSV.")
    parser.add_argument("--executive-db", type=Path, default=DATA_DIR / "executive_control.db")
    parser.add_argument("--salience-db", type=Path, default=DATA_DIR / "salience_network.db")
    parser.add_argument("--chatbot-db", type=Path, default=DATA_DIR / "chat_bot.db")
    parser.add_argument("--vision-db", type=Path, default=DATA_DIR / "vision.db")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    db_paths = [args.executive_db, args.salience_db, args.chatbot_db, args.vision_db]
    run_id = _first_run_id(db_paths)
    export_name = _safe_name(run_id) if run_id else time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DATA_DIR / "exports" / export_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    _export_db("executive", args.executive_db, out_dir)
    _export_db("salience", args.salience_db, out_dir)
    _export_db("chatbot", args.chatbot_db, out_dir)
    _export_db("vision", args.vision_db, out_dir)
    print(f"export_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
