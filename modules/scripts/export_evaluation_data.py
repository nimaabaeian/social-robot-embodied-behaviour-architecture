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
DATA_DIR = ROOT / "modules" / "data_collection"


EXPORTS: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    ("executive", "v_interactions_clean", "executive_interactions_clean.csv", None),
    (
        "executive",
        "hunger_level_events",
        "executive_hunger_level_events_clean.csv",
        "SELECT * FROM hunger_level_events WHERE valid_for_analysis = 1",
    ),
    ("salience", "v_interaction_attempts_clean", "salience_interaction_attempts_clean.csv", None),
    ("salience", "v_target_selections_clean", "salience_target_selections_clean.csv", None),
    (
        "salience",
        "v_homeostatic_learning_changes_clean",
        "salience_homeostatic_learning_changes_clean.csv",
        None,
    ),
    (
        "salience",
        "v_face_ips_timeline",
        "salience_face_ips_timeline_clean.csv",
        "SELECT * FROM v_face_ips_timeline WHERE valid_for_analysis = 1",
    ),
    (
        "salience",
        "face_ips_events",
        "salience_face_ips_events_clean.csv",
        "SELECT * FROM face_ips_events WHERE valid_for_analysis = 1",
    ),
    ("chatbot", "v_chat_session_metrics", "chatbot_session_metrics.csv", None),
    ("chatbot", "v_chat_events_clean", "chatbot_events_clean.csv", None),
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
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    db_paths = [args.executive_db, args.salience_db, args.chatbot_db]
    run_id = _first_run_id(db_paths)
    export_name = _safe_name(run_id) if run_id else time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DATA_DIR / "exports" / export_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    _export_db("executive", args.executive_db, out_dir)
    _export_db("salience", args.salience_db, out_dir)
    _export_db("chatbot", args.chatbot_db, out_dir)
    print(f"export_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
