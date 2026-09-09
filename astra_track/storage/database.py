"""
SQLite local database for simulation sessions, telemetry, and SIH benchmark results.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional


class DatabaseManager:
    """Manages SQLite storage for sessions, metrics, and frame-level logs."""

    def __init__(self, db_path: str = "astra_track.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table: simulations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    duration REAL NOT NULL,
                    resolution TEXT NOT NULL,
                    fps REAL NOT NULL,
                    config_json TEXT
                )
            """)

            # Table: metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    acquisition_time REAL,
                    avg_error REAL,
                    max_error REAL,
                    loss_percentage REAL,
                    reacquisition_time REAL,
                    avg_fps REAL,
                    lock_retention REAL,
                    result TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES simulations (id)
                )
            """)

            # Table: frames
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    frame_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    actual_x REAL,
                    actual_y REAL,
                    detected_x REAL,
                    detected_y REAL,
                    filtered_x REAL,
                    filtered_y REAL,
                    predicted_x REAL,
                    predicted_y REAL,
                    pan REAL,
                    tilt REAL,
                    error_px REAL,
                    confidence REAL,
                    status TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES simulations (id)
                )
            """)
            conn.commit()

    def save_simulation(self, summary_data: Dict[str, Any], frame_records: Optional[List[Dict[str, Any]]] = None) -> int:
        """Saves simulation metadata, metrics, and optional frame telemetry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO simulations (timestamp, scenario, duration, resolution, fps, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                summary_data.get("timestamp", ""),
                summary_data.get("scenario", "Simulation"),
                summary_data.get("duration_s", 0.0),
                summary_data.get("resolution", "640x480"),
                summary_data.get("avg_fps", 30.0),
                json.dumps(summary_data.get("config", {}))
            ))
            sim_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO metrics (
                    simulation_id, acquisition_time, avg_error, max_error,
                    loss_percentage, reacquisition_time, avg_fps, lock_retention, result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sim_id,
                summary_data.get("acquisition_time_s", 0.0),
                summary_data.get("avg_error_px", 0.0),
                summary_data.get("max_error_px", 0.0),
                summary_data.get("target_loss_pct", 0.0),
                summary_data.get("avg_reacquisition_time_s", 0.0),
                summary_data.get("avg_fps", 30.0),
                summary_data.get("lock_retention_pct", 100.0),
                "PASS" if summary_data.get("overall_pass", True) else "FAIL"
            ))

            if frame_records:
                frame_tuples = [
                    (
                        sim_id,
                        f.get("frame_id", 0),
                        f.get("timestamp", 0.0),
                        f.get("actual_x", -1.0),
                        f.get("actual_y", -1.0),
                        f.get("detected_x", -1.0),
                        f.get("detected_y", -1.0),
                        f.get("filtered_x", 0.0),
                        f.get("filtered_y", 0.0),
                        f.get("predicted_x", 0.0),
                        f.get("predicted_y", 0.0),
                        f.get("pan", 0.0),
                        f.get("tilt", 0.0),
                        f.get("error_px", 0.0),
                        f.get("confidence", 0.0),
                        f.get("status", "")
                    )
                    for f in frame_records
                ]
                cursor.executemany("""
                    INSERT INTO frames (
                        simulation_id, frame_id, timestamp, actual_x, actual_y,
                        detected_x, detected_y, filtered_x, filtered_y,
                        predicted_x, predicted_y, pan, tilt, error_px, confidence, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, frame_tuples)

            conn.commit()
            return sim_id

    def get_all_simulations(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.id, s.timestamp, s.scenario, s.duration, s.fps,
                       m.acquisition_time, m.avg_error, m.max_error,
                       m.loss_percentage, m.reacquisition_time, m.avg_fps,
                       m.lock_retention, m.result
                FROM simulations s
                LEFT JOIN metrics m ON s.id = m.simulation_id
                ORDER BY s.id DESC
            """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
