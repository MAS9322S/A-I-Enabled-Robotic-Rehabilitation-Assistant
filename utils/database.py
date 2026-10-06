import os
import sqlite3
from pathlib import Path


class Database:
    """Supabase REST backend for cloud deployments, SQLite fallback for local use."""

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
        try:
            import streamlit as st
            self.supabase_url = str(st.secrets.get("SUPABASE_URL", self.supabase_url)).rstrip("/")
            self.supabase_key = str(st.secrets.get("SUPABASE_ANON_KEY", self.supabase_key))
        except Exception:
            pass
        self.use_supabase = bool(self.supabase_url and self.supabase_key)
        self.mode_label = "Supabase cloud database" if self.use_supabase else "Local SQLite fallback"
        if not self.use_supabase:
            self.db_path = Path(__file__).resolve().parent.parent / "data" / "rehab.db"
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()

    def _init_sqlite(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS patients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, age INTEGER NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, exercise TEXT NOT NULL, repetitions INTEGER NOT NULL, avg_angle REAL NOT NULL, max_range REAL NOT NULL, form_score REAL NOT NULL, feedback TEXT, created_at TEXT NOT NULL)"
            )
            conn.commit()

    @staticmethod
    def _now():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def _headers(self):
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }

    def _rest(self, method, table, payload=None, params=None, prefer=None):
        import requests
        url = f"{self.supabase_url}/rest/v1/{table}"
        headers = self._headers()
        if prefer:
            headers["Prefer"] = prefer
        response = requests.request(method, url, headers=headers, json=payload, params=params, timeout=20)
        response.raise_for_status()
        if not response.text:
            return []
        return response.json()

    def list_patients(self):
        if self.use_supabase:
            return self._rest("GET", "patients", params={"select": "id,name,age,created_at", "order": "created_at.desc"})
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(x) for x in conn.execute("SELECT id,name,age,created_at FROM patients ORDER BY created_at DESC")]

    def create_patient(self, name, age):
        if self.use_supabase:
            rows = self._rest("POST", "patients", {"name": name, "age": age, "created_at": self._now()}, prefer="return=representation")
            return rows[0]["id"]
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("INSERT INTO patients(name, age, created_at) VALUES (?, ?, ?)", (name, age, self._now()))
            conn.commit()
            return cur.lastrowid

    def get_patient(self, patient_id):
        if self.use_supabase:
            rows = self._rest("GET", "patients", params={"select": "id,name,age,created_at", "id": f"eq.{patient_id}"})
            return rows[0] if rows else None
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT id,name,age,created_at FROM patients WHERE id=?", (patient_id,)).fetchone()
            return dict(row) if row else None

    def save_session(self, patient_id, exercise, repetitions, avg_angle, max_range, form_score, feedback):
        payload = {
            "patient_id": patient_id,
            "exercise": exercise,
            "repetitions": int(repetitions),
            "avg_angle": float(avg_angle),
            "max_range": float(max_range),
            "form_score": float(form_score),
            "feedback": feedback,
            "created_at": self._now(),
        }
        if self.use_supabase:
            self._rest("POST", "sessions", payload)
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions(patient_id,exercise,repetitions,avg_angle,max_range,form_score,feedback,created_at) VALUES (?,?,?,?,?,?,?,?)",
                tuple(payload.values()),
            )
            conn.commit()

    def list_sessions(self, patient_id):
        if self.use_supabase:
            return self._rest(
                "GET",
                "sessions",
                params={
                    "select": "exercise,repetitions,avg_angle,max_range,form_score,feedback,created_at",
                    "patient_id": f"eq.{patient_id}",
                    "order": "created_at.desc",
                    "limit": "50",
                },
            )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(x) for x in conn.execute(
                "SELECT exercise,repetitions,avg_angle,max_range,form_score,feedback,created_at FROM sessions WHERE patient_id=? ORDER BY created_at DESC LIMIT 50",
                (patient_id,),
            )]
