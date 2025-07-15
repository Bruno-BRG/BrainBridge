import sqlite3
from datetime import datetime
from typing import List, Dict
from bci.config import get_database_path

class DatabaseManager:
    """Gerencia o banco SQLite de pacientes e gravações."""
    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_database_path()
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                affected_hand TEXT,
                time_since_event INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                filename TEXT,
                start_time TIMESTAMP,
                duration INTEGER,
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        ''')
        conn.commit()
        conn.close()

    def add_patient(self, name: str, age: int, sex: str,
                    affected_hand: str, time_since_event: int,
                    notes: str = "") -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO patients (name, age, sex, affected_hand, time_since_event, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, age, sex, affected_hand, time_since_event, notes))
        pid = c.lastrowid
        conn.commit()
        conn.close()
        return pid

    def get_all_patients(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT id, name, age, sex, affected_hand, time_since_event, created_at, notes FROM patients ORDER BY created_at DESC')
        rows = c.fetchall()
        conn.close()
        return [
            {"id": r[0], "name": r[1], "age": r[2], "sex": r[3],
             "affected_hand": r[4], "time_since_event": r[5],
             "created_at": r[6], "notes": r[7]}
            for r in rows
        ]

    def add_recording(self, patient_id: int, filename: str, notes: str = ""):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO recordings (patient_id, filename, start_time, notes)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, filename, datetime.now().isoformat(), notes))
        conn.commit()
        conn.close()
