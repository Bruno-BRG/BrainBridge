#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database manager for BCI System
Handles patient data, sessions, recordings, and model information
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BCIDatabaseManager:
    """Database manager for BCI system"""
    
    def __init__(self, db_path: str = "bci_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    patient_id TEXT UNIQUE NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    affected_hand TEXT,
                    stroke_date TEXT,
                    time_since_stroke INTEGER,
                    medical_info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_type TEXT DEFAULT 'training',
                    duration_minutes INTEGER,
                    notes TEXT,
                    session_data TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            """)
            
            # Recordings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    patient_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    recording_type TEXT DEFAULT 'eeg',
                    sample_rate REAL DEFAULT 125.0,
                    channels INTEGER DEFAULT 16,
                    duration_seconds REAL,
                    annotations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id),
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    model_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    model_type TEXT DEFAULT 'EEGInceptionERP',
                    training_accuracy REAL,
                    validation_accuracy REAL,
                    test_accuracy REAL,
                    model_params TEXT,
                    is_finetuned BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def add_patient(self, patient_data: Dict) -> str:
        """Add new patient to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO patients (name, patient_id, age, gender, affected_hand, 
                                    stroke_date, time_since_stroke, medical_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_data['name'],
                patient_data['patient_id'],
                patient_data.get('age'),
                patient_data.get('gender'),
                patient_data.get('affected_hand'),
                patient_data.get('stroke_date'),
                patient_data.get('time_since_stroke'),
                patient_data.get('medical_info', '')
            ))
            
            conn.commit()
            logger.info(f"Patient {patient_data['patient_id']} added successfully")
            return patient_data['patient_id']
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get patient by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients ORDER BY name")
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    def update_patient(self, patient_id: str, updates: Dict):
        """Update patient information"""
        if not updates:
            return
            
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [patient_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE patients 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
                WHERE patient_id = ?
            """, values)
            conn.commit()
    
    def delete_patient(self, patient_id: str):
        """Delete patient and all related data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM recordings WHERE patient_id = ?", (patient_id,))
            cursor.execute("DELETE FROM models WHERE patient_id = ?", (patient_id,))
            cursor.execute("DELETE FROM sessions WHERE patient_id = ?", (patient_id,))
            cursor.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
            conn.commit()
    
    def add_session(self, session_data: Dict) -> int:
        """Add new session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (patient_id, session_type, duration_minutes, 
                                    notes, session_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_data['patient_id'],
                session_data.get('session_type', 'training'),
                session_data.get('duration_minutes'),
                session_data.get('notes', ''),
                json.dumps(session_data.get('session_data', {}))
            ))
            
            session_id = cursor.lastrowid
            conn.commit()
            return session_id
    
    def update_session(self, session_id: int, updates: Dict):
        """Update session information"""
        if not updates:
            return
            
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [session_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE sessions 
                SET {set_clause}
                WHERE id = ?
            """, values)
            conn.commit()
    
    def get_patient_sessions(self, patient_id: str) -> List[Dict]:
        """Get all sessions for a patient"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions 
                WHERE patient_id = ? 
                ORDER BY session_date DESC
            """, (patient_id,))
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            sessions = [dict(zip(columns, row)) for row in rows]
            
            # Parse session_data JSON
            for session in sessions:
                if session['session_data']:
                    try:
                        session['session_data'] = json.loads(session['session_data'])
                    except json.JSONDecodeError:
                        session['session_data'] = {}
                        
            return sessions
    
    def add_recording(self, recording_data: Dict) -> int:
        """Add new recording"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO recordings (session_id, patient_id, file_path, recording_type,
                                      sample_rate, channels, duration_seconds, annotations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recording_data['session_id'],
                recording_data['patient_id'],
                recording_data['file_path'],
                recording_data.get('recording_type', 'eeg'),
                recording_data.get('sample_rate', 125.0),
                recording_data.get('channels', 16),
                recording_data.get('duration_seconds'),
                json.dumps(recording_data.get('annotations', []))
            ))
            
            recording_id = cursor.lastrowid
            conn.commit()
            return recording_id
    
    def get_patient_recordings(self, patient_id: str) -> List[Dict]:
        """Get all recordings for a patient"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, s.session_date 
                FROM recordings r
                JOIN sessions s ON r.session_id = s.id
                WHERE r.patient_id = ? 
                ORDER BY r.created_at DESC
            """, (patient_id,))
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            recordings = [dict(zip(columns, row)) for row in rows]
            
            # Parse annotations JSON
            for recording in recordings:
                if recording['annotations']:
                    try:
                        recording['annotations'] = json.loads(recording['annotations'])
                    except json.JSONDecodeError:
                        recording['annotations'] = []
                        
            return recordings
    
    def add_model(self, model_data: Dict) -> int:
        """Add model information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO models (patient_id, model_name, model_path, model_type,
                                  training_accuracy, validation_accuracy, test_accuracy,
                                  model_params, is_finetuned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_data.get('patient_id'),
                model_data['model_name'],
                model_data['model_path'],
                model_data.get('model_type', 'EEGInceptionERP'),
                model_data.get('training_accuracy'),
                model_data.get('validation_accuracy'),
                model_data.get('test_accuracy'),
                json.dumps(model_data.get('model_params', {})),
                model_data.get('is_finetuned', False)
            ))
            
            model_id = cursor.lastrowid
            conn.commit()
            return model_id
    
    def get_patient_models(self, patient_id: str) -> List[Dict]:
        """Get all models for a patient"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM models 
                WHERE patient_id = ? OR patient_id IS NULL
                ORDER BY is_finetuned DESC, created_at DESC
            """, (patient_id,))
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            models = [dict(zip(columns, row)) for row in rows]
            
            # Parse model_params JSON
            for model in models:
                if model['model_params']:
                    try:
                        model['model_params'] = json.loads(model['model_params'])
                    except json.JSONDecodeError:
                        model['model_params'] = {}
                        
            return models
    
    def get_all_models(self) -> List[Dict]:
        """Get all available models"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM models 
                ORDER BY is_finetuned DESC, created_at DESC
            """)
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            models = [dict(zip(columns, row)) for row in rows]
            
            # Parse model_params JSON
            for model in models:
                if model['model_params']:
                    try:
                        model['model_params'] = json.loads(model['model_params'])
                    except json.JSONDecodeError:
                        model['model_params'] = {}
                        
            return models
    
    def get_latest_session(self, patient_id: str) -> Optional[Dict]:
        """Get patient's latest session"""
        sessions = self.get_patient_sessions(patient_id)
        return sessions[0] if sessions else None

if __name__ == "__main__":
    # Test the database
    db = BCIDatabaseManager("test.db")
    print("Database initialized successfully")
    print("Available methods:", [method for method in dir(db) if not method.startswith('_')])