"""
Adapter SQLite para repositório de pacientes
"""
import sqlite3
import os
import sys
from typing import List, Optional
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.application.port.out.patient_repository_port import (
    PatientRepositoryPort,
    PatientAlreadyExistsError,
    PatientNotFoundError,
    RepositoryError
)
from src.domain.model.patient import Patient, PatientId


class SqlitePatientRepository(PatientRepositoryPort):
    """
    Adapter SQLite para persistência de pacientes
    
    Implementa PatientRepositoryPort usando SQLite como banco de dados.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = None
        self._init_database()
    
    def _get_connection(self):
        """Obtém conexão SQLite"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute("PRAGMA foreign_keys = ON")
        return self.connection
    
    def _init_database(self):
        """Inicializa o banco de dados"""
        try:
            # Criar diretório se necessário
            if self.db_path != ":memory:":
                db_dir = os.path.dirname(self.db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
            
            # Criar tabela
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    time_since_brain_event INTEGER NOT NULL,
                    brain_event_type TEXT NOT NULL,
                    affected_side TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
            print("Tabela 'patients' inicializada com sucesso")
        except Exception as e:
            print(f"Erro detalhado na inicialização: {e}")
            raise RepositoryError(f"Erro ao inicializar banco de dados: {str(e)}") from e
    
    def save(self, patient: Patient) -> Patient:
        """Salva um paciente no banco de dados"""
        try:
            conn = self._get_connection()
            
            if patient.id is None:
                # Inserir novo paciente
                cursor = conn.execute("""
                    INSERT INTO patients (name, age, gender, time_since_brain_event,
                                        brain_event_type, affected_side, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient.name,
                    patient.age,
                    patient.gender,
                    patient.time_since_brain_event,
                    patient.brain_event_type,
                    patient.affected_side,
                    patient.notes,
                    patient.created_at.isoformat() if patient.created_at else datetime.now().isoformat()
                ))
                
                # Retornar paciente com ID atribuído
                new_id = cursor.lastrowid
                conn.commit()
                
                return Patient(
                    id=PatientId(new_id),
                    name=patient.name,
                    age=patient.age,
                    gender=patient.gender,
                    time_since_brain_event=patient.time_since_brain_event,
                    brain_event_type=patient.brain_event_type,
                    affected_side=patient.affected_side,
                    notes=patient.notes,
                    created_at=patient.created_at or datetime.now()
                )
            else:
                # Atualizar paciente existente
                cursor = conn.execute("""
                    UPDATE patients 
                    SET name = ?, age = ?, gender = ?, time_since_brain_event = ?,
                        brain_event_type = ?, affected_side = ?, notes = ?
                    WHERE id = ?
                """, (
                    patient.name,
                    patient.age,
                    patient.gender,
                    patient.time_since_brain_event,
                    patient.brain_event_type,
                    patient.affected_side,
                    patient.notes,
                    patient.id.value
                ))
                
                if cursor.rowcount == 0:
                    raise PatientNotFoundError(f"Paciente com ID {patient.id.value} não encontrado")
                
                conn.commit()
                return patient
                
        except sqlite3.Error as e:
            raise RepositoryError(f"Erro SQLite: {str(e)}") from e
        except Exception as e:
            raise RepositoryError(f"Erro inesperado: {str(e)}") from e
    
    def _ensure_table_exists(self):
        """Garante que a tabela existe"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    time_since_brain_event INTEGER NOT NULL,
                    brain_event_type TEXT NOT NULL,
                    affected_side TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
    def find_by_id(self, patient_id: PatientId) -> Optional[Patient]:
        """Busca paciente por ID"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM patients WHERE id = ?
            """, (patient_id.value,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_patient(row)
            return None
            
        except sqlite3.Error as e:
            raise RepositoryError(f"Erro SQLite: {str(e)}") from e
        except Exception as e:
            raise RepositoryError(f"Erro inesperado: {str(e)}") from e
    
    def find_all(self) -> List[Patient]:
        """Retorna todos os pacientes"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM patients ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
            return [self._row_to_patient(row) for row in rows]
            
        except sqlite3.Error as e:
            raise RepositoryError(f"Erro SQLite: {str(e)}") from e
        except Exception as e:
            raise RepositoryError(f"Erro inesperado: {str(e)}") from e
    
    def delete(self, patient_id: PatientId) -> bool:
        """Remove paciente do banco de dados"""
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                DELETE FROM patients WHERE id = ?
            """, (patient_id.value,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            raise RepositoryError(f"Erro SQLite: {str(e)}") from e
        except Exception as e:
            raise RepositoryError(f"Erro inesperado: {str(e)}") from e
    
    def _row_to_patient(self, row: sqlite3.Row) -> Patient:
        """Converte linha do banco para entidade Patient"""
        created_at = datetime.fromisoformat(row['created_at']) if row['created_at'] else None
        
        return Patient(
            id=PatientId(row['id']),
            name=row['name'],
            age=row['age'],
            gender=row['gender'],
            time_since_brain_event=row['time_since_brain_event'],
            brain_event_type=row['brain_event_type'],
            affected_side=row['affected_side'],
            notes=row['notes'],
            created_at=created_at
        )
