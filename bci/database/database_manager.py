import sqlite3
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from ..configs.config import get_database_path

class DatabaseManager:
    """Gerenciador do banco de dados SQLite para pacientes"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = get_database_path()
        self.db_path = str(db_path)  # Converter Path para string se necessário
        
        # Garantir que o diretório do banco existe
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"Diretório do banco criado: {db_dir}")
        
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados"""
        try:
            # Verificar se o arquivo de banco existe
            db_exists = os.path.exists(self.db_path)
            
            if not db_exists:
                print(f"Criando novo banco de dados: {self.db_path}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Criar tabela de pacientes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    sex TEXT NOT NULL,
                    affected_hand TEXT NOT NULL,
                    time_since_event INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')
            
            # Criar tabela de gravações
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration INTEGER,
                    notes TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            if not db_exists:
                print("✅ Banco de dados criado com sucesso!")
            else:
                print("✅ Banco de dados carregado com sucesso!")
                
        except Exception as e:
            print(f"❌ Erro ao inicializar banco de dados: {e}")
            raise
    
    def add_patient(self, name: str, age: int, sex: str, affected_hand: str, 
                   time_since_event: int, notes: str = ""):
        """Adiciona um novo paciente"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO patients (name, age, sex, affected_hand, time_since_event, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, age, sex, affected_hand, time_since_event, notes))
            
            patient_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"Paciente {name} adicionado com ID: {patient_id}")
            return patient_id
            
        except Exception as e:
            print(f"Erro ao adicionar paciente: {e}")
            if conn:
                conn.close()
            raise
    
    def test_connection(self):
        """Testa a conexão com o banco de dados"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM patients")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"Conexão com banco OK. Pacientes cadastrados: {count}")
            return True
        except Exception as e:
            print(f"Erro ao conectar com banco: {e}")
            return False
    
    def get_all_patients(self) -> List[Dict]:
        """Retorna todos os pacientes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
            patients = cursor.fetchall()
            conn.close()
            
            return [{"id": p[0], "name": p[1], "age": p[2], "sex": p[3], 
                    "affected_hand": p[4], "time_since_event": p[5], 
                    "created_at": p[6], "notes": p[7]} for p in patients]
                    
        except Exception as e:
            print(f"Erro ao buscar pacientes: {e}")
            return []
    
    def delete_patient(self, patient_id: int):
        """Remove um paciente do banco de dados"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Primeiro, remover gravações associadas
            cursor.execute('DELETE FROM recordings WHERE patient_id = ?', (patient_id,))
            
            # Depois, remover o paciente
            cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
            
            conn.commit()
            affected_rows = cursor.rowcount
            conn.close()
            
            if affected_rows > 0:
                print(f"Paciente {patient_id} removido com sucesso")
                return True
            else:
                print(f"Paciente {patient_id} não encontrado")
                return False
                
        except Exception as e:
            print(f"Erro ao remover paciente: {e}")
            return False
    
    def add_recording(self, patient_id: int, filename: str, task_type: str, notes: str = ""):
        """Adiciona uma nova gravação"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_time = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO recordings (patient_id, filename, task_type, start_time, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, filename, task_type, start_time, notes))
            
            recording_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"✅ Gravação {filename} adicionada com ID: {recording_id}")
            return recording_id
            
        except Exception as e:
            print(f"❌ Erro ao adicionar gravação: {e}")
            if conn:
                conn.close()
            raise
    
    def update_recording_end_time(self, recording_id: int, duration: int):
        """Atualiza o tempo de fim e duração de uma gravação"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            end_time = datetime.now().isoformat()
            
            cursor.execute('''
                UPDATE recordings 
                SET end_time = ?, duration = ?
                WHERE id = ?
            ''', (end_time, duration, recording_id))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Gravação {recording_id} finalizada. Duração: {duration}s")
            
        except Exception as e:
            print(f"❌ Erro ao atualizar gravação: {e}")
            if conn:
                conn.close()
    
    def get_patient_recordings(self, patient_id: int) -> List[Dict]:
        """Retorna todas as gravações de um paciente"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM recordings 
                WHERE patient_id = ? 
                ORDER BY start_time DESC
            ''', (patient_id,))
            
            recordings = cursor.fetchall()
            conn.close()
            
            return [{"id": r[0], "patient_id": r[1], "filename": r[2], 
                    "task_type": r[3], "start_time": r[4], "end_time": r[5],
                    "duration": r[6], "notes": r[7]} for r in recordings]
                    
        except Exception as e:
            print(f"❌ Erro ao buscar gravações: {e}")
            return []

