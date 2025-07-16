class DatabaseManager:
    """Gerenciador do banco de dados SQLite para pacientes"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = get_database_path()
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
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
        
        cursor.execute('''
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
    
    def add_patient(self, name: str, age: int, sex: str, affected_hand: str, 
                   time_since_event: int, notes: str = ""):
        """Adiciona um novo paciente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients (name, age, sex, affected_hand, time_since_event, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, age, sex, affected_hand, time_since_event, notes))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return patient_id
    
    def get_all_patients(self) -> List[Dict]:
        """Retorna todos os pacientes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
        patients = cursor.fetchall()
        conn.close()
        
        return [{"id": p[0], "name": p[1], "age": p[2], "sex": p[3], 
                "affected_hand": p[4], "time_since_event": p[5], 
                "created_at": p[6], "notes": p[7]} for p in patients]
    
    def add_recording(self, patient_id: int, filename: str, notes: str = ""):
        """Adiciona uma nova gravação"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recordings (patient_id, filename, start_time, notes)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, filename, datetime.now().isoformat(), notes))
        
        conn.commit()
        conn.close()

