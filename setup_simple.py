"""
Script simples para configurar o sistema BCI
"""
import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import database module
from database import BCIDatabaseManager

def main():
    print("🔧 Configuração Inicial do Sistema BCI")
    print("=" * 40)
    
    # Create database
    print("\n📊 Criando banco de dados...")
    db = BCIDatabaseManager()
    
    # Add sample patients
    print("\n👥 Adicionando pacientes de exemplo...")
    
    sample_patients = [
        {
            'patient_id': 'P001',
            'name': 'João Silva',
            'age': 45,
            'gender': 'Masculino',
            'affected_hand': 'Direita',
            'stroke_date': '2024-01-15',
            'time_since_stroke': 150,
            'medical_info': 'AVC isquêmico. Hemiparesia direita moderada.'
        },
        {
            'patient_id': 'P002', 
            'name': 'Maria Santos',
            'age': 38,
            'gender': 'Feminino',
            'affected_hand': 'Esquerda',
            'stroke_date': '2024-03-20',
            'time_since_stroke': 85,
            'medical_info': 'AVC hemorrágico. Recuperação motora em progresso.'
        }
    ]
    
    for patient_data in sample_patients:
        try:
            existing = db.get_patient(patient_data['patient_id'])
            if existing:
                print(f"⚠️  Paciente {patient_data['patient_id']} já existe")
                continue
                
            db.add_patient(patient_data)
            print(f"✅ Paciente {patient_data['name']} ({patient_data['patient_id']}) adicionado")
            
        except Exception as e:
            print(f"❌ Erro ao adicionar paciente {patient_data['patient_id']}: {e}")
    
    # Add models
    print("\n🧠 Configurando modelos...")
    models_dir = Path("../models/teste")
    
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt"))
        print(f"🔍 Encontrados {len(model_files)} modelos")
        
        for model_file in model_files:
            model_name = model_file.stem
            
            # Check if model already exists
            existing_models = db.get_patient_models("")
            if any(m['model_name'] == model_name for m in existing_models):
                print(f"⚠️  Modelo {model_name} já existe no banco")
                continue
            
            # Add model to database
            model_data = {
                'patient_id': None,
                'model_name': model_name,
                'model_path': str(model_file.absolute()),
                'model_type': 'EEGInceptionERP',
                'model_params': {
                    'n_chans': 16,
                    'n_times': 500,
                    'sfreq': 125.0,
                    'n_outputs': 2
                },
                'is_finetuned': False
            }
            
            model_id = db.add_model(model_data)
            print(f"✅ Modelo {model_name} adicionado (ID: {model_id})")
    else:
        print("❌ Diretório models/teste não encontrado!")
    
    print("\n🎉 Configuração concluída!")
    print("\n💡 Agora você pode:")
    print("   - Executar: python run_bci_system.py")
    print("   - Testar com: python test_eeg_simulator.py")

if __name__ == "__main__":
    main()
