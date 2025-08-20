"""
Exemplo de uso completo do sistema BCI com arquitetura hexagonal
"""
import sys
import os
from datetime import datetime

# Adicionar o diretório raiz ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.domain.model.patient import Patient, PatientId
from src.application.usecase.simple_patient_usecase import SimplePatientManagementUseCase
from src.infrastructure.adapter.out.sqlite_patient_repository import SqlitePatientRepository
from src.interface.dto.patient_dto import CreatePatientDTO, UpdatePatientNotesDTO
from src.interface.mapper.patient_mapper import PatientMapper


def main():
    """Demonstração completa do sistema BCI"""
    print("=" * 70)
    print("🧠 SISTEMA BCI - DEMONSTRAÇÃO ARQUITETURA HEXAGONAL")
    print("=" * 70)
    
    # 1. CONFIGURAÇÃO DAS DEPENDÊNCIAS (Dependency Injection)
    print("\n📋 1. Configurando dependências...")
    
    # Adapter de persistência (SQLite)
    repository = SqlitePatientRepository(":memory:")
    
    # Use Case da aplicação
    patient_management = SimplePatientManagementUseCase(repository)
    
    print("✅ Repositório SQLite e Use Case configurados")
    
    # 2. REGISTRO DE PACIENTES
    print("\n👤 2. Registrando pacientes...")
    
    # Paciente 1
    patient_data_1 = CreatePatientDTO(
        name="João Silva",
        age=45,
        gender="M",
        time_since_brain_event=12,
        brain_event_type="AVC",
        affected_side="Esquerdo",
        notes="Paciente colaborativo, boa evolução"
    )
    
    patient_1 = patient_management.register_patient(patient_data_1)
    print(f"✅ Paciente registrado: {patient_1.name} (ID: {patient_1.id})")
    print(f"   Faixa etária: {patient_1.age_group}")
    print(f"   Evento recente: {'Sim' if patient_1.is_recent_event else 'Não'}")
    
    # Paciente 2
    patient_data_2 = CreatePatientDTO(
        name="Maria Santos",
        age=28,
        gender="F",
        time_since_brain_event=3,
        brain_event_type="TCE",
        affected_side="Direito",
        notes="Primeira sessão de BCI"
    )
    
    patient_2 = patient_management.register_patient(patient_data_2)
    print(f"✅ Paciente registrado: {patient_2.name} (ID: {patient_2.id})")
    print(f"   Faixa etária: {patient_2.age_group}")
    print(f"   Evento recente: {'Sim' if patient_2.is_recent_event else 'Não'}")
    
    # 3. BUSCA DE PACIENTES
    print("\n🔍 3. Buscando pacientes...")
    
    # Buscar por ID
    found_patient = patient_management.find_patient_by_id(patient_1.id)
    if found_patient:
        print(f"✅ Paciente encontrado: {found_patient.name}")
    
    # Listar todos
    all_patients = patient_management.list_all_patients()
    print(f"✅ Total de pacientes cadastrados: {len(all_patients)}")
    
    for patient in all_patients:
        print(f"   - {patient.name} ({patient.age} anos, {patient.brain_event_type})")
    
    # 4. ATUALIZAÇÃO DE DADOS
    print("\n✏️ 4. Atualizando observações...")
    
    update_data = UpdatePatientNotesDTO(
        patient_id=patient_1.id,
        notes="Paciente apresentou melhora significativa na coordenação motora após 5 sessões de BCI"
    )
    
    updated_patient = patient_management.update_patient_notes(update_data)
    print(f"✅ Observações atualizadas para {updated_patient.name}")
    print(f"   Nova observação: {updated_patient.notes}")
    
    # 5. DEMONSTRAÇÃO DA REGRA DE DOMÍNIO
    print("\n⚡ 5. Demonstrando regras de domínio...")
    
    try:
        # Tentar criar paciente com dados inválidos
        invalid_patient = CreatePatientDTO(
            name="",  # Nome vazio - deve falhar
            age=45,
            gender="M",
            time_since_brain_event=12,
            brain_event_type="AVC",
            affected_side="Esquerdo",
            notes=""
        )
        patient_management.register_patient(invalid_patient)
    except Exception as e:
        print(f"✅ Validação funcionando: {str(e)}")
    
    # 6. DEMONSTRAÇÃO DA SEPARAÇÃO DE RESPONSABILIDADES
    print("\n🏗️ 6. Arquitetura Hexagonal em ação...")
    print("   📁 Domínio: Regras de negócio (Patient, PatientId)")
    print("   🔧 Aplicação: Casos de uso (PatientManagementUseCase)")
    print("   🔌 Ports: Interfaces (PatientRepositoryPort, PatientManagementInPort)")
    print("   📦 Adapters: Implementações (SqlitePatientRepository)")
    print("   📊 Interface: DTOs e Mappers")
    
    # 7. ESTATÍSTICAS FINAIS
    print("\n📊 7. Estatísticas do sistema...")
    print(f"   Total de pacientes: {len(all_patients)}")
    print(f"   Eventos recentes (< 6 meses): {sum(1 for p in all_patients if p.is_recent_event)}")
    print(f"   Pacientes adultos: {sum(1 for p in all_patients if p.age_group == 'Adulto')}")
    
    print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("✅ Sistema BCI com arquitetura hexagonal funcionando perfeitamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
