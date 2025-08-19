"""
Controller de Pacientes - Seguindo MVP e SOLID

Este controller gerencia a interação entre a view de pacientes
e os serviços de negócio, seguindo o padrão MVP.
"""

from typing import Dict, Any, Optional, List
from ..interfaces.ui_interfaces import IPatientView, IController
from ..services.patient_service import PatientService
from ..models.patient import Patient, Gender, AffectedHand


class PatientController(IController):
    """
    Controller de pacientes seguindo MVP
    
    Responsabilidade: Mediar entre view e serviços de pacientes
    """
    
    def __init__(self, patient_service: PatientService):
        """
        Inicializa controller com dependências injetadas
        
        Args:
            patient_service: Serviço de pacientes
        """
        self._patient_service = patient_service
        self._view: Optional[IPatientView] = None
        self._current_patients: List[Patient] = []
    
    def set_view(self, view: IPatientView) -> None:
        """Define a view associada"""
        self._view = view
    
    def initialize(self) -> bool:
        """
        Inicializa o controller
        
        Returns:
            True se inicializado com sucesso
        """
        try:
            # Carregar dados iniciais
            self._load_patients()
            return True
        except Exception as e:
            print(f"Erro ao inicializar controller de pacientes: {e}")
            return False
    
    def handle_action(self, action: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Manipula ações da interface
        
        Args:
            action: Nome da ação
            data: Dados da ação (opcional)
            
        Returns:
            True se ação foi processada com sucesso
        """
        try:
            if action == "create_patient":
                return self._handle_create_patient(data or {})
            elif action == "update_patient":
                return self._handle_update_patient(data or {})
            elif action == "delete_patient":
                return self._handle_delete_patient(data or {})
            elif action == "load_patients":
                return self._load_patients()
            elif action == "search_patients":
                return self._handle_search_patients(data or {})
            elif action == "clear_form":
                return self._handle_clear_form()
            elif action == "select_patient":
                return self._handle_select_patient(data or {})
            else:
                print(f"Ação não reconhecida: {action}")
                return False
                
        except Exception as e:
            print(f"Erro ao processar ação '{action}': {e}")
            return False
    
    def get_patients(self) -> List[Patient]:
        """Retorna lista atual de pacientes"""
        return self._current_patients.copy()
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Patient]:
        """Busca paciente por ID"""
        return self._patient_service.get_patient(patient_id)
    
    def get_gender_options(self) -> List[str]:
        """Retorna opções de gênero"""
        return [gender.value for gender in Gender]
    
    def get_affected_hand_options(self) -> List[str]:
        """Retorna opções de mão afetada"""
        return [hand.value for hand in AffectedHand]
    
    def _handle_create_patient(self, data: Dict[str, Any]) -> bool:
        """
        Manipula criação de paciente
        
        Args:
            data: Dados do paciente
            
        Returns:
            True se criado com sucesso
        """
        try:
            # Validar dados obrigatórios
            required_fields = ['name', 'age', 'gender', 'affected_hand', 'time_since_event']
            for field in required_fields:
                if field not in data or data[field] is None:
                    self._show_error(f"Campo '{field}' é obrigatório")
                    return False
            
            # Criar paciente
            success, message, patient_id = self._patient_service.create_patient(data)
            
            if success:
                self._show_success(message)
                self._load_patients()  # Recarregar lista
                if self._view:
                    self._view.clear_form()
                return True
            else:
                self._show_error(message)
                return False
                
        except Exception as e:
            self._show_error(f"Erro inesperado: {str(e)}")
            return False
    
    def _handle_update_patient(self, data: Dict[str, Any]) -> bool:
        """
        Manipula atualização de paciente
        
        Args:
            data: Dados para atualização
            
        Returns:
            True se atualizado com sucesso
        """
        try:
            patient_id = data.get('patient_id')
            if not patient_id:
                self._show_error("ID do paciente é obrigatório para atualização")
                return False
            
            # Remover patient_id dos dados de atualização
            update_data = {k: v for k, v in data.items() if k != 'patient_id'}
            
            # Atualizar paciente
            success, message = self._patient_service.update_patient(patient_id, update_data)
            
            if success:
                self._show_success(message)
                self._load_patients()  # Recarregar lista
                return True
            else:
                self._show_error(message)
                return False
                
        except Exception as e:
            self._show_error(f"Erro inesperado: {str(e)}")
            return False
    
    def _handle_delete_patient(self, data: Dict[str, Any]) -> bool:
        """
        Manipula exclusão de paciente
        
        Args:
            data: Dados contendo ID do paciente
            
        Returns:
            True se excluído com sucesso
        """
        try:
            patient_id = data.get('patient_id')
            if not patient_id:
                self._show_error("ID do paciente é obrigatório para exclusão")
                return False
            
            # Confirmar exclusão (a view deveria fazer isso, mas vamos verificar)
            patient = self._patient_service.get_patient(patient_id)
            if not patient:
                self._show_error("Paciente não encontrado")
                return False
            
            # Excluir paciente
            success, message = self._patient_service.delete_patient(patient_id)
            
            if success:
                self._show_success(message)
                self._load_patients()  # Recarregar lista
                if self._view:
                    self._view.clear_form()
                return True
            else:
                self._show_error(message)
                return False
                
        except Exception as e:
            self._show_error(f"Erro inesperado: {str(e)}")
            return False
    
    def _load_patients(self) -> bool:
        """
        Carrega lista de pacientes
        
        Returns:
            True se carregado com sucesso
        """
        try:
            self._current_patients = self._patient_service.get_all_patients()
            
            if self._view:
                # Converter para formato da view
                patients_data = [patient.to_dict() for patient in self._current_patients]
                self._view.display_patients(patients_data)
            
            return True
            
        except Exception as e:
            self._show_error(f"Erro ao carregar pacientes: {str(e)}")
            return False
    
    def _handle_search_patients(self, data: Dict[str, Any]) -> bool:
        """
        Manipula busca de pacientes
        
        Args:
            data: Dados contendo termo de busca
            
        Returns:
            True se busca executada com sucesso
        """
        try:
            query = data.get('query', '').strip()
            
            if not query:
                # Se busca vazia, mostrar todos
                return self._load_patients()
            
            # Executar busca
            found_patients = self._patient_service.search_patients(query)
            
            if self._view:
                # Converter para formato da view
                patients_data = [patient.to_dict() for patient in found_patients]
                self._view.display_patients(patients_data)
            
            return True
            
        except Exception as e:
            self._show_error(f"Erro na busca: {str(e)}")
            return False
    
    def _handle_clear_form(self) -> bool:
        """
        Manipula limpeza do formulário
        
        Returns:
            True se formulário limpo com sucesso
        """
        try:
            if self._view:
                self._view.clear_form()
            return True
        except Exception as e:
            print(f"Erro ao limpar formulário: {e}")
            return False
    
    def _handle_select_patient(self, data: Dict[str, Any]) -> bool:
        """
        Manipula seleção de paciente
        
        Args:
            data: Dados contendo ID do paciente selecionado
            
        Returns:
            True se seleção processada com sucesso
        """
        try:
            patient_id = data.get('patient_id')
            if not patient_id:
                return False
            
            # Buscar dados do paciente
            patient = self._patient_service.get_patient(patient_id)
            if not patient:
                self._show_error("Paciente não encontrado")
                return False
            
            # Mostrar no formulário
            if self._view:
                self._view.show_patient_form(patient.to_dict())
            
            return True
            
        except Exception as e:
            self._show_error(f"Erro ao selecionar paciente: {str(e)}")
            return False
    
    def _show_error(self, message: str) -> None:
        """Exibe mensagem de erro"""
        # Em uma implementação real, isso seria feito através da view
        print(f"ERRO: {message}")
        
        # Se view estiver disponível, usar método específico
        if self._view and hasattr(self._view, 'show_error'):
            self._view.show_error(message)
    
    def _show_success(self, message: str) -> None:
        """Exibe mensagem de sucesso"""
        # Em uma implementação real, isso seria feito através da view
        print(f"SUCESSO: {message}")
        
        # Se view estiver disponível, usar método específico
        if self._view and hasattr(self._view, 'show_success'):
            self._view.show_success(message)
