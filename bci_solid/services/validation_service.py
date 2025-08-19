"""
Serviço de Validação - Seguindo SRP

Este serviço tem uma única responsabilidade:
validar dados de entrada do sistema.
"""

from typing import Dict, List, Any
import re
from ..interfaces.service_interfaces import IValidationService


class ValidationService(IValidationService):
    """
    Serviço de validação seguindo SRP
    
    Responsabilidade única: Validar dados de entrada
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o serviço de validação
        
        Args:
            config: Configurações de validação
        """
        self._config = config or {}
        
        # Configurações padrão
        self._min_name_length = self._config.get('min_name_length', 2)
        self._max_name_length = self._config.get('max_name_length', 100)
        self._min_age = self._config.get('min_age', 0)
        self._max_age = self._config.get('max_age', 150)
        self._max_time_since_event = self._config.get('max_time_since_event', 600)  # 50 anos em meses
    
    def validate_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Valida dados do paciente
        
        Args:
            patient_data: Dados do paciente
            
        Returns:
            Dicionário com erros por campo (vazio se válido)
        """
        errors = {}
        
        # Validar nome
        name_errors = self._validate_name(patient_data.get('name'))
        if name_errors:
            errors['name'] = name_errors
        
        # Validar idade
        age_errors = self._validate_age(patient_data.get('age'))
        if age_errors:
            errors['age'] = age_errors
        
        # Validar gênero
        gender_errors = self._validate_gender(patient_data.get('gender'))
        if gender_errors:
            errors['gender'] = gender_errors
        
        # Validar mão afetada
        hand_errors = self._validate_affected_hand(patient_data.get('affected_hand'))
        if hand_errors:
            errors['affected_hand'] = hand_errors
        
        # Validar tempo desde evento
        time_errors = self._validate_time_since_event(patient_data.get('time_since_event'))
        if time_errors:
            errors['time_since_event'] = time_errors
        
        # Validar notas (opcional)
        if 'notes' in patient_data:
            notes_errors = self._validate_notes(patient_data['notes'])
            if notes_errors:
                errors['notes'] = notes_errors
        
        return errors
    
    def validate_session_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Valida configuração da sessão
        
        Args:
            config: Configuração da sessão
            
        Returns:
            Dicionário com erros por campo (vazio se válido)
        """
        errors = {}
        
        # Validar ID do paciente
        patient_id_errors = self._validate_patient_id(config.get('patient_id'))
        if patient_id_errors:
            errors['patient_id'] = patient_id_errors
        
        # Validar tipo de sessão
        type_errors = self._validate_session_type(config.get('session_type'))
        if type_errors:
            errors['session_type'] = type_errors
        
        # Validar nome do arquivo
        filename_errors = self._validate_filename(config.get('filename'))
        if filename_errors:
            errors['filename'] = filename_errors
        
        # Validar configurações de streaming
        if 'streaming' in config:
            streaming_errors = self._validate_streaming_config(config['streaming'])
            if streaming_errors:
                errors['streaming'] = streaming_errors
        
        return errors
    
    def _validate_name(self, name: Any) -> List[str]:
        """Valida nome do paciente"""
        errors = []
        
        if name is None:
            errors.append("Nome é obrigatório")
            return errors
        
        if not isinstance(name, str):
            errors.append("Nome deve ser um texto")
            return errors
        
        name = name.strip()
        
        if len(name) < self._min_name_length:
            errors.append(f"Nome deve ter pelo menos {self._min_name_length} caracteres")
        
        if len(name) > self._max_name_length:
            errors.append(f"Nome deve ter no máximo {self._max_name_length} caracteres")
        
        # Verificar caracteres válidos
        if not re.match(r'^[a-zA-ZÀ-ÿ\s\'-]+$', name):
            errors.append("Nome contém caracteres inválidos")
        
        return errors
    
    def _validate_age(self, age: Any) -> List[str]:
        """Valida idade do paciente"""
        errors = []
        
        if age is None:
            errors.append("Idade é obrigatória")
            return errors
        
        if not isinstance(age, int):
            try:
                age = int(age)
            except (ValueError, TypeError):
                errors.append("Idade deve ser um número inteiro")
                return errors
        
        if age < self._min_age:
            errors.append(f"Idade deve ser pelo menos {self._min_age}")
        
        if age > self._max_age:
            errors.append(f"Idade deve ser no máximo {self._max_age}")
        
        return errors
    
    def _validate_gender(self, gender: Any) -> List[str]:
        """Valida gênero do paciente"""
        errors = []
        
        if gender is None:
            errors.append("Gênero é obrigatório")
            return errors
        
        valid_genders = ["Masculino", "Feminino", "Outro"]
        
        if gender not in valid_genders:
            errors.append(f"Gênero deve ser um dos: {', '.join(valid_genders)}")
        
        return errors
    
    def _validate_affected_hand(self, hand: Any) -> List[str]:
        """Valida mão afetada"""
        errors = []
        
        if hand is None:
            errors.append("Mão afetada é obrigatória")
            return errors
        
        valid_hands = ["Esquerda", "Direita", "Ambas", "Nenhuma"]
        
        if hand not in valid_hands:
            errors.append(f"Mão afetada deve ser uma das: {', '.join(valid_hands)}")
        
        return errors
    
    def _validate_time_since_event(self, time_value: Any) -> List[str]:
        """Valida tempo desde evento"""
        errors = []
        
        if time_value is None:
            errors.append("Tempo desde evento é obrigatório")
            return errors
        
        if not isinstance(time_value, int):
            try:
                time_value = int(time_value)
            except (ValueError, TypeError):
                errors.append("Tempo desde evento deve ser um número inteiro")
                return errors
        
        if time_value < 0:
            errors.append("Tempo desde evento deve ser não-negativo")
        
        if time_value > self._max_time_since_event:
            errors.append(f"Tempo desde evento deve ser no máximo {self._max_time_since_event} meses")
        
        return errors
    
    def _validate_notes(self, notes: Any) -> List[str]:
        """Valida notas do paciente"""
        errors = []
        
        if notes is not None and not isinstance(notes, str):
            errors.append("Notas devem ser um texto")
        
        if isinstance(notes, str) and len(notes) > 1000:
            errors.append("Notas devem ter no máximo 1000 caracteres")
        
        return errors
    
    def _validate_patient_id(self, patient_id: Any) -> List[str]:
        """Valida ID do paciente"""
        errors = []
        
        if patient_id is None:
            errors.append("ID do paciente é obrigatório")
            return errors
        
        if not isinstance(patient_id, int):
            try:
                patient_id = int(patient_id)
            except (ValueError, TypeError):
                errors.append("ID do paciente deve ser um número inteiro")
                return errors
        
        if patient_id <= 0:
            errors.append("ID do paciente deve ser positivo")
        
        return errors
    
    def _validate_session_type(self, session_type: Any) -> List[str]:
        """Valida tipo de sessão"""
        errors = []
        
        if session_type is None:
            errors.append("Tipo de sessão é obrigatório")
            return errors
        
        valid_types = ["baseline", "treino", "teste", "jogo"]
        
        if session_type not in valid_types:
            errors.append(f"Tipo de sessão deve ser um dos: {', '.join(valid_types)}")
        
        return errors
    
    def _validate_filename(self, filename: Any) -> List[str]:
        """Valida nome do arquivo"""
        errors = []
        
        if filename is None:
            errors.append("Nome do arquivo é obrigatório")
            return errors
        
        if not isinstance(filename, str):
            errors.append("Nome do arquivo deve ser um texto")
            return errors
        
        filename = filename.strip()
        
        if not filename:
            errors.append("Nome do arquivo não pode estar vazio")
            return errors
        
        # Verificar caracteres válidos para nome de arquivo
        invalid_chars = r'[<>:"/\\|?*]'
        if re.search(invalid_chars, filename):
            errors.append("Nome do arquivo contém caracteres inválidos")
        
        return errors
    
    def _validate_streaming_config(self, config: Dict[str, Any]) -> List[str]:
        """Valida configuração de streaming"""
        errors = []
        
        # Validar taxa de amostragem
        sampling_rate = config.get('sampling_rate')
        if sampling_rate is not None:
            if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
                errors.append("Taxa de amostragem deve ser um número positivo")
        
        # Validar número de canais
        num_channels = config.get('num_channels')
        if num_channels is not None:
            if not isinstance(num_channels, int) or num_channels <= 0:
                errors.append("Número de canais deve ser um inteiro positivo")
        
        return errors
