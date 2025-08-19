"""
Factory de Serviços - Seguindo Factory Pattern e DIP

Esta factory cria instâncias de serviços com suas dependências
injetadas corretamente, seguindo o padrão Factory.
"""

from typing import Dict, Any, Optional
from ..interfaces.service_interfaces import IDataStreamingService, IDataProcessingService, IModelService, IValidationService
from ..interfaces.data_interfaces import IPatientRepository, ISessionRepository, IDataLogger
from ..services.patient_service import PatientService
from ..services.data_streaming_service import DataStreamingService


class ServiceFactory:
    """
    Factory para criação de serviços
    
    Responsabilidade: Criar e configurar serviços com dependências corretas
    """
    
    def __init__(self):
        """Inicializa a factory"""
        self._repositories: Dict[str, Any] = {}
        self._services: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configura a factory com parâmetros
        
        Args:
            config: Configurações da factory
        """
        self._config = config.copy()
    
    def register_repository(self, name: str, repository: Any) -> None:
        """
        Registra um repositório na factory
        
        Args:
            name: Nome do repositório
            repository: Instância do repositório
        """
        self._repositories[name] = repository
    
    def create_patient_service(self, 
                              patient_repository: Optional[IPatientRepository] = None,
                              validation_service: Optional[IValidationService] = None) -> PatientService:
        """
        Cria serviço de pacientes
        
        Args:
            patient_repository: Repositório de pacientes (opcional)
            validation_service: Serviço de validação (opcional)
            
        Returns:
            Instância de PatientService
        """
        # Usar repositório fornecido ou buscar registrado
        if patient_repository is None:
            patient_repository = self._repositories.get('patient_repository')
        
        if patient_repository is None:
            raise ValueError("PatientRepository é obrigatório para criar PatientService")
        
        # Usar validação fornecida ou buscar registrada
        if validation_service is None:
            validation_service = self._services.get('validation_service')
        
        # Criar serviço
        service = PatientService(
            patient_repository=patient_repository,
            validation_service=validation_service
        )
        
        # Registrar para reutilização
        self._services['patient_service'] = service
        
        return service
    
    def create_data_streaming_service(self,
                                    data_receiver: Optional[Any] = None,
                                    data_logger: Optional[IDataLogger] = None,
                                    buffer_size: Optional[int] = None) -> DataStreamingService:
        """
        Cria serviço de streaming de dados
        
        Args:
            data_receiver: Receptor de dados (opcional)
            data_logger: Logger de dados (opcional)
            buffer_size: Tamanho do buffer (opcional)
            
        Returns:
            Instância de DataStreamingService
        """
        # Usar configurações padrão se não fornecidas
        if buffer_size is None:
            buffer_size = self._config.get('streaming_buffer_size', 1000)
        
        # Buscar dependências registradas se não fornecidas
        if data_receiver is None:
            data_receiver = self._services.get('data_receiver')
        
        if data_logger is None:
            data_logger = self._services.get('data_logger')
        
        # Criar serviço
        service = DataStreamingService(
            data_receiver=data_receiver,
            data_logger=data_logger,
            buffer_size=buffer_size
        )
        
        # Registrar para reutilização
        self._services['data_streaming_service'] = service
        
        return service
    
    def create_validation_service(self) -> 'ValidationService':
        """
        Cria serviço de validação
        
        Returns:
            Instância de ValidationService
        """
        from ..services.validation_service import ValidationService
        
        # Criar serviço com configurações
        service = ValidationService(
            config=self._config.get('validation', {})
        )
        
        # Registrar para reutilização
        self._services['validation_service'] = service
        
        return service
    
    def create_session_service(self,
                              session_repository: Optional[ISessionRepository] = None,
                              validation_service: Optional[IValidationService] = None) -> 'SessionService':
        """
        Cria serviço de sessões
        
        Args:
            session_repository: Repositório de sessões (opcional)
            validation_service: Serviço de validação (opcional)
            
        Returns:
            Instância de SessionService
        """
        from ..services.session_service import SessionService
        
        # Usar repositório fornecido ou buscar registrado
        if session_repository is None:
            session_repository = self._repositories.get('session_repository')
        
        if session_repository is None:
            raise ValueError("SessionRepository é obrigatório para criar SessionService")
        
        # Usar validação fornecida ou buscar registrada
        if validation_service is None:
            validation_service = self._services.get('validation_service')
        
        # Criar serviço
        service = SessionService(
            session_repository=session_repository,
            validation_service=validation_service
        )
        
        # Registrar para reutilização
        self._services['session_service'] = service
        
        return service
    
    def get_service(self, name: str) -> Optional[Any]:
        """
        Obtém serviço registrado
        
        Args:
            name: Nome do serviço
            
        Returns:
            Instância do serviço ou None
        """
        return self._services.get(name)
    
    def get_repository(self, name: str) -> Optional[Any]:
        """
        Obtém repositório registrado
        
        Args:
            name: Nome do repositório
            
        Returns:
            Instância do repositório ou None
        """
        return self._repositories.get(name)
    
    def create_all_services(self) -> Dict[str, Any]:
        """
        Cria todos os serviços necessários
        
        Returns:
            Dicionário com todos os serviços criados
        """
        services = {}
        
        try:
            # Criar serviços na ordem correta de dependências
            
            # 1. Serviços sem dependências
            services['validation'] = self.create_validation_service()
            
            # 2. Serviços com repositórios
            if 'patient_repository' in self._repositories:
                services['patient'] = self.create_patient_service(
                    validation_service=services['validation']
                )
            
            if 'session_repository' in self._repositories:
                services['session'] = self.create_session_service(
                    validation_service=services['validation']
                )
            
            # 3. Serviços de streaming
            services['data_streaming'] = self.create_data_streaming_service()
            
            return services
            
        except Exception as e:
            print(f"Erro ao criar serviços: {e}")
            return services
    
    def reset(self) -> None:
        """Reseta a factory, removendo todas as instâncias criadas"""
        self._services.clear()
        self._repositories.clear()
        self._config.clear()
