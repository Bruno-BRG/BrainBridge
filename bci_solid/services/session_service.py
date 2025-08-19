"""
Serviço de Sessões - Seguindo DIP e SRP

Este serviço depende de abstrações e tem uma única responsabilidade:
gerenciar operações de negócio relacionadas a sessões.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from ..interfaces.data_interfaces import ISessionRepository
from ..interfaces.service_interfaces import IValidationService
from ..models.session import Session, SessionType, SessionStatus


class SessionService:
    """
    Serviço de sessões seguindo DIP e SRP
    
    Responsabilidade única: Lógica de negócio para sessões
    Depende de abstrações (ISessionRepository, IValidationService)
    """
    
    def __init__(self, 
                 session_repository: ISessionRepository,
                 validation_service: Optional[IValidationService] = None):
        """
        Inicializa o serviço com dependências injetadas
        
        Args:
            session_repository: Repositório de dados de sessões
            validation_service: Serviço de validação (opcional)
        """
        self._session_repository = session_repository
        self._validation_service = validation_service
    
    def create_session(self, session_data: Dict[str, Any]) -> tuple[bool, str, Optional[int]]:
        """
        Cria uma nova sessão
        
        Args:
            session_data: Dados da sessão
            
        Returns:
            Tupla (sucesso, mensagem, session_id)
        """
        try:
            # Validar dados se serviço de validação estiver disponível
            if self._validation_service:
                validation_result = self._validation_service.validate_session_config(session_data)
                if validation_result:
                    errors = []
                    for field, field_errors in validation_result.items():
                        errors.extend(field_errors)
                    return False, "; ".join(errors), None
            
            # Criar modelo de sessão (validação automática)
            session = self._create_session_model(session_data)
            
            # Salvar no repositório
            session_id = self._session_repository.create_session(session.to_dict())
            
            return True, "Sessão criada com sucesso", session_id
            
        except ValueError as e:
            return False, f"Dados inválidos: {str(e)}", None
        except Exception as e:
            return False, f"Erro interno: {str(e)}", None
    
    def get_session(self, session_id: int) -> Optional[Session]:
        """
        Busca sessão por ID
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Sessão ou None se não encontrada
        """
        try:
            session_data = self._session_repository.get_session_by_id(session_id)
            if session_data:
                return Session.from_dict(session_data)
            return None
        except Exception:
            return None
    
    def get_sessions_by_patient(self, patient_id: int) -> List[Session]:
        """
        Retorna todas as sessões de um paciente
        
        Args:
            patient_id: ID do paciente
            
        Returns:
            Lista de sessões do paciente
        """
        try:
            sessions_data = self._session_repository.get_sessions_by_patient(patient_id)
            return [Session.from_dict(data) for data in sessions_data]
        except Exception:
            return []
    
    def start_session(self, session_id: int) -> tuple[bool, str]:
        """
        Inicia uma sessão
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False, "Sessão não encontrada"
            
            # Verificar se pode ser iniciada
            if session.status != SessionStatus.CREATED:
                return False, f"Sessão não pode ser iniciada no status '{session.status.value}'"
            
            # Iniciar sessão
            session.start_session()
            
            # Salvar no repositório
            success = self._session_repository.update_session(session_id, session.to_dict())
            
            if success:
                return True, "Sessão iniciada com sucesso"
            else:
                return False, "Erro ao iniciar sessão"
                
        except ValueError as e:
            return False, f"Erro de validação: {str(e)}"
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def pause_session(self, session_id: int) -> tuple[bool, str]:
        """
        Pausa uma sessão
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False, "Sessão não encontrada"
            
            # Pausar sessão
            session.pause_session()
            
            # Salvar no repositório
            success = self._session_repository.update_session(session_id, session.to_dict())
            
            if success:
                return True, "Sessão pausada com sucesso"
            else:
                return False, "Erro ao pausar sessão"
                
        except ValueError as e:
            return False, f"Erro de validação: {str(e)}"
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def resume_session(self, session_id: int) -> tuple[bool, str]:
        """
        Resume uma sessão pausada
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False, "Sessão não encontrada"
            
            # Resumir sessão
            session.resume_session()
            
            # Salvar no repositório
            success = self._session_repository.update_session(session_id, session.to_dict())
            
            if success:
                return True, "Sessão resumida com sucesso"
            else:
                return False, "Erro ao resumir sessão"
                
        except ValueError as e:
            return False, f"Erro de validação: {str(e)}"
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def complete_session(self, session_id: int) -> tuple[bool, str]:
        """
        Completa uma sessão
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False, "Sessão não encontrada"
            
            # Completar sessão
            session.complete_session()
            
            # Salvar no repositório
            success = self._session_repository.update_session(session_id, session.to_dict())
            
            if success:
                return True, "Sessão completada com sucesso"
            else:
                return False, "Erro ao completar sessão"
                
        except ValueError as e:
            return False, f"Erro de validação: {str(e)}"
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def cancel_session(self, session_id: int) -> tuple[bool, str]:
        """
        Cancela uma sessão
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False, "Sessão não encontrada"
            
            # Cancelar sessão
            session.cancel_session()
            
            # Salvar no repositório
            success = self._session_repository.update_session(session_id, session.to_dict())
            
            if success:
                return True, "Sessão cancelada com sucesso"
            else:
                return False, "Erro ao cancelar sessão"
                
        except ValueError as e:
            return False, f"Erro de validação: {str(e)}"
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def add_marker_to_session(self, session_id: int, marker_type: str) -> tuple[bool, str]:
        """
        Adiciona marcador a uma sessão
        
        Args:
            session_id: ID da sessão
            marker_type: Tipo do marcador
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False, "Sessão não encontrada"
            
            # Verificar se sessão está ativa
            if not session.is_active():
                return False, "Marcadores só podem ser adicionados em sessões ativas"
            
            # Adicionar marcador
            session.add_marker(marker_type)
            
            # Salvar no repositório
            success = self._session_repository.update_session(session_id, session.to_dict())
            
            if success:
                return True, f"Marcador {marker_type} adicionado com sucesso"
            else:
                return False, "Erro ao adicionar marcador"
                
        except Exception as e:
            return False, f"Erro interno: {str(e)}"
    
    def get_session_statistics(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Retorna estatísticas de uma sessão
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Dicionário com estatísticas ou None
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            return {
                'session_id': session.session_id,
                'patient_id': session.patient_id,
                'type': session.session_type.value,
                'status': session.status.value,
                'duration': session.get_duration_string(),
                'marker_count': session.marker_count,
                'start_time': session.start_time.isoformat() if session.start_time else None,
                'end_time': session.end_time.isoformat() if session.end_time else None
            }
            
        except Exception:
            return None
    
    def get_patient_session_summary(self, patient_id: int) -> Dict[str, Any]:
        """
        Retorna resumo das sessões de um paciente
        
        Args:
            patient_id: ID do paciente
            
        Returns:
            Dicionário com resumo
        """
        try:
            sessions = self.get_sessions_by_patient(patient_id)
            
            if not sessions:
                return {
                    'total_sessions': 0,
                    'by_type': {},
                    'by_status': {},
                    'total_duration_seconds': 0
                }
            
            # Calcular estatísticas
            total_duration = sum(
                s.duration_seconds for s in sessions 
                if s.duration_seconds is not None
            )
            
            return {
                'total_sessions': len(sessions),
                'by_type': self._count_by_attribute(sessions, 'session_type'),
                'by_status': self._count_by_attribute(sessions, 'status'),
                'total_duration_seconds': total_duration,
                'last_session': max(sessions, key=lambda s: s.start_time or datetime.min).to_dict() if sessions else None
            }
            
        except Exception:
            return {
                'total_sessions': 0,
                'by_type': {},
                'by_status': {},
                'total_duration_seconds': 0
            }
    
    def _create_session_model(self, session_data: Dict[str, Any]) -> Session:
        """
        Cria modelo de sessão a partir dos dados
        
        Args:
            session_data: Dados da sessão
            
        Returns:
            Instância de Session
        """
        # Converter string para enum se necessário
        session_type = SessionType(session_data['session_type']) if isinstance(session_data['session_type'], str) else session_data['session_type']
        
        return Session(
            patient_id=session_data['patient_id'],
            session_type=session_type,
            filename=session_data['filename'],
            notes=session_data.get('notes')
        )
    
    def _count_by_attribute(self, sessions: List[Session], attribute: str) -> Dict[str, int]:
        """
        Conta sessões por atributo
        
        Args:
            sessions: Lista de sessões
            attribute: Nome do atributo
            
        Returns:
            Dicionário com contagens
        """
        counts = {}
        for session in sessions:
            value = getattr(session, attribute)
            if hasattr(value, 'value'):  # Enum
                value = value.value
            counts[value] = counts.get(value, 0) + 1
        return counts
