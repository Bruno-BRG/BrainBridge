"""
Factories do Sistema BCI seguindo SOLID

Factories que implementam a criação de objetos complexos,
seguindo o padrão Factory e facilitando a injeção de dependências.
"""

from .service_factory import ServiceFactory

__all__ = [
    'ServiceFactory'
]
