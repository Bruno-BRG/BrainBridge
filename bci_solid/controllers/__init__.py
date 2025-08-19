"""
Controllers do Sistema BCI seguindo SOLID

Controllers que implementam a lógica de apresentação,
seguindo o padrão MVP e os princípios SOLID.
"""

from .patient_controller import PatientController

__all__ = [
    'PatientController'
]
