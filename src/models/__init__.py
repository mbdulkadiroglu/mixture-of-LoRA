"""
Model interfaces for the Adaptive SLM Framework.
"""

from .teacher import TeacherModel
from .student import StudentModel

__all__ = ["TeacherModel", "StudentModel"]
