"""
Components for InternVL2 model architecture.
"""
from models.components.projection_head import ClassificationHead, CrossAttention, ResponseGenerator
from models.components.template_system import TemplateSelector

__all__ = ["ClassificationHead", "CrossAttention", "ResponseGenerator", "TemplateSelector"]