"""
Ab initio data generators for the InternVL2 Receipt Counter project.

This package provides high-quality, first-principles-based implementations for
generating synthetic receipts and tax documents for training and evaluation.
"""

from data.data_generators_new.receipt_generator import create_receipt
from data.data_generators_new.tax_document_generator import create_tax_document
from data.data_generators_new.create_multimodal_data import (
    create_synthetic_multimodal_data,
    generate_question_templates,
    generate_answer_templates,
    generate_qa_pair
)

__all__ = [
    # Receipt generation
    'create_receipt',
    
    # Tax document generation
    'create_tax_document',
    
    # Multimodal data generation
    'create_synthetic_multimodal_data',
    'generate_question_templates',
    'generate_answer_templates',
    'generate_qa_pair'
]