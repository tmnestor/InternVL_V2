"""
Ab initio data generators for the InternVL2 Receipt Counter project.

This package provides high-quality, first-principles-based implementations for
generating synthetic receipts and tax documents for training and evaluation.
"""

from data.generators.create_multimodal_data import (
    create_synthetic_multimodal_data,
    generate_answer_templates,
    generate_qa_pair,
    generate_question_templates,
)
from data.generators.receipt_generator import create_receipt
from data.generators.tax_document_generator import create_tax_document

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