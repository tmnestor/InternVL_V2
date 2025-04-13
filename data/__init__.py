"""
Data handling module for the InternVL2 receipt counter.
"""
from data.dataset import ReceiptDataset, create_dataloaders

# Import the ab initio data generators
try:
    from data.data_generators_new import (
        create_receipt,
        create_synthetic_multimodal_data,
        create_tax_document,
    )
    
    # Update __all__ with the new components
    __all__ = [
        # Dataset components
        "ReceiptDataset", 
        "create_dataloaders",
        
        # Ab initio data generators
        "create_receipt",
        "create_tax_document",
        "create_synthetic_multimodal_data"
    ]
except ImportError:
    # Keep original __all__ if new components aren't available
    __all__ = ["ReceiptDataset", "create_dataloaders"]