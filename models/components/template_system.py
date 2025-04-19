"""
Template system for structured response generation.

This module implements a hierarchical template system for generating
structured responses to questions about receipts and tax documents.
"""
from typing import Dict, Optional


class TemplateSelector:
    """
    Template selector for generating structured responses.
    
    Provides a hierarchical template system with templates for different
    question types and document categories.
    """
    
    def __init__(self):
        """Initialize template selector with predefined templates."""
        self._template_registry = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize the template registry with templates for each question type.
        
        Returns:
            Dictionary of templates organized by question type
        """
        templates = {
            # Document type templates
            "DOCUMENT_TYPE": {
                "tax_document": "This is a tax document from {issuer}.",
                "receipt": "This is a receipt from {store_name}.",
                "default": "This appears to be a {document_type}."
            },
            
            # Counting templates
            "COUNTING": {
                "zero": "There are no receipts in this image.",
                "single": "There is 1 receipt in this image.",
                "multiple": "There are {count} receipts in this image.",
                "default": "I can see {count} receipt(s) in this image."
            },
            
            # Existence templates
            "EXISTENCE": {
                "yes_single": "Yes, there is a receipt visible.",
                "yes_multiple": "Yes, there are {count} receipts visible.",
                "no": "No, there are no receipts visible in this image.",
                "default": "Yes, there {is_are} {count} receipt(s) visible."
            },
            
            # Detail extraction templates
            "DETAIL_EXTRACTION": {
                "store_name": "The receipt is from {store_name}.",
                "date": "The receipt is dated {date}.",
                "items": "The items purchased include: {items}.",
                "tax_document_detail": "This tax document contains information about {tax_detail}.",
                "default": "The receipt shows a purchase from {store_name} on {date}."
            },
            
            # Payment method templates
            "PAYMENT_INFO": {
                "card": "Payment was made using a credit/debit card.",
                "cash": "Payment was made in cash.",
                "other": "Payment was made using {payment_method}.",
                "default": "Payment was made using {payment_method}."
            },
            
            # Amount templates
            "AMOUNT": {
                "total": "The total amount on the receipt is ${amount}.",
                "subtotal": "The subtotal before tax is ${subtotal}.",
                "tax": "The tax amount is ${tax}.",
                "default": "The total amount is ${amount}."
            },
            
            # Tax document templates
            "TAX_INFO": {
                "form_type": "This is an {form_type} tax document.",
                "tax_year": "This tax document is for the {tax_year} financial year.",
                "abn": "The ABN on this document is {abn}.",
                "default": "This tax document contains information from the Australian Taxation Office."
            },
            
            # Fallback template
            "DEFAULT": {
                "default": "This image contains a {document_type}."
            }
        }
        
        return templates
    
    def select_template(
        self, 
        question_type: str, 
        document_class: int, 
        extracted_details: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Select the appropriate template based on question type and document details.
        
        Args:
            question_type: Classified question type (DOCUMENT_TYPE, COUNTING, etc.)
            document_class: Classified document class (0=tax_doc, 1-5=receipts)
            extracted_details: Dictionary of extracted document details
            
        Returns:
            Filled template string
        """
        if extracted_details is None:
            extracted_details = {}
        
        # Get templates for the question type, falling back to DEFAULT if not found
        templates = self._template_registry.get(question_type, self._template_registry.get("DEFAULT", {}))
        
        # Select and fill template based on document class and question type
        if question_type == "DOCUMENT_TYPE":
            if document_class == 0:
                return templates["tax_document"].format(
                    issuer=extracted_details.get("issuer", "Australian Taxation Office")
                )
            else:
                return templates["receipt"].format(
                    store_name=extracted_details.get("store_name", "a store")
                )
                
        elif question_type == "COUNTING":
            count = document_class if document_class > 0 else 0
            if count == 0:
                return templates["zero"]
            elif count == 1:
                return templates["single"]
            else:
                return templates["multiple"].format(count=count)
                
        elif question_type == "EXISTENCE":
            count = document_class if document_class > 0 else 0
            if count == 0:
                return templates["no"]
            elif count == 1:
                return templates["yes_single"]
            else:
                return templates["yes_multiple"].format(count=count)
                
        elif question_type == "DETAIL_EXTRACTION":
            # Handle store name
            if "store_name" in extracted_details and "store_name" in templates:
                return templates["store_name"].format(
                    store_name=extracted_details["store_name"]
                )
            # Handle date
            elif "date" in extracted_details and "date" in templates:
                return templates["date"].format(
                    date=extracted_details["date"]
                )
            # Handle items
            elif "items" in extracted_details and "items" in templates:
                return templates["items"].format(
                    items=extracted_details["items"]
                )
            # Handle tax document details
            elif document_class == 0 and "tax_detail" in extracted_details:
                return templates["tax_document_detail"].format(
                    tax_detail=extracted_details["tax_detail"]
                )
            # Default case
            else:
                return templates["default"].format(
                    store_name=extracted_details.get("store_name", "a store"),
                    date=extracted_details.get("date", "an unspecified date")
                )
                
        elif question_type == "PAYMENT_INFO":
            payment_method = extracted_details.get("payment_method", "an unspecified method")
            if "card" in payment_method.lower():
                return templates["card"]
            elif "cash" in payment_method.lower():
                return templates["cash"]
            else:
                return templates["other"].format(payment_method=payment_method)
                
        elif question_type == "AMOUNT":
            if "amount" in extracted_details:
                return templates["total"].format(amount=extracted_details["amount"])
            elif "subtotal" in extracted_details:
                return templates["subtotal"].format(subtotal=extracted_details["subtotal"])
            elif "tax" in extracted_details:
                return templates["tax"].format(tax=extracted_details["tax"])
            else:
                return templates["default"].format(amount="unknown")
                
        elif question_type == "TAX_INFO":
            if "form_type" in extracted_details:
                return templates["form_type"].format(form_type=extracted_details["form_type"])
            elif "tax_year" in extracted_details:
                return templates["tax_year"].format(tax_year=extracted_details["tax_year"])
            elif "abn" in extracted_details:
                return templates["abn"].format(abn=extracted_details["abn"])
            else:
                return templates["default"]
        
        # Fallback template
        default_templates = self._template_registry.get("DEFAULT", {})
        default_template = default_templates.get("default", "This image contains a {document_type}.")
        
        return default_template.format(
            document_type="tax document" if document_class == 0 else "receipt",
            count=document_class if document_class > 0 else 0
        )