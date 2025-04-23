# Ab Initio Zero-Shot Receipt Information Extraction Using Llama-Vision

## 1. Overview

This document outlines the implementation plan for building a zero-shot information extraction system for receipts using Llama-3.2-11B-Vision. The system will extract structured information from receipts without requiring any receipt-specific training examples, operating purely "ab initio" (from first principles).

### 1.1 Key Components

1. **Model Layer**: Llama-3.2-11B-Vision for vision-language understanding
2. **Extraction Framework**: Prompt engineering and structured output generation
3. **Post-processing Layer**: Validation, normalization, and structured data conversion
4. **Evaluation Module**: Metrics calculation against synthetic datasets

### 1.2 Target Information Fields

The system will extract the following fields from receipts without prior training:

- Store/Business Name
- Date of Purchase
- Time of Purchase
- Total Amount
- Payment Method
- Receipt Number/ID
- Individual Items (with prices)
- Tax Information
- Discounts/Promotions

## 2. Implementation Plan

### 2.1 Model Layer: Llama-Vision Integration

#### 2.1.1 Model Initialization

```python
# models/extractors/llama_vision_extractor.py
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaVisionExtractor:
    """Zero-shot receipt information extractor using Llama-Vision."""
    
    def __init__(
        self,
        model_path: str = "/path/to/meta-llama/Llama-3.2-11B-Vision",
        device: str = "cuda",
        use_8bit: bool = True,
        max_new_tokens: int = 1024,
    ):
        """Initialize the Llama-Vision extractor.
        
        Args:
            model_path: Path to Llama-Vision model
            device: Device to run inference on
            use_8bit: Whether to use 8-bit quantization
            max_new_tokens: Maximum number of tokens to generate
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Check if model path exists
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load model with appropriate settings
        self.logger.info(f"Loading Llama-Vision model from {model_path}")
        
        # Configure quantization if needed
        if use_8bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["vision_model", "vision_tower"]
                )
                self.logger.info("Using 8-bit quantization for memory efficiency")
            except ImportError:
                self.logger.warning("BitsAndBytesConfig not available, using default precision")
                quantization_config = None
        else:
            quantization_config = None
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        self.logger.info("Llama-Vision model loaded successfully")
```

#### 2.1.2 Image Processing

```python
# models/extractors/llama_vision_extractor.py (continued)
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class LlamaVisionExtractor:
    # ... previous code ...
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Preprocess image for Llama-Vision.
        
        Args:
            image_path: Path to receipt image
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not image_path.exists():
            raise ValueError(f"Image does not exist: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Apply preprocessing
        # Note: Llama-Vision preprocessing may vary - check model documentation
        transform = transforms.Compose([
            transforms.Resize(
                (336, 336),  # Llama-Vision optimal resolution
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        return transform(image).unsqueeze(0)  # Add batch dimension
```

### 2.2 Extraction Framework

#### 2.2.1 Structured Extraction Prompts

```python
# models/extractors/llama_vision_extractor.py (continued)
import json

class LlamaVisionExtractor:
    # ... previous code ...
    
    def _get_extraction_prompt(self, field: Optional[str] = None) -> str:
        """Get appropriate extraction prompt.
        
        Args:
            field: Specific field to extract, or None for all fields
            
        Returns:
            Formatted extraction prompt
        """
        base_prompt = """<image>
You are a receipt information extraction assistant. Analyze the receipt in the image and extract the following information accurately.
"""
        
        # Full extraction prompt (all fields)
        full_extraction_prompt = base_prompt + """
Extract ALL of the following information from the receipt:
1. Store/Business Name
2. Date of Purchase (YYYY-MM-DD format)
3. Time of Purchase (HH:MM format)
4. Total Amount (include currency)
5. Payment Method
6. Receipt Number/ID
7. Individual Items (with prices)
8. Tax Information (amount and/or percentage)
9. Discounts/Promotions (if any)

Provide your answer as a structured JSON object with these fields. For any field that cannot be found in the receipt, use null.
Format your response as a valid JSON object only, with no additional commentary or explanation. Ensure that numeric values are formatted accordingly.
"""
        
        # Field-specific prompts if needed
        field_prompts = {
            "store_name": base_prompt + "What is the store or business name on this receipt? Extract just the name.",
            "date": base_prompt + "What is the date of purchase on this receipt? Format as YYYY-MM-DD.",
            "total": base_prompt + "What is the total amount on this receipt? Include the currency symbol if visible.",
            "items": base_prompt + """Extract all individual items with their prices from this receipt.
Format your answer as a valid JSON array of objects with 'item_name', 'quantity', and 'price' fields.""",
        }
        
        if field is not None and field in field_prompts:
            return field_prompts[field]
        else:
            return full_extraction_prompt
```

#### 2.2.2 Extraction Functions

```python
# models/extractors/llama_vision_extractor.py (continued)
class LlamaVisionExtractor:
    # ... previous code ...
    
    def extract_field(self, image_path: Union[str, Path], field: str) -> Any:
        """Extract a specific field from receipt.
        
        Args:
            image_path: Path to receipt image
            field: Field to extract
            
        Returns:
            Extracted field value
        """
        # Process image
        pixel_values = self._preprocess_image(image_path)
        
        # Create prompt
        prompt = self._get_extraction_prompt(field)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pixel_values = pixel_values.to(self.device)
        
        # Extract the field using the model
        with torch.no_grad():
            # Prepare inputs with both text and image for Llama-Vision
            model_inputs = self.model.prepare_inputs_for_generation(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=pixel_values
            )
            
            # Generate response
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,  # Low temperature for deterministic extraction
                do_sample=False,  # No sampling for consistent results
                return_dict_in_generate=True,
                output_scores=False
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Extract the response part (after the prompt)
        response = generated_text[len(prompt):].strip()
        
        # Parse and return the field value
        return self._parse_field(response, field)
    
    def extract_all_fields(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract all receipt information fields.
        
        Args:
            image_path: Path to receipt image
            
        Returns:
            Dictionary with all extracted fields
        """
        # Process image
        pixel_values = self._preprocess_image(image_path)
        
        # Create prompt for full extraction
        prompt = self._get_extraction_prompt()
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pixel_values = pixel_values.to(self.device)
        
        # Extract all fields using the model
        with torch.no_grad():
            # Prepare inputs with both text and image for Llama-Vision
            model_inputs = self.model.prepare_inputs_for_generation(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=pixel_values
            )
            
            # Generate response
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=False
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Extract the response part (after the prompt)
        response = generated_text[len(prompt):].strip()
        
        # Parse the JSON response
        return self._parse_json_response(response)
```

### 2.3 Post-processing Layer

#### 2.3.1 Response Parsing and Validation

```python
# models/extractors/llama_vision_extractor.py (continued)
import re
import json
from datetime import datetime

class LlamaVisionExtractor:
    # ... previous code ...
    
    def _parse_field(self, response: str, field: str) -> Any:
        """Parse field-specific response.
        
        Args:
            response: Raw response from model
            field: Field type
            
        Returns:
            Parsed and validated field value
        """
        if field == "date":
            # Extract date in YYYY-MM-DD format
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', response)
            if date_match:
                return date_match.group(0)
            
            # Try other date formats and convert
            patterns = [
                # MM/DD/YYYY
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
                # DD/MM/YYYY
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"),
                # MM-DD-YYYY
                (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
            ]
            
            for pattern, formatter in patterns:
                match = re.search(pattern, response)
                if match:
                    return formatter(match)
            
            return None
            
        elif field == "total":
            # Extract amount with currency
            amount_match = re.search(r'(\$|€|£|\d)[\d\s,.]+', response)
            if amount_match:
                return amount_match.group(0).strip()
            return None
            
        elif field == "items":
            # Try to extract JSON array from response
            try:
                # Find JSON-like structure in response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    items_json = json_match.group(0)
                    return json.loads(items_json)
                return []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse items JSON: {response}")
                return []
                
        elif field == "store_name":
            # Return the first line or full response if short
            if '\n' in response:
                return response.split('\n')[0].strip()
            return response.strip()
            
        # Default case: return the response as is
        return response.strip()
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate JSON response.
        
        Args:
            response: Raw JSON response from model
            
        Returns:
            Validated and normalized extraction results
        """
        # Standard structure for output
        result = {
            "store_name": None,
            "date": None,
            "time": None,
            "total_amount": None,
            "payment_method": None,
            "receipt_id": None,
            "items": [],
            "tax_info": None,
            "discounts": None,
            "raw_extraction": response  # Store raw response for debugging
        }
        
        # Sanitize response to extract valid JSON
        # Sometimes model outputs extra text before/after JSON
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if not json_match:
            self.logger.warning(f"No valid JSON found in response: {response[:100]}...")
            return result
            
        json_str = json_match.group(1)
        
        # Parse JSON with error handling
        try:
            extracted = json.loads(json_str)
            
            # Map extracted fields to standard structure
            field_mappings = {
                "store_name": ["store_name", "store", "business_name", "business", "merchant", "merchant_name"],
                "date": ["date", "date_of_purchase", "purchase_date"],
                "time": ["time", "time_of_purchase", "purchase_time"],
                "total_amount": ["total_amount", "total", "amount", "total_price"],
                "payment_method": ["payment_method", "payment", "payment_type", "method"],
                "receipt_id": ["receipt_id", "receipt_number", "id", "transaction_id", "receipt_no"],
                "items": ["items", "line_items", "products", "purchases"],
                "tax_info": ["tax_info", "tax", "tax_amount", "tax_rate", "gst", "vat"],
                "discounts": ["discounts", "promotions", "discount_amount", "savings"]
            }
            
            # Fill in the result structure from extracted data
            for result_key, possible_keys in field_mappings.items():
                for key in possible_keys:
                    if key in extracted and extracted[key] is not None:
                        result[result_key] = extracted[key]
                        break
            
            # Apply validation to specific fields
            if result["date"]:
                # Normalize date format
                date_match = re.search(r'\d{4}-\d{1,2}-\d{1,2}', result["date"])
                if date_match:
                    # Ensure proper zero-padding
                    date_parts = result["date"].split('-')
                    if len(date_parts) == 3:
                        y, m, d = date_parts
                        result["date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            
            # Normalize total amount format
            if result["total_amount"]:
                # Strip any extra text around the amount
                amount_match = re.search(r'(\$|€|£|\d)[\d\s,.]+', str(result["total_amount"]))
                if amount_match:
                    result["total_amount"] = amount_match.group(0).strip()
                    
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Invalid JSON: {json_str[:100]}...")
            return result
```

### 2.4 Evaluation Module

#### 2.4.1 Evaluation Metrics and Framework

```python
# evaluation/receipt_extractor_evaluator.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from collections import defaultdict

class ReceiptExtractorEvaluator:
    """Evaluator for receipt information extraction using synthetic datasets."""
    
    def __init__(
        self,
        extractor,
        ground_truth_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize receipt extractor evaluator.
        
        Args:
            extractor: Initialized receipt extractor model
            ground_truth_path: Path to ground truth data file or directory
            output_dir: Path to save evaluation results
        """
        self.logger = logging.getLogger(__name__)
        self.extractor = extractor
        
        # Load ground truth data
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.logger.info(f"Loaded {len(self.ground_truth)} ground truth samples")
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_ground_truth(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load ground truth data.
        
        Args:
            path: Path to ground truth file or directory
            
        Returns:
            List of ground truth samples
        """
        path = Path(path)
        
        if path.is_file():
            # Load single file
            if path.suffix.lower() == ".json":
                with open(path, "r") as f:
                    return json.load(f)
            elif path.suffix.lower() == ".csv":
                return pd.read_csv(path).to_dict(orient="records")
        elif path.is_dir():
            # Load all JSON files in directory
            data = []
            for json_file in path.glob("*.json"):
                with open(json_file, "r") as f:
                    data.extend(json.load(f) if json_file.name == "metadata.json" else [json.load(f)])
            return data
        
        raise ValueError(f"Unsupported ground truth path: {path}")
    
    def evaluate(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate receipt extractor on ground truth data.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Evaluation metrics
        """
        # Select samples
        samples = self.ground_truth
        if sample_size and sample_size < len(samples):
            samples = np.random.choice(samples, size=sample_size, replace=False).tolist()
        
        self.logger.info(f"Evaluating on {len(samples)} samples")
        
        # Track metrics
        results = []
        field_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "error": 0})
        
        # Process each sample
        for i, sample in enumerate(samples):
            self.logger.info(f"Processing sample {i+1}/{len(samples)}")
            
            # Get image path
            image_path = sample.get("image_path")
            if not image_path:
                # Try to construct image path from metadata
                filename = sample.get("filename")
                if filename:
                    image_dir = Path(self.ground_truth_path).parent / "images"
                    image_path = image_dir / filename
            
            if not Path(image_path).exists():
                self.logger.warning(f"Image does not exist: {image_path}")
                continue
            
            # Extract all fields
            try:
                extraction = self.extractor.extract_all_fields(image_path)
                
                # Compare with ground truth for each field
                result = {
                    "image_path": str(image_path),
                    "metrics": {}
                }
                
                # Evaluate each field
                self._evaluate_field(result, sample, extraction, "store_name", field_metrics)
                self._evaluate_field(result, sample, extraction, "date", field_metrics)
                self._evaluate_field(result, sample, extraction, "total_amount", field_metrics)
                self._evaluate_field(result, sample, extraction, "receipt_id", field_metrics)
                self._evaluate_field(result, sample, extraction, "payment_method", field_metrics)
                
                # Items evaluation (if available)
                if "items" in sample and sample["items"] and "items" in extraction and extraction["items"]:
                    result["metrics"]["items"] = self._evaluate_items(sample["items"], extraction["items"])
                    
                    # Update field metrics for items
                    if result["metrics"]["items"]["match_rate"] >= 0.8:
                        field_metrics["items"]["correct"] += 1
                    elif result["metrics"]["items"]["match_rate"] >= 0.5:
                        field_metrics["items"]["partial"] = field_metrics["items"].get("partial", 0) + 1
                    
                    field_metrics["items"]["total"] += 1
                
                # Save full extraction for analysis
                result["extraction"] = extraction
                result["ground_truth"] = sample
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process sample {image_path}: {e}")
                field_metrics["overall"]["error"] += 1
                
        # Calculate overall metrics
        metrics = self._calculate_overall_metrics(field_metrics)
        
        # Save detailed results
        with open(self.output_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save summary metrics
        with open(self.output_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _evaluate_field(
        self, 
        result: Dict[str, Any], 
        ground_truth: Dict[str, Any], 
        extraction: Dict[str, Any], 
        field: str,
        field_metrics: Dict[str, Dict[str, int]]
    ) -> None:
        """Evaluate a specific extraction field.
        
        Args:
            result: Result dictionary to update
            ground_truth: Ground truth data
            extraction: Extracted data
            field: Field name to evaluate
            field_metrics: Metrics dictionary to update
        """
        # Skip if field is not in ground truth
        if field not in ground_truth or not ground_truth[field]:
            return
        
        # Get values
        gt_value = str(ground_truth[field]).strip().lower()
        extracted_value = str(extraction.get(field, "")).strip().lower()
        
        # Check for exact match
        exact_match = gt_value == extracted_value
        
        # Check for partial match (for longer text fields)
        partial_match = False
        if len(gt_value) > 5:
            # Calculate string similarity
            partial_match = (
                extracted_value in gt_value or
                gt_value in extracted_value or
                self._string_similarity(gt_value, extracted_value) > 0.8
            )
        
        # Update metrics
        result["metrics"][field] = {
            "ground_truth": ground_truth[field],
            "extracted": extraction.get(field),
            "exact_match": exact_match,
            "partial_match": partial_match if not exact_match else False
        }
        
        # Update field metrics
        field_metrics[field]["total"] += 1
        if exact_match:
            field_metrics[field]["correct"] += 1
            field_metrics["overall"]["correct"] += 1
        elif partial_match:
            field_metrics[field]["partial"] = field_metrics[field].get("partial", 0) + 1
            field_metrics["overall"]["partial"] = field_metrics["overall"].get("partial", 0) + 1
        
        field_metrics["overall"]["total"] += 1
    
    def _evaluate_items(self, ground_truth_items: List[Dict], extracted_items: List[Dict]) -> Dict[str, Any]:
        """Evaluate extracted items against ground truth.
        
        Args:
            ground_truth_items: List of ground truth items
            extracted_items: List of extracted items
            
        Returns:
            Item evaluation metrics
        """
        # Count matching items
        matched_items = 0
        partial_matches = 0
        
        # Track exact item matches
        item_matches = []
        
        # Compare each ground truth item with extracted items
        for gt_item in ground_truth_items:
            gt_name = str(gt_item.get("item_name", "")).lower()
            gt_price = str(gt_item.get("price", "")).lower()
            
            best_match = None
            best_score = 0
            
            # Find best matching extracted item
            for ext_item in extracted_items:
                ext_name = str(ext_item.get("item_name", "")).lower()
                ext_price = str(ext_item.get("price", "")).lower()
                
                # Calculate match score
                name_sim = self._string_similarity(gt_name, ext_name)
                
                # Price exact match adds weight
                price_match = gt_price in ext_price or ext_price in gt_price
                
                # Combined score
                score = name_sim + (0.5 if price_match else 0)
                
                # Update best match
                if score > best_score:
                    best_score = score
                    best_match = {
                        "item": ext_item,
                        "score": score,
                        "name_sim": name_sim,
                        "price_match": price_match
                    }
            
            # Record match quality
            if best_match:
                item_matches.append(best_match)
                
                if best_match["score"] > 1.3:  # High confidence match
                    matched_items += 1
                elif best_match["score"] > 0.8:  # Partial match
                    partial_matches += 1
        
        # Calculate metrics
        total_gt_items = len(ground_truth_items)
        total_extracted_items = len(extracted_items)
        
        precision = matched_items / total_extracted_items if total_extracted_items > 0 else 0
        recall = matched_items / total_gt_items if total_gt_items > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Include partial matches in match rate
        match_rate = (matched_items + 0.5 * partial_matches) / total_gt_items if total_gt_items > 0 else 0
        
        return {
            "matched_items": matched_items,
            "partial_matches": partial_matches,
            "total_gt_items": total_gt_items,
            "total_extracted_items": total_extracted_items,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "match_rate": match_rate
        }
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simple ratio-based approach).
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        # Simple implementation - in production, use difflib or another string similarity library
        if not str1 or not str2:
            return 0
            
        # Check for substring
        if str1 in str2 or str2 in str1:
            shorter = min(len(str1), len(str2))
            longer = max(len(str1), len(str2))
            return shorter / longer
        
        # Count common characters
        common = sum(1 for c in str1 if c in str2)
        total = max(len(str1), len(str2))
        
        return common / total if total > 0 else 0
    
    def _calculate_overall_metrics(self, field_metrics: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Calculate overall evaluation metrics.
        
        Args:
            field_metrics: Field-level metrics
            
        Returns:
            Overall metrics dictionary
        """
        metrics = {
            "fields": {},
            "overall": {}
        }
        
        # Calculate per-field metrics
        for field, stats in field_metrics.items():
            if field == "overall":
                continue
                
            total = stats["total"]
            if total == 0:
                continue
                
            correct = stats["correct"]
            partial = stats.get("partial", 0)
            
            accuracy = correct / total
            # Include partial matches at half weight
            match_rate = (correct + 0.5 * partial) / total
            
            metrics["fields"][field] = {
                "accuracy": accuracy,
                "match_rate": match_rate,
                "count": total
            }
        
        # Calculate overall metrics
        overall = field_metrics["overall"]
        total = overall["total"]
        if total > 0:
            metrics["overall"] = {
                "accuracy": overall["correct"] / total,
                "match_rate": (overall["correct"] + 0.5 * overall.get("partial", 0)) / total,
                "error_rate": overall.get("error", 0) / (total + overall.get("error", 0)),
                "total_fields_evaluated": total
            }
        
        return metrics
```

#### 2.4.2 Evaluation Script

```python
# scripts/evaluation/evaluate_receipt_extractor.py
import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from models.extractors.llama_vision_extractor import LlamaVisionExtractor
from evaluation.receipt_extractor_evaluator import ReceiptExtractorEvaluator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate receipt information extractor")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Llama-Vision model"
    )
    
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth data file or directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization for memory efficiency"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{args.output_dir}/evaluation.log")
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log arguments
    logger.info(f"Evaluation arguments: {args}")
    
    # Initialize receipt extractor
    logger.info(f"Initializing Llama-Vision extractor from {args.model_path}")
    extractor = LlamaVisionExtractor(
        model_path=args.model_path,
        device=args.device,
        use_8bit=args.use_8bit
    )
    
    # Initialize evaluator
    logger.info(f"Initializing evaluator with ground truth from {args.ground_truth}")
    evaluator = ReceiptExtractorEvaluator(
        extractor=extractor,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    logger.info("Starting evaluation")
    metrics = evaluator.evaluate(sample_size=args.sample_size)
    
    # Log results
    logger.info("Evaluation complete")
    logger.info(f"Overall accuracy: {metrics['overall']['accuracy']:.4f}")
    logger.info(f"Overall match rate: {metrics['overall']['match_rate']:.4f}")
    
    # Per-field results
    logger.info("Per-field accuracy:")
    for field, field_metrics in metrics["fields"].items():
        logger.info(f"  {field}: {field_metrics['accuracy']:.4f} (n={field_metrics['count']})")
    
    logger.info(f"Detailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

## 3. Usage Examples

### 3.1 Basic Extraction

```python
# Example of basic receipt information extraction
from models.extractors.llama_vision_extractor import LlamaVisionExtractor

# Initialize extractor
extractor = LlamaVisionExtractor(
    model_path="/path/to/meta-llama/Llama-3.2-11B-Vision",
    use_8bit=True
)

# Extract all fields from a receipt
result = extractor.extract_all_fields("path/to/receipt.jpg")

# Print structured results
print(f"Store: {result['store_name']}")
print(f"Date: {result['date']}")
print(f"Total: {result['total_amount']}")
print(f"Items: {len(result['items'])} items found")
for item in result['items']:
    print(f"  - {item.get('item_name')}: {item.get('price')}")
```

### 3.2 Batch Processing

```python
# Example of batch processing multiple receipts
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from models.extractors.llama_vision_extractor import LlamaVisionExtractor

# Initialize extractor
extractor = LlamaVisionExtractor(
    model_path="/path/to/meta-llama/Llama-3.2-11B-Vision",
    use_8bit=True
)

# Process a directory of receipt images
receipts_dir = Path("data/receipts")
output_file = "extracted_data.csv"

# Collect all receipt images
image_files = [f for f in receipts_dir.glob("*.jpg") if os.path.isfile(f)]
print(f"Found {len(image_files)} receipt images")

# Process each receipt
results = []
for image_file in tqdm(image_files, desc="Processing receipts"):
    # Extract information
    extraction = extractor.extract_all_fields(image_file)
    
    # Add filename to results
    extraction["filename"] = image_file.name
    
    # Append to results list
    results.append(extraction)

# Convert to dataframe
df = pd.DataFrame(results)

# Save results
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
```

### 3.3 Running Evaluation

```bash
# Run evaluation on synthetic dataset
python scripts/evaluation/evaluate_receipt_extractor.py \
    --model-path /path/to/meta-llama/Llama-3.2-11B-Vision \
    --ground-truth data/synthetic_receipts/metadata.json \
    --output-dir evaluation_results \
    --use-8bit
```

## 4. Implementation Timeline

| Phase | Task | Timeline | Dependencies |
|-------|------|----------|--------------|
| **1. Setup** | Environment preparation, model download | Day 1-2 | Llama-Vision model access |
| **2. Core Extraction** | Basic extractor implementation | Day 3-4 | Setup completion |
| **3. Post-processing** | Result parsing and validation | Day 5-6 | Core extraction |
| **4. Evaluation Framework** | Metrics implementation | Day 7-8 | Post-processing |
| **5. Testing** | Initial testing on sample data | Day 9-10 | All previous components |
| **6. Evaluation Run** | Full evaluation on synthetic dataset | Day 11-12 | Testing completion |
| **7. Optimization** | Performance improvements, error handling | Day 13-14 | Evaluation results |
| **8. Documentation** | Usage guides, API docs | Day 15 | All implementation |

## 5. Memory and Performance Considerations

### 5.1 Memory Requirements

- **Minimum GPU Memory**:
  - **Full Precision**: 40GB+ (not recommended)
  - **8-bit Quantization**: 20GB minimum (recommended)
  - **4-bit Quantization**: 12GB minimum (reduced accuracy)

### 5.2 Performance Optimization

- Batch size 1 for largest receipts
- Implement caching for multiple extractions of the same receipt
- Use quantization as needed based on available hardware
- Consider ROI cropping for better receipt visibility

### 5.3 Throughput Estimates

- Average processing time per receipt: 2-5 seconds
- Daily throughput (single GPU): ~15,000-20,000 receipts
- Concurrent processing for higher throughput

## 6. Conclusion

This implementation plan provides a comprehensive approach to "ab initio" zero-shot receipt information extraction using Llama-Vision, with no training required. The system leverages the pre-trained capabilities of Llama-3.2-11B-Vision and enhances them with specialized receipt-focused prompt engineering and post-processing.

The evaluation framework provides robust metrics for measuring extraction accuracy against synthetic datasets, allowing for continuous improvement in prompt design and output parsing without requiring model fine-tuning.

By following this plan, you will be able to build a production-ready receipt information extraction system that operates completely from first principles, leveraging the zero-shot capabilities of the Llama-Vision foundation model.