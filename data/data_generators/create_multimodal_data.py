#!/usr/bin/env python3
"""
Generate multimodal vision-language data for receipt counting.

This script creates data with receipt images and corresponding question-answer pairs
for training and evaluation of the multimodal receipt counter model.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from data.data_generators.create_receipt_collages import load_receipt_images
from data.data_generators.create_synthetic_receipts import create_receipt_image
from data.data_generators.receipt_processor import create_blank_image


def generate_question_templates() -> Dict[str, List[str]]:
    """
    Generate question templates for different types of queries.
    
    Returns:
        Dictionary with question types and templates
    """
    templates = {
        "counting": [
            "How many receipts are in this image?",
            "How many receipts do you see?",
            "How many receipts are visible?",
            "Count the number of receipts in this image.",
            "Count the receipts.",
            "How many receipts can you identify?",
            "What is the receipt count in this image?",
            "Tell me the number of receipts present.",
            "How many receipts are shown?",
            "Identify the number of receipts in this image.",
        ],
        "existence": [
            "Are there any receipts in this image?",
            "Does this image contain any receipts?",
            "Can you see any receipts?",
            "Is there at least one receipt in this image?",
            "Are receipts present in this image?",
            "Do you see any receipts in the image?",
            "Does this image show any receipts?",
            "Can you find any receipts in this image?",
            "Are receipts visible in this image?",
            "Is there evidence of receipts in this picture?",
        ],
        "value": [
            "What is the total value of the receipts?",
            "What is the combined value of all receipts?",
            "What's the total amount shown on these receipts?",
            "How much do these receipts sum up to?",
            "What's the total dollar amount of these receipts?",
            "What is the aggregate value of all receipts?",
            "How much is the total value shown on the receipts?",
            "What's the sum of the amounts on these receipts?",
            "What do these receipts add up to?",
            "What's the total sum of these receipts?",
        ],
        "detail": [
            "Which receipt has the highest value?",
            "Can you identify any restaurant receipts?",
            "Are there any grocery store receipts?",
            "What's the date on the receipt?",
            "When were these purchases made?",
            "What types of items were purchased?",
            "Which stores do these receipts come from?",
            "Are any of these receipts from the same store?",
            "What payment methods were used?",
            "How many different stores are represented in these receipts?",
        ]
    }
    
    return templates


def generate_answer_templates() -> Dict[str, List[str]]:
    """
    Generate answer templates for different types of queries.
    
    Returns:
        Dictionary with answer types and templates
    """
    templates = {
        "counting": [
            "There {is_are} {count} receipt{plural} in this image.",
            "I can see {count} receipt{plural}.",
            "The image contains {count} receipt{plural}.",
            "There {is_are} {count} receipt{plural} visible.",
            "I count {count} receipt{plural} in the image.",
            "This image shows {count} receipt{plural}.",
            "There {is_are} {count} receipt{plural} present.",
        ],
        "existence_yes": [
            "Yes, there {is_are} {count} receipt{plural} in this image.",
            "Yes, the image contains {count} receipt{plural}.",
            "Yes, I can see {count} receipt{plural}.",
            "Yes, there {is_are} receipt{plural} present in this image.",
            "Yes, the image shows {count} receipt{plural}.",
        ],
        "existence_no": [
            "No, there are no receipts in this image.",
            "No, I don't see any receipts.",
            "No, the image doesn't contain any receipts.",
            "No, there are no receipts visible.",
            "No, I can't find any receipts in this image.",
        ],
        "value": [
            "The total value of the receipts is ${total_value:.2f}.",
            "The combined value of all receipts is ${total_value:.2f}.",
            "The receipts sum up to ${total_value:.2f}.",
            "The total amount on these receipts is ${total_value:.2f}.",
            "The sum of the receipts is ${total_value:.2f}.",
        ],
        "detail_high_value": [
            "The {store_name} receipt has the highest value at ${highest_value:.2f}.",
            "The highest value receipt is from {store_name} at ${highest_value:.2f}.",
            "The most expensive receipt is from {store_name}, totaling ${highest_value:.2f}.",
            "The receipt with the highest amount is from {store_name} (${highest_value:.2f}).",
        ],
        "detail_stores": [
            "These receipts are from {store_list}.",
            "The stores on these receipts are {store_list}.",
            "The receipts come from {store_list}.",
            "These receipts are from the following stores: {store_list}.",
        ],
        "detail_date": [
            "The receipt is dated {date}.",
            "This receipt is from {date}.",
            "The purchase was made on {date}.",
            "The transaction occurred on {date}.",
        ],
        "detail_payment": [
            "The payment was made using {payment_method}.",
            "The receipt shows payment by {payment_method}.",
            "The customer paid with {payment_method}.",
            "{payment_method} was used for this transaction.",
        ]
    }
    
    return templates


def generate_qa_pair(receipt_count: int, receipt_values: Optional[List[float]] = None,
                     store_names: Optional[List[str]] = None, dates: Optional[List[str]] = None,
                     payment_methods: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Generate a question-answer pair for a given receipt image.
    
    Args:
        receipt_count: Number of receipts in the image
        receipt_values: List of receipt values (if available)
        store_names: List of store names (if available)
        dates: List of dates (if available)
        payment_methods: List of payment methods (if available)
        
    Returns:
        Dictionary with question and answer
    """
    # Get templates
    question_templates = generate_question_templates()
    answer_templates = generate_answer_templates()
    
    # Determine question types based on available information
    available_types = ["counting", "existence"]
    
    if receipt_values:
        available_types.append("value")
        
    if store_names and receipt_values:
        available_types.append("detail")
    
    # Weights for question types
    type_weights = {
        "counting": 0.4,
        "existence": 0.3,
        "value": 0.2,
        "detail": 0.1
    }
    
    # Filter to available types and normalize weights
    filtered_weights = {k: v for k, v in type_weights.items() if k in available_types}
    total_weight = sum(filtered_weights.values())
    normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}
    
    # Select question type
    question_type = random.choices(
        list(normalized_weights.keys()),
        weights=list(normalized_weights.values()),
        k=1
    )[0]
    
    # Select question template
    question = random.choice(question_templates[question_type])
    
    # Generate answer based on question type
    if question_type == "counting":
        is_are = "is" if receipt_count == 1 else "are"
        plural = "" if receipt_count == 1 else "s"
        answer_template = random.choice(answer_templates["counting"])
        answer = answer_template.format(
            count=receipt_count,
            is_are=is_are,
            plural=plural
        )
    
    elif question_type == "existence":
        if receipt_count > 0:
            is_are = "is" if receipt_count == 1 else "are"
            plural = "" if receipt_count == 1 else "s"
            answer_template = random.choice(answer_templates["existence_yes"])
            answer = answer_template.format(
                count=receipt_count,
                is_are=is_are,
                plural=plural
            )
        else:
            answer = random.choice(answer_templates["existence_no"])
    
    elif question_type == "value" and receipt_values:
        total_value = sum(receipt_values)
        answer_template = random.choice(answer_templates["value"])
        answer = answer_template.format(total_value=total_value)
    
    elif question_type == "detail" and store_names and receipt_values:
        # Choose detail sub-type
        detail_types = []
        
        if receipt_values and store_names and len(receipt_values) == len(store_names) and len(receipt_values) > 0:
            detail_types.append("detail_high_value")
        
        if store_names and len(store_names) > 0:
            detail_types.append("detail_stores")
            
        if dates and len(dates) > 0:
            detail_types.append("detail_date")
            
        if payment_methods and len(payment_methods) > 0:
            detail_types.append("detail_payment")
            
        if not detail_types:
            # Fallback to counting if no detailed info available
            is_are = "is" if receipt_count == 1 else "are"
            plural = "" if receipt_count == 1 else "s"
            answer_template = random.choice(answer_templates["counting"])
            answer = answer_template.format(
                count=receipt_count,
                is_are=is_are,
                plural=plural
            )
        else:
            detail_type = random.choice(detail_types)
            
            if detail_type == "detail_high_value" and receipt_values and store_names:
                # Find highest value receipt
                max_idx = receipt_values.index(max(receipt_values))
                highest_value = receipt_values[max_idx]
                store_name = store_names[max_idx]
                
                answer_template = random.choice(answer_templates["detail_high_value"])
                answer = answer_template.format(
                    store_name=store_name,
                    highest_value=highest_value
                )
                
            elif detail_type == "detail_stores" and store_names:
                if len(store_names) == 1:
                    store_list = store_names[0]
                elif len(store_names) == 2:
                    store_list = f"{store_names[0]} and {store_names[1]}"
                else:
                    store_list = ", ".join(store_names[:-1]) + f", and {store_names[-1]}"
                    
                answer_template = random.choice(answer_templates["detail_stores"])
                answer = answer_template.format(store_list=store_list)
                
            elif detail_type == "detail_date" and dates:
                date = random.choice(dates)
                answer_template = random.choice(answer_templates["detail_date"])
                answer = answer_template.format(date=date)
                
            elif detail_type == "detail_payment" and payment_methods:
                payment_method = random.choice(payment_methods)
                answer_template = random.choice(answer_templates["detail_payment"])
                answer = answer_template.format(payment_method=payment_method)
                
            else:
                # Final fallback to counting
                is_are = "is" if receipt_count == 1 else "are"
                plural = "" if receipt_count == 1 else "s"
                answer_template = random.choice(answer_templates["counting"])
                answer = answer_template.format(
                    count=receipt_count,
                    is_are=is_are,
                    plural=plural
                )
    else:
        # Fallback to counting for any other case
        is_are = "is" if receipt_count == 1 else "are"
        plural = "" if receipt_count == 1 else "s"
        answer_template = random.choice(answer_templates["counting"])
        answer = answer_template.format(
            count=receipt_count,
            is_are=is_are,
            plural=plural
        )
    
    return {
        "question": question,
        "answer": answer
    }


def create_synthetic_multimodal_data(num_samples: int, output_dir: Union[str, Path], 
                                    image_size: int = 448, seed: int = 42) -> pd.DataFrame:
    """
    Create a multimodal dataset with synthetic receipt images and QA pairs.
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save the generated data
        image_size: Size of output images
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with image filenames, receipt counts, questions, and answers
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Distribution of receipt counts (0-5)
    count_probs = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
    
    # Generate data
    data = []
    
    for i in range(num_samples):
        # Determine receipt count
        receipt_count = np.random.choice(len(count_probs), p=count_probs)
        
        # Create receipts with metadata
        receipts_info = []
        total_receipts = 0
        
        for _ in range(receipt_count):
            # Create single receipt with random items
            width = random.randint(300, 600)
            height = random.randint(800, 1500)
            items_count = random.randint(5, 15)
            receipt = create_receipt_image(width, height, items_count)
            
            # Extract value information (with some randomness since we can't perfectly parse the synthetic data)
            receipt_value = round(random.uniform(10.0, 200.0), 2)
            
            # Extract store names
            store_names = [
                "GROCERY WORLD", "SUPERMARKET PLUS", "FOOD MART", "MARKET PLACE", 
                "CONVENIENCE STORE", "FRESH FOODS", "MEGA MART", "VALUE STORE",
                "QUICK SHOP", "DAILY GOODS", "FAMILY MARKET"
            ]
            store_name = random.choice(store_names)
            
            # Extract dates
            months = ["January", "February", "March", "April", "May", "June", "July", 
                     "August", "September", "October", "November", "December"]
            month = random.choice(months)
            day = random.randint(1, 28)
            year = random.randint(2020, 2023)
            date = f"{month} {day}, {year}"
            
            # Extract payment methods
            payment_methods = ["VISA", "MASTERCARD", "AMEX", "CASH", "DEBIT"]
            payment_method = random.choice(payment_methods)
            
            # Scale receipt for collage
            scale_factor = min(image_size / 2 / receipt.width, image_size / 2 / receipt.height)
            new_width = int(receipt.width * scale_factor)
            new_height = int(receipt.height * scale_factor)
            receipt = receipt.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            receipts_info.append({
                "image": receipt,
                "value": receipt_value,
                "store_name": store_name,
                "date": date,
                "payment_method": payment_method
            })
            
            total_receipts += 1
        
        # Create collage
        collage = create_blank_image(image_size, image_size, 'white')
        
        if receipt_count == 0:
            # For no receipts, create a background with some texture
            collage = create_blank_image(image_size, image_size, 'white')
            draw = Image.new('RGB', (image_size, image_size), color=(255, 255, 255))
            collage = draw
        else:
            # Place receipts in collage
            for idx, receipt_info in enumerate(receipts_info):
                receipt = receipt_info["image"]
                
                if receipt_count == 1:
                    # Center the receipt
                    x_pos = (image_size - receipt.width) // 2
                    y_pos = (image_size - receipt.height) // 2
                else:
                    # Distribute receipts across the image
                    if idx % 2 == 0:  # Left side
                        x_pos = random.randint(10, image_size // 2 - receipt.width - 10)
                    else:  # Right side
                        x_pos = random.randint(image_size // 2 + 10, image_size - receipt.width - 10)
                        
                    y_pos = random.randint(10, image_size - receipt.height - 10)
                
                # Paste the receipt onto the collage
                collage.paste(receipt, (x_pos, y_pos))
        
        # Save image
        filename = f"multimodal_receipt_{i:05d}.png"
        collage.save(images_dir / filename)
        
        # Generate 3 different QA pairs for each image
        receipt_values = [info["value"] for info in receipts_info] if receipts_info else None
        store_names = [info["store_name"] for info in receipts_info] if receipts_info else None
        dates = [info["date"] for info in receipts_info] if receipts_info else None
        payment_methods = [info["payment_method"] for info in receipts_info] if receipts_info else None
        
        qa_pairs = []
        for _ in range(3):  # Generate 3 QA pairs per image
            qa_pair = generate_qa_pair(
                receipt_count=total_receipts,
                receipt_values=receipt_values,
                store_names=store_names,
                dates=dates,
                payment_methods=payment_methods
            )
            qa_pairs.append(qa_pair)
        
        # Add to dataset
        for qa_idx, qa_pair in enumerate(qa_pairs):
            data.append({
                "filename": filename,
                "receipt_count": total_receipts,
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "qa_pair_idx": qa_idx
            })
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    # Create and save metadata
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    # Save QA pairs as JSON for easier review
    qa_data = df[["filename", "receipt_count", "question", "answer", "qa_pair_idx"]].to_dict(orient="records")
    with open(output_dir / "qa_pairs.json", "w") as f:
        json.dump(qa_data, f, indent=2)
    
    print(f"Dataset generation complete: {num_samples} images, {len(data)} QA pairs")
    print(f"Distribution of receipt counts: {df['receipt_count'].value_counts().sort_index()}")
    
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate multimodal receipt dataset")
    parser.add_argument("--output_dir", default="multimodal_receipts", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--image_size", type=int, default=448, help="Size of output images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    df = create_synthetic_multimodal_data(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        image_size=args.image_size,
        seed=args.seed
    )