#!/usr/bin/env python3
"""
Standalone receipt and tax document generator.

This script is a self-contained implementation that generates realistic
Australian receipts and tax documents without external dependencies
beyond the Python standard library and PIL/Pillow.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Create output directory
OUTPUT_DIR = "standalone_samples"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Font utility function
def get_font(size, bold=False):
    """Get a font of specified size with fallbacks."""
    # Try common fonts
    font_families = [
        "Arial", "Helvetica", "DejaVuSans", "FreeSans", "Liberation Sans", 
        "Nimbus Sans L", "Calibri", "Verdana", "Tahoma"
    ]
    
    if bold:
        # Try bold fonts
        for family in font_families:
            try:
                return ImageFont.truetype(f"{family} Bold", size)
            except IOError:
                try:
                    return ImageFont.truetype(f"{family}-Bold", size)
                except IOError:
                    continue
    
    # Try regular fonts
    for family in font_families:
        try:
            return ImageFont.truetype(family, size)
        except IOError:
            continue
    
    # Last resort fallback
    return ImageFont.load_default()


#===== RECEIPT GENERATOR =====

def create_receipt(image_size=2048):
    """
    Create a realistic Australian receipt.
    
    Args:
        image_size: Base size for the image (height will be adjusted)
        
    Returns:
        PIL Image of the receipt
    """
    # Determine receipt dimensions
    width = random.randint(300, 450)
    height = random.randint(1000, 1500)
    
    # Create blank white receipt
    receipt = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(receipt)
    
    # Get fonts with realistic sizing
    font_header = get_font(50, bold=True)
    font_subheader = get_font(40)
    font_body = get_font(30)
    font_small = get_font(20)
    
    # Set margin
    margin = 30
    
    # Generate store name (Australian stores)
    stores = ["WOOLWORTHS", "COLES", "ALDI", "IGA", "KMART", "TARGET", 
             "OFFICEWORKS", "DAN MURPHY'S", "JB HI-FI", "MYER", "BIG W"]
    store_name = random.choice(stores)
    
    # Center the store name
    store_name_width = font_header.getlength(store_name)
    store_name_x = max(margin, (width - store_name_width) // 2)
    draw.text((store_name_x, margin), store_name, fill="black", font=font_header)
    
    # Generate store address
    street_num = random.randint(1, 999)
    streets = ["Main St", "George St", "King St", "Queen St", "Market St"]
    street = random.choice(streets)
    cities = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
    city = random.choice(cities)
    states = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "NT", "ACT"]
    state = random.choice(states)
    postcode = random.randint(1000, 9999)
    
    store_address = f"{street_num} {street}, {city} {state} {postcode}"
    
    # Center the address
    address_width = font_small.getlength(store_address)
    address_x = max(margin, (width - address_width) // 2)
    current_y = margin + font_header.size + 10
    draw.text((address_x, current_y), store_address, fill="black", font=font_small)
    
    # Add separator line
    current_y += font_small.size + 15
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Add date and time
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2023)
    date_str = f"{day:02d}/{month:02d}/{year}"
    
    hour = random.randint(8, 21)
    minute = random.randint(0, 59)
    am_pm = "AM" if hour < 12 else "PM"
    if hour > 12:
        hour -= 12
    time_str = f"{hour}:{minute:02d} {am_pm}"
    
    draw.text((margin, current_y), "DATE:", fill="black", font=font_body)
    date_text_width = font_body.getlength("DATE:")
    draw.text((margin + date_text_width + 10, current_y), date_str, 
              fill="black", font=font_body)
    
    current_y += font_body.size + 10
    draw.text((margin, current_y), "TIME:", fill="black", font=font_body)
    time_text_width = font_body.getlength("TIME:")
    draw.text((margin + time_text_width + 10, current_y), time_str, 
              fill="black", font=font_body)
    
    # Add receipt number
    receipt_number = f"#{random.randint(10000, 999999)}"
    receipt_text = "RECEIPT:"
    receipt_text_width = font_body.getlength(receipt_text)
    
    current_y += font_body.size + 10
    draw.text((margin, current_y), receipt_text, fill="black", font=font_body)
    draw.text((margin + receipt_text_width + 10, current_y), receipt_number, 
              fill="black", font=font_body)
    
    # Add second separator
    current_y += font_body.size + 15
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Add item headers
    draw.text((margin, current_y), "ITEM", fill="black", font=font_body)
    
    # Calculate column positions
    qty_x = width - margin - 200
    price_x = width - margin - 120
    total_x = width - margin - 60
    
    draw.text((qty_x, current_y), "QTY", fill="black", font=font_body)
    draw.text((price_x, current_y), "PRICE", fill="black", font=font_body)
    draw.text((total_x, current_y), "TOTAL", fill="black", font=font_body)
    
    # Add header separator
    current_y += font_body.size + 5
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 10
    
    # Generate items
    items = [
        ("Milk 2L", 3.99),
        ("Bread", 2.49),
        ("Eggs 12pk", 3.29),
        ("Cheese", 4.99),
        ("Apples", 1.49),
        ("Bananas", 0.59),
        ("Chicken", 6.99),
        ("Rice 1kg", 3.49),
        ("Coffee", 7.99),
        ("Cereal", 4.29)
    ]
    
    # Select random items
    item_count = random.randint(5, 8)
    selected_items = random.sample(items, min(item_count, len(items)))
    
    # Track subtotal
    subtotal = 0
    
    # Draw items
    item_height = font_body.size + 10
    for name, price in selected_items:
        # Skip if we'd go past the bottom margin
        if current_y > height - 200:
            break
        
        # Randomize quantity (mostly 1, sometimes more)
        qty = 1
        if random.random() < 0.3:
            qty = random.randint(2, 5)
        
        # Calculate total
        total = round(price * qty, 2)
        subtotal += total
        
        # Format strings
        qty_str = str(qty)
        price_str = f"${price:.2f}"
        total_str = f"${total:.2f}"
        
        # Draw item details
        draw.text((margin, current_y), name, fill="black", font=font_body)
        
        # Right-align quantity, price, and total
        qty_width = font_body.getlength(qty_str)
        price_width = font_body.getlength(price_str)
        total_width = font_body.getlength(total_str)
        
        draw.text((qty_x + 30 - qty_width, current_y), qty_str, fill="black", font=font_body)
        draw.text((price_x + 30 - price_width, current_y), price_str, fill="black", font=font_body)
        draw.text((total_x + 30 - total_width, current_y), total_str, fill="black", font=font_body)
        
        current_y += item_height
    
    # Add separator before totals
    current_y += 10
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Calculate tax and total
    tax_rate = 0.10  # 10% GST in Australia
    tax = round(subtotal * tax_rate, 2)
    total = subtotal + tax
    
    # Format totals
    subtotal_str = f"${subtotal:.2f}"
    tax_str = f"${tax:.2f}"
    total_str = f"${total:.2f}"
    
    # Add subtotal
    draw.text((margin, current_y), "Subtotal:", fill="black", font=font_body)
    subtotal_width = font_body.getlength(subtotal_str)
    draw.text((width - margin - subtotal_width, current_y), subtotal_str, 
              fill="black", font=font_body)
    
    # Add tax
    current_y += item_height
    draw.text((margin, current_y), "GST (10%):", fill="black", font=font_body)
    tax_width = font_body.getlength(tax_str)
    draw.text((width - margin - tax_width, current_y), tax_str, 
              fill="black", font=font_body)
    
    # Add total with emphasis
    current_y += item_height + 5
    draw.text((margin, current_y), "TOTAL:", fill="black", font=font_subheader)
    total_width = font_subheader.getlength(total_str)
    draw.text((width - margin - total_width, current_y), total_str, 
              fill="black", font=font_subheader)
    
    # Add payment method
    current_y += font_subheader.size + 15
    payment_methods = ["EFTPOS", "VISA", "MASTERCARD", "CASH"]
    payment = random.choice(payment_methods)
    draw.text((margin, current_y), f"PAYMENT: {payment}", fill="black", font=font_body)
    
    # Add footer
    footer_y = height - margin - font_body.size
    thank_you = "THANK YOU FOR SHOPPING WITH US"
    thank_you_width = font_body.getlength(thank_you)
    thank_you_x = max(margin, (width - thank_you_width) // 2)
    
    draw.text((thank_you_x, footer_y), thank_you, fill="black", font=font_body)
    
    # Apply slight random rotation for realism
    if random.random() < 0.7:
        angle = random.uniform(-0.8, 0.8)
        receipt = receipt.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
    
    # Resize receipt to fit in the requested image size while maintaining aspect ratio
    scale_factor = min(image_size / receipt.width, image_size / receipt.height)
    new_width = int(receipt.width * scale_factor)
    new_height = int(receipt.height * scale_factor)
    receipt = receipt.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a white background image of the requested size
    background = Image.new('RGB', (image_size, image_size), 'white')
    
    # Paste receipt in the center
    paste_x = (image_size - new_width) // 2
    paste_y = (image_size - new_height) // 2
    background.paste(receipt, (paste_x, paste_y))
    
    return background


#===== TAX DOCUMENT GENERATOR =====

def create_tax_document(image_size=2048):
    """
    Create an Australian Taxation Office document.
    
    Args:
        image_size: Size of the output image
        
    Returns:
        PIL Image of a tax document
    """
    # Create white background
    doc = Image.new('RGB', (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(doc)
    
    # Define ATO blue color
    ato_blue = (0, 51, 160)
    
    # Define margin
    margin = int(image_size * 0.05)
    
    # Create ATO header
    # Draw blue banner at top
    banner_height = 80
    draw.rectangle([(0, 0), (image_size, banner_height)], fill=ato_blue)
    
    # Add ATO text
    header_font = get_font(60, bold=True)
    draw.text((20, 20), "Australian Taxation Office", fill="white", font=header_font)
    
    # Choose document type
    doc_types = [
        "Notice of Assessment",
        "Tax Return Summary",
        "Business Activity Statement",
        "PAYG Payment Summary",
        "Income Tax Assessment",
        "Superannuation Statement"
    ]
    doc_type = random.choice(doc_types)
    
    # Add document title
    title_y = 120
    title_font = get_font(50, bold=True)
    title_width = title_font.getlength(doc_type)
    title_x = (image_size - title_width) // 2
    draw.text((title_x, title_y), doc_type, fill="black", font=title_font)
    
    # Add document date
    date_y = title_y + title_font.size + 20
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2023)
    date_str = f"Date: {day:02d}/{month:02d}/{year}"
    
    date_font = get_font(30)
    date_width = date_font.getlength(date_str)
    date_x = image_size - margin - date_width
    draw.text((date_x, date_y), date_str, fill="black", font=date_font)
    
    # Add reference number
    ref_y = date_y + date_font.size + 10
    ref_num = f"Reference: {random.randint(100000, 999999)}"
    ref_width = date_font.getlength(ref_num)
    ref_x = image_size - margin - ref_width
    draw.text((ref_x, ref_y), ref_num, fill="black", font=date_font)
    
    # Add recipient section
    recipient_y = ref_y + date_font.size + 40
    draw.text((margin, recipient_y), "Taxpayer Details", 
             fill=ato_blue, font=get_font(40, bold=True))
    
    # Generate taxpayer name
    first_names = ["John", "Emma", "Michael", "Sarah", "David"]
    last_names = ["Smith", "Jones", "Williams", "Brown", "Wilson"]
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    
    # Anonymize some characters
    name_chars = list(name)
    for _ in range(3):
        pos = random.randint(0, len(name) - 1)
        if name_chars[pos] != " ":
            name_chars[pos] = "X"
    name = "".join(name_chars)
    
    # Add name
    name_y = recipient_y + 50
    draw.text((margin + 20, name_y), f"Name: {name}", 
             fill="black", font=date_font)
    
    # Add TFN
    tfn_y = name_y + date_font.size + 10
    tfn = f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    tfn_chars = list(tfn)
    for _ in range(5):
        pos = random.randint(0, len(tfn) - 1)
        if tfn_chars[pos] != " ":
            tfn_chars[pos] = "X"
    tfn = "".join(tfn_chars)
    
    draw.text((margin + 20, tfn_y), f"TFN: {tfn}", 
             fill="black", font=date_font)
    
    # Add address
    address_y = tfn_y + date_font.size + 10
    street_num = random.randint(1, 999)
    streets = ["Main St", "George St", "King St", "Queen St", "Market St"]
    street = random.choice(streets)
    cities = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
    city = random.choice(cities)
    states = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "NT", "ACT"]
    state = random.choice(states)
    postcode = random.randint(1000, 9999)
    
    address = f"{street_num} {street}, {city} {state} {postcode}"
    draw.text((margin + 20, address_y), f"Address: {address}", 
             fill="black", font=date_font)
    
    # Add separator
    sep_y = address_y + date_font.size + 30
    draw.line([(margin, sep_y), (image_size - margin, sep_y)], 
             fill=ato_blue, width=2)
    
    # Add financial details section
    finance_y = sep_y + 30
    draw.text((margin, finance_y), "Financial Summary", 
             fill=ato_blue, font=get_font(40, bold=True))
    
    # Create financial table
    table_y = finance_y + 60
    table_font = get_font(30)
    
    # Generate financial data
    gross_income = random.randint(50000, 120000)
    deductions = random.randint(2000, 15000)
    taxable_income = gross_income - deductions
    tax = int(taxable_income * 0.3)  # Simplified tax calculation
    medicare = int(taxable_income * 0.02)  # 2% Medicare levy
    tax_withheld = int(tax * random.uniform(0.9, 1.1))
    balance = tax + medicare - tax_withheld
    
    # Format table items
    items = [
        ("Gross Income", f"${gross_income:,}"),
        ("Deductions", f"${deductions:,}"),
        ("Taxable Income", f"${taxable_income:,}"),
        ("Tax on Taxable Income", f"${tax:,}"),
        ("Medicare Levy", f"${medicare:,}"),
        ("PAYG Tax Withheld", f"${tax_withheld:,}")
    ]
    
    # Draw table
    row_height = table_font.size + 20
    # First column width (not used but kept as reference for table design)
    # col1_width = 350
    
    for i, (label, value) in enumerate(items):
        row_y = table_y + i * row_height
        
        # Add zebra striping
        if i % 2 == 0:
            draw.rectangle([(margin, row_y), 
                           (image_size - margin, row_y + row_height)],
                          fill=(248, 248, 248))
        
        # Add text
        draw.text((margin + 20, row_y + 10), label, fill="black", font=table_font)
        
        # Right-align value
        value_width = table_font.getlength(value)
        draw.text((image_size - margin - value_width - 20, row_y + 10), 
                 value, fill="black", font=table_font)
    
    # Add result line
    result_y = table_y + len(items) * row_height + 20
    
    # Format result based on positive/negative balance
    if balance > 0:
        result_text = "Amount Payable:"
        balance_text = f"${balance:,}"
        color = ato_blue
    else:
        result_text = "Refund Amount:"
        balance_text = f"${-balance:,}"
        color = (0, 150, 0)  # Green for refund
    
    # Draw result with highlighting
    draw.rectangle([(margin, result_y), 
                   (image_size - margin, result_y + row_height + 10)],
                  fill=(240, 240, 250))
    
    # Add result text with emphasis
    result_font = get_font(35, bold=True)
    draw.text((margin + 20, result_y + 15), result_text, 
             fill="black", font=result_font)
    
    # Right-align result value
    balance_width = result_font.getlength(balance_text)
    draw.text((image_size - margin - balance_width - 20, result_y + 15), 
             balance_text, fill=color, font=result_font)
    
    # Add payment instructions if balance due
    if balance > 0:
        payment_y = result_y + row_height + 50
        due_day = random.randint(day, day + 30) % 28
        if due_day == 0:
            due_day = 28
        due_month = month if due_day > day else (month % 12) + 1
        
        payment_text = f"Payment due by: {due_day:02d}/{due_month:02d}/{year}"
        draw.text((margin, payment_y), payment_text, 
                 fill=ato_blue, font=table_font)
    
    # Add footer
    footer_y = image_size - 100
    draw.rectangle([(0, footer_y - 3), (image_size, footer_y)], 
                  fill=ato_blue, width=3)
    
    # Add footer text
    footer_font = get_font(25)
    footer_text = "THIS DOCUMENT MUST BE RETAINED FOR TAXATION PURPOSES"
    footer_width = footer_font.getlength(footer_text)
    footer_x = (image_size - footer_width) // 2
    
    draw.text((footer_x, footer_y + 15), footer_text, 
             fill="black", font=footer_font)
    
    # Apply subtle finishing effect
    if random.random() < 0.5:
        angle = random.uniform(-0.3, 0.3)
        doc = doc.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
        
        # Resize to original dimensions
        doc = doc.resize((image_size, image_size), Image.LANCZOS)
    
    return doc


def main():
    """Generate sample receipts and tax documents."""
    print(f"Generating samples in '{OUTPUT_DIR}'...")
    
    # Generate receipt samples
    for i in range(2):
        try:
            receipt = create_receipt(image_size=2048)
            receipt.save(f"{OUTPUT_DIR}/receipt_sample{i+1}.png")
            print(f"✅ Created {OUTPUT_DIR}/receipt_sample{i+1}.png")
        except Exception as e:
            print(f"❌ Error generating receipt sample {i+1}: {e}")
    
    # Generate tax document samples
    for i in range(2):
        try:
            tax_doc = create_tax_document(image_size=2048)
            tax_doc.save(f"{OUTPUT_DIR}/tax_document_sample{i+1}.png")
            print(f"✅ Created {OUTPUT_DIR}/tax_document_sample{i+1}.png")
        except Exception as e:
            print(f"❌ Error generating tax document sample {i+1}: {e}")
    
    print("\nSample generation complete!")


if __name__ == "__main__":
    main()