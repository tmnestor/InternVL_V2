#!/usr/bin/env python3
"""
Ab initio receipt generator for the InternVL2 receipt counter project.

This module creates synthetic receipts from first principles, with realistic 
variations in layout, content, and styling to train robust receipt classification models.
"""
import random

from PIL import Image, ImageDraw, ImageFont

# Only import numpy if available, otherwise use fallbacks
try:
    import numpy as np
except ImportError:
    # Fallback implementations for numpy functions we use
    class NumpyFallback:
        def random(self):
            class Random:
                def uniform(self, low, high):
                    return random.uniform(low, high)
                def choice(self, arr, p=None, size=None):
                    if p is None:
                        return random.choice(arr)
                    # Simple weighted random choice
                    import bisect
                    cumulative_sum = []
                    cumsum = 0
                    for item in p:
                        cumsum += item
                        cumulative_sum.append(cumsum)
                    x = random.random()
                    i = bisect.bisect(cumulative_sum, x)
                    return arr[i]
                def randint(self, low, high):
                    return random.randint(low, high)
                def seed(self, seed):
                    random.seed(seed)
            return Random()
    np = NumpyFallback()

# Realistic parameters for receipt generation
RECEIPT_WIDTHS = range(300, 650)  # Australian receipt widths in pixels
RECEIPT_HEIGHTS = range(800, 2000)  # Heights with realistic aspect ratios
RECEIPT_ITEM_COUNTS = range(3, 20)  # Typical item counts
RECEIPT_MARGIN = 50  # Base margin in pixels
FONT_SIZES = {
    "header": range(45, 65),
    "subheader": range(35, 50),
    "body": range(25, 35),
    "small": range(18, 25)
}

# Australian store names for realistic receipts
STORE_NAMES = [
    "WOOLWORTHS", "COLES", "ALDI", "IGA", "BUNNINGS", "KMART", "TARGET", 
    "OFFICEWORKS", "BIG W", "DAN MURPHY'S", "BWS", "CHEMIST WAREHOUSE",
    "JB HI-FI", "HARVEY NORMAN", "REBEL", "SUPERCHEAP AUTO", "LIQUORLAND",
    "PRICELINE", "DAVID JONES", "MYER", "SPOTLIGHT", "THE GOOD GUYS",
    "PETBARN", "BCFAUSTRALIA", "RED ROOSTER", "KFC", "MACCAS", "HUNGRY JACK'S"
]

# Australian locations
CITIES = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra", 
          "Hobart", "Darwin", "Gold Coast", "Newcastle", "Wollongong", "Geelong"]
STATES = [
    ("NSW", range(2000, 2999)),
    ("VIC", range(3000, 3999)),
    ("QLD", range(4000, 4999)),
    ("SA", range(5000, 5999)),
    ("WA", range(6000, 6999)),
    ("TAS", range(7000, 7999)),
    ("NT", range(800, 899)),   # Fixed leading zero issue
    ("ACT", range(2600, 2620))
]

# Australian items and pricing
ITEMS = [
    # Groceries
    ("Milk 2L", range(300, 500), "ea"),
    ("Bread", range(280, 450), "loaf"),
    ("Free Range Eggs", range(400, 700), "dozen"),
    ("Bananas", range(250, 400), "kg"),
    ("Apples", range(350, 550), "kg"),
    ("Avocado", range(200, 500), "ea"),
    ("Tim Tams", range(350, 500), "pack"),
    ("Vegemite", range(400, 700), "jar"),
    ("Milo", range(800, 1100), "tin"),
    ("Weet-Bix", range(400, 550), "box"),
    ("Cheese", range(600, 900), "block"),
    ("Chicken Breast", range(900, 1300), "kg"),
    ("Mince Beef", range(800, 1200), "kg"),
    ("Coffee Pods", range(800, 1200), "box"),
    ("Tea Bags", range(400, 600), "box"),
    ("Pasta", range(200, 300), "pack"),
    ("Pasta Sauce", range(200, 400), "jar"),
    ("Rice 1kg", range(200, 400), "pack"),
    ("Frozen Peas", range(200, 350), "bag"),
    ("Toilet Paper", range(700, 1200), "pack"),
    ("Paper Towels", range(350, 550), "roll"),
    ("Washing Powder", range(800, 1500), "box"),
    ("Dishwashing Liquid", range(250, 400), "bottle"),
    # Beverages
    ("Coca-Cola 2L", range(300, 500), "bottle"),
    ("Bottled Water", range(200, 400), "pack"),
    ("Orange Juice", range(400, 600), "bottle"),
    ("Beer 6-pack", range(1800, 2500), "pack"),
    ("Wine", range(1200, 3000), "bottle"),
    # Household
    ("Batteries", range(500, 800), "pack"),
    ("Light Bulbs", range(600, 900), "pack"),
    ("Hand Soap", range(200, 400), "bottle"),
    ("Toothpaste", range(300, 500), "tube"),
    ("Shampoo", range(500, 800), "bottle"),
    ("Deodorant", range(300, 600), "can"),
    # Snacks
    ("Potato Chips", range(300, 500), "bag"),
    ("Chocolate", range(200, 400), "bar"),
    ("Ice Cream", range(500, 800), "tub"),
    ("Biscuits", range(200, 400), "pack"),
    # Fresh items
    ("Fresh Salad", range(400, 700), "pack"),
    ("Sushi Pack", range(800, 1200), "pack")
]

# Payment methods
PAYMENT_METHODS = [
    "EFTPOS", "VISA", "MASTERCARD", "AMEX", "CASH", "ZIP", "AFTERPAY", "PAYPAL"
]

# Receipt footer messages
FOOTER_MESSAGES = [
    "THANK YOU FOR SHOPPING WITH US",
    "PLEASE COME AGAIN",
    "SAVE YOUR RECEIPT FOR RETURNS",
    "GST INCLUDED IN TOTAL WHERE APPLICABLE",
    "ITEMS CANNOT BE RETURNED WITHOUT RECEIPT",
    "RECEIPT REQUIRED FOR EXCHANGES",
    "ITEMS SUBJECT TO AVAILABILITY",
    "PRICES VALID ON DATE OF PURCHASE ONLY",
    "OPEN 7 DAYS A WEEK"
]


def get_font(style="body", fallbacks=True):
    """
    Get a font of appropriate size for the given style with fallbacks.
    
    Args:
        style: Font style (header, subheader, body, small)
        fallbacks: Whether to try fallback fonts
        
    Returns:
        PIL ImageFont object
    """
    # Get random size from appropriate range
    size = random.choice(list(FONT_SIZES.get(style, FONT_SIZES["body"])))
    
    # Try to use different fonts in priority order
    font_families = [
        "Arial", "Helvetica", "DejaVuSans", "FreeSans", "Liberation Sans", 
        "Nimbus Sans L", "Courier New", "Courier", "Verdana", "Tahoma"
    ]
    
    if style in ["header", "subheader"]:
        # Try bold fonts for headers
        for family in font_families:
            try:
                return ImageFont.truetype(f"{family} Bold", size)
            except IOError:
                try:
                    return ImageFont.truetype(f"{family}-Bold", size)
                except IOError:
                    continue
    
    # Try regular fonts
    if fallbacks:
        for family in font_families:
            try:
                return ImageFont.truetype(family, size)
            except IOError:
                continue
    
    # Last resort fallback
    return ImageFont.load_default()


def create_standard_receipt(width, height, items_count=None):
    """
    Create a standard format receipt with store header, items, and totals.
    
    Args:
        width: Receipt width in pixels
        height: Receipt height in pixels
        items_count: Number of items (if None, randomized)
        
    Returns:
        PIL Image of the receipt
    """
    # Create blank white receipt
    receipt = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(receipt)
    
    # Get fonts with realistic sizing
    font_header = get_font("header")
    font_subheader = get_font("subheader")
    font_body = get_font("body")
    font_small = get_font("small")
    
    # Determine margins based on width
    margin = max(RECEIPT_MARGIN, int(width * 0.08))  # Dynamic margin
    
    # Generate store name and header
    store_name = random.choice(STORE_NAMES)
    
    # Center the store name
    store_name_width = font_header.getlength(store_name)
    store_name_x = max(margin, (width - store_name_width) // 2)
    draw.text((store_name_x, margin), store_name, fill="black", font=font_header)
    
    # Generate store address
    state, postcodes = random.choice(STATES)
    postcode = random.choice(list(postcodes))
    
    street_num = random.randint(1, 999)
    streets = ["Main St", "George St", "King St", "Queen St", "Market St", 
              "Elizabeth St", "Victoria St", "Park Rd", "Church St", "High St"]
    street = random.choice(streets)
    city = random.choice(CITIES)
    
    store_address = f"{street_num} {street}, {city} {state} {postcode}"
    
    # Center the address
    address_width = font_small.getlength(store_address)
    address_x = max(margin, (width - address_width) // 2)
    current_y = margin + font_header.size + 10
    draw.text((address_x, current_y), store_address, fill="black", font=font_small)
    
    # Add phone number (Australian format)
    area_code = random.choice([2, 3, 7, 8])  # Australian area codes
    phone_prefix = random.randint(1000, 9999)
    phone_suffix = random.randint(1000, 9999)
    phone = f"(0{area_code}) {phone_prefix} {phone_suffix}"
    
    phone_width = font_small.getlength(phone)
    phone_x = max(margin, (width - phone_width) // 2)
    current_y += font_small.size + 5
    draw.text((phone_x, current_y), phone, fill="black", font=font_small)
    
    # Add separator line
    current_y += font_small.size + 15
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Add date and time
    current_date = generate_date()
    current_time = generate_time()
    
    draw.text((margin, current_y), "DATE:", fill="black", font=font_body)
    date_text_width = font_body.getlength("DATE:")
    draw.text((margin + date_text_width + 10, current_y), current_date, 
              fill="black", font=font_body)
    
    current_y += font_body.size + 10
    draw.text((margin, current_y), "TIME:", fill="black", font=font_body)
    time_text_width = font_body.getlength("TIME:")
    draw.text((margin + time_text_width + 10, current_y), current_time, 
              fill="black", font=font_body)
    
    # Add receipt number
    receipt_number = f"#{random.randint(10000, 999999)}"
    receipt_text = "RECEIPT:"
    
    # Position receipt number on the right
    receipt_number_width = font_body.getlength(receipt_number)
    receipt_number_x = width - margin - receipt_number_width
    
    current_y += font_body.size + 10
    draw.text((margin, current_y), receipt_text, fill="black", font=font_body)
    draw.text((receipt_number_x, current_y), receipt_number, fill="black", font=font_body)
    
    # Add second separator
    current_y += font_body.size + 15
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Add item headers
    draw.text((margin, current_y), "ITEM", fill="black", font=font_body)
    
    # Calculate column positions based on width
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
    
    # Determine number of items
    if items_count is None:
        items_count = random.choice(list(RECEIPT_ITEM_COUNTS))
    
    # Select random items
    selected_items = random.sample(ITEMS, min(items_count, len(ITEMS)))
    if len(selected_items) < items_count:
        # Add some duplicates to reach requested count
        additional = random.choices(ITEMS, k=items_count - len(selected_items))
        selected_items.extend(additional)
    
    # Randomize order
    random.shuffle(selected_items)
    
    # Calculate item spacing
    item_height = font_body.size + 10
    
    # Track subtotal
    subtotal = 0
    
    # Draw items
    for name, price_range, _ in selected_items:  # use _ since we don't use the unit variable
        # Skip if we'd go past the bottom margin
        if current_y > height - 200:
            break
        
        # Randomize quantity (mostly 1, sometimes more)
        qty = 1
        if random.random() < 0.3:  # 30% chance of quantity > 1
            qty = random.randint(2, 5)
        
        # Calculate price with slight variation
        base_price = random.choice(list(price_range))
        price_variation = random.uniform(0.95, 1.05)
        price = round(base_price * price_variation) / 100.0  # Convert to dollars
        
        # Calculate total
        total = round(price * qty, 2)
        subtotal += total
        
        # Format strings
        qty_str = str(qty)
        price_str = f"${price:.2f}"
        total_str = f"${total:.2f}"
        
        # Truncate item name if too long
        max_name_width = qty_x - margin - 10
        item_name = name
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        
        # Draw item details
        draw.text((margin, current_y), item_name, fill="black", font=font_body)
        
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
    draw.text((margin, current_y), "SUBTOTAL:", fill="black", font=font_body)
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
    payment = random.choice(PAYMENT_METHODS)
    draw.text((margin, current_y), "PAYMENT METHOD:", fill="black", font=font_body)
    
    current_y += item_height
    draw.text((margin, current_y), payment, fill="black", font=font_body)
    
    # Add card details if applicable
    if payment not in ["CASH", "EFTPOS"]:
        current_y += item_height
        card_number = "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))
        draw.text((margin, current_y), card_number, fill="black", font=font_body)
        
        current_y += item_height
        auth_code = f"AUTH: {random.randint(100000, 999999)}"
        draw.text((margin, current_y), auth_code, fill="black", font=font_small)
    
    # Add footer message
    # Move to bottom of receipt with margin
    footer_y = height - margin - font_body.size * 2
    
    # Add a separator line
    draw.line([(margin, footer_y - 15), (width - margin, footer_y - 15)], 
              fill="black", width=1)
    
    # Select and center a footer message
    footer = random.choice(FOOTER_MESSAGES)
    footer_width = font_body.getlength(footer)
    footer_x = max(margin, (width - footer_width) // 2)
    
    draw.text((footer_x, footer_y), footer, fill="black", font=font_body)
    
    # Apply very slight random rotation for realism
    if random.random() < 0.7:  # 70% chance of slight rotation
        angle = random.uniform(-0.8, 0.8)
        receipt = receipt.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
    
    return receipt


def create_detailed_receipt(width, height, items_count=None):
    """
    Create a more detailed receipt with structured layout and advanced formatting.
    
    Args:
        width: Receipt width in pixels
        height: Receipt height in pixels
        items_count: Number of items (if None, randomized)
        
    Returns:
        PIL Image of the receipt
    """
    # Create blank white receipt
    receipt = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(receipt)
    
    # Get fonts with realistic sizing
    font_header = get_font("header")
    font_subheader = get_font("subheader")
    font_body = get_font("body")
    font_small = get_font("small")
    
    # Determine margins based on width
    margin = max(RECEIPT_MARGIN, int(width * 0.08))
    
    # Generate store name and header
    store_name = random.choice(STORE_NAMES)
    
    # Create a header box
    header_height = margin + font_header.size + 20
    draw.rectangle([(0, 0), (width, header_height)], 
                  fill=(240, 240, 240))  # Light gray background
    
    # Center the store name in the header
    store_name_width = font_header.getlength(store_name)
    store_name_x = max(margin, (width - store_name_width) // 2)
    draw.text((store_name_x, margin//2), store_name, fill="black", font=font_header)
    
    current_y = header_height + 10
    
    # Add receipt title
    receipt_title = "TAX INVOICE"
    title_width = font_subheader.getlength(receipt_title)
    title_x = max(margin, (width - title_width) // 2)
    draw.text((title_x, current_y), receipt_title, fill="black", font=font_subheader)
    
    current_y += font_subheader.size + 15
    
    # Generate store address and details
    state, postcodes = random.choice(STATES)
    postcode = random.choice(list(postcodes))
    
    street_num = random.randint(1, 999)
    streets = ["Main St", "George St", "King St", "Queen St", "Market St", 
              "Elizabeth St", "Victoria St", "Park Rd", "Church St", "High St"]
    street = random.choice(streets)
    city = random.choice(CITIES)
    
    store_address = f"{street_num} {street}, {city} {state} {postcode}"
    
    # Generate ABN (Australian Business Number)
    abn_parts = [
        f"{random.randint(10, 99)}",
        f"{random.randint(100, 999)}",
        f"{random.randint(100, 999)}",
        f"{random.randint(100, 999)}"
    ]
    abn = " ".join(abn_parts)
    
    # Center the address
    address_width = font_small.getlength(store_address)
    address_x = max(margin, (width - address_width) // 2)
    draw.text((address_x, current_y), store_address, fill="black", font=font_small)
    
    # Add ABN
    current_y += font_small.size + 5
    abn_text = f"ABN: {abn}"
    abn_width = font_small.getlength(abn_text)
    abn_x = max(margin, (width - abn_width) // 2)
    draw.text((abn_x, current_y), abn_text, fill="black", font=font_small)
    
    # Add phone number
    current_y += font_small.size + 5
    area_code = random.choice([2, 3, 7, 8])
    phone_prefix = random.randint(1000, 9999)
    phone_suffix = random.randint(1000, 9999)
    phone = f"Phone: (0{area_code}) {phone_prefix} {phone_suffix}"
    
    phone_width = font_small.getlength(phone)
    phone_x = max(margin, (width - phone_width) // 2)
    draw.text((phone_x, current_y), phone, fill="black", font=font_small)
    
    # Add separator line
    current_y += font_small.size + 15
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Add transaction details in two-column layout
    left_col_x = margin
    right_col_x = width // 2 + 20
    
    # Generate date in Australian format (DD/MM/YYYY)
    current_date = generate_date()
    current_time = generate_time()
    receipt_number = f"#{random.randint(10000, 999999)}"
    
    # Staff details
    staff_names = ["John", "Sarah", "Michael", "Emma", "David", "Jessica", "James"]
    staff_name = random.choice(staff_names)
    register_num = random.randint(1, 9)
    
    # Transaction details - left column
    draw.text((left_col_x, current_y), "Date:", fill="black", font=font_body)
    date_text_width = font_body.getlength("Date:")
    draw.text((left_col_x + date_text_width + 10, current_y), current_date, 
              fill="black", font=font_body)
    
    current_y += font_body.size + 10
    draw.text((left_col_x, current_y), "Time:", fill="black", font=font_body)
    time_text_width = font_body.getlength("Time:")
    draw.text((left_col_x + time_text_width + 10, current_y), current_time, 
              fill="black", font=font_body)
    
    # Reset Y position for right column
    detail_start_y = current_y - font_body.size - 10
    
    # Transaction details - right column
    draw.text((right_col_x, detail_start_y), "Receipt:", fill="black", font=font_body)
    receipt_text_width = font_body.getlength("Receipt:")
    draw.text((right_col_x + receipt_text_width + 10, detail_start_y), receipt_number, 
              fill="black", font=font_body)
    
    draw.text((right_col_x, detail_start_y + font_body.size + 10), 
              f"Served by: {staff_name} (Reg {register_num})", fill="black", font=font_body)
    
    # Continue from the end of the left column
    current_y += font_body.size + 15
    
    # Add second separator before items
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Add item headers
    draw.text((margin, current_y), "ITEM", fill="black", font=font_body)
    
    # Calculate column positions based on width
    qty_x = margin + int(width * 0.5)
    price_x = margin + int(width * 0.65)
    total_x = margin + int(width * 0.8)
    
    draw.text((qty_x, current_y), "QTY", fill="black", font=font_body)
    draw.text((price_x, current_y), "PRICE", fill="black", font=font_body)
    draw.text((total_x, current_y), "TOTAL", fill="black", font=font_body)
    
    # Add header separator
    current_y += font_body.size + 5
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 10
    
    # Determine number of items
    if items_count is None:
        items_count = random.choice(list(RECEIPT_ITEM_COUNTS))
    
    # Select random items
    selected_items = random.sample(ITEMS, min(items_count, len(ITEMS)))
    if len(selected_items) < items_count:
        # Add some duplicates to reach requested count
        additional = random.choices(ITEMS, k=items_count - len(selected_items))
        selected_items.extend(additional)
    
    # Randomize order
    random.shuffle(selected_items)
    
    # Calculate item spacing
    item_height = font_body.size + 15  # More spacing for detailed receipt
    
    # Track subtotal
    subtotal = 0
    
    # Draw items with alternating background for readability
    for i, (name, price_range, unit) in enumerate(selected_items):
        # Skip if we'd go past the bottom margin
        if current_y > height - 250:
            break
        
        # Add zebra striping
        if i % 2 == 0:
            draw.rectangle([(margin - 5, current_y - 3), 
                           (width - margin + 5, current_y + font_body.size + 5)],
                          fill=(248, 248, 248))  # Very light gray
        
        # Randomize quantity (mostly 1, sometimes more)
        qty = 1
        if random.random() < 0.3:  # 30% chance of quantity > 1
            qty = random.randint(2, 5)
        
        # Calculate price with slight variation
        base_price = random.choice(list(price_range))
        price_variation = random.uniform(0.95, 1.05)
        price = round(base_price * price_variation) / 100.0  # Convert to dollars
        
        # Calculate total
        total = round(price * qty, 2)
        subtotal += total
        
        # Format strings
        qty_str = str(qty)
        price_str = f"${price:.2f}"
        total_str = f"${total:.2f}"
        
        # Format item name with unit
        item_name = f"{name}"
        if unit and random.random() < 0.7:  # 70% chance to include unit
            item_name = f"{name} ({unit})"
            
        # Truncate if too long
        max_name_width = qty_x - margin - 20
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        
        # Draw item details
        draw.text((margin, current_y), item_name, fill="black", font=font_body)
        
        # Right-align quantity, price, and total
        qty_width = font_body.getlength(qty_str)
        price_width = font_body.getlength(price_str)
        total_width = font_body.getlength(total_str)
        
        draw.text((qty_x + 20 - qty_width, current_y), qty_str, fill="black", font=font_body)
        draw.text((price_x + 30 - price_width, current_y), price_str, fill="black", font=font_body)
        draw.text((total_x + 30 - total_width, current_y), total_str, fill="black", font=font_body)
        
        current_y += item_height
    
    # Add separator before totals
    current_y += 5
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
    
    # Add subtotal with right alignment
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
    
    # Highlight total with background
    total_bg_padding = 10
    total_text_width = font_subheader.getlength("TOTAL:") + 10 + font_subheader.getlength(total_str)
    total_bg_left = width - margin - total_text_width - total_bg_padding
    total_bg_right = width - margin + total_bg_padding
    total_bg_top = current_y - total_bg_padding
    total_bg_bottom = current_y + font_subheader.size + total_bg_padding
    
    draw.rectangle([(total_bg_left, total_bg_top), 
                   (total_bg_right, total_bg_bottom)],
                  fill=(240, 240, 240))  # Light gray background
    
    draw.text((margin, current_y), "TOTAL:", fill="black", font=font_subheader)
    total_width = font_subheader.getlength(total_str)
    draw.text((width - margin - total_width, current_y), total_str, 
              fill="black", font=font_subheader)
    
    current_y += font_subheader.size + 20
    
    # Add payment method in a box
    payment_box_top = current_y
    payment_box_height = 80
    
    # Draw box
    draw.rectangle([(margin, payment_box_top), 
                   (width - margin, payment_box_top + payment_box_height)],
                  outline="black", width=1)
    
    # Add payment method header
    payment_header_y = payment_box_top + 10
    payment_header = "PAYMENT DETAILS"
    payment_header_width = font_body.getlength(payment_header)
    payment_header_x = max(margin + 10, (width - payment_header_width) // 2)
    draw.text((payment_header_x, payment_header_y), payment_header, 
              fill="black", font=font_body)
    
    # Add payment type and amount
    payment = random.choice(PAYMENT_METHODS)
    
    # Add details specific to payment type
    payment_details_y = payment_header_y + font_body.size + 10
    
    # Left column - payment type
    draw.text((margin + 10, payment_details_y), f"Method: {payment}", 
              fill="black", font=font_body)
    
    # Right column - amount
    amount_text = f"Amount: {total_str}"
    amount_width = font_body.getlength(amount_text)
    draw.text((width - margin - amount_width - 10, payment_details_y), 
              amount_text, fill="black", font=font_body)
    
    # Add card details if applicable
    if payment not in ["CASH"]:
        current_y = payment_box_top + payment_box_height + 15
        
        # Card number and authorization
        card_number = "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))
        draw.text((margin, current_y), card_number, fill="black", font=font_body)
        
        current_y += font_body.size + 5
        auth_code = f"Authorization: {random.randint(100000, 999999)}"
        draw.text((margin, current_y), auth_code, fill="black", font=font_small)
        
        # Status (approved)
        current_y += font_small.size + 15
        status = "APPROVED"
        status_width = font_body.getlength(status)
        status_x = max(margin, (width - status_width) // 2)
        
        draw.text((status_x, current_y), status, fill="black", font=font_body)
    
    # Add footer - move to bottom of receipt with margin
    footer_y = height - margin - font_body.size * 3
    
    # Add a separator line
    draw.line([(margin, footer_y - 15), (width - margin, footer_y - 15)], 
              fill="black", width=1)
    
    # Add thank you message
    thank_you = "THANK YOU FOR SHOPPING WITH US"
    thank_you_width = font_body.getlength(thank_you)
    thank_you_x = max(margin, (width - thank_you_width) // 2)
    
    draw.text((thank_you_x, footer_y), thank_you, fill="black", font=font_body)
    
    # Add GST inclusive message
    gst_message = "All prices include GST where applicable"
    gst_width = font_small.getlength(gst_message)
    gst_x = max(margin, (width - gst_width) // 2)
    
    draw.text((gst_x, footer_y + font_body.size + 10), gst_message, 
              fill="black", font=font_small)
    
    # Apply slight random rotation for realism
    if random.random() < 0.7:  # 70% chance of rotation
        angle = random.uniform(-0.8, 0.8)
        receipt = receipt.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
    
    return receipt


def create_fancy_receipt(width, height, items_count=None):
    """
    Create a fancy styled receipt with decorative elements.
    
    Args:
        width: Receipt width in pixels
        height: Receipt height in pixels
        items_count: Number of items (if None, randomized)
        
    Returns:
        PIL Image of the receipt
    """
    # Create blank receipt with very light gray background for fancy look
    receipt = Image.new('RGB', (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(receipt)
    
    # Get fonts with realistic sizing
    font_header = get_font("header")
    font_subheader = get_font("subheader")
    font_body = get_font("body")
    font_small = get_font("small")
    
    # Determine margins based on width
    margin = max(RECEIPT_MARGIN, int(width * 0.1))  # Larger margin for fancy receipt
    
    # Generate store name
    store_name = random.choice(STORE_NAMES)
    
    # Create decorative header
    header_height = margin * 2 + font_header.size
    
    # Draw decorative pattern at top
    for i in range(5):
        y_pos = 20 + i * 8
        draw.line([(0, y_pos), (width, y_pos)], fill=(200, 200, 200), width=1)
    
    # Add store logo placeholder (circle or square)
    logo_size = min(width // 4, 100)
    logo_x = (width - logo_size) // 2
    logo_y = 50
    
    if random.random() < 0.5:
        # Circle logo
        draw.ellipse([(logo_x, logo_y), (logo_x + logo_size, logo_y + logo_size)], 
                    outline=(100, 100, 100), width=2)
    else:
        # Square logo
        draw.rectangle([(logo_x, logo_y), (logo_x + logo_size, logo_y + logo_size)], 
                      outline=(100, 100, 100), width=2)
    
    # Add store name below logo
    store_name_y = logo_y + logo_size + 20
    store_name_width = font_header.getlength(store_name)
    store_name_x = max(margin, (width - store_name_width) // 2)
    
    draw.text((store_name_x, store_name_y), store_name, fill="black", font=font_header)
    
    # Current Y position after header
    current_y = store_name_y + font_header.size + 15
    
    # Add receipt title with decorative elements
    receipt_title = "PURCHASE RECEIPT"
    title_width = font_subheader.getlength(receipt_title)
    title_x = max(margin, (width - title_width) // 2)
    
    # Draw decorative lines around title
    title_line_left = title_x - 20
    title_line_right = title_x + title_width + 20
    title_line_y = current_y + font_subheader.size // 2
    
    draw.line([(margin, title_line_y), (title_line_left, title_line_y)], 
              fill=(100, 100, 100), width=1)
    draw.line([(title_line_right, title_line_y), (width - margin, title_line_y)], 
              fill=(100, 100, 100), width=1)
    
    draw.text((title_x, current_y), receipt_title, fill="black", font=font_subheader)
    
    current_y += font_subheader.size + 25
    
    # Generate store address and details
    state, postcodes = random.choice(STATES)
    postcode = random.choice(list(postcodes))
    
    street_num = random.randint(1, 999)
    streets = ["Main St", "George St", "King St", "Queen St", "Market St", 
              "Elizabeth St", "Victoria St", "Park Rd", "Church St", "High St"]
    street = random.choice(streets)
    city = random.choice(CITIES)
    
    store_address = f"{street_num} {street}, {city} {state} {postcode}"
    
    # Center the address
    address_width = font_small.getlength(store_address)
    address_x = max(margin, (width - address_width) // 2)
    draw.text((address_x, current_y), store_address, fill="black", font=font_small)
    
    # Add phone number
    current_y += font_small.size + 5
    area_code = random.choice([2, 3, 7, 8])
    phone_prefix = random.randint(1000, 9999)
    phone_suffix = random.randint(1000, 9999)
    phone = f"(0{area_code}) {phone_prefix} {phone_suffix}"
    
    phone_width = font_small.getlength(phone)
    phone_x = max(margin, (width - phone_width) // 2)
    draw.text((phone_x, current_y), phone, fill="black", font=font_small)
    
    # Add decorative separator
    current_y += font_small.size + 20
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill=(150, 150, 150), width=1)
    
    # Add dotted line below
    for i in range(margin, width - margin, 10):
        draw.line([(i, current_y + 5), (i + 5, current_y + 5)], 
                  fill=(200, 200, 200), width=1)
    
    current_y += 15
    
    # Add transaction details in a fancy box
    box_margin = 20
    box_left = margin
    box_right = width - margin
    box_top = current_y
    box_height = font_body.size * 4 + box_margin * 2
    
    # Draw box with rounded corners simulation
    draw.rectangle([(box_left, box_top), (box_right, box_top + box_height)], 
                  outline=(150, 150, 150), width=1)
    
    # Generate date and receipt number
    current_date = generate_date()
    current_time = generate_time()
    receipt_number = f"#{random.randint(10000, 999999)}"
    
    # Transaction details inside box - left column
    detail_x = box_left + box_margin
    detail_y = box_top + box_margin
    
    # Date and time
    draw.text((detail_x, detail_y), "Date:", fill="black", font=font_body)
    date_text_width = font_body.getlength("Date:")
    draw.text((detail_x + date_text_width + 10, detail_y), current_date, 
              fill="black", font=font_body)
    
    detail_y += font_body.size + 10
    draw.text((detail_x, detail_y), "Time:", fill="black", font=font_body)
    time_text_width = font_body.getlength("Time:")
    draw.text((detail_x + time_text_width + 10, detail_y), current_time, 
              fill="black", font=font_body)
    
    # Right column - receipt number
    right_detail_x = width - margin - 150
    detail_y = box_top + box_margin
    
    draw.text((right_detail_x, detail_y), "Receipt:", fill="black", font=font_body)
    receipt_text_width = font_body.getlength("Receipt:")
    draw.text((right_detail_x + receipt_text_width + 10, detail_y), receipt_number, 
              fill="black", font=font_body)
    
    # Add staff member
    staff_names = ["John", "Sarah", "Michael", "Emma", "David", "Jessica", "James"]
    staff_name = random.choice(staff_names)
    
    detail_y += font_body.size + 10
    draw.text((right_detail_x, detail_y), f"Served by: {staff_name}", 
              fill="black", font=font_body)
    
    # Update current position after box
    current_y = box_top + box_height + 20
    
    # Decorative item section header
    item_header = "YOUR ITEMS"
    item_header_width = font_subheader.getlength(item_header)
    item_header_x = max(margin, (width - item_header_width) // 2)
    
    draw.text((item_header_x, current_y), item_header, fill="black", font=font_subheader)
    
    current_y += font_subheader.size + 15
    
    # Add item table with decorative header
    table_left = margin
    table_right = width - margin
    
    # Draw shaded header
    header_top = current_y
    header_height = font_body.size + 15
    draw.rectangle([(table_left, header_top), (table_right, header_top + header_height)], 
                  fill=(240, 240, 240))
    
    # Add column headers
    draw.text((table_left + 10, header_top + 7), "ITEM", fill="black", font=font_body)
    
    # Calculate column positions
    qty_x = table_left + int((table_right - table_left) * 0.6)
    price_x = table_left + int((table_right - table_left) * 0.75)
    total_x = table_left + int((table_right - table_left) * 0.9)
    
    draw.text((qty_x, header_top + 7), "QTY", fill="black", font=font_body)
    draw.text((price_x, header_top + 7), "PRICE", fill="black", font=font_body)
    draw.text((total_x, header_top + 7), "TOTAL", fill="black", font=font_body)
    
    current_y = header_top + header_height + 5
    
    # Determine number of items
    if items_count is None:
        items_count = random.choice(list(RECEIPT_ITEM_COUNTS))
    
    # Select random items
    selected_items = random.sample(ITEMS, min(items_count, len(ITEMS)))
    if len(selected_items) < items_count:
        # Add some duplicates to reach requested count
        additional = random.choices(ITEMS, k=items_count - len(selected_items))
        selected_items.extend(additional)
    
    # Randomize order
    random.shuffle(selected_items)
    
    # Calculate item spacing
    item_height = font_body.size + 15
    
    # Track subtotal
    subtotal = 0
    
    # Draw items with alternating background for readability
    for i, (name, price_range, unit) in enumerate(selected_items):
        # Skip if we'd go past the bottom margin
        if current_y > height - 300:
            break
        
        # Add zebra striping
        if i % 2 == 0:
            draw.rectangle([(table_left, current_y - 3), 
                           (table_right, current_y + font_body.size + 8)],
                          fill=(248, 248, 248))
        
        # Randomize quantity (mostly 1, sometimes more)
        qty = 1
        if random.random() < 0.3:
            qty = random.randint(2, 5)
        
        # Calculate price with slight variation
        base_price = random.choice(list(price_range))
        price_variation = random.uniform(0.95, 1.05)
        price = round(base_price * price_variation) / 100.0
        
        # Calculate total
        total = round(price * qty, 2)
        subtotal += total
        
        # Format strings
        qty_str = str(qty)
        price_str = f"${price:.2f}"
        total_str = f"${total:.2f}"
        
        # Format item name with unit
        item_name = f"{name}"
        if unit and random.random() < 0.7:
            item_name = f"{name} ({unit})"
            
        # Truncate if too long
        max_name_width = qty_x - table_left - 20
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        
        # Draw item details
        draw.text((table_left + 10, current_y), item_name, fill="black", font=font_body)
        
        # Right-align quantity, price, and total
        qty_width = font_body.getlength(qty_str)
        price_width = font_body.getlength(price_str)
        total_width = font_body.getlength(total_str)
        
        draw.text((qty_x + 20 - qty_width, current_y), qty_str, fill="black", font=font_body)
        draw.text((price_x + 30 - price_width, current_y), price_str, fill="black", font=font_body)
        draw.text((total_x + 30 - total_width, current_y), total_str, fill="black", font=font_body)
        
        current_y += item_height
    
    # Add separator after items
    draw.line([(table_left, current_y + 5), (table_right, current_y + 5)], 
              fill=(150, 150, 150), width=1)
    current_y += 15
    
    # Calculate tax and total
    tax_rate = 0.10  # 10% GST
    tax = round(subtotal * tax_rate, 2)
    total = subtotal + tax
    
    # Format totals
    subtotal_str = f"${subtotal:.2f}"
    tax_str = f"${tax:.2f}"
    total_str = f"${total:.2f}"
    
    # Create fancy totals section
    totals_left = width // 2
    
    # Draw subtotal
    draw.text((totals_left, current_y), "Subtotal:", fill="black", font=font_body)
    subtotal_width = font_body.getlength(subtotal_str)
    draw.text((table_right - subtotal_width - 10, current_y), subtotal_str, 
              fill="black", font=font_body)
    
    # Add tax
    current_y += item_height
    draw.text((totals_left, current_y), "GST (10%):", fill="black", font=font_body)
    tax_width = font_body.getlength(tax_str)
    draw.text((table_right - tax_width - 10, current_y), tax_str, 
              fill="black", font=font_body)
    
    # Add separator before total
    current_y += item_height - 5
    draw.line([(totals_left, current_y), (table_right, current_y)], 
              fill=(100, 100, 100), width=1)
    current_y += 10
    
    # Add highlighted total
    total_bg_height = font_subheader.size + 16
    draw.rectangle([(totals_left - 10, current_y - 8), 
                   (table_right + 10, current_y + total_bg_height)],
                  fill=(240, 240, 240))
    
    draw.text((totals_left, current_y), "TOTAL:", fill="black", font=font_subheader)
    total_width = font_subheader.getlength(total_str)
    draw.text((table_right - total_width - 10, current_y), total_str, 
              fill="black", font=font_subheader)
    
    current_y += total_bg_height + 20
    
    # Add payment method
    payment = random.choice(PAYMENT_METHODS)
    
    # Create fancy payment box
    payment_box_top = current_y
    payment_box_height = 60
    
    # Draw decorative payment method box
    draw.rectangle([(table_left, payment_box_top), 
                   (table_right, payment_box_top + payment_box_height)],
                  outline=(150, 150, 150), width=1)
    
    # Add payment text
    payment_text = f"PAID BY {payment}"
    payment_width = font_body.getlength(payment_text)
    payment_x = max(table_left + 10, (table_left + table_right - payment_width) // 2)
    
    draw.text((payment_x, payment_box_top + 20), payment_text, fill="black", font=font_body)
    
    # Add card details if applicable
    if payment not in ["CASH", "EFTPOS"]:
        current_y = payment_box_top + payment_box_height + 15
        
        # Card number
        card_number = "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))
        card_width = font_small.getlength(card_number)
        card_x = max(margin, (width - card_width) // 2)
        
        draw.text((card_x, current_y), card_number, fill="black", font=font_small)
        
        # Authorization
        current_y += font_small.size + 5
        auth_code = f"Auth: {random.randint(100000, 999999)}"
        auth_width = font_small.getlength(auth_code)
        auth_x = max(margin, (width - auth_width) // 2)
        
        draw.text((auth_x, current_y), auth_code, fill="black", font=font_small)
    
    # Add footer with decorative elements
    # Move to bottom of receipt with margin
    footer_y = height - margin - font_body.size * 3
    
    # Add decorative pattern
    for i in range(margin, width - margin, 10):
        draw.line([(i, footer_y - 15), (i + 5, footer_y - 15)], 
                  fill=(200, 200, 200), width=1)
    
    # Add thank you message
    thank_you = "THANK YOU FOR YOUR PURCHASE"
    thank_you_width = font_body.getlength(thank_you)
    thank_you_x = max(margin, (width - thank_you_width) // 2)
    
    draw.text((thank_you_x, footer_y), thank_you, fill="black", font=font_body)
    
    # Add custom message
    messages = [
        "We appreciate your business!",
        "Please come back soon!",
        "Save your receipt for returns.",
        "Rate our service online!",
        "Follow us on social media.",
        "Join our rewards program!"
    ]
    
    custom_message = random.choice(messages)
    message_width = font_small.getlength(custom_message)
    message_x = max(margin, (width - message_width) // 2)
    
    draw.text((message_x, footer_y + font_body.size + 10), custom_message, 
              fill="black", font=font_small)
    
    # Add decorative bottom pattern
    for i in range(5):
        y = height - 20 - i * 8
        draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)
    
    # Apply slight random rotation for realism
    if random.random() < 0.7:
        angle = random.uniform(-0.8, 0.8)
        receipt = receipt.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
    
    return receipt


def create_minimal_receipt(width, height, items_count=None):
    """
    Create a minimalist receipt with clean, simple layout.
    
    Args:
        width: Receipt width in pixels
        height: Receipt height in pixels
        items_count: Number of items (if None, randomized)
        
    Returns:
        PIL Image of the receipt
    """
    # Create blank white receipt
    receipt = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(receipt)
    
    # Get fonts with realistic sizing
    font_header = get_font("header")
    font_body = get_font("body")
    # font_small isn't used in this style
    
    # Determine margins based on width
    margin = max(RECEIPT_MARGIN, int(width * 0.1))  # Larger margin for minimal look
    
    # Generate store name
    store_name = random.choice(STORE_NAMES)
    
    # Center the store name
    store_name_width = font_header.getlength(store_name)
    store_name_x = max(margin, (width - store_name_width) // 2)
    draw.text((store_name_x, margin), store_name, fill="black", font=font_header)
    
    # Add simple divider
    current_y = margin + font_header.size + 15
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Add date and time
    current_date = generate_date()
    current_time = generate_time()
    
    draw.text((margin, current_y), f"Date: {current_date}", fill="black", font=font_body)
    current_y += font_body.size + 10
    draw.text((margin, current_y), f"Time: {current_time}", fill="black", font=font_body)
    
    # Add receipt ID
    current_y += font_body.size + 10
    receipt_id = f"#{random.randint(10000, 999999)}"
    draw.text((margin, current_y), f"Receipt: {receipt_id}", fill="black", font=font_body)
    
    # Add payment method
    current_y += font_body.size + 10
    payment = random.choice(PAYMENT_METHODS)
    draw.text((margin, current_y), f"Payment: {payment}", fill="black", font=font_body)
    
    # Add second divider before items
    current_y += font_body.size + 15
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Determine number of items
    if items_count is None:
        items_count = random.choice(list(RECEIPT_ITEM_COUNTS))
    
    # Select random items with minimal info
    selected_items = random.sample(ITEMS, min(items_count, len(ITEMS)))
    if len(selected_items) < items_count:
        # Add some duplicates to reach requested count
        additional = random.choices(ITEMS, k=items_count - len(selected_items))
        selected_items.extend(additional)
    
    # Randomize order
    random.shuffle(selected_items)
    
    # Calculate item spacing
    item_height = font_body.size + 5  # Minimal spacing
    
    # Track subtotal
    subtotal = 0
    
    # Draw items - simplified format with just item and price for minimal look
    for name, price_range, _ in selected_items:
        # Skip if we'd go past the bottom margin
        if current_y > height - 150:
            break
        
        # Calculate price with slight variation
        base_price = random.choice(list(price_range))
        price_variation = random.uniform(0.95, 1.05)
        price = round(base_price * price_variation) / 100.0
        
        # For minimal receipt, we don't show quantity (always 1)
        total = price  # qty is always 1 
        subtotal += total
        
        # Format strings
        total_str = f"${total:.2f}"
        
        # Truncate item name if too long
        item_name = name
        max_name_width = width - margin * 2 - font_body.getlength(total_str) - 20
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        
        # Draw item and price only
        draw.text((margin, current_y), item_name, fill="black", font=font_body)
        
        # Right-align price
        total_width = font_body.getlength(total_str)
        draw.text((width - margin - total_width, current_y), total_str, 
                  fill="black", font=font_body)
        
        current_y += item_height
    
    # Add separator before totals
    current_y += 10
    draw.line([(margin, current_y), (width - margin, current_y)], 
              fill="black", width=1)
    current_y += 15
    
    # Calculate tax and total
    tax_rate = 0.10  # 10% GST
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
    current_y += item_height + 5
    draw.text((margin, current_y), "GST (10%):", fill="black", font=font_body)
    tax_width = font_body.getlength(tax_str)
    draw.text((width - margin - tax_width, current_y), tax_str, 
              fill="black", font=font_body)
    
    # Add total
    current_y += item_height + 5
    draw.text((margin, current_y), "Total:", fill="black", font=font_header)
    total_width = font_header.getlength(total_str)
    draw.text((width - margin - total_width, current_y), total_str, 
              fill="black", font=font_header)
    
    # Add minimal footer at bottom
    footer_y = height - margin - font_body.size
    thank_you = "THANK YOU"
    thank_you_width = font_body.getlength(thank_you)
    thank_you_x = max(margin, (width - thank_you_width) // 2)
    
    draw.text((thank_you_x, footer_y), thank_you, fill="black", font=font_body)
    
    # Apply slight random rotation for realism
    if random.random() < 0.7:
        angle = random.uniform(-0.5, 0.5)  # Minimal rotation for minimal receipt
        receipt = receipt.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
    
    return receipt


def generate_date():
    """
    Generate a realistic date string in Australian format.
    
    Returns:
        Date string (DD/MM/YYYY)
    """
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2023)
    
    # Format options
    formats = [
        f"{day:02d}/{month:02d}/{year}",  # DD/MM/YYYY
        f"{day:02d}-{month:02d}-{year}",  # DD-MM-YYYY
        f"{day:02d}.{month:02d}.{year}"   # DD.MM.YYYY
    ]
    
    return random.choice(formats)


def generate_time():
    """
    Generate a realistic time string.
    
    Returns:
        Time string (HH:MM AM/PM)
    """
    hour = random.randint(7, 22)  # Store hours
    minute = random.randint(0, 59)
    
    # 12-hour format
    am_pm = "AM" if hour < 12 else "PM"
    hour_12 = hour if hour <= 12 else hour - 12
    if hour_12 == 0:
        hour_12 = 12
    
    # Format options
    formats = [
        f"{hour_12}:{minute:02d} {am_pm}",  # 9:30 AM
        f"{hour_12}:{minute:02d}{am_pm.lower()}",  # 9:30am
        f"{hour:02d}:{minute:02d}"  # 24-hour format: 14:30
    ]
    
    return random.choice(formats)


def create_receipt(image_size=2048):
    """
    Create a receipt image by randomly selecting a style.
    
    Args:
        image_size: Base size for the image (will be adjusted to maintain realistic proportions)
        
    Returns:
        PIL Image of the receipt
    """
    # Determine receipt style
    receipt_styles = ["standard", "detailed", "minimal", "fancy"]
    style = random.choice(receipt_styles)
    
    # Determine realistic dimensions based on style
    if style == "standard":
        width = random.randint(350, 450)
        height = random.randint(1000, 1500)
    elif style == "detailed":
        width = random.randint(400, 500)
        height = random.randint(1200, 1800)
    elif style == "minimal":
        width = random.randint(300, 400)
        height = random.randint(800, 1200)
    else:  # fancy
        width = random.randint(400, 550)
        height = random.randint(1300, 2000)
    
    # Randomly determine number of items
    items_count = random.randint(5, 15)
    
    # Create receipt based on style
    if style == "standard":
        receipt = create_standard_receipt(width, height, items_count)
    elif style == "detailed":
        receipt = create_detailed_receipt(width, height, items_count)
    elif style == "minimal":
        receipt = create_minimal_receipt(width, height, items_count)
    else:  # fancy
        receipt = create_fancy_receipt(width, height, items_count)
    
    # Resize to maintain realistic proportions while fitting within image_size
    scale_factor = min(image_size / 2 / receipt.width, image_size / 1.2 / receipt.height)
    new_width = int(receipt.width * scale_factor)
    new_height = int(receipt.height * scale_factor)
    
    # Use high-quality resize
    receipt = receipt.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return receipt


if __name__ == "__main__":
    # For testing receipt generation
    receipt = create_receipt()
    receipt.save("test_receipt.png")
    print("Created test receipt: test_receipt.png")