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
RECEIPT_WIDTHS = range(500, 800)  # Wider range for Australian receipt widths in pixels
RECEIPT_HEIGHTS = range(800, 2000)  # Heights with realistic aspect ratios
RECEIPT_ITEM_COUNTS = range(3, 20)  # Typical item counts
RECEIPT_MARGIN = 60  # Increased base margin in pixels
FONT_SIZES = {
    "header": range(40, 55),      # Slightly reduced header font size
    "subheader": range(30, 45),   # Slightly reduced subheader font size
    "body": range(20, 28),        # Reduced body font size for better fit
    "small": range(16, 22)        # Slightly reduced small font size
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
    
    # Calculate column positions based on width - improved spacing
    # Reserve 40% of receipt width for item names
    item_width = int((width - 2 * margin) * 0.4)
    
    # Calculate column positions with more space between columns
    qty_x = margin + item_width + 20
    price_x = qty_x + 80  # More space for quantity
    total_x = price_x + 100  # More space for price
    
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
        
        # Truncate item name if too long - ensure it fits in the allocated item space
        max_name_width = item_width - 10  # Use the calculated item_width instead
        item_name = name
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        if len(item_name) < len(name) and len(item_name) > 3:
            item_name = item_name[:-1] + "."  # Add ellipsis if truncated
        
        # Draw item details
        draw.text((margin, current_y), item_name, fill="black", font=font_body)
        
        # Center-align quantity in its column
        qty_width = font_body.getlength(qty_str)
        qty_center = qty_x + 40  # Center of quantity column
        draw.text((qty_center - qty_width // 2, current_y), qty_str, fill="black", font=font_body)
        
        # Right-align price and total for better readability
        price_width = font_body.getlength(price_str)
        total_width = font_body.getlength(total_str)
        
        draw.text((price_x + 50 - price_width, current_y), price_str, fill="black", font=font_body)
        draw.text((total_x + 50 - total_width, current_y), total_str, fill="black", font=font_body)
        
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
    
    # Calculate column positions based on width - improved spacing for detailed receipt
    # Reserve 35% of receipt width for item names
    item_width = int((width - 2 * margin) * 0.35)
    
    # Calculate column positions with better spacing
    qty_x = margin + item_width + 30
    price_x = qty_x + 90  # More space for quantity 
    total_x = price_x + 120  # More space for price
    
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
            
        # Truncate if too long - with improved space calculation
        max_name_width = item_width - 10
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        if len(item_name) < len(name) and len(item_name) > 3:
            item_name = item_name[:-1] + "."  # Add ellipsis if truncated
        
        # Draw item details
        draw.text((margin, current_y), item_name, fill="black", font=font_body)
        
        # Center-align quantity in its column
        qty_width = font_body.getlength(qty_str)
        qty_center = qty_x + 45  # Center of quantity column
        draw.text((qty_center - qty_width // 2, current_y), qty_str, fill="black", font=font_body)
        
        # Right-align price and total with more space
        price_width = font_body.getlength(price_str)
        total_width = font_body.getlength(total_str)
        
        draw.text((price_x + 60 - price_width, current_y), price_str, fill="black", font=font_body)
        draw.text((total_x + 60 - total_width, current_y), total_str, fill="black", font=font_body)
        
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
    
    # Calculate column positions with more space - fancy receipt
    # Reserve 40% of table width for item names
    available_width = table_right - table_left
    item_width = int(available_width * 0.40)
    
    # Improved spacing between columns for fancy receipt
    qty_x = table_left + item_width + 30
    price_x = qty_x + 100
    total_x = price_x + 120
    
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
            
        # Truncate if too long - use calculated item_width
        max_name_width = item_width - 20
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        if len(item_name) < len(name) and len(item_name) > 3:
            item_name = item_name[:-1] + "."  # Add ellipsis if truncated
        
        # Draw item details
        draw.text((table_left + 10, current_y), item_name, fill="black", font=font_body)
        
        # Center-align quantity in its column for consistency
        qty_width = font_body.getlength(qty_str)
        qty_center = qty_x + 50  # Center of quantity column
        draw.text((qty_center - qty_width // 2, current_y), qty_str, fill="black", font=font_body)
        
        # Right-align price and total with more space
        price_width = font_body.getlength(price_str)
        total_width = font_body.getlength(total_str)
        
        draw.text((price_x + 60 - price_width, current_y), price_str, fill="black", font=font_body)
        draw.text((total_x + 60 - total_width, current_y), total_str, fill="black", font=font_body)
        
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
        
        # Calculate available space for item name - minimal version
        # Reserve at least 70px for the price
        price_column_width = max(70, font_body.getlength(total_str) + 30)
        max_name_width = width - margin * 2 - price_column_width
        
        # Truncate item name if too long
        item_name = name
        while font_body.getlength(item_name) > max_name_width and len(item_name) > 3:
            item_name = item_name[:-1]
        if len(item_name) < len(name) and len(item_name) > 3:
            item_name = item_name[:-1] + "."  # Add ellipsis if truncated
        
        # Draw item and price only - with better spacing
        draw.text((margin, current_y), item_name, fill="black", font=font_body)
        
        # Right-align price with consistent positioning
        total_width = font_body.getlength(total_str)
        price_x = width - margin - price_column_width + 30  # More consistent positioning
        draw.text((price_x, current_y), total_str, fill="black", font=font_body)
        
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
    
    # Determine realistic dimensions based on style - ensuring minimum widths for readability
    if style == "standard":
        width = random.randint(550, 650)  # Increased minimum width
        height = random.randint(1000, 1500)
    elif style == "detailed":
        width = random.randint(600, 700)  # Increased minimum width
        height = random.randint(1200, 1800)
    elif style == "minimal":
        width = random.randint(500, 600)  # Increased minimum width
        height = random.randint(800, 1200)
    else:  # fancy
        width = random.randint(600, 750)  # Increased minimum width
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




def generate_dataset(output_dir, num_collages=300, count_probs=None, image_size=2048, 
                    stapled_ratio=0.0, seed=42):
    """
    Generate a dataset of receipt collages with varying receipt counts.
    
    Args:
        output_dir: Directory to save generated images and metadata
        num_collages: Number of collage images to generate
        count_probs: Probability distribution for receipt counts (0-5)
        image_size: Size of output images
        stapled_ratio: Ratio of images with stapled receipts (only applies to receipt_count > 1)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with image filenames and receipt counts
    """
    import pandas as pd
    from pathlib import Path
    import random
    
    try:
        import numpy as np
    except ImportError:
        # Fallback for numpy
        class NumpyFallback:
            def random(self):
                class Random:
                    def choice(self, arr, p=None):
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
                    def seed(self, seed):
                        random.seed(seed)
                return Random()
        np = NumpyFallback()
    
    # Import create_tax_document from the same directory
    from data.generators.tax_document_generator import create_tax_document
    
    # Set random seeds
    random.seed(seed)
    try:
        np.random.seed(seed)
    except AttributeError:
        pass
    
    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Default distribution if not provided
    if count_probs is None:
        count_probs = [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]  # Probabilities for 0, 1, 2, 3, 4, 5 receipts
    
    # Normalize probabilities
    count_probs = np.array(count_probs) if hasattr(np, 'array') else count_probs
    try:
        count_probs = count_probs / sum(count_probs)
    except (TypeError, AttributeError):
        count_sum = sum(count_probs)
        count_probs = [p / count_sum for p in count_probs]
    
    # Function to create a single collage
    def create_receipt_collage(receipt_count, image_size=2048, stapled=False):
        """Create a collage with a specified number of receipts."""
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
        
        # If no receipts, return a tax document (class 0)
        if receipt_count == 0:
            return create_tax_document(image_size)
        
        # Create blank background
        collage = Image.new('RGB', (image_size, image_size), 'white')
        
        # Generate receipts
        receipts = []
        for _ in range(receipt_count):
            # Create a receipt with ab initio generator
            receipt = create_receipt(image_size)
            receipts.append(receipt)
        
        # For stapled receipts, create a stack with offsets
        if stapled and receipt_count > 1:
            # Find the largest receipt dimensions
            max_width = max(r.width for r in receipts)
            max_height = max(r.height for r in receipts)
            
            # Center position for the stack
            center_x = (image_size - max_width) // 2
            center_y = (image_size - max_height) // 2
            
            # Place receipts with small offsets
            for _, receipt in enumerate(receipts):
                # Calculate offset for this receipt in the stack
                offset_x = random.randint(-15, 15)
                offset_y = random.randint(-15, 15)
                
                # Ensure receipt stays within bounds
                x_pos = max(20, min(center_x + offset_x, image_size - receipt.width - 20))
                y_pos = max(20, min(center_y + offset_y, image_size - receipt.height - 20))
                
                # Paste receipt onto collage
                collage.paste(receipt, (x_pos, y_pos))
            
            # Add a staple mark
            draw = ImageDraw.Draw(collage)
            if random.random() > 0.5:  # Top staple
                staple_x = center_x + max_width // 2
                staple_y = center_y - 10
                draw.line([(staple_x-8, staple_y), (staple_x+8, staple_y)], fill="black", width=3)
                draw.line([(staple_x-5, staple_y-5), (staple_x-5, staple_y+5)], fill="black", width=3)
                draw.line([(staple_x+5, staple_y-5), (staple_x+5, staple_y+5)], fill="black", width=3)
            else:  # Side staple
                staple_x = center_x - 10
                staple_y = center_y + max_height // 2
                draw.line([(staple_x, staple_y-8), (staple_x, staple_y+8)], fill="black", width=3)
                draw.line([(staple_x-5, staple_y-5), (staple_x+5, staple_y-5)], fill="black", width=3)
                draw.line([(staple_x-5, staple_y+5), (staple_x+5, staple_y+5)], fill="black", width=3)
        else:
            # Distribute receipts across the image
            for idx, receipt in enumerate(receipts):
                if receipt_count == 1:
                    # Center the receipt
                    x_pos = (image_size - receipt.width) // 2
                    y_pos = (image_size - receipt.height) // 2
                else:
                    # Distribute receipts with some randomness
                    # Calculate left and right side positions with range checking
                    left_side_start = 30
                    left_side_end = max(left_side_start + 1, image_size // 2 - receipt.width - 30)
                    
                    right_side_start = image_size // 2 + 30
                    right_side_end = max(right_side_start + 1, image_size - receipt.width - 30)
                    
                    # If we can't fit the receipt in the intended area, use center positioning
                    if left_side_start >= left_side_end or right_side_start >= right_side_end:
                        # Fall back to center positioning
                        x_pos = max(30, (image_size - receipt.width) // 2)
                    else:
                        # Place on left or right as intended
                        if idx % 2 == 0:  # Left side
                            x_pos = random.randint(left_side_start, left_side_end)
                        else:  # Right side
                            x_pos = random.randint(right_side_start, right_side_end)
                    
                    # Handle potential vertical positioning issues
                    y_start = 30
                    y_end = max(y_start + 1, image_size - receipt.height - 30)
                    y_pos = random.randint(y_start, y_end)
                
                # Paste receipt onto collage
                collage.paste(receipt, (x_pos, y_pos))
                
                # Add random rotation to some receipts when there are multiple
                if receipt_count > 1 and random.random() < 0.5:
                    # Create a new layer for the rotated receipt
                    layer = Image.new('RGBA', (image_size, image_size), (0, 0, 0, 0))
                    layer.paste(receipt, (x_pos, y_pos))
                    
                    # Apply rotation
                    angle = random.uniform(-10, 10)
                    rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=False)
                    
                    # Composite onto main collage
                    collage = Image.alpha_composite(collage.convert('RGBA'), rotated)
                    collage = collage.convert('RGB')
        
        # Apply slight blur to ~20% of images for realism
        if random.random() < 0.2:
            blur_radius = random.uniform(0.3, 1.0)
            collage = collage.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Occasionally enhance contrast slightly
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(collage)
            collage = enhancer.enhance(random.uniform(1.0, 1.2))
        
        return collage
    
    # Generate collages
    data = []
    
    for i in range(num_collages):
        # Determine number of receipts based on probability distribution
        if hasattr(np.random, 'choice'):
            receipt_count = np.random.choice(len(count_probs), p=count_probs)
        else:
            receipt_count = np.random().choice(range(len(count_probs)), p=count_probs)
        
        # Determine if this should be stapled (only for multiple receipts)
        is_stapled = False
        if receipt_count > 1 and random.random() < stapled_ratio:
            is_stapled = True
        
        # Create collage
        try:
            collage = create_receipt_collage(receipt_count, image_size, stapled=is_stapled)
            
            # Save image
            filename = f"receipt_collage_{i:05d}.png"
            collage.save(images_dir / filename)
            
            # Add to dataset
            data.append({
                "filename": filename,
                "receipt_count": receipt_count,
                "is_stapled": is_stapled
            })
            
            # Progress update
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Generated {i + 1}/{num_collages} collages")
                
        except Exception as e:
            print(f"Error generating collage {i}: {e}")
    
    # Create and save metadata
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    # Print statistics
    print(f"\nDataset generation complete: {num_collages} images")
    print(f"Distribution of receipt counts: {df['receipt_count'].value_counts().sort_index()}")
    print(f"ATO tax documents (0 receipts): {len(df[df['receipt_count'] == 0])}")
    print(f"High-resolution images: {image_size}×{image_size}")
    
    if stapled_ratio > 0:
        stapled_count = df['is_stapled'].sum()
        print(f"Stapled receipts: {stapled_count} ({stapled_count/num_collages:.1%})")
    
    # Split dataset functionality
    try:
        # Try to use sklearn for stratified splitting
        from sklearn.model_selection import train_test_split
        
        # Check if we have enough samples in each class for stratified split
        class_counts = df['receipt_count'].value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count >= 4:  # Need at least 2 samples per split (train/val/test)
            # First split into train and temp (val+test combined)
            train_ratio = 0.7  # Fixed ratio for train split
            val_ratio = 0.15   # Fixed ratio for validation split
            
            train_df, temp_df = train_test_split(
                df, 
                train_size=train_ratio,
                stratify=df['receipt_count'],
                random_state=42
            )
            
            # Then split temp into val and test
            val_ratio_adjusted = val_ratio / (1 - train_ratio)  # Adjust for remaining portion
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_ratio_adjusted,
                stratify=temp_df['receipt_count'],
                random_state=42
            )
            print("Used sklearn for stratified dataset splitting")
        else:
            # Not enough samples for stratified split, fall back to random
            raise ValueError("Not enough samples for stratified split")
    
    except (ImportError, ValueError) as e:
        # Manual split if sklearn is not available or not enough samples
        print(f"Using non-stratified split: {e}")
        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1, random_state=42)
        
        # Calculate split indices
        train_ratio = 0.7
        val_ratio = 0.15
        train_end = int(len(df_shuffled) * train_ratio)
        val_end = train_end + int(len(df_shuffled) * val_ratio)
        
        # Split the dataframe
        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:]
    
    # Save split metadata
    train_df.to_csv(output_dir / "metadata_train.csv", index=False)
    val_df.to_csv(output_dir / "metadata_val.csv", index=False)
    test_df.to_csv(output_dir / "metadata_test.csv", index=False)
    
    # Print split statistics
    print("\nDataset split statistics:")
    print(f"  Training set:   {len(train_df)} images ({train_ratio:.1%})")
    print(f"  Validation set: {len(val_df)} images ({val_ratio:.1%})")
    print(f"  Test set:       {len(test_df)} images ({1-train_ratio-val_ratio:.1%})")
    
    # Print class distribution in each split
    print("\nReceipt count distribution:")
    train_dist = train_df['receipt_count'].value_counts().sort_index()
    val_dist = val_df['receipt_count'].value_counts().sort_index()
    test_dist = test_df['receipt_count'].value_counts().sort_index()
    
    for count in sorted(df['receipt_count'].unique()):
        train_count = train_dist.get(count, 0)
        val_count = val_dist.get(count, 0)
        test_count = test_dist.get(count, 0)
        total_count = train_count + val_count + test_count
        
        print(f"  Class {count}: {train_count} train, {val_count} val, {test_count} test "
              f"(total: {total_count})")
    
    return df


if __name__ == "__main__":
    # For testing receipt generation
    receipt = create_receipt()
    receipt.save("test_receipt.png")
    print("Created test receipt: test_receipt.png")