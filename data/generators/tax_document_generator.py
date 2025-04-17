#!/usr/bin/env python3
"""
Ab initio tax document generator for the InternVL2 receipt counter project.

This module creates Australian Taxation Office (ATO) document images from first principles,
with realistic variations in layout, content, and styling to ensure they're
visually distinct from receipts.
"""
import random

from PIL import Image, ImageDraw, ImageFont

# Only import numpy if available, otherwise use fallbacks
try:
    import numpy as np
except ImportError:
    # Fallback implementations for numpy functions we use
    class NumpyFallback:
        def pi(self):
            return 3.14159265358979
        def sin(self, x):
            import math
            return math.sin(x)
        def cos(self, x):
            import math
            return math.cos(x)
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

# ATO color palette
ATO_COLORS = {
    "blue": (0, 51, 160),          # Primary ATO blue
    "light_blue": (227, 242, 253), # Light blue for backgrounds
    "navy": (0, 38, 100),          # Darker blue for headers
    "gray": (88, 88, 88),          # Standard text color
    "light_gray": (240, 240, 240), # Light gray for alternating rows
    "red": (157, 0, 0),            # ATO red for warnings/important info
    "gold": (183, 156, 0)          # Gold for official seals/emblems
}

# Australian Government Departments
DEPARTMENTS = [
    "Australian Taxation Office",
    "Services Australia",
    "Department of Home Affairs",
    "Department of Social Services",
    "Medicare Australia",
    "Australian Securities & Investments Commission",
    "Department of Veterans' Affairs"
]

# Document types per department
DOCUMENT_TYPES = {
    "Australian Taxation Office": [
        "Notice of Assessment",
        "Tax Return Summary",
        "Business Activity Statement",
        "PAYG Payment Summary",
        "Income Tax Assessment",
        "Superannuation Statement",
        "Tax File Number Notification",
        "Tax Refund Notification"
    ],
    "Services Australia": [
        "Centrelink Payment Summary",
        "Family Tax Benefit Statement",
        "JobSeeker Payment Details",
        "Disability Support Pension Statement",
        "Age Pension Statement"
    ],
    "Department of Home Affairs": [
        "Citizenship Certificate Notice",
        "Visa Grant Notice",
        "Immigration Status Document"
    ],
    "Medicare Australia": [
        "Medicare Benefit Statement",
        "Medicare Claim Summary",
        "Pharmaceutical Benefits Scheme Statement"
    ],
    "Department of Social Services": [
        "Social Security Payment Summary",
        "Carer Allowance Statement",
        "Youth Allowance Summary"
    ],
    "Australian Securities & Investments Commission": [
        "Company Registration Document",
        "Business Name Registration",
        "Financial Services License"
    ],
    "Department of Veterans' Affairs": [
        "DVA Service Pension Statement",
        "Veterans' Entitlements Summary",
        "War Widow Pension Statement"
    ]
}

# Australian city postcodes
POSTCODES = {
    "Sydney": range(2000, 2250),
    "Melbourne": range(3000, 3210),
    "Brisbane": range(4000, 4180),
    "Perth": range(6000, 6200),
    "Adelaide": range(5000, 5100),
    "Canberra": range(2600, 2620),
    "Hobart": range(7000, 7020),
    "Darwin": range(800, 820)  # No leading zeros here
}

# Australian states and territories
STATES = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA": "South Australia",
    "WA": "Western Australia",
    "TAS": "Tasmania",
    "NT": "Northern Territory",
    "ACT": "Australian Capital Territory"
}


def get_font(style="body", bold=False, size_range=None):
    """
    Get a font of appropriate size with fallbacks.
    
    Args:
        style: Font style (header, subheader, body, small)
        bold: Whether to use bold font
        size_range: Optional custom size range tuple (min, max)
        
    Returns:
        PIL ImageFont object
    """
    # Define size ranges for different styles
    size_ranges = {
        "header": (45, 60),
        "subheader": (35, 45),
        "body": (25, 35),
        "small": (18, 25),
        "tiny": (12, 16)
    }
    
    # Use provided size range or get from predefined ranges
    if size_range:
        size_min, size_max = size_range
    else:
        size_min, size_max = size_ranges.get(style, size_ranges["body"])
    
    # Get random size from range
    size = random.randint(size_min, size_max)
    
    # Font families to try
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


def create_background_with_watermark(width, height, dept="Australian Taxation Office"):
    """
    Create a document background with subtle watermark.
    
    Args:
        width: Image width
        height: Image height
        dept: Government department name
        
    Returns:
        PIL Image with background and watermark
    """
    # Create base document with light color
    background_color = (255, 255, 255)  # White
    if random.random() < 0.7:  # 70% chance of light colored background
        background_color = ATO_COLORS["light_blue"]
    
    background = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(background)
    
    # Add subtle pattern or texture
    if random.random() < 0.5:
        # Create grid pattern
        grid_spacing = random.randint(20, 50)
        grid_color = (240, 240, 245)  # Very light gray-blue
        
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Create watermark text
    watermark_text = dept
    watermark_font = get_font("subheader", size_range=(60, 80))
    
    # Create separate image for watermark
    watermark = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    watermark_draw = ImageDraw.Draw(watermark)
    
    # Calculate text size
    text_width = watermark_font.getlength(watermark_text)
    text_height = watermark_font.size
    
    # Draw watermark text diagonally across the background
    angle = 45  # degrees
    
    # Draw multiple watermarks
    for i in range(-2, 3):
        for j in range(-1, 3):
            # Position watermark
            x = width * 0.5 + i * text_width * 0.8
            y = height * 0.3 + j * text_height * 2
            
            # Draw watermark with very light color
            watermark_draw.text((x, y), watermark_text, 
                              fill=(0, 0, 0, 15), font=watermark_font)
    
    # Rotate watermark
    watermark = watermark.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # Composite watermark onto background
    background = Image.alpha_composite(background.convert('RGBA'), watermark)
    
    return background.convert('RGB')


def create_header(draw, width, height, margin, dept, doc_type):
    """
    Create an official-looking header for the document.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        dept: Department name
        doc_type: Document type
        
    Returns:
        Y-position after the header
    """
    # Determine header style
    header_style = random.choice(["banner", "logo", "classic"])
    
    if header_style == "banner":
        # Full-width colored banner
        banner_height = random.randint(60, 100)
        draw.rectangle([(0, 0), (width, banner_height)], 
                      fill=ATO_COLORS["blue"])
        
        # Add Australian Government text
        gov_font = get_font("subheader", bold=True, size_range=(28, 35))
        gov_text = "Australian Government"
        
        # White text on blue background
        draw.text((margin, 15), gov_text, fill="white", font=gov_font)
        
        # Add department name below or beside
        dept_font = get_font("body", bold=True, size_range=(22, 30))
        
        if random.random() < 0.5:
            # Department name beside government text
            draw.text((margin + gov_font.getlength(gov_text) + 30, 15), 
                      dept, fill="white", font=dept_font)
            
            # Current position is after banner
            current_y = banner_height + 15
        else:
            # Department name below government text
            draw.text((margin, 15 + gov_font.size + 5), 
                      dept, fill="white", font=dept_font)
            
            # Current position is after banner
            current_y = banner_height + 15
        
        # Add document type header
        doc_font = get_font("header", bold=True)
        doc_width = doc_font.getlength(doc_type)
        doc_x = max(margin, (width - doc_width) // 2)  # Center text
        
        draw.text((doc_x, current_y), doc_type, fill="black", font=doc_font)
        current_y += doc_font.size + 20
    
    elif header_style == "logo":
        # Add Australian Government coat of arms placeholder
        logo_size = random.randint(70, 100)
        
        # Draw crest placeholder
        if random.random() < 0.5:
            # Circular emblem
            draw.ellipse([(margin, 20), (margin + logo_size, 20 + logo_size)], 
                        outline=ATO_COLORS["gold"], width=2)
            
            # Add simplified coat of arms elements
            center_x = margin + logo_size // 2
            center_y = 20 + logo_size // 2
            
            # Australian stars
            star_points = [(center_x, center_y - 15)]
            for i in range(1, 7):
                angle = 2 * np.pi * i / 6
                x = center_x + 20 * np.sin(angle)
                y = center_y - 15 + 20 * np.cos(angle)
                star_points.append((x, y))
                
            draw.polygon(star_points, outline=ATO_COLORS["gold"], width=1)
        else:
            # Shield emblem
            shield_points = [
                (margin + 10, 30),
                (margin + logo_size - 10, 30),
                (margin + logo_size, 50),
                (margin + logo_size - 10, logo_size + 10),
                (margin + logo_size // 2, logo_size + 20),
                (margin + 10, logo_size + 10),
                (margin, 50)
            ]
            draw.polygon(shield_points, outline=ATO_COLORS["gold"], width=2)
        
        # Add text beside logo
        gov_font = get_font("subheader", bold=True, size_range=(30, 38))
        gov_text = "Australian Government"
        
        # Position beside logo
        gov_x = margin + logo_size + 20
        gov_y = 20
        
        draw.text((gov_x, gov_y), gov_text, fill="black", font=gov_font)
        
        # Add department name below
        dept_font = get_font("body", bold=True)
        dept_y = gov_y + gov_font.size + 5
        
        draw.text((gov_x, dept_y), dept, fill=ATO_COLORS["blue"], font=dept_font)
        
        # Add colored separator bar
        bar_y = 20 + logo_size + 20
        draw.rectangle([(0, bar_y), (width, bar_y + 3)], 
                      fill=ATO_COLORS["blue"])
        
        # Add document type below bar
        doc_y = bar_y + 20
        doc_font = get_font("header", bold=True)
        doc_width = doc_font.getlength(doc_type)
        doc_x = max(margin, (width - doc_width) // 2)  # Center text
        
        draw.text((doc_x, doc_y), doc_type, fill="black", font=doc_font)
        current_y = doc_y + doc_font.size + 20
    
    else:  # classic
        # Simple text header with department name and document type
        gov_font = get_font("body", bold=True)
        gov_text = "Australian Government"
        gov_width = gov_font.getlength(gov_text)
        gov_x = max(margin, (width - gov_width) // 2)  # Center text
        
        draw.text((gov_x, margin), gov_text, fill="black", font=gov_font)
        
        # Add department name below
        dept_font = get_font("subheader", bold=True)
        dept_width = dept_font.getlength(dept)
        dept_x = max(margin, (width - dept_width) // 2)
        dept_y = margin + gov_font.size + 10
        
        draw.text((dept_x, dept_y), dept, fill=ATO_COLORS["blue"], font=dept_font)
        
        # Add separator line
        line_y = dept_y + dept_font.size + 15
        draw.line([(width // 4, line_y), (width * 3 // 4, line_y)], 
                 fill=ATO_COLORS["blue"], width=2)
        
        # Add document type below separator
        doc_y = line_y + 20
        doc_font = get_font("header", bold=True)
        doc_width = doc_font.getlength(doc_type)
        doc_x = max(margin, (width - doc_width) // 2)
        
        draw.text((doc_x, doc_y), doc_type, fill="black", font=doc_font)
        current_y = doc_y + doc_font.size + 20
    
    return current_y


def create_reference_section(draw, width, height, margin, current_y):
    """
    Create a reference section with document identifiers and date.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        current_y: Starting Y position
        
    Returns:
        Y-position after the reference section
    """
    # Generate document reference number
    ref_number = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(10000, 99999)}"
    
    # Generate document date
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2023)
    # Month names for date formatting
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    date_format = random.choice([
        f"{day:02d}/{month:02d}/{year}",
        f"{day:02d} {months[month-1]} {year}",
        f"{day:02d}-{month:02d}-{year}"
    ])
    
    ref_font = get_font("body")
    
    # Choose reference style
    ref_style = random.choice(["box", "right_aligned", "table"])
    
    if ref_style == "box":
        # Create a reference box
        box_padding = 15
        box_width = width - 2 * margin
        box_height = ref_font.size * 3 + box_padding * 2
        
        # Draw box with light background
        draw.rectangle([(margin, current_y), 
                       (margin + box_width, current_y + box_height)],
                      fill=ATO_COLORS["light_gray"], outline="black", width=1)
        
        # Add reference information inside box
        text_x = margin + box_padding
        text_y = current_y + box_padding
        
        draw.text((text_x, text_y), f"Reference: {ref_number}", 
                 fill="black", font=ref_font)
        
        draw.text((text_x, text_y + ref_font.size + 5), f"Date: {date_format}", 
                 fill="black", font=ref_font)
        
        current_y += box_height + 20
    
    elif ref_style == "right_aligned":
        # Right-aligned reference information
        ref_text = f"Reference: {ref_number}"
        date_text = f"Date: {date_format}"
        
        ref_width = ref_font.getlength(ref_text)
        date_width = ref_font.getlength(date_text)
        
        draw.text((width - margin - ref_width, current_y), ref_text, 
                 fill="black", font=ref_font)
        
        draw.text((width - margin - date_width, current_y + ref_font.size + 10), 
                 date_text, fill="black", font=ref_font)
        
        current_y += ref_font.size * 2 + 30
    
    else:  # table
        # Create a two-column mini-table for reference info
        table_width = width // 2
        table_x = width - margin - table_width
        
        # Draw table outline
        draw.rectangle([(table_x, current_y), 
                       (width - margin, current_y + ref_font.size * 2 + 20)],
                      outline="black", width=1)
        
        # Draw divider between rows
        row_y = current_y + ref_font.size + 10
        draw.line([(table_x, row_y), (width - margin, row_y)], 
                 fill="black", width=1)
        
        # Add reference information
        draw.text((table_x + 10, current_y + 5), "Reference:", 
                 fill="black", font=ref_font)
        draw.text((table_x + 150, current_y + 5), ref_number, 
                 fill="black", font=ref_font)
        
        draw.text((table_x + 10, row_y + 5), "Date:", 
                 fill="black", font=ref_font)
        draw.text((table_x + 150, row_y + 5), date_format, 
                 fill="black", font=ref_font)
        
        current_y += ref_font.size * 2 + 30
    
    return current_y


def create_recipient_section(draw, width, height, margin, current_y):
    """
    Create a recipient section with taxpayer details.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        current_y: Starting Y position
        
    Returns:
        Y-position after the recipient section
    """
    recipient_font = get_font("body")
    
    # Generate taxpayer details
    first_names = ["John", "Emma", "Michael", "Sarah", "David", "Jessica", "Mark", "Lisa"]
    last_names = ["Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "Johnson", "White"]
    
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    full_name = f"{first_name} {last_name}"
    
    # Generate TFN (Tax File Number) with partial masking
    tfn = f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    tfn_masked = list(tfn)
    
    # Mask several digits
    for _ in range(5):
        pos = random.randint(0, len(tfn) - 1)
        if tfn_masked[pos] not in [' ', 'X']:
            tfn_masked[pos] = 'X'
    
    tfn_masked = ''.join(tfn_masked)
    
    # Generate address
    street_num = random.randint(1, 999)
    streets = ["Main St", "George St", "King St", "Queen St", "Market St", 
               "Elizabeth St", "Victoria St", "Park Rd", "Church St", "High St"]
    street = random.choice(streets)
    
    city = random.choice(list(POSTCODES.keys()))
    postcode_range = POSTCODES[city]
    postcode = random.choice(list(postcode_range))
    
    state_code = next((code for code, state in STATES.items() 
                     if any(city in state for state, _ in POSTCODES.items())), "NSW")
    
    address = f"{street_num} {street}, {city} {state_code} {postcode}"
    
    # Choose recipient style
    recipient_style = random.choice(["simple", "box", "letterhead"])
    
    if recipient_style == "simple":
        # Simple recipient details
        draw.text((margin, current_y), "Taxpayer details:", 
                 fill=ATO_COLORS["blue"], font=get_font("body", bold=True))
        
        current_y += recipient_font.size + 10
        draw.text((margin, current_y), f"Name: {full_name}", 
                 fill="black", font=recipient_font)
        
        current_y += recipient_font.size + 5
        draw.text((margin, current_y), f"TFN: {tfn_masked}", 
                 fill="black", font=recipient_font)
        
        current_y += recipient_font.size + 5
        draw.text((margin, current_y), f"Address: {address}", 
                 fill="black", font=recipient_font)
        
        current_y += recipient_font.size + 15
    
    elif recipient_style == "box":
        # Boxed recipient details
        box_padding = 15
        box_width = width - 2 * margin
        box_height = recipient_font.size * 4 + box_padding * 2
        
        # Draw box
        draw.rectangle([(margin, current_y), 
                       (margin + box_width, current_y + box_height)],
                      outline="black", width=1)
        
        # Add header inside box
        header_y = current_y + box_padding
        draw.text((margin + box_padding, header_y), "Taxpayer details", 
                 fill=ATO_COLORS["blue"], font=get_font("body", bold=True))
        
        # Add taxpayer information
        text_x = margin + box_padding
        text_y = header_y + recipient_font.size + 10
        
        draw.text((text_x, text_y), f"Name: {full_name}", 
                 fill="black", font=recipient_font)
        
        draw.text((text_x, text_y + recipient_font.size + 5), f"TFN: {tfn_masked}", 
                 fill="black", font=recipient_font)
        
        draw.text((text_x, text_y + recipient_font.size * 2 + 10), f"Address: {address}", 
                 fill="black", font=recipient_font)
        
        current_y += box_height + 20
    
    else:  # letterhead
        # Letterhead style recipient with address on top right
        name_font = get_font("subheader")
        
        # Add recipient name on left
        draw.text((margin, current_y), full_name, 
                 fill="black", font=name_font)
        
        # Add TFN below name
        current_y += name_font.size + 5
        draw.text((margin, current_y), f"TFN: {tfn_masked}", 
                 fill="black", font=recipient_font)
        
        # Add address on right
        address_lines = address.split(", ")
        address_x = width - margin - 200
        address_y = current_y - name_font.size
        
        for line in address_lines:
            draw.text((address_x, address_y), line, 
                     fill="black", font=recipient_font)
            address_y += recipient_font.size + 5
        
        current_y += recipient_font.size + 15
    
    # Add separator after recipient details
    draw.line([(margin, current_y), (width - margin, current_y)], 
             fill=ATO_COLORS["gray"], width=1)
    current_y += 20
    
    return current_y


def create_financial_section(draw, width, height, margin, current_y, doc_type):
    """
    Create a financial data section appropriate for the document type.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        current_y: Starting Y position
        doc_type: Document type
        
    Returns:
        Y-position after the financial section
    """
    # Use heading font for the section title
    heading_font = get_font("subheader", bold=True)
    
    # Add section heading
    heading_text = "Financial Details"
    
    if "Assessment" in doc_type:
        heading_text = "Assessment Summary"
    elif "Payment" in doc_type:
        heading_text = "Payment Details"
    elif "Activity" in doc_type:
        heading_text = "Activity Statement Summary"
    elif "Benefit" in doc_type:
        heading_text = "Benefit Details"
    
    # Draw heading
    draw.text((margin, current_y), heading_text, 
             fill=ATO_COLORS["blue"], font=heading_font)
    
    current_y += heading_font.size + 15
    
    # Choose table style based on document type
    if "BAS" in doc_type or "Activity Statement" in doc_type:
        # Business Activity Statement style table
        current_y = create_bas_table(draw, width, height, margin, current_y)
    
    elif "Assessment" in doc_type or "Tax" in doc_type:
        # Tax assessment style table
        current_y = create_tax_assessment_table(draw, width, height, margin, current_y)
    
    elif "Payment" in doc_type or "Pension" in doc_type or "Allowance" in doc_type:
        # Payment summary style table
        current_y = create_payment_table(draw, width, height, margin, current_y)
    
    else:
        # Generic financial table
        current_y = create_generic_table(draw, width, height, margin, current_y)
    
    # Add section end spacing
    current_y += 20
    
    return current_y


def create_bas_table(draw, width, height, margin, current_y):
    """
    Create a Business Activity Statement table.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        current_y: Starting Y position
        
    Returns:
        Y-position after the table
    """
    table_font = get_font("body")
    # We don't use small_font in this function but we might in the future
    
    # Define table dimensions
    table_width = width - 2 * margin
    col1_width = int(table_width * 0.6)  # Description column
    col2_width = int(table_width * 0.4)  # Amount column
    
    # Draw table header
    header_height = table_font.size + 20
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + header_height)],
                  fill=ATO_COLORS["blue"])
    
    # Add header text
    draw.text((margin + 10, current_y + 10), "GST AND PAYG SUMMARY", 
             fill="white", font=get_font("body", bold=True))
    
    current_y += header_height
    
    # Generate realistic BAS data
    sales = random.randint(10000, 500000)
    gst_on_sales = int(sales * 0.1)
    purchases = random.randint(5000, sales)
    gst_on_purchases = int(purchases * 0.1)
    net_gst = gst_on_sales - gst_on_purchases
    payg_withholding = random.randint(5000, 50000)
    total_amount = net_gst + payg_withholding
    
    # BAS line items
    bas_items = [
        ("G1. Total sales (including GST)", f"${sales:,}"),
        ("G3. GST on sales", f"${gst_on_sales:,}"),
        ("G10. Purchases (including GST)", f"${purchases:,}"),
        ("G11. GST on purchases", f"${gst_on_purchases:,}"),
        ("G20. Net GST", f"${net_gst:,}"),
        ("W4. PAYG withholding", f"${payg_withholding:,}")
    ]
    
    # Draw table rows with alternating colors
    row_height = table_font.size + 20
    for i, (label, value) in enumerate(bas_items):
        # Use alternating row colors
        if i % 2 == 0:
            row_bg = ATO_COLORS["light_gray"]
            draw.rectangle([(margin, current_y), 
                           (margin + table_width, current_y + row_height)],
                          fill=row_bg)
        
        # Draw table row outline
        draw.rectangle([(margin, current_y), 
                       (margin + table_width, current_y + row_height)],
                      outline="black", width=1)
        
        # Draw divider between columns
        draw.line([(margin + col1_width, current_y), 
                  (margin + col1_width, current_y + row_height)],
                 fill="black", width=1)
        
        # Add text to cells
        text_y = current_y + (row_height - table_font.size) // 2
        draw.text((margin + 10, text_y), label, fill="black", font=table_font)
        
        # Right-align amount
        amount_width = table_font.getlength(value)
        amount_x = margin + col1_width + col2_width - amount_width - 10
        draw.text((amount_x, text_y), value, fill="black", font=table_font)
        
        current_y += row_height
    
    # Add total row with highlighting
    total_height = table_font.size + 30
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + total_height)],
                  fill=(240, 240, 250))
    
    # Draw total row outline
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + total_height)],
                  outline="black", width=2)
    
    # Add total text
    total_y = current_y + (total_height - table_font.size) // 2
    draw.text((margin + 10, total_y), "TOTAL AMOUNT PAYABLE", 
             fill=ATO_COLORS["blue"], font=get_font("body", bold=True))
    
    # Right-align total amount
    total_text = f"${total_amount:,}"
    total_width = table_font.getlength(total_text)
    total_x = margin + col1_width + col2_width - total_width - 10
    
    # Highlight total amount
    if total_amount > 0:
        total_color = ATO_COLORS["blue"]
    else:
        total_color = "green"
        total_text = f"${abs(total_amount):,} CR"
    
    draw.text((total_x, total_y), total_text, 
             fill=total_color, font=get_font("body", bold=True))
    
    current_y += total_height + 10
    
    # Add payment due date
    due_day = random.randint(15, 28)
    due_month = random.randint(1, 12)
    due_year = random.randint(2020, 2023)
    due_date = f"{due_day:02d}/{due_month:02d}/{due_year}"
    
    draw.text((margin, current_y), f"Payment due by: {due_date}", 
             fill="black", font=table_font)
    
    current_y += table_font.size + 20
    
    return current_y


def create_tax_assessment_table(draw, width, height, margin, current_y):
    """
    Create a tax assessment summary table.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        current_y: Starting Y position
        
    Returns:
        Y-position after the table
    """
    table_font = get_font("body")
    
    # Define table dimensions
    table_width = width - 2 * margin
    col1_width = int(table_width * 0.7)  # Description column
    col2_width = int(table_width * 0.3)  # Amount column
    
    # Generate realistic tax data
    gross_income = random.randint(50000, 120000)
    deductions = random.randint(2000, 12000)
    taxable_income = gross_income - deductions
    
    # Calculate tax based on ATO tax brackets (simplified)
    if taxable_income <= 18200:
        tax = 0
    elif taxable_income <= 45000:
        tax = (taxable_income - 18200) * 0.19
    elif taxable_income <= 120000:
        tax = 5092 + (taxable_income - 45000) * 0.325
    elif taxable_income <= 180000:
        tax = 29467 + (taxable_income - 120000) * 0.37
    else:
        tax = 51667 + (taxable_income - 180000) * 0.45
    
    tax = int(tax)
    
    # Other components
    medicare_levy = int(taxable_income * 0.02)
    tax_offsets = random.randint(0, 1000)
    tax_withheld = int((tax + medicare_levy) * random.uniform(0.9, 1.1))
    
    refund_or_payable = tax + medicare_levy - tax_offsets - tax_withheld
    
    # Tax assessment items
    tax_items = [
        ("Gross Income", f"${gross_income:,}"),
        ("Deductions", f"${deductions:,}"),
        ("Taxable Income", f"${taxable_income:,}"),
        ("Tax on Taxable Income", f"${tax:,}"),
        ("Medicare Levy", f"${medicare_levy:,}"),
        ("Tax Offsets", f"${tax_offsets:,}"),
        ("PAYG Tax Withheld", f"${tax_withheld:,}")
    ]
    
    # Draw table header
    header_height = table_font.size + 20
    header_bg = ATO_COLORS["blue"]
    
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + header_height)],
                  fill=header_bg)
    
    # Add header text
    draw.text((margin + 10, current_y + 10), "TAX ASSESSMENT SUMMARY", 
             fill="white", font=get_font("body", bold=True))
    
    current_y += header_height
    
    # Draw column headers with gray background
    col_header_height = table_font.size + 15
    col_header_bg = (220, 220, 220)
    
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + col_header_height)],
                  fill=col_header_bg)
    
    # Draw column header text
    col_header_y = current_y + (col_header_height - table_font.size) // 2
    draw.text((margin + 10, col_header_y), "Description", 
             fill="black", font=get_font("body", bold=True))
    
    draw.text((margin + col1_width + 10, col_header_y), "Amount", 
             fill="black", font=get_font("body", bold=True))
    
    # Draw column divider
    draw.line([(margin + col1_width, current_y), 
              (margin + col1_width, current_y + col_header_height)],
             fill="black", width=1)
    
    current_y += col_header_height
    
    # Draw table rows with alternating colors
    row_height = table_font.size + 15
    for i, (label, value) in enumerate(tax_items):
        # Use alternating row colors
        if i % 2 == 0:
            row_bg = ATO_COLORS["light_gray"]
            draw.rectangle([(margin, current_y), 
                           (margin + table_width, current_y + row_height)],
                          fill=row_bg)
        
        # Draw table row outline
        draw.rectangle([(margin, current_y), 
                       (margin + table_width, current_y + row_height)],
                      outline="black", width=1)
        
        # Draw divider between columns
        draw.line([(margin + col1_width, current_y), 
                  (margin + col1_width, current_y + row_height)],
                 fill="black", width=1)
        
        # Add text to cells
        text_y = current_y + (row_height - table_font.size) // 2
        draw.text((margin + 10, text_y), label, fill="black", font=table_font)
        
        # Right-align amount
        amount_width = table_font.getlength(value)
        amount_x = margin + col1_width + col2_width - amount_width - 10
        draw.text((amount_x, text_y), value, fill="black", font=table_font)
        
        current_y += row_height
    
    # Add separator before total
    draw.line([(margin, current_y), (margin + table_width, current_y)],
             fill="black", width=2)
    
    # Add result row
    result_height = table_font.size + 25
    
    # Different styling based on refund or amount due
    if refund_or_payable > 0:
        result_label = "Amount Due"
        result_value = f"${refund_or_payable:,}"
        result_color = ATO_COLORS["blue"]
    else:
        result_label = "Refund Amount"
        result_value = f"${abs(refund_or_payable):,}"
        result_color = "green"
    
    # Draw result row with emphasis
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + result_height)],
                  fill=(240, 240, 250))
    
    # Draw result row outline
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + result_height)],
                  outline="black", width=2)
    
    # Add result text
    result_y = current_y + (result_height - table_font.size) // 2
    draw.text((margin + 10, result_y), result_label, 
             fill="black", font=get_font("body", bold=True))
    
    # Right-align result amount
    result_width = table_font.getlength(result_value)
    result_x = margin + col1_width + col2_width - result_width - 10
    
    draw.text((result_x, result_y), result_value, 
             fill=result_color, font=get_font("body", bold=True))
    
    current_y += result_height + 15
    
    # Add payment/refund details if applicable
    if refund_or_payable > 0:
        due_day = random.randint(15, 28)
        due_month = random.randint(1, 12)
        due_year = random.randint(2020, 2023)
        due_date = f"{due_day:02d}/{due_month:02d}/{due_year}"
        
        draw.text((margin, current_y), f"Payment due by: {due_date}", 
                 fill=ATO_COLORS["blue"], font=table_font)
        
        # Add payment methods note
        current_y += table_font.size + 10
        draw.text((margin, current_y), 
                 "Payment options: BPAY, Direct Credit, or myGov", 
                 fill="black", font=get_font("small"))
    else:
        # Add refund note
        draw.text((margin, current_y), 
                 "Your refund will be credited to your nominated bank account within 14 days.", 
                 fill="black", font=table_font)
    
    current_y += table_font.size + 20
    
    return current_y


def create_payment_table(draw, width, height, margin, current_y):
    """
    Create a payment summary table for government benefits.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        current_y: Starting Y position
        
    Returns:
        Y-position after the table
    """
    table_font = get_font("body")
    # We don't use small_font in this function but we might in the future
    
    # Define table dimensions
    table_width = width - 2 * margin
    
    # Generate payment period
    start_day = 1
    start_month = random.randint(1, 12)
    year = random.randint(2020, 2023)
    
    # End date is typically two weeks or a month later
    if random.random() < 0.7:  # 70% chance of fortnightly payment
        period_type = "Fortnightly"
        end_day = min(start_day + 14, 28)
        end_month = start_month
    else:
        period_type = "Monthly"
        end_day = start_day
        end_month = start_month + 1
        if end_month > 12:
            end_month = 1
    
    start_date = f"{start_day:02d}/{start_month:02d}/{year}"
    end_date = f"{end_day:02d}/{end_month:02d}/{year}"
    
    # Generate realistic payment amounts
    if "Pension" in period_type:
        base_payment = random.randint(800, 1000)
    elif "Disability" in period_type:
        base_payment = random.randint(700, 950)
    elif "JobSeeker" in period_type:
        base_payment = random.randint(500, 650)
    else:
        base_payment = random.randint(600, 800)
    
    energy_supplement = random.randint(10, 30)
    rent_assistance = random.randint(0, 150)
    
    # Calculate deductions
    tax_withheld = int(base_payment * random.uniform(0, 0.1))
    other_deductions = random.randint(0, 50)
    
    # Calculate net payment
    gross_payment = base_payment + energy_supplement + rent_assistance
    total_deductions = tax_withheld + other_deductions
    net_payment = gross_payment - total_deductions
    
    # Draw section header
    draw.text((margin, current_y), f"{period_type} Payment Summary", 
             fill=ATO_COLORS["blue"], font=get_font("subheader", bold=True))
    
    current_y += get_font("subheader").size + 15
    
    # Draw payment period
    draw.text((margin, current_y), f"Payment period: {start_date} to {end_date}", 
             fill="black", font=table_font)
    
    current_y += table_font.size + 15
    
    # Create payment table
    # Table header
    header_height = table_font.size + 15
    header_bg = ATO_COLORS["blue"]
    
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + header_height)],
                  fill=header_bg)
    
    # Add header text
    draw.text((margin + 10, current_y + 5), "PAYMENT DETAILS", 
             fill="white", font=get_font("body", bold=True))
    
    current_y += header_height
    
    # Define payment components
    payments = [
        ("Base Payment", f"${base_payment:.2f}"),
        ("Energy Supplement", f"${energy_supplement:.2f}"),
        ("Rent Assistance", f"${rent_assistance:.2f}")
    ]
    
    deductions = [
        ("Tax Withheld", f"${tax_withheld:.2f}"),
        ("Other Deductions", f"${other_deductions:.2f}")
    ]
    
    # Draw payment section
    section_header_height = table_font.size + 10
    
    # Payments header
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + section_header_height)],
                  fill=ATO_COLORS["light_gray"])
    
    draw.text((margin + 10, current_y + 3), "Payments", 
             fill="black", font=get_font("body", bold=True))
    
    current_y += section_header_height
    
    # Payment rows
    row_height = table_font.size + 10
    for label, amount in payments:
        draw.rectangle([(margin, current_y), 
                       (margin + table_width, current_y + row_height)],
                      outline="black", width=1)
        
        # Add text
        draw.text((margin + 20, current_y + 3), label, 
                 fill="black", font=table_font)
        
        # Right-align amount
        amount_width = table_font.getlength(amount)
        amount_x = margin + table_width - amount_width - 20
        
        draw.text((amount_x, current_y + 3), amount, 
                 fill="black", font=table_font)
        
        current_y += row_height
    
    # Gross payment row
    gross_text = f"Gross Payment: ${gross_payment:.2f}"
    gross_width = table_font.getlength(gross_text)
    gross_x = margin + table_width - gross_width - 20
    
    draw.text((gross_x, current_y + 5), gross_text, 
             fill="black", font=get_font("body", bold=True))
    
    current_y += table_font.size + 15
    
    # Deductions header
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + section_header_height)],
                  fill=ATO_COLORS["light_gray"])
    
    draw.text((margin + 10, current_y + 3), "Deductions", 
             fill="black", font=get_font("body", bold=True))
    
    current_y += section_header_height
    
    # Deduction rows
    for label, amount in deductions:
        draw.rectangle([(margin, current_y), 
                       (margin + table_width, current_y + row_height)],
                      outline="black", width=1)
        
        # Add text
        draw.text((margin + 20, current_y + 3), label, 
                 fill="black", font=table_font)
        
        # Right-align amount
        amount_width = table_font.getlength(amount)
        amount_x = margin + table_width - amount_width - 20
        
        draw.text((amount_x, current_y + 3), amount, 
                 fill="black", font=table_font)
        
        current_y += row_height
    
    # Total deductions row
    deductions_text = f"Total Deductions: ${total_deductions:.2f}"
    deductions_width = table_font.getlength(deductions_text)
    deductions_x = margin + table_width - deductions_width - 20
    
    draw.text((deductions_x, current_y + 5), deductions_text, 
             fill="black", font=get_font("body", bold=True))
    
    current_y += table_font.size + 15
    
    # Net payment row with highlighting
    net_height = table_font.size + 20
    
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + net_height)],
                  fill=(240, 240, 250), outline="black", width=2)
    
    # Net payment text
    draw.text((margin + 20, current_y + 6), "NET PAYMENT", 
             fill=ATO_COLORS["blue"], font=get_font("body", bold=True))
    
    # Right-align net amount
    net_text = f"${net_payment:.2f}"
    net_width = table_font.getlength(net_text)
    net_x = margin + table_width - net_width - 20
    
    draw.text((net_x, current_y + 6), net_text, 
             fill=ATO_COLORS["blue"], font=get_font("body", bold=True))
    
    current_y += net_height + 15
    
    # Add payment method
    draw.text((margin, current_y), "Payment method: Direct Credit to nominated account", 
             fill="black", font=table_font)
    
    current_y += table_font.size + 5
    
    # Add payment date
    payment_day = random.randint(1, 28)
    payment_month = random.randint(1, 12)
    payment_date = f"{payment_day:02d}/{payment_month:02d}/{year}"
    
    draw.text((margin, current_y), f"Payment date: {payment_date}", 
             fill="black", font=table_font)
    
    current_y += table_font.size + 20
    
    return current_y


def create_generic_table(draw, width, height, margin, current_y):
    """
    Create a generic financial table suitable for various document types.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        current_y: Starting Y position
        
    Returns:
        Y-position after the table
    """
    table_font = get_font("body")
    
    # Define table dimensions
    table_width = width - 2 * margin
    col1_width = int(table_width * 0.7)  # Description column
    col2_width = int(table_width * 0.3)  # Amount column
    
    # Generate random financial data
    main_amount = random.randint(5000, 50000)
    secondary_amount = random.randint(1000, 10000)
    tertiary_amount = random.randint(500, 5000)
    quaternary_amount = random.randint(100, 2000)
    
    total_amount = main_amount + secondary_amount + tertiary_amount - quaternary_amount
    
    # Generic financial table items
    if random.random() < 0.5:
        # Income style items
        table_items = [
            ("Primary Income", f"${main_amount:,}"),
            ("Additional Income", f"${secondary_amount:,}"),
            ("Interest Earned", f"${tertiary_amount:,}"),
            ("Fees and Charges", f"${quaternary_amount:,}")
        ]
    else:
        # Asset style items
        table_items = [
            ("Primary Asset Value", f"${main_amount:,}"),
            ("Secondary Assets", f"${secondary_amount:,}"),
            ("Financial Investments", f"${tertiary_amount:,}"),
            ("Liabilities", f"${quaternary_amount:,}")
        ]
    
    # Draw table header
    header_height = table_font.size + 20
    header_bg = ATO_COLORS["blue"]
    
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + header_height)],
                  fill=header_bg)
    
    # Add header text
    draw.text((margin + 10, current_y + 10), "FINANCIAL SUMMARY", 
             fill="white", font=get_font("body", bold=True))
    
    current_y += header_height
    
    # Draw column headers with gray background
    col_header_height = table_font.size + 15
    col_header_bg = (220, 220, 220)
    
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + col_header_height)],
                  fill=col_header_bg)
    
    # Draw column header text
    col_header_y = current_y + (col_header_height - table_font.size) // 2
    draw.text((margin + 10, col_header_y), "Description", 
             fill="black", font=get_font("body", bold=True))
    
    draw.text((margin + col1_width + 10, col_header_y), "Amount", 
             fill="black", font=get_font("body", bold=True))
    
    # Draw column divider
    draw.line([(margin + col1_width, current_y), 
              (margin + col1_width, current_y + col_header_height)],
             fill="black", width=1)
    
    current_y += col_header_height
    
    # Draw table rows with alternating colors
    row_height = table_font.size + 15
    for i, (label, value) in enumerate(table_items):
        # Use alternating row colors
        if i % 2 == 0:
            row_bg = ATO_COLORS["light_gray"]
            draw.rectangle([(margin, current_y), 
                           (margin + table_width, current_y + row_height)],
                          fill=row_bg)
        
        # Draw table row outline
        draw.rectangle([(margin, current_y), 
                       (margin + table_width, current_y + row_height)],
                      outline="black", width=1)
        
        # Draw divider between columns
        draw.line([(margin + col1_width, current_y), 
                  (margin + col1_width, current_y + row_height)],
                 fill="black", width=1)
        
        # Add text to cells
        text_y = current_y + (row_height - table_font.size) // 2
        draw.text((margin + 10, text_y), label, fill="black", font=table_font)
        
        # Right-align amount
        amount_width = table_font.getlength(value)
        amount_x = margin + col1_width + col2_width - amount_width - 10
        draw.text((amount_x, text_y), value, fill="black", font=table_font)
        
        current_y += row_height
    
    # Add separator before total
    draw.line([(margin, current_y), (margin + table_width, current_y)],
             fill="black", width=2)
    
    # Add total row
    total_height = table_font.size + 25
    
    # Draw total row with emphasis
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + total_height)],
                  fill=(240, 240, 250))
    
    # Draw total row outline
    draw.rectangle([(margin, current_y), 
                   (margin + table_width, current_y + total_height)],
                  outline="black", width=2)
    
    # Add total text
    total_y = current_y + (total_height - table_font.size) // 2
    draw.text((margin + 10, total_y), "TOTAL", 
             fill="black", font=get_font("body", bold=True))
    
    # Right-align total amount
    total_text = f"${total_amount:,}"
    total_width = table_font.getlength(total_text)
    total_x = margin + col1_width + col2_width - total_width - 10
    
    # Color based on positive/negative
    total_color = ATO_COLORS["blue"] if total_amount >= 0 else "red"
    draw.text((total_x, total_y), total_text, 
             fill=total_color, font=get_font("body", bold=True))
    
    current_y += total_height + 20
    
    return current_y


def create_footer(draw, width, height, margin, dept):
    """
    Create an official footer for the document.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        dept: Department name
        
    Returns:
        None (draws directly on the image)
    """
    footer_font = get_font("body")
    small_font = get_font("small")
    tiny_font = get_font("tiny")
    
    # Choose footer style
    footer_style = random.choice(["bar", "box", "minimal"])
    
    if footer_style == "bar":
        # Colored bar footer
        bar_height = 60
        bar_y = height - bar_height - margin
        
        # Draw colored bar
        draw.rectangle([(0, bar_y), (width, bar_y + 5)], 
                      fill=ATO_COLORS["blue"])
        
        # Add document identifier
        doc_id = f"Document ID: {random.randint(1000000, 9999999)}"
        draw.text((margin, bar_y + 15), doc_id, fill="black", font=small_font)
        
        # Add page number
        page_text = "Page 1 of 1"
        page_width = small_font.getlength(page_text)
        draw.text((width - margin - page_width, bar_y + 15), 
                 page_text, fill="black", font=small_font)
        
        # Add confidentiality notice
        conf_text = "This document contains confidential information."
        conf_width = tiny_font.getlength(conf_text)
        conf_x = max(margin, (width - conf_width) // 2)
        
        draw.text((conf_x, bar_y + 35), conf_text, 
                 fill=ATO_COLORS["gray"], font=tiny_font)
    
    elif footer_style == "box":
        # Boxed footer
        box_height = 70
        box_y = height - box_height - margin
        
        # Draw footer box
        draw.rectangle([(margin, box_y), (width - margin, box_y + box_height)],
                      fill=ATO_COLORS["light_gray"], outline="black", width=1)
        
        # Add department contact info
        if dept == "Australian Taxation Office":
            contact = "For more information visit ato.gov.au or call 13 28 61"
        elif "Medicare" in dept:
            contact = "For more information visit servicesaustralia.gov.au or call 132 011"
        else:
            contact = f"For more information visit {dept.lower().replace(' ', '')}.gov.au"
        
        contact_width = footer_font.getlength(contact)
        contact_x = max(margin + 10, (width - contact_width) // 2)
        
        draw.text((contact_x, box_y + 10), contact, 
                 fill="black", font=footer_font)
        
        # Add document identifier and page number
        doc_id = f"Ref: {random.randint(1000000, 9999999)}"
        page_text = "Page 1 of 1"
        
        draw.text((margin + 20, box_y + 40), doc_id, 
                 fill="black", font=small_font)
        
        page_width = small_font.getlength(page_text)
        draw.text((width - margin - page_width - 10, box_y + 40), 
                 page_text, fill="black", font=small_font)
    
    else:  # minimal
        # Simple text footer
        footer_y = height - margin - footer_font.size - small_font.size - 10
        
        # Add department name
        dept_text = f"{dept} © {random.randint(2020, 2023)}"
        dept_width = footer_font.getlength(dept_text)
        dept_x = max(margin, (width - dept_width) // 2)
        
        draw.text((dept_x, footer_y), dept_text, 
                 fill=ATO_COLORS["gray"], font=footer_font)
        
        # Add confidentiality note
        conf_text = "This document contains official information. Keep it secure."
        conf_width = small_font.getlength(conf_text)
        conf_x = max(margin, (width - conf_width) // 2)
        
        draw.text((conf_x, footer_y + footer_font.size + 5), 
                 conf_text, fill=ATO_COLORS["gray"], font=small_font)


def add_official_marks(draw, width, height, margin):
    """
    Add official-looking marks like stamps, signatures, or QR codes.
    
    Args:
        draw: ImageDraw object
        width: Document width
        height: Document height
        margin: Margin size
        
    Returns:
        None (draws directly on the image)
    """
    # Decide what marks to add
    marks = []
    
    if random.random() < 0.5:
        marks.append("signature")
    
    if random.random() < 0.4:
        marks.append("stamp")
    
    if random.random() < 0.3:
        marks.append("qr")
    
    # Place marks in appropriate locations
    for mark in marks:
        if mark == "signature":
            # Add signature placeholder at the bottom section
            sig_y = height - margin - 150
            sig_width = 200
            sig_height = 70
            
            # Random x position in bottom section
            sig_x = random.randint(margin, width - margin - sig_width)
            
            # Draw signature box with dashed line
            for i in range(sig_x, sig_x + sig_width, 5):
                line_length = min(3, sig_x + sig_width - i)
                draw.line([(i, sig_y + sig_height), 
                          (i + line_length, sig_y + sig_height)],
                         fill=ATO_COLORS["gray"], width=1)
            
            # Add random signature-like curves
            for _ in range(3):
                start_x = random.randint(sig_x + 20, sig_x + 80)
                start_y = random.randint(sig_y + 30, sig_y + 60)
                
                control1_x = random.randint(start_x + 10, start_x + 40)
                control1_y = random.randint(start_y - 20, start_y + 20)
                
                control2_x = random.randint(control1_x + 10, control1_x + 40)
                control2_y = random.randint(start_y - 20, start_y + 20)
                
                end_x = random.randint(control2_x + 10, sig_x + sig_width - 20)
                end_y = random.randint(start_y - 10, start_y + 10)
                
                # Draw signature curve
                draw.line([(start_x, start_y), (control1_x, control1_y), 
                          (control2_x, control2_y), (end_x, end_y)],
                         fill="black", width=2)
            
            # Add signature label
            sig_label = "Authorized Signature"
            sig_label_width = get_font("small").getlength(sig_label)
            sig_label_x = sig_x + (sig_width - sig_label_width) // 2
            
            draw.text((sig_label_x, sig_y + sig_height + 5), 
                     sig_label, fill="black", font=get_font("small"))
        
        elif mark == "stamp":
            # Add official-looking stamp
            stamp_size = random.randint(100, 150)
            
            # Position stamp in bottom right or middle right
            if random.random() < 0.5:
                # Bottom right
                stamp_x = width - margin - stamp_size - 50
                stamp_y = height - margin - stamp_size - 80
            else:
                # Middle right
                stamp_x = width - margin - stamp_size - 50
                stamp_y = height // 2
            
            # Choose stamp style
            stamp_style = random.choice(["circle", "square", "oval"])
            stamp_color = ATO_COLORS["red"] if random.random() < 0.7 else ATO_COLORS["blue"]
            
            # Draw stamp outline
            if stamp_style == "circle":
                draw.ellipse([(stamp_x, stamp_y), 
                             (stamp_x + stamp_size, stamp_y + stamp_size)],
                            outline=stamp_color, width=2)
                
                # Add inner circle
                inner_margin = stamp_size // 8
                draw.ellipse([(stamp_x + inner_margin, stamp_y + inner_margin), 
                             (stamp_x + stamp_size - inner_margin, 
                              stamp_y + stamp_size - inner_margin)],
                            outline=stamp_color, width=1)
            
            elif stamp_style == "square":
                draw.rectangle([(stamp_x, stamp_y), 
                               (stamp_x + stamp_size, stamp_y + stamp_size)],
                              outline=stamp_color, width=2)
                
                # Add inner square
                inner_margin = stamp_size // 8
                draw.rectangle([(stamp_x + inner_margin, stamp_y + inner_margin), 
                               (stamp_x + stamp_size - inner_margin, 
                                stamp_y + stamp_size - inner_margin)],
                              outline=stamp_color, width=1)
            
            else:  # oval
                # Draw oval stamp
                draw.ellipse([(stamp_x, stamp_y), 
                             (stamp_x + stamp_size, stamp_y + stamp_size * 0.7)],
                            outline=stamp_color, width=2)
                
                # Add inner oval
                inner_margin = stamp_size // 8
                draw.ellipse([(stamp_x + inner_margin, stamp_y + inner_margin * 0.7), 
                             (stamp_x + stamp_size - inner_margin, 
                              stamp_y + stamp_size * 0.7 - inner_margin * 0.7)],
                            outline=stamp_color, width=1)
            
            # Add text to stamp
            stamp_text = random.choice([
                "APPROVED", "PROCESSED", "OFFICIAL", 
                "VERIFIED", "CONFIRMED", "AUTHORIZED"
            ])
            
            stamp_font = get_font("body", bold=True, 
                                 size_range=(stamp_size // 8, stamp_size // 6))
            
            stamp_text_width = stamp_font.getlength(stamp_text)
            stamp_text_x = stamp_x + (stamp_size - stamp_text_width) // 2
            stamp_text_y = stamp_y + stamp_size // 2 - stamp_font.size // 2
            
            draw.text((stamp_text_x, stamp_text_y), stamp_text, 
                     fill=stamp_color, font=stamp_font)
            
            # Add date to stamp
            date_day = random.randint(1, 28)
            date_month = random.randint(1, 12)
            date_year = random.randint(2020, 2023)
            date_text = f"{date_day:02d}/{date_month:02d}/{date_year}"
            
            date_font = get_font("small", size_range=(stamp_size // 10, stamp_size // 8))
            date_width = date_font.getlength(date_text)
            date_x = stamp_x + (stamp_size - date_width) // 2
            date_y = stamp_text_y + stamp_font.size + 5
            
            draw.text((date_x, date_y), date_text, 
                     fill=stamp_color, font=date_font)
        
        elif mark == "qr":
            # Add QR code placeholder
            qr_size = random.randint(70, 100)
            
            # Position QR code in top right or bottom right
            if random.random() < 0.5:
                # Top right
                qr_x = width - margin - qr_size - 20
                qr_y = margin + 100
            else:
                # Bottom right
                qr_x = width - margin - qr_size - 20
                qr_y = height - margin - qr_size - 50
            
            # Draw QR code placeholder
            draw.rectangle([(qr_x, qr_y), (qr_x + qr_size, qr_y + qr_size)],
                          outline="black", width=1)
            
            # Add QR code-like pattern
            cell_size = qr_size // 10
            for i in range(10):
                for j in range(10):
                    # Randomly fill cells
                    if random.random() < 0.4:
                        cell_x = qr_x + i * cell_size
                        cell_y = qr_y + j * cell_size
                        
                        draw.rectangle([(cell_x, cell_y), 
                                       (cell_x + cell_size, cell_y + cell_size)],
                                      fill="black")
            
            # Add position markers (corners)
            marker_size = cell_size * 3
            
            # Top-left marker
            draw.rectangle([(qr_x, qr_y), 
                           (qr_x + marker_size, qr_y + marker_size)],
                          fill="black")
            draw.rectangle([(qr_x + cell_size, qr_y + cell_size), 
                           (qr_x + marker_size - cell_size, qr_y + marker_size - cell_size)],
                          fill="white")
            
            # Top-right marker
            draw.rectangle([(qr_x + qr_size - marker_size, qr_y), 
                           (qr_x + qr_size, qr_y + marker_size)],
                          fill="black")
            draw.rectangle([(qr_x + qr_size - marker_size + cell_size, qr_y + cell_size), 
                           (qr_x + qr_size - cell_size, qr_y + marker_size - cell_size)],
                          fill="white")
            
            # Bottom-left marker
            draw.rectangle([(qr_x, qr_y + qr_size - marker_size), 
                           (qr_x + marker_size, qr_y + qr_size)],
                          fill="black")
            draw.rectangle([(qr_x + cell_size, qr_y + qr_size - marker_size + cell_size), 
                           (qr_x + marker_size - cell_size, qr_y + qr_size - cell_size)],
                          fill="white")
            
            # Add QR label
            qr_label = "Scan for details"
            qr_label_width = get_font("tiny").getlength(qr_label)
            qr_label_x = qr_x + (qr_size - qr_label_width) // 2
            
            draw.text((qr_label_x, qr_y + qr_size + 5), 
                     qr_label, fill="black", font=get_font("tiny"))


def create_tax_document(image_size=2048):
    """
    Create an Australian Taxation Office document.
    
    Args:
        image_size: Size of the output image (square)
        
    Returns:
        PIL Image containing a tax document
    """
    # Choose a department
    dept = random.choice(DEPARTMENTS)
    
    # Choose a document type for the selected department
    doc_types = DOCUMENT_TYPES.get(dept, DOCUMENT_TYPES["Australian Taxation Office"])
    doc_type = random.choice(doc_types)
    
    # Create background with subtle watermark
    background = create_background_with_watermark(image_size, image_size, dept)
    draw = ImageDraw.Draw(background)
    
    # Set margin
    margin = int(image_size * 0.05)
    
    # Create header
    current_y = create_header(draw, image_size, image_size, margin, dept, doc_type)
    
    # Add reference section
    current_y = create_reference_section(draw, image_size, image_size, margin, current_y)
    
    # Add recipient details
    current_y = create_recipient_section(draw, image_size, image_size, margin, current_y)
    
    # Add financial information
    current_y = create_financial_section(draw, image_size, image_size, margin, current_y, doc_type)
    
    # Add official marks (signatures, stamps, etc.)
    add_official_marks(draw, image_size, image_size, margin)
    
    # Create footer
    create_footer(draw, image_size, image_size, margin, dept)
    
    # Apply subtle finishing effect
    # Slight rotation for realism
    if random.random() < 0.5:
        angle = random.uniform(-0.3, 0.3)
        background = background.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
        
        # Resize to original dimensions
        background = background.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return background


if __name__ == "__main__":
    # Test document generation
    doc = create_tax_document(2048)
    doc.save("test_tax_document.png")
    print("Created test tax document: test_tax_document.png")