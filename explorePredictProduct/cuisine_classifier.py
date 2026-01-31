"""
Product-to-Cuisine Classifier

STATUS: Designed but not fully implemented
REASON: 33K unique products is too many for simple keyword matching

This script was designed to classify individual products into cuisine
categories based on their names, instead of using vendor.primary_cuisine.
"""

import re
from typing import Optional, Dict, List

# Cuisine keyword patterns
CUISINE_PATTERNS: Dict[str, List[str]] = {
    "japanese": [
        "sushi", "maki", "nigiri", "teriyaki", "ramen", "udon",
        "tempura", "sashimi", "sake", "edamame", "gyoza", "yakitori"
    ],
    "italian": [
        "pizza", "pasta", "risotto", "lasagna", "bolognese",
        "carbonara", "margherita", "calzone", "penne", "spaghetti"
    ],
    "thai": [
        "pad thai", "curry", "green curry", "red curry", "satay",
        "tom yum", "tom kha", "panang", "massaman"
    ],
    "indian": [
        "biryani", "tikka", "korma", "tandoori", "naan", "masala",
        "vindaloo", "samosa", "pakora", "dal", "paneer"
    ],
    "chinese": [
        "wok", "szechuan", "dim sum", "chow mein", "kung pao",
        "sweet and sour", "spring roll", "fried rice"
    ],
    "mexican": [
        "taco", "burrito", "quesadilla", "nachos", "enchilada",
        "fajita", "guacamole", "salsa"
    ],
    "greek": [
        "souvlaki", "gyros", "tzatziki", "moussaka", "feta"
    ],
    "burger": [
        "burger", "hamburgare", "cheeseburger"
    ],
    "kebab": [
        "kebab", "döner", "falafel", "hummus", "shawarma"
    ],
    "swedish": [
        "köttbullar", "gravlax", "räkmacka", "tunnbröd"
    ],
}


def normalize_product_name(name: str) -> str:
    """
    Normalize product name for classification.
    - Convert to lowercase
    - Remove size indicators (small, medium, large, etc.)
    - Remove quantity indicators
    - Remove price information
    """
    if not name:
        return ""

    normalized = name.lower()

    # Remove common size patterns
    size_patterns = [
        r'\b(small|medium|large|xl|xxl)\b',
        r'\b(liten|mellan|stor)\b',  # Swedish sizes
        r'\d+\s*(ml|cl|l|g|kg)\b',   # Volume/weight
        r'\d+\s*st\b',               # Quantity
    ]

    for pattern in size_patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)

    # Remove extra whitespace
    normalized = ' '.join(normalized.split())

    return normalized.strip()


def classify_product(product_name: str) -> Optional[str]:
    """
    Classify a product into a cuisine category based on keywords.

    Args:
        product_name: The name of the product

    Returns:
        Cuisine category string, or None if no match found
    """
    normalized = normalize_product_name(product_name)

    if not normalized:
        return None

    # Check each cuisine's keywords
    for cuisine, keywords in CUISINE_PATTERNS.items():
        for keyword in keywords:
            if keyword in normalized:
                return cuisine

    return None


def get_classification_confidence(product_name: str) -> Dict[str, float]:
    """
    Get confidence scores for all possible cuisine classifications.

    NOTE: This is a stub - would need ML model for proper confidence scores.
    Currently returns 1.0 for matched cuisine, 0.0 for others.
    """
    normalized = normalize_product_name(product_name)
    scores = {cuisine: 0.0 for cuisine in CUISINE_PATTERNS.keys()}

    for cuisine, keywords in CUISINE_PATTERNS.items():
        matches = sum(1 for kw in keywords if kw in normalized)
        if matches > 0:
            scores[cuisine] = min(1.0, matches * 0.5)

    return scores


# Example usage (for reference)
if __name__ == "__main__":
    test_products = [
        "Lax Teriyaki",
        "Pad Thai Nudlar",
        "Margherita Pizza",
        "Chicken Tikka Masala",
        "Classic Burger",
        "Veggie Wok",
        "Random Product Name",
    ]

    print("Product Classification Examples:")
    print("-" * 50)
    for product in test_products:
        cuisine = classify_product(product)
        print(f"{product:30} -> {cuisine or 'UNKNOWN'}")
