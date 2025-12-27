"""
Food delivery dataset adapter.

Parses Singapore food delivery dataset (JSONL format) into structured types
for agent processing.
"""

import re
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class OrderRecord:
    """Structured order record."""
    idx: int
    day: str        # Mon, Tue, Wed, Thu, Fri, Sat, Sun
    hour: int       # 0-23
    cuisine: str    # pizza, burgare, mexikanskt, etc.
    price: float    # Normalized 0.0-1.0


@dataclass
class CandidateProduct:
    """Candidate product for recommendation."""
    name: str
    cuisine: str
    price: float


@dataclass
class RecommendationRequest:
    """Complete recommendation request."""
    user_id: str                    # Assigned during parsing
    orders: List[OrderRecord]       # Recent order history
    candidate: CandidateProduct     # Product to evaluate
    ground_truth: Optional[bool]    # For evaluation (Yes -> True, No -> False)
    system_prompt: Optional[str] = None


def parse_order_table(table_text: str) -> List[OrderRecord]:
    """
    Parse markdown order table into list of OrderRecords.

    Args:
        table_text: Markdown table text

    Returns:
        List of OrderRecord objects
    """
    orders = []

    # Split into lines
    lines = table_text.strip().split('\n')

    # Find table rows (skip header and separator)
    for line in lines:
        if not line.strip() or line.startswith('|--'):
            continue
        if 'idx' in line and 'day' in line:  # Header row
            continue

        # Parse row: | idx | day | hour | cuisine | price |
        parts = [p.strip() for p in line.split('|') if p.strip()]

        if len(parts) >= 5:
            try:
                idx = int(parts[0])
                day = parts[1]
                hour = int(parts[2])
                cuisine = parts[3]
                # Price may have '$' prefix, remove it
                price_str = parts[4].replace('$', '').strip()
                price = float(price_str)

                orders.append(OrderRecord(
                    idx=idx,
                    day=day,
                    hour=hour,
                    cuisine=cuisine,
                    price=price
                ))
            except (ValueError, IndexError):
                # Skip malformed rows
                continue

    return orders


def parse_candidate_product(candidate_text: str) -> CandidateProduct:
    """
    Parse candidate product line.

    Examples:
        "- Buffalo Wings (cuisine: mexikanskt, price: $0.32)"
        "Buffalo Wings (cuisine: mexikanskt, price: $0.32)"

    Args:
        candidate_text: Candidate product text

    Returns:
        CandidateProduct object
    """
    # Remove leading dash if present
    text = candidate_text.strip().lstrip('-').strip()

    # Pattern: Name (cuisine: TYPE, price: $PRICE)
    pattern = r'(.+?)\s*\(cuisine:\s*(\w+),\s*price:\s*\$?([\d.]+)\)'
    match = re.search(pattern, text)

    if match:
        name = match.group(1).strip()
        cuisine = match.group(2).strip()
        price = float(match.group(3))

        return CandidateProduct(
            name=name,
            cuisine=cuisine,
            price=price
        )
    else:
        # Fallback: try to extract what we can
        # Look for price pattern
        price_match = re.search(r'\$?([\d.]+)', text)
        price = float(price_match.group(1)) if price_match else 0.0

        # Extract name (text before parenthesis)
        name_match = re.match(r'(.+?)\s*\(', text)
        name = name_match.group(1).strip() if name_match else text

        return CandidateProduct(
            name=name,
            cuisine='unknown',
            price=price
        )


def generate_user_id(orders: List[OrderRecord]) -> str:
    """
    Generate deterministic user ID from order pattern.

    Uses hash of cuisine sequence to create consistent IDs.

    Args:
        orders: List of orders

    Returns:
        User ID string
    """
    if not orders:
        return "user_unknown"

    # Create signature from first few orders
    signature_parts = []
    for order in orders[:5]:  # Use first 5 orders
        signature_parts.append(f"{order.cuisine}:{order.price:.2f}")

    signature = "-".join(signature_parts)

    # Hash to create short ID
    hash_obj = hashlib.md5(signature.encode())
    hash_hex = hash_obj.hexdigest()[:8]

    return f"user_{hash_hex}"


def parse_jsonl_record(record: Dict[str, Any]) -> RecommendationRequest:
    """
    Parse JSONL record into structured RecommendationRequest.

    Args:
        record: Dictionary from JSONL line with keys:
            - instruction: Order table + candidate product
            - input: Usually empty
            - output: "Yes" or "No"
            - system: System prompt
            - history: Conversation history (usually empty)

    Returns:
        RecommendationRequest object
    """
    instruction = record.get('instruction', '')
    output = record.get('output', '')
    system_prompt = record.get('system', '')

    # Split instruction into order table and candidate
    # Pattern: table ends before "Candidate product:"
    if 'Candidate product:' in instruction or 'candidate product:' in instruction.lower():
        parts = re.split(r'[Cc]andidate [Pp]roduct:', instruction, maxsplit=1)
        table_text = parts[0].strip()
        candidate_text = parts[1].strip() if len(parts) > 1 else ''
    else:
        # Fallback: assume entire instruction is table
        table_text = instruction
        candidate_text = ''

    # Parse components
    orders = parse_order_table(table_text)
    candidate = parse_candidate_product(candidate_text) if candidate_text else CandidateProduct("Unknown", "unknown", 0.0)

    # Generate user ID
    user_id = generate_user_id(orders)

    # Parse ground truth
    ground_truth = None
    if output:
        output_lower = output.strip().lower()
        if output_lower == 'yes':
            ground_truth = True
        elif output_lower == 'no':
            ground_truth = False

    return RecommendationRequest(
        user_id=user_id,
        orders=orders,
        candidate=candidate,
        ground_truth=ground_truth,
        system_prompt=system_prompt
    )


def format_order_history_table(orders: List[OrderRecord]) -> str:
    """
    Format orders as markdown table.

    Args:
        orders: List of OrderRecord objects

    Returns:
        Markdown table string
    """
    if not orders:
        return "No order history available."

    lines = [
        "| # | Day | Hour | Cuisine | Price |",
        "|---|-----|------|---------|-------|"
    ]

    for i, order in enumerate(orders, 1):
        lines.append(
            f"| {i} | {order.day} | {order.hour} | {order.cuisine} | ${order.price:.2f} |"
        )

    return "\n".join(lines)


def format_candidate_product(candidate: CandidateProduct) -> str:
    """
    Format candidate product as string.

    Args:
        candidate: CandidateProduct object

    Returns:
        Formatted string
    """
    return f"- {candidate.name} (cuisine: {candidate.cuisine}, price: ${candidate.price:.2f})"
