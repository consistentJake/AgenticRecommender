# Product-to-Cuisine Research

**Status:** On Hold

## Overview

This folder contains research on mapping individual products to cuisine categories, rather than using the vendor's primary cuisine for all products.

## Problem Statement

Currently, the system assigns `vendor.primary_cuisine` to all products from that vendor:
- A sushi restaurant's Pad Thai gets labeled "sushi"
- Order with multiple products → `basket_size = 1` for cuisine dimension
- Example: Order 32 has "Lax Teriyaki" + "Pad Thai Nudlar", both labeled "sushi"

## Proposed Solution

Map `product_id → product_name → cuisine category` using keyword-based classification.

## Research Findings

- **33,754 unique products** in the dataset
- ~26% have clear cuisine keywords in their names
- Product names are in Swedish + English mix
- Current system uses `vendor.primary_cuisine` (37 categories)

## Why On Hold

33K unique product names creates too many categories. Would need:
- Hierarchical cuisine taxonomy
- ML-based classification approach
- Manual validation of edge cases

## Files in This Folder

- `README.md` - This summary
- `cuisine_classifier.py` - Designed classifier (not fully implemented)
- `analysis_notes.md` - Detailed research notes and findings
