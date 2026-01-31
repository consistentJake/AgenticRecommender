# Product-to-Cuisine Analysis Notes

## Dataset Statistics

- **Total unique products:** 33,754
- **Products with clear cuisine keywords:** ~26%
- **Language mix:** Swedish + English
- **Current cuisine categories:** 37 (from `vendor.primary_cuisine`)

## Problem Identified

### Current Behavior
All products from a vendor inherit `vendor.primary_cuisine`:
```
vendor_id: 123 (primary_cuisine: "sushi")
├── product: "Lax Teriyaki"     → cuisine: "sushi"
├── product: "Pad Thai Nudlar"  → cuisine: "sushi"  ❌ Should be Thai
└── product: "Edamame"          → cuisine: "sushi"  ✓ Correct
```

### Impact on Basket Size
Order with multiple products from different actual cuisines:
- Current: `basket_size = 1` (all same vendor cuisine)
- Expected: `basket_size > 1` (different product cuisines)

### Example: Order 32
| Product | Current Label | Actual Cuisine |
|---------|---------------|----------------|
| Lax Teriyaki | sushi | Japanese |
| Pad Thai Nudlar | sushi | Thai |

Both labeled "sushi" because vendor is a sushi restaurant.

## Keyword Patterns Identified

### Strong Indicators (High Confidence)
```
Japanese: sushi, maki, nigiri, teriyaki, ramen, udon, tempura
Italian:  pizza, pasta, risotto, lasagna, bolognese
Thai:     pad thai, curry (green/red), satay, tom yum
Indian:   biryani, tikka, korma, tandoori, naan, masala
Chinese:  wok, szechuan, dim sum
Mexican:  taco, burrito, quesadilla
Greek:    souvlaki, gyros
Burger:   burger, hamburgare
```

### Ambiguous Keywords (Need Context)
- "curry" - Could be Thai, Indian, or Japanese
- "noodles/nudlar" - Could be Thai, Chinese, or Japanese
- "fried rice" - Could be Chinese, Thai, or general Asian
- "chicken" - Universal, no cuisine signal

### Swedish Language Considerations
Common Swedish food terms:
- "nudlar" = noodles
- "ris" = rice
- "kyckling" = chicken
- "nötkött" = beef
- "fläsk" = pork
- "räkor" = shrimp

## Classification Challenges

### 1. Volume of Unique Products
33,754 unique product names makes simple classification difficult:
- Manual mapping not feasible
- Keyword matching covers only ~26%
- Remaining 74% would be "unknown"

### 2. Name Variations
Same dish, different names:
- "Pad Thai", "Pad-Thai", "Padthai", "Pad Thai Nudlar"
- "Chicken Teriyaki", "Teriyaki Chicken", "Kyckling Teriyaki"

### 3. Combo/Mixed Products
Products that span cuisines:
- "Sushi & Pad Thai Combo"
- "Asian Fusion Bowl"
- "Mix Meny"

### 4. Generic Names
Names with no cuisine signal:
- "Chef's Special"
- "Today's Special"
- "Mix 1", "Mix 2"
- "Lunch Menu A"

## Proposed Classification Approach (Not Implemented)

### Phase 1: Keyword Matching
```python
def classify_by_keywords(product_name):
    # See cuisine_classifier.py
    pass
```

### Phase 2: ML Classification (Future)
- Train classifier on products with known cuisine
- Use product descriptions if available
- Consider vendor cuisine as prior probability

### Phase 3: Hierarchical Taxonomy
Group 37 cuisines into broader categories:
```
Asian
├── Japanese (sushi, ramen)
├── Chinese (wok, dim sum)
├── Thai (pad thai, curry)
├── Indian (tikka, biryani)
└── Korean (bibimbap, kimchi)

European
├── Italian (pizza, pasta)
├── Greek (souvlaki, gyros)
└── Swedish (köttbullar)

American
├── Burger
├── Mexican (taco, burrito)
└── BBQ
```

## Decision: On Hold

### Reasons
1. 33K unique products requires significant effort
2. Current vendor-based approach works for most cases
3. Would need ML model for proper classification
4. Manual validation of edge cases required

### Future Work (If Revisited)
1. Build training dataset from high-confidence keyword matches
2. Train text classifier (BERT/transformers)
3. Handle Swedish language properly
4. Create cuisine hierarchy for grouping
5. Validate with domain expert

## References

- Data source: `data/orders.csv`, `data/products.csv`
- Current cuisine mapping: `vendor.primary_cuisine` field
- Related code: `features/` feature extraction modules
