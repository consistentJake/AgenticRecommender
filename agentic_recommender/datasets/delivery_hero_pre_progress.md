## Data Analysis Results (Dec 21)

### Historic Event Length Distribution Analysis

Analyzed 1,000 customers from Delivery Hero dataset:

**Customer Statistics:**
- Total customers: 1,000
- Min events per customer: 1
- Max events per customer: 61
- Average events per customer: 5.69
- Median: 3 events

**Distribution:**
- 68% of customers have 1-5 historic events
- 17.3% have 6-10 events
- 10.5% have 11-20 events
- Only 4.2% have 20+ events

**Data Preservation vs Sequence Length:**
| Seq Length | Customers Preserved | % Kept | Data Loss |
|------------|-------------------|--------|-----------|
| 3          | ~50%              | 50%    | 50%       |
| 5          | 32%               | 32%    | 68%       |
| 10         | 14.7%             | 14.7%  | 85.3%     |
| 15         | 8.3%              | 8.3%   | 91.7%     |

**Recommendation:** Use seq_len=5 as reasonable trade-off between data preservation (32%) and sufficient context.

### Updated Approach

**New Strategy:**
1. Use min/max thresholds instead of fixed sequence length
2. Add sliding window option for customers with many events
3. Improved negative sampling (exclude same product names)

**Command Examples:**
```bash
# Basic conversion with optimized settings
python agentic_recommender/datasets/convert_to_jsonl.py \
    --source agentic_recommender/datasets/delivery_hero \
    --output-dir agentic_recommender/datasets \
    --min-history-len 3 \
    --max-history-len 5 \
    --enable-sliding-windows

# Sample customers for testing
python agentic_recommender/datasets/convert_to_jsonl.py \
    --source agentic_recommender/datasets/delivery_hero \
    --output-dir agentic_recommender/datasets \
    --min-history-len 3 \
    --max-history-len 5 \
    --sample-customers 1000 \
    --enable-sliding-windows

# Without sliding windows (one sample per user)
python agentic_recommender/datasets/convert_to_jsonl.py \
    --min-history-len 3 \
    --max-history-len 5 \
    --sample-customers 1000
```

## Previous Questions (Dec 20)

1. ✅ we need to integrate the hour of when order is placed, the day of week, like wed, sun, mon
2. ✅ instruction and system are duplicated.