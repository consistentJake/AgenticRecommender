# Dataset Sources

## Amazon Beauty Dataset

### Source
The dataset was found [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) in "per-category files". The necessary files are the Beauty [reviews](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz) and the Beauty [metadata](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz). Note this is the older version of the Amazon Beauty dataset. This is intentional, because this way it matches the dataset with most research papers (BERT4Rec, SASRec) using Beauty.

### Required Files
- `reviews_Beauty.json.gz` - User review interactions data
- `meta_Beauty.json.gz` - Product metadata including titles and descriptions

### Download Instructions
```bash
# Download review data
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz

# Download metadata
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz

# Extract files
gunzip reviews_Beauty.json.gz
gunzip meta_Beauty.json.gz
```

### Processing
The raw JSON files are processed to create:
1. Session-based interaction data (sessions.csv)
2. Product lookup table (products_lookup.csv)
3. Formatted timestamps (formatted_sessions.csv)
4. Final dataset objects for training and evaluation

### Data Format
- **Reviews**: Each line is a JSON object with fields like `reviewerID`, `asin`, `unixReviewTime`, `overall`
- **Metadata**: Each line is a JSON object with fields like `asin`, `title`, `categories`, `description`