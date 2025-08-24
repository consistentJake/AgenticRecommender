# LLM-Based Sequential Recommendation Implementation Analysis

## Overview
This document provides a detailed analysis of the implementation from the paper "Improving Sequential Recommendations with LLMs" and its corresponding codebase.

## Code Reference Guide - Absolute File Paths

This section provides a comprehensive mapping of all code references mentioned in this analysis to their absolute file paths for easy reference in future project implementations:

### Dataset Preparation & Core Data Structures
- **Session Dataset Class**: `previousWorks/LLM-Sequential-Recommendation/main/data/session_dataset.py`
- **Beauty Dataset Creation**: `previousWorks/LLM-Sequential-Recommendation/beauty/create_sessions.ipynb`
- **Embedding Attachment**: `previousWorks/LLM-Sequential-Recommendation/beauty/attach_embeddings.ipynb`

### Neural Model Implementations
- **SASRec Model**: `previousWorks/LLM-Sequential-Recommendation/main/transformer/sasrec/sasrec_model.py`
- **Projection Head (Neural Prediction)**: `previousWorks/LLM-Sequential-Recommendation/main/utils/neural_utils/custom_layers/projection_head.py`

### LLM-Based Approaches
- **OpenAI Embedding Utils**: `previousWorks/LLM-Sequential-Recommendation/main/llm_based/embedding_utils/openai_utils.py`
- **Create Embeddings Notebook**: `previousWorks/LLM-Sequential-Recommendation/main/llm_based/embedding_utils/create_embeddings.ipynb`
- **Fine-tuning Prompt Generation**: `previousWorks/LLM-Sequential-Recommendation/main/llm_based/prompt_model/genitem/generate_finetune_prompts.ipynb`
- **GPT Prediction & Hallucination Mapping**: `previousWorks/LLM-Sequential-Recommendation/main/llm_based/prompt_model/genitem/predict_gpt.ipynb`

### Key Function Locations by Topic
- **Data Splitting (`_prepare_to_predict`, `_extract_ground_truths`)**: Found in `previousWorks/LLM-Sequential-Recommendation/main/data/session_dataset.py`
- **Neural Model Vocabulary Prediction**: Found in `previousWorks/LLM-Sequential-Recommendation/main/utils/neural_utils/custom_layers/projection_head.py`
- **Embedding Generation**: Found in `previousWorks/LLM-Sequential-Recommendation/main/llm_based/embedding_utils/openai_utils.py`
- **Hallucination Resolution**: Found in `previousWorks/LLM-Sequential-Recommendation/main/llm_based/prompt_model/genitem/predict_gpt.ipynb`
- **Clustering Implementation**: Found in `previousWorks/LLM-Sequential-Recommendation/beauty/attach_embeddings.ipynb`

## 1. Dataset Preparation

### 1.1 Amazon Beauty Dataset
The Amazon Beauty dataset preparation follows these steps:

**Data Loading and Preprocessing:**
- Raw data is loaded from `reviews_Beauty.json` containing user reviews
- Each interaction consists of:
  - `SessionId`: reviewer ID (e.g., "A39HTATAQ9V7YF")
  - `ItemId`: product ASIN (e.g., "0205616461")
  - `Time`: Unix timestamp
  - `Reward`: Fixed at 1.0 (implicit feedback)

**5-core Filtering:**
```python
MIN_NUM_INTERACTIONS = 5
```
The dataset applies iterative p-core filtering where both users and items must have at least 5 interactions. This process:
1. Removes users with < 5 interactions
2. Removes items with < 5 interactions  
3. Repeats until convergence

**Final Statistics (Beauty):**
- Sessions: 22,363
- Items: 12,101
- Interactions: 198,502
- Average session length: 8.9
- Density: 0.073%

**Product Metadata:**
Product names are extracted from `meta_Beauty.json` and cleaned:
- HTML entities are replaced (e.g., `&amp;` → "and")
- Special characters are handled
- Product names are mapped to item IDs

**Data Structure Example:**
```
SessionId: 0 (mapped from "A39HTATAQ9V7YF")
ItemId: 1234 (mapped from "B0000535UX")
Time: 2023-01-15 10:30:00
Product: "Bio-Active Anti-Aging Serum"
```

### 1.2 Delivery Hero Dataset
The Delivery Hero dataset is a proprietary QCommerce dataset for dark store and local shop orders:

**Key Characteristics:**
- No p-core filtering applied (to simulate real-world setting)
- Only sessions with >1 interaction kept
- Contains anonymous session data

**Final Statistics (Delivery Hero):**
- Sessions: 258,710
- Items: 38,246
- Interactions: 1,474,658
- Average session length: 5.7
- Density: 0.015%

**Notable Differences:**
- Shorter average sessions than Beauty dataset
- Lower density (sparser data)
- Higher semantic similarity within sessions (items more homogeneous)
- No publicly available product metadata

### 1.3 Steam Dataset
Gaming dataset with additional preprocessing:
- 5-core filtering applied
- Items with insufficient metadata removed
- Game titles concatenated with tags for better semantic representation

**Final Statistics (Steam):**
- Sessions: 279,290
- Items: 11,784
- Interactions: 3,456,395
- Average session length: 12.4
- Density: 0.105%

## 2. Recommendation Tasks

The implementation supports multiple recommendation task formulations:

### 2.1 Next-Item Prediction (Primary Task)
**Leave-one-out evaluation:**
- For each session of n items:
  - Prompt: items 1 to n-1
  - Ground truth: item n
  - Task: Predict the next item

**Data Split (from `previousWorks/LLM-Sequential-Recommendation/main/data/session_dataset.py`):**
```python
def _prepare_to_predict(self, data, n_withheld=1):
    prompt_items = session_items[:-n_withheld]
    return prompt_items

def _extract_ground_truths(self, data, n_withheld=1):
    ground_truth = session_items[-n_withheld:]
    return ground_truth
```

### 2.2 Classification Task (LLMSeqPromptClass)
Uses K-means clustering for item categorization:

**Implementation:**
1. Generate item embeddings (OpenAI/Google)
2. Cluster items using K-means (K=200)
3. Select most popular item per cluster as representative
4. For classification:
   - Find K nearest clusters to ground truth embedding
   - Present cluster representatives as categories
   - Model selects from available categories

**Example Prompt:**
```
User's previous purchases:
- Moisturizing Face Cream
- Anti-aging Serum
- Eye Makeup Remover

Available products (200 categories):
1. Lip Gloss
2. Foundation Brush
3. Hair Conditioner
...

Select 20 products from available list.
```

### 2.3 Generation Tasks
**Single Item Generation (LLMSeqPromptGenItem):**
- Model generates one item name
- Repeated k times for top-k recommendations
- Hallucinations mapped to catalog via embedding similarity

**List Generation (LLMSeqPromptGenList):**
- Model generates ranked list of k items in one response
- Trained on recommendations from high-performing baseline model

### 2.4 Ranking Task (LLMSeqPromptRank)
- Candidate items provided in prompt
- Model reranks the provided slate
- No hallucination handling needed

## 3. Target Item Selection in Classification

The classification task uses an intelligent target selection mechanism:

**Ground Truth Mapping (implemented in clustering notebooks like `previousWorks/LLM-Sequential-Recommendation/beauty/attach_embeddings.ipynb`):**
```python
# For each ground truth item:
gt_embedding = product_embeddings[ground_truth_item]
distances = euclidean_distances(kmeans.cluster_centers_, gt_embedding)
nearest_clusters = distances.argsort()[:TOP_K]

# Map clusters to representative items
target_categories = [cluster_to_popular_item[c] for c in nearest_clusters]
```

**Key Points:**
- Ground truth may not be in the exact cluster center
- Top-K nearest clusters selected based on embedding distance
- Ensures ground truth is semantically close to selected categories
- Popular items used as cluster representatives for better generalization

## 4. Top-K Selection Calculation

Top-K recommendations are calculated differently across approaches:

### 4.1 Neural Models (BERT4Rec, SASRec, GRU4Rec)
**Implementation in `previousWorks/LLM-Sequential-Recommendation/main/transformer/sasrec/sasrec_model.py` and projection layers:**
```python
def predict(self, predict_data, top_k=10):
    scores = model.forward(session_items)  # Get scores for all items
    top_k_indices = scores.argsort()[-top_k:][::-1]  # Sort and take top-k
    return catalog_items[top_k_indices]
```

### 4.2 LLMSeqSim (Embedding Similarity)
**Implementation using embeddings from `previousWorks/LLM-Sequential-Recommendation/main/llm_based/embedding_utils/openai_utils.py`:**
```python
# 1. Compute session embedding
session_emb = aggregate(item_embeddings)  # Average, weighted, etc.

# 2. Compute similarities to all catalog items
similarities = cosine_similarity(session_emb, catalog_embeddings)

# 3. Return top-k most similar
top_k_items = similarities.argsort()[-top_k:][::-1]
```

### 4.3 LLMSeqPrompt (Generation)
**Implementation in `previousWorks/LLM-Sequential-Recommendation/main/llm_based/prompt_model/genitem/predict_gpt.ipynb`:**
```python
# For single item generation:
recommendations = []
for i in range(k):
    item = llm.generate(prompt, temperature=0.7)
    recommendations.append(item)

# Deduplicate and rank by frequency
ranked_recs = Counter(recommendations).most_common(k)
```

### 4.4 Evaluation Metrics
**Top-K metrics calculated in evaluation scripts (implementation distributed across various eval notebooks):**

**NDCG@K:**
```python
def ndcg_at_k(predictions, ground_truth, k):
    dcg = sum([1/log2(i+2) for i, item in enumerate(predictions[:k]) 
               if item in ground_truth])
    idcg = sum([1/log2(i+2) for i in range(min(len(ground_truth), k))])
    return dcg / idcg if idcg > 0 else 0
```

**Hit Rate@K:**
```python
def hit_rate_at_k(predictions, ground_truth, k):
    return 1.0 if any(item in ground_truth for item in predictions[:k]) else 0.0
```

## 5. Data Examples

### Beauty Dataset Session Example:
```json
{
  "session_id": 123,
  "items": [
    {"id": 456, "name": "Moisturizing Face Cream", "time": "2023-01-01"},
    {"id": 789, "name": "Eye Makeup Remover", "time": "2023-01-02"},
    {"id": 012, "name": "Lip Gloss", "time": "2023-01-03"},
    {"id": 345, "name": "Foundation Brush", "time": "2023-01-04"}
  ]
}
```
**Training:** Items 1-3 as prompt, Item 4 as ground truth

### Delivery Hero Session Example:
```json
{
  "session_id": 456,
  "items": [
    {"id": 111, "name": "Milk 1L", "time": "2023-02-01"},
    {"id": 222, "name": "Bread", "time": "2023-02-01"},
    {"id": 333, "name": "Eggs 12pk", "time": "2023-02-01"}
  ]
}
```
**Note:** Shorter sessions, same-day purchases typical

## 6. Key Implementation Details

### Embedding Dimensions:
- OpenAI: 1536 dimensions
- Google PaLM: 768 dimensions
- Reduced to: 128 or 512 via PCA/LDA

### Fine-tuning Parameters:
- Training epochs: Until validation loss converges
- Temperature: 0.0-2.0 (task-dependent)
- Validation set: 500 samples (OpenAI), 250 (PaLM)

### Hybrid Models:
- **LLMSeqSim & Sequential:** Combines semantic similarity with neural models
- **LLMSeqSim & Popularity:** Adds popularity threshold to semantic recommendations

### Performance Optimizations:
- Batch processing for embeddings
- Caching of computed similarities
- Multiprocessing for evaluation
- Early stopping in hyperparameter search

## 7. Detailed Analysis of Follow-up Questions

### 7.1 Next-Item Prediction Without Candidate Pools

In Section 2.1, **there is NO explicit candidate pool** for next-item prediction. The neural models predict over the **entire item vocabulary** using softmax.

**Code Evidence from `previousWorks/LLM-Sequential-Recommendation/main/utils/neural_utils/custom_layers/projection_head.py:20-22`:**
```python
# Neural models predict over full vocabulary
logits = tf.matmul(sequence_output, self.embedding_table, transpose_b=True)
return logits  # Shape: [batch_size, vocab_size]
```

**SASRec implementation from `previousWorks/LLM-Sequential-Recommendation/main/transformer/sasrec/sasrec_model.py:80-85`:**
```python
def call(self, x, training=False):
    seq_output = self.seq_layer(x, training=training)
    seq_output = self.layer_norm(seq_output)
    # Predict next item from ENTIRE vocabulary
    test_item_emb = self.projection_head(seq_output)
    return test_item_emb[:, -1, :]  # Last position prediction
```

The models use **leave-one-out evaluation** where each session's last item becomes the ground truth target across ALL possible items.

### 7.2 Hallucination Mapping via Embedding Similarity

The LLM generates item names that may not exist in the catalog. These "hallucinations" are mapped back using **dot product similarity** between embeddings.

**Key implementation from `previousWorks/LLM-Sequential-Recommendation/main/llm_based/prompt_model/genitem/predict_gpt.ipynb:285-300`:**
```python
# Get embedding for hallucinated item
item_embedding = unmappable_items_embeddings[item_name]
item_embedding = np.array([item_embedding], dtype=np.float64)

# Compute similarity with ALL catalog items
predictions = (product_index_to_embedding @ item_embedding.T).T[0]

# Find most similar items
top_k_item_ids_indices = predictions.argsort()[::-1][:count + TOP_K]
top_k_item_ids = [
    product_index_to_id[item_index] for item_index in top_k_item_ids_indices
]
```

This maps the generated text to the **closest real catalog item** via cosine similarity in OpenAI's embedding space.

### 7.3 Top-K Selection and Clustering Importance

**Top-K (K=200) clustering** is crucial for the **classification variant** (LLM2Sequential approach). 

**Clustering reduces the massive vocabulary** (12K+ items) into 200 manageable classes, making classification feasible for LLMs.

**Code from `previousWorks/LLM-Sequential-Recommendation/beauty/attach_embeddings.ipynb:7-15`:**
```python
# Each item gets assigned to one of K=200 clusters
embeddings = pd.read_csv(f"product_embeddings_{embedding_source_name}.csv.gz")
# 'class' column contains cluster assignment (0-199)
print(embeddings['class'].nunique())  # Should be 200
```

This **transforms** the next-item prediction from a **12K-way classification** to a **200-way classification**, which is more suitable for LLM fine-tuning.

### 7.4 Embedding Generation Process

Embeddings are generated using **only the item name** - no additional descriptions.

**Implementation from `previousWorks/LLM-Sequential-Recommendation/main/llm_based/embedding_utils/openai_utils.py:83-87`:**
```python
def set_embeddings_from_df(df: pd.DataFrame) -> pd.DataFrame:
    names = list(df["name"])  # ONLY item names used
    embeddings = openai_get_embeddings(names, EMBEDDING_ENGINE)
    # text-embedding-ada-002 produces 1536-dimensional vectors
    df["ada_embedding"] = embeddings
```

**Example from `previousWorks/LLM-Sequential-Recommendation/main/llm_based/embedding_utils/create_embeddings.ipynb:40-45`:**
```python
# Raw item names used directly:
# "WAWO 15 Color Professionl Makeup Eyeshadow Cam..."
# "Xtreme Brite Brightening Gel 1oz."
# "Prada Candy By Prada Eau De Parfum Spray 1.7 O..."
```

The embeddings are **1536-dimensional** vectors from OpenAI's `text-embedding-ada-002` model.

### 7.5 Fine-tuning Task Breakdown

**Input/Output Structure from `previousWorks/LLM-Sequential-Recommendation/main/llm_based/prompt_model/genitem/generate_finetune_prompts.ipynb:110-130`:**

```python
SYSTEM_PROMPT = """Provide a unique item recommendation that is complementary to the user's item list. 
Ensure the recommendation is from items included in the data you are fine-tuned with. List only the item name."""

USER_PROMPT_TEMPLATE = """The user's item list:\n{user_item_list}"""

ASSISTANT_PROMPT_TEMPLATE = """{ground_truth}"""

# Example training case:
{
  "messages": [
    {"role": "system", "content": "Provide a unique item recommendation..."},
    {"role": "user", "content": "The user's item list:\nItem A\nItem B\nItem C"},
    {"role": "assistant", "content": "Item D"}
  ]
}
```

**Training Data Preparation from `previousWorks/LLM-Sequential-Recommendation/main/llm_based/prompt_model/genitem/generate_finetune_prompts.ipynb` lines 135-145:**
```python
# Split sessions: first n-1 items = prompt, last item = ground truth
train_prompts[session_id] = items[:-1]  # Input sequence
train_ground_truths[session_id] = items[-1:]  # Target item

# Convert to item names for LLM training
textified_train_prompts[session] = [
    product_id_to_name[product_id] for product_id in rec_items
]
```

The fine-tuning uses **standard supervised learning** where the model learns to predict the next item name given a sequence of previous item names, formatted as conversational prompts.

## 8. Summary

The implementation demonstrates three orthogonal approaches for leveraging LLMs in sequential recommendation:

1. **LLMSeqSim:** Pure semantic similarity using LLM embeddings
2. **LLMSeqPrompt:** Fine-tuned LLMs for direct recommendation generation
3. **LLM2Sequential:** Enhanced traditional models with LLM embeddings

Key findings show that LLM embeddings significantly boost performance, especially for datasets with rich semantic information. The classification approach provides a middle ground between free generation and constrained ranking, reducing hallucinations while maintaining flexibility.