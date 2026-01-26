Sure — I’ll write it in a clean, paper-ready **Markdown** style, suitable for reports, papers, or README docs. I’ll also make it rigorous enough for ML / recommender-system evaluation.

---

# Evaluation Metrics for Next-Basket Recommendation

In a next-basket recommendation task, each user has a sequence of orders.
Given the historical orders as input, the model predicts a ranked list of items for the next order.

Let:

* $R_K = [r_1, r_2, \dots, r_K]$ be the top-$K$ recommended items.
* $G$ be the ground-truth set of items in the last order (basket).
* $|G| = m$ be the basket size.
* $R_K \cap G$ be the set of correctly predicted items.

---

## 1. Hit Rate@K (Hit@K)

Hit Rate measures whether the model successfully predicts at least one item in the target basket.

$$
\text{Hit@K} =
\begin{cases}
1, & \text{if } |R_K \cap G| > 0 \\
0, & \text{otherwise}
\end{cases}
$$

This metric evaluates whether the model can “hit” any relevant item in the recommendation list.

---

## 2. Recall@K

Recall@K measures the fraction of ground-truth items that appear in the recommendation list.

$$
\text{Recall@K} = \frac{|R_K \cap G|}{|G|}
$$

This metric is particularly suitable for multi-item baskets, as it reflects how many true items are recovered by the model.

---

## 3. Precision@K

Precision@K measures the fraction of recommended items that are relevant.

$$
\text{Precision@K} = \frac{|R_K \cap G|}{K}
$$

This metric evaluates the quality of the recommendation list from the perspective of correctness.

---

## 4. NDCG@K (Normalized Discounted Cumulative Gain)

NDCG@K measures both the correctness and the ranking quality of the recommendations, giving higher weight to correctly predicted items appearing earlier in the ranked list.

### 4.1 DCG@K (Discounted Cumulative Gain)

We define binary relevance for each recommended item:

$$
\text{rel}(r_i) =
\begin{cases}
1, & \text{if } r_i \in G \\
0, & \text{otherwise}
\end{cases}
$$

Then:

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}(r_i)}{\log_2(i+1)}
$$

---

### 4.2 IDCG@K (Ideal DCG)

IDCG@K corresponds to the ideal ranking where all ground-truth items appear at the top of the recommendation list:

$$
\text{IDCG@K} = \sum_{i=1}^{\min(|G|, K)} \frac{1}{\log_2(i+1)}
$$

---

### 4.3 NDCG@K

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

NDCG@K ranges from 0 to 1, where 1 indicates a perfect ranking of the ground-truth items.

---

## 5. MRR (Mean Reciprocal Rank) for Multi-Item Targets

For basket prediction, we define MRR based on the rank of the first correctly predicted item:

$$
\text{MRR} = \frac{1}{\min \{ i \mid r_i \in G \}}
$$

If no relevant item appears in the top-$K$, MRR is set to 0.

---

## 6. Aggregation Across Users

Metrics are computed for each user (or test instance) and then averaged:

$$
\text{Metric} = \frac{1}{N} \sum_{u=1}^{N} \text{Metric}_u
$$

where $N$ is the number of test users.

To avoid bias toward users with many interactions, metrics can be averaged per user before global averaging.

---

## 7. Seen-Item Filtering

During evaluation, items that the user has already interacted with in historical orders are typically removed from the candidate set:

$$
R_K = \text{Top-}K \text{ items excluding previously seen items}
$$

This prevents trivial recommendations of previously purchased items and ensures a fair evaluation of future-item prediction.

---
