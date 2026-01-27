import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import os

# ==========================================
# 1. Configuration & Data Preparation
# ==========================================
class Config:
    embedding_dim = 64
    n_layers = 3        # Number of graph propagation layers (standard is 3)
    lr = 0.001
    epochs = 50        # Keep it small for demo, use 100+ for real data
    batch_size = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mock Data Generation (Replace this with your real csv loading)
# Format: List of [user_id, item_id] interactions
def generate_mock_data(num_users=1000, num_items=5000, num_interactions=20000):
    print("Generating mock dataset...")
    users = np.random.randint(0, num_users, num_interactions)
    items = np.random.randint(0, num_items, num_interactions)
    # Remove duplicates
    data = np.unique(np.stack([users, items], axis=1), axis=0)
    return data, num_users, num_items

# ==========================================
# 2. LightGCN Model Architecture
# ==========================================
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, config, graph):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.final_emb = None # Store for inference
        
        # 1. Initialize Embeddings (User + Item combined)
        self.embedding = nn.Embedding(num_users + num_items, config.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        # 2. Graph Storage (Sparse Adjacency Matrix)
        self.graph = graph 
        self.n_layers = config.n_layers

    def forward(self):
        # LightGCN Propagation
        all_embs = [self.embedding.weight]
        emb = self.embedding.weight
        
        for _ in range(self.n_layers):
            # The Magic: Multiply Graph by Embeddings (Message Passing)
            emb = torch.sparse.mm(self.graph, emb) 
            all_embs.append(emb)
        
        # Average all layers (The "Light" part)
        final_embs = torch.stack(all_embs, dim=1)
        final_embs = torch.mean(final_embs, dim=1)
        
        return final_embs

    def get_loss(self, users, pos_items, neg_items):
        # Retrieve current embeddings
        final_embs = self.forward()
        
        # Split into User and Item parts
        user_embs = final_embs[users]
        pos_embs = final_embs[self.num_users + pos_items] # Offset item IDs
        neg_embs = final_embs[self.num_users + neg_items]
        
        # Calculate Scores (Dot Product)
        pos_scores = torch.sum(user_embs * pos_embs, dim=1)
        neg_scores = torch.sum(user_embs * neg_embs, dim=1)
        
        # BPR Loss (Bayesian Personalized Ranking)
        # We want pos_score > neg_score
        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        
        # L2 Regularization (Prevent overfitting)
        reg_loss = (1/2) * (self.embedding.weight.norm(2).pow(2)) / float(len(users))
        
        return loss + (1e-4 * reg_loss)

# ==========================================
# 3. Graph Building Helper
# ==========================================
def build_sparse_graph(train_data, num_users, num_items, device):
    print("Building graph...")
    # Create Adjacency Matrix: 
    # [  0   , R_ui ]
    # [ R_ui.T,  0   ]
    
    users = train_data[:, 0]
    items = train_data[:, 1]
    
    # Coordinate format for sparse matrix
    # Row: User -> Item,  Col: Item -> User
    row = np.concatenate([users, items + num_users])
    col = np.concatenate([items + num_users, users])
    data = np.ones(len(row))
    
    # Create Scipy Sparse Matrix
    adj_mat = sp.coo_matrix((data, (row, col)), shape=(num_users + num_items, num_users + num_items))
    
    # Normalize Graph (D^-1/2 * A * D^-1/2) - Critical for GCN stability
    rowsum = np.array(adj_mat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
    
    # Convert to PyTorch Sparse Tensor
    norm_adj = norm_adj.tocoo()
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape).to(device)

# ==========================================
# 4. Training Loop
# ==========================================
def train():
    # Load Data
    data, num_users, num_items = generate_mock_data()
    train_data = data # In real life, split train/test
    
    config = Config()
    graph = build_sparse_graph(train_data, num_users, num_items, config.device)
    
    model = LightGCN(num_users, num_items, config, graph).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    print(f"Start Training on {config.device}...")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        # Simple Batch Loader
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        
        for i in range(0, len(train_data), config.batch_size):
            batch_idx = indices[i:i + config.batch_size]
            batch_users = train_data[batch_idx, 0]
            batch_pos = train_data[batch_idx, 1]
            
            # Negative Sampling (Randomly pick items users haven't seen)
            # In production, ensure these are true negatives
            batch_neg = np.random.randint(0, num_items, size=len(batch_idx))
            
            # Tensor conversions
            u = torch.tensor(batch_users).to(config.device)
            pos = torch.tensor(batch_pos).to(config.device)
            neg = torch.tensor(batch_neg).to(config.device)
            
            optimizer.zero_grad()
            loss = model.get_loss(u, pos, neg)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    # ==========================================
    # 5. Saving for Agent Consumption
    # ==========================================
    print("Saving embeddings for Agent...")
    model.eval()
    with torch.no_grad():
        final_embs = model.forward().cpu().numpy()
    
    user_embs = final_embs[:num_users]
    item_embs = final_embs[num_users:]
    
    # Save to disk
    np.save('user_embeddings.npy', user_embs)
    np.save('item_embeddings.npy', item_embs)
    
    # Save Biases (Optional: LightGCN doesn't use biases by default, 
    # but you can calculate item popularity here for the Critic)
    item_pop = np.bincount(train_data[:, 1], minlength=num_items)
    np.save('item_popularity.npy', item_pop)
    
    print("Done! Files 'user_embeddings.npy' and 'item_embeddings.npy' are ready.")

if __name__ == "__main__":
    train()


# =============================================================================
# LightGCN EMBEDDING MANAGER (for Enhanced Rerank Evaluation)
# =============================================================================

import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set


@dataclass
class LightGCNConfig:
    """Configuration for LightGCN training."""
    embedding_dim: int = 64
    n_layers: int = 3
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 1024
    reg_weight: float = 1e-4


class LightGCNEmbeddingManager:
    """
    Manages LightGCN embeddings for user-cuisine collaborative filtering.

    Features:
    - Trains on user-cuisine interaction graph
    - Caches embeddings to disk for fast reloading
    - Provides similarity computation methods

    Cache location: ~/.cache/agentic_recommender/lightgcn/{dataset_name}_embeddings.pkl
    """

    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "lightgcn"

    def __init__(self, config: LightGCNConfig = None):
        self.config = config or LightGCNConfig()
        self.user_embeddings: Optional[np.ndarray] = None
        self.cuisine_embeddings: Optional[np.ndarray] = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.cuisine_to_idx: Dict[str, int] = {}
        self.idx_to_cuisine: Dict[int, str] = {}
        self.cache_key: Optional[str] = None
        self._fitted = False

    def _compute_cache_key(self, interactions: List[Tuple[str, str]]) -> str:
        """Compute MD5 hash of interactions for cache validation."""
        # Sort for consistency
        sorted_interactions = sorted(interactions)
        data_str = str(sorted_interactions)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_cache_path(self, dataset_name: str, method: str = "full", prediction_target: str = "cuisine") -> Path:
        """
        Get cache file path for a dataset, method, and prediction target.

        Args:
            dataset_name: Name for cache file (e.g., "data_se")
            method: Evaluation method ("method1", "method2", or "full")
            prediction_target: "cuisine", "vendor_cuisine", "vendor", or "product"

        Returns:
            Path to cache file

        Cache naming:
            - {dataset}_{method}_{target}_lightgcn.pkl (with method and target)
            - {dataset}_{target}_lightgcn.pkl (without method, with target)
            - {dataset}_embeddings.pkl (legacy, cuisine only, no method)
        """
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Build cache filename with prediction_target for non-default targets
        if method and method != "full":
            if prediction_target and prediction_target != "cuisine":
                return self.CACHE_DIR / f"{dataset_name}_{method}_{prediction_target}_lightgcn.pkl"
            return self.CACHE_DIR / f"{dataset_name}_{method}_lightgcn.pkl"
        else:
            if prediction_target and prediction_target != "cuisine":
                return self.CACHE_DIR / f"{dataset_name}_{prediction_target}_lightgcn.pkl"
            return self.CACHE_DIR / f"{dataset_name}_embeddings.pkl"

    def load_or_train(
        self,
        dataset_name: str,
        interactions: List[Tuple[str, str]],
        method: str = "full",
        prediction_target: str = "cuisine",
        force_retrain: bool = False,
        verbose: bool = True
    ) -> 'LightGCNEmbeddingManager':
        """
        Load embeddings from cache or train new model.

        Args:
            dataset_name: Name for cache file (e.g., "data_se")
            interactions: List of (user_id, item) tuples
            method: Evaluation method for cache naming ("method1", "method2", or "full")
            prediction_target: "cuisine", "vendor_cuisine", "vendor", or "product"
            force_retrain: Force retraining even if cache exists
            verbose: Print progress

        Returns:
            self (for chaining)
        """
        cache_path = self._get_cache_path(dataset_name, method, prediction_target)
        new_cache_key = self._compute_cache_key(interactions)

        # Try to load from cache
        if not force_retrain and cache_path.exists():
            if verbose:
                print(f"[LightGCNManager] Loading/training LightGCN with {method} cache")
                print(f"[LightGCNManager] Found cache at {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                if cached_data.get('cache_key') == new_cache_key:
                    if verbose:
                        print("[LightGCNManager] Cache key matches, loading embeddings...")
                    self._load_from_cache(cached_data)
                    return self
                else:
                    if verbose:
                        print("[LightGCNManager] Cache key mismatch, retraining...")
            except Exception as e:
                if verbose:
                    print(f"[LightGCNManager] Failed to load cache: {e}")

        # Train new model
        if verbose:
            print(f"[LightGCNManager] Training new model on {len(interactions)} interactions...")

        self._train(interactions, verbose)
        self.cache_key = new_cache_key

        # Save to cache
        self._save_to_cache(cache_path, verbose)

        return self

    def _train(self, interactions: List[Tuple[str, str]], verbose: bool = True):
        """Train LightGCN on user-cuisine interactions."""
        # Build ID mappings
        users = sorted(set(uid for uid, _ in interactions))
        cuisines = sorted(set(cuisine for _, cuisine in interactions))

        self.user_to_idx = {uid: idx for idx, uid in enumerate(users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.cuisine_to_idx = {c: idx for idx, c in enumerate(cuisines)}
        self.idx_to_cuisine = {idx: c for c, idx in self.cuisine_to_idx.items()}

        num_users = len(users)
        num_cuisines = len(cuisines)

        if verbose:
            print(f"[LightGCNManager] Users: {num_users}, Cuisines: {num_cuisines}")

        # Convert interactions to numpy array with indices
        train_data = np.array([
            [self.user_to_idx[uid], self.cuisine_to_idx[cuisine]]
            for uid, cuisine in interactions
        ])
        train_data = np.unique(train_data, axis=0)  # Remove duplicates

        if verbose:
            print(f"[LightGCNManager] Unique interactions: {len(train_data)}")

        # Build graph
        device = self.config.embedding_dim  # Just use CPU for simplicity
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        graph = build_sparse_graph(train_data, num_users, num_cuisines, device)

        # Create and train model
        model = LightGCN(num_users, num_cuisines, self._to_config_obj(), graph).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        if verbose:
            print(f"[LightGCNManager] Training on {device}...")

        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0

            indices = np.arange(len(train_data))
            np.random.shuffle(indices)

            for i in range(0, len(train_data), self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                batch_users = train_data[batch_idx, 0]
                batch_pos = train_data[batch_idx, 1]
                batch_neg = np.random.randint(0, num_cuisines, size=len(batch_idx))

                u = torch.tensor(batch_users).to(device)
                pos = torch.tensor(batch_pos).to(device)
                neg = torch.tensor(batch_neg).to(device)

                optimizer.zero_grad()
                loss = model.get_loss(u, pos, neg)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch % 10 == 0 or epoch == self.config.epochs - 1):
                print(f"[LightGCNManager] Epoch {epoch}: Loss = {total_loss:.4f}")

        # Extract embeddings
        model.eval()
        with torch.no_grad():
            final_embs = model.forward().cpu().numpy()

        self.user_embeddings = final_embs[:num_users]
        self.cuisine_embeddings = final_embs[num_users:]
        self._fitted = True

        if verbose:
            print(f"[LightGCNManager] Training complete!")
            print(f"[LightGCNManager] User embedding shape: {self.user_embeddings.shape}")
            print(f"[LightGCNManager] Cuisine embedding shape: {self.cuisine_embeddings.shape}")

    def _to_config_obj(self):
        """Convert dataclass to Config-like object for LightGCN."""
        class ConfigObj:
            pass
        cfg = ConfigObj()
        cfg.embedding_dim = self.config.embedding_dim
        cfg.n_layers = self.config.n_layers
        cfg.lr = self.config.learning_rate
        cfg.epochs = self.config.epochs
        cfg.batch_size = self.config.batch_size
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cfg

    def _load_from_cache(self, cached_data: Dict):
        """Load embeddings and mappings from cache dict."""
        self.user_embeddings = cached_data['user_embeddings']
        self.cuisine_embeddings = cached_data['cuisine_embeddings']
        self.user_to_idx = cached_data['user_to_idx']
        self.idx_to_user = cached_data['idx_to_user']
        self.cuisine_to_idx = cached_data['cuisine_to_idx']
        self.idx_to_cuisine = cached_data['idx_to_cuisine']
        self.cache_key = cached_data['cache_key']
        self._fitted = True

    def _save_to_cache(self, cache_path: Path, verbose: bool = True):
        """Save embeddings and mappings to cache."""
        cached_data = {
            'user_embeddings': self.user_embeddings,
            'cuisine_embeddings': self.cuisine_embeddings,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'cuisine_to_idx': self.cuisine_to_idx,
            'idx_to_cuisine': self.idx_to_cuisine,
            'cache_key': self.cache_key,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        if verbose:
            print(f"[LightGCNManager] Saved embeddings to {cache_path}")

    def get_user_cuisine_similarity(self, user_id: str, cuisine: str) -> float:
        """
        Compute similarity between a user and a cuisine using dot product.

        Args:
            user_id: User identifier
            cuisine: Cuisine name

        Returns:
            Similarity score (dot product of embeddings)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call load_or_train() first.")

        user_idx = self.user_to_idx.get(user_id)
        cuisine_idx = self.cuisine_to_idx.get(cuisine)

        if user_idx is None or cuisine_idx is None:
            return 0.0

        user_emb = self.user_embeddings[user_idx]
        cuisine_emb = self.cuisine_embeddings[cuisine_idx]

        return float(np.dot(user_emb, cuisine_emb))

    def get_user_cuisines_similarities(
        self,
        user_id: str,
        cuisines: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Get similarities between a user and multiple cuisines.

        Args:
            user_id: User identifier
            cuisines: List of cuisine names

        Returns:
            List of (cuisine, similarity) tuples, sorted by similarity descending
        """
        similarities = []
        for cuisine in cuisines:
            sim = self.get_user_cuisine_similarity(user_id, cuisine)
            similarities.append((cuisine, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities

    def rerank_by_similarity(
        self,
        user_id: str,
        cuisines: List[str]
    ) -> List[str]:
        """
        Rerank cuisines by user-cuisine similarity.

        Args:
            user_id: User identifier
            cuisines: List of cuisine names to rerank

        Returns:
            Cuisines sorted by similarity (highest first)
        """
        similarities = self.get_user_cuisines_similarities(user_id, cuisines)
        return [cuisine for cuisine, _ in similarities]

    def get_top_cuisines_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        exclude: Set[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k cuisines for a user based on embedding similarity.

        Args:
            user_id: User identifier
            top_k: Number of cuisines to return
            exclude: Cuisines to exclude from results

        Returns:
            List of (cuisine, similarity) tuples
        """
        exclude = exclude or set()
        all_cuisines = [c for c in self.cuisine_to_idx.keys() if c not in exclude]
        similarities = self.get_user_cuisines_similarities(user_id, all_cuisines)
        return similarities[:top_k]

    def get_stats(self) -> Dict:
        """Get statistics about the embedding manager."""
        return {
            'fitted': self._fitted,
            'num_users': len(self.user_to_idx),
            'num_cuisines': len(self.cuisine_to_idx),
            'embedding_dim': self.config.embedding_dim,
            'n_layers': self.config.n_layers,
            'cache_key': self.cache_key,
        }


# =============================================================================
# HELPER FUNCTIONS FOR DATA PREPARATION
# =============================================================================

def filter_interactions_leave_last_out(
    orders_df,  # pd.DataFrame
    prediction_target: str = "cuisine",
) -> List[Tuple[str, str]]:
    """
    Get (user_id, item) interactions EXCLUDING the last order per user.

    Used for Method 1 (PureTrainingData) to prevent data leakage.
    The last order per user is held out for testing.

    Args:
        orders_df: DataFrame with order data (must have customer_id, order_id, and target column)
        prediction_target: "cuisine", "vendor", "product", or "vendor_cuisine"

    Returns:
        List of (user_id, item) tuples from N-1 orders per user.
    """
    from .item_algorithm import VendorCuisineItemAlgorithm

    # Map prediction target to column (None = special handling)
    target_column = {
        'cuisine': 'cuisine',
        'vendor': 'vendor_id',
        'product': 'product_id',
        'vendor_cuisine': None,  # Special handling below
    }.get(prediction_target, 'cuisine')

    # For vendor_cuisine, create the algorithm instance
    vendor_cuisine_algo = VendorCuisineItemAlgorithm() if prediction_target == 'vendor_cuisine' else None

    interactions = []

    # Group by customer
    for customer_id, group in orders_df.groupby('customer_id'):
        # Sort by time
        if 'day_num' in group.columns:
            sorted_group = group.sort_values(['day_num', 'hour'])
        else:
            sorted_group = group

        # Get unique orders
        unique_orders = sorted_group.drop_duplicates('order_id')

        # Need at least 2 orders to have N-1
        if len(unique_orders) < 2:
            continue

        # Exclude last order (last row after sorting)
        excluded_last = unique_orders.iloc[:-1]

        # Get all items from the remaining orders
        remaining_order_ids = set(excluded_last['order_id'].unique())

        # Get all rows (items) from non-last orders
        training_rows = sorted_group[sorted_group['order_id'].isin(remaining_order_ids)]

        for _, row in training_rows.iterrows():
            if prediction_target == 'vendor_cuisine':
                # Create composite item: vendor_id||cuisine
                item = vendor_cuisine_algo.extract_item_from_row(row)
            else:
                item = str(row.get(target_column, 'unknown'))
            interactions.append((str(customer_id), item))

    return interactions


def get_all_interactions(
    orders_df,  # pd.DataFrame
    prediction_target: str = "cuisine",
) -> List[Tuple[str, str]]:
    """
    Get ALL (user_id, item) interactions from orders.

    Used for Method 2 (FullHistoryTest) where all training data is used.

    Args:
        orders_df: DataFrame with order data
        prediction_target: "cuisine", "vendor", "product", or "vendor_cuisine"

    Returns:
        List of (user_id, item) tuples.
    """
    from .item_algorithm import VendorCuisineItemAlgorithm

    # Map prediction target to column (None = special handling)
    target_column = {
        'cuisine': 'cuisine',
        'vendor': 'vendor_id',
        'product': 'product_id',
        'vendor_cuisine': None,  # Special handling below
    }.get(prediction_target, 'cuisine')

    # For vendor_cuisine, create the algorithm instance
    vendor_cuisine_algo = VendorCuisineItemAlgorithm() if prediction_target == 'vendor_cuisine' else None

    interactions = []

    for _, row in orders_df.iterrows():
        customer_id = str(row['customer_id'])
        if prediction_target == 'vendor_cuisine':
            # Create composite item: vendor_id||cuisine
            item = vendor_cuisine_algo.extract_item_from_row(row)
        else:
            item = str(row.get(target_column, 'unknown'))
        interactions.append((customer_id, item))

    return interactions
