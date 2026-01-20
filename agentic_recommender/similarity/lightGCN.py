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
