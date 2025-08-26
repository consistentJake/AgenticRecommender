"""
Delivery Hero dataset processing for food delivery data.
"""

import pandas as pd
import random
from pathlib import Path
from typing import Dict, Any, List
from .base_dataset import SequentialDataset


class DeliveryHeroDataset(SequentialDataset):
    """
    Delivery Hero dataset processor for food delivery data.
    
    Data format:
    - Orders: customer_id, geohash, order_id, vendor_id, product_id, day_of_week, order_time, order_day
    - Products: vendor_id, product_id, name, unit_price  
    - Vendors: vendor_id, chain_id, geohash, primary_cuisine
    
    Reference: LLM_Sequential_Recommendation_Analysis.md:76-93
    """
    
    def __init__(self, city: str = "sg", data_path: str = None, products_path: str = None, **kwargs):
        # Set default paths to the actual data location
        self.city = city
        if data_path is None:
            data_path = f"/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/data/inputs/delivery_hero/data_{city}/orders_{city}.txt"
        if products_path is None:
            products_path = f"/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/data/inputs/delivery_hero/data_{city}/products_{city}.txt"
        
        self.products_path = products_path
        super().__init__(data_path, **kwargs)
        
        # DH-specific settings - no p-core filtering for real-world sparsity
        self.min_interactions = 1  # Keep real-world sparsity
        self.min_session_length = 5  # Customers with at least 5 orders (like Beauty 5-core)
    
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load Delivery Hero orders data.
        
        Expected format: customer_id, geohash, order_id, vendor_id, product_id, day_of_week, order_time, order_day
        """
        try:
            print(f"📊 Loading Delivery Hero {self.city.upper()} dataset...")
            df = pd.read_csv(self.data_path)
            
            print(f"Raw data loaded: {len(df):,} order items")
            
            # Convert DH format to our standard format
            df_clean = df.rename(columns={
                'customer_id': 'user_id',
                'product_id': 'item_id',
                'order_id': 'session_id'  # Use order_id as session
            })
            
            # Create a timestamp from order_day and order_time
            # For now, use order_day as a simple timestamp
            df_clean['timestamp'] = df_clean['order_day'].str.extract(r'(\d+)').astype(int)
            
            # Select only the columns we need
            df_clean = df_clean[['user_id', 'item_id', 'timestamp']].copy()
            
            # Remove any rows with missing values
            df_clean = df_clean.dropna()
            
            print(f"After cleaning: {len(df_clean):,} interactions")
            
            # No need to filter by order size - we'll create sequential sessions per customer
            print(f"Ready for session creation: {len(df_clean):,} interactions")
            return df_clean
            
        except FileNotFoundError:
            print(f"⚠️ DH data file not found: {self.data_path}")
            print("Creating synthetic grocery data...")
            return pd.DataFrame(self._create_synthetic_grocery_data())
        except Exception as e:
            print(f"⚠️ Error loading DH data: {e}")
            print("Creating synthetic grocery data...")
            return pd.DataFrame(self._create_synthetic_grocery_data())
    
    def _create_sessions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DH data to sequential sessions.
        Following Beauty dataset approach: each customer becomes a session 
        with their orders over time creating the sequence.
        """
        sessions = []
        
        # Group by customer (like Beauty groups by reviewer)
        grouped = data.groupby('user_id')
        
        for user_id, user_data in grouped:
            # Sort by timestamp to create chronological sequence
            user_data = user_data.sort_values('timestamp')
            
            # Create session from all items ordered by this customer over time
            items = user_data['item_id'].tolist()
            timestamps = user_data['timestamp'].tolist()
            
            if len(items) >= self.min_session_length:
                session = {
                    'session_id': len(sessions),
                    'user_id': user_id,
                    'items': items,
                    'timestamps': timestamps,
                    'length': len(items)
                }
                sessions.append(session)
                
                # Track user items for negative sampling
                self.user_items[user_id] = set(items)
                self.all_items.update(items)
        
        return sessions
    
    def _process_metadata(self) -> Dict[str, str]:
        """Process food product names from products file"""
        item_names = {}
        
        if self.products_path and Path(self.products_path).exists():
            try:
                print("🍔 Loading product metadata...")
                products_df = pd.read_csv(self.products_path)
                
                # Create item name mapping
                for _, row in products_df.iterrows():
                    product_id = row['product_id']
                    name = row.get('name', f"Food_Item_{product_id}")
                    
                    # Clean product name
                    if pd.notna(name):
                        name = str(name).strip()
                        if len(name) > 100:
                            name = name[:97] + "..."
                        item_names[product_id] = name
                    
                print(f"Loaded {len(item_names):,} product names")
                    
            except Exception as e:
                print(f"⚠️ Error processing products metadata: {e}")
        
        # For items without metadata, create generic names
        for item_id in self.all_items:
            if item_id not in item_names:
                item_names[item_id] = f"Food_Item_{item_id}"
        
        return item_names
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic data for testing (alias for grocery data)"""
        return self._create_synthetic_grocery_data()
    
    def _create_synthetic_grocery_data(self) -> List[Dict[str, Any]]:
        """Create synthetic grocery data"""
        print("🛒 Creating synthetic grocery dataset...")
        
        # Grocery categories
        categories = [
            "Dairy", "Meat", "Vegetables", "Fruits", "Bread", 
            "Snacks", "Beverages", "Cleaning", "Personal_Care", "Frozen"
        ]
        
        # Generate items
        items = []
        for cat in categories:
            for i in range(50):  # 50 items per category
                items.append({
                    'id': f"{cat}_{i:03d}",
                    'name': f"{cat} Product {i+1}",
                    'category': cat
                })
        
        # Generate sessions (shorter than beauty)
        data = []
        for session_id in range(1000):  # 1000 sessions
            # Each session has 2-10 items (shorter than beauty)
            n_items = random.randint(2, 10)
            
            session_items = []
            # Grocery sessions tend to be more coherent
            if random.random() < 0.6:  # 60% chance for coherent session
                # Pick 1-2 main categories
                main_cats = random.sample(categories, random.randint(1, 2))
                for _ in range(n_items):
                    if random.random() < 0.8:  # 80% from main categories
                        cat = random.choice(main_cats)
                        cat_items = [item for item in items if item['category'] == cat]
                    else:
                        cat_items = items
                    
                    item = random.choice(cat_items)
                    if item['id'] not in [si['item_id'] for si in session_items]:
                        session_items.append({
                            'user_id': f"DH_Session_{session_id}",
                            'item_id': item['id'],
                            'timestamp': 1600000000 + len(session_items) * 3600,  # 1 hour apart
                        })
            
            else:  # Random session
                random_items = random.sample(items, min(n_items, len(items)))
                for i, item in enumerate(random_items):
                    session_items.append({
                        'user_id': f"DH_Session_{session_id}",
                        'item_id': item['id'],
                        'timestamp': 1600000000 + i * 3600,
                    })
            
            data.extend(session_items)
        
        # Store item names
        for item in items:
            self.item_to_name[item['id']] = item['name']
        
        return data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        info = {
            'name': f'Delivery Hero Dataset ({self.city.upper()})',
            'source': f'Food delivery data from {self.city.upper()}',
            'filtering': 'Minimal (2+ items per order)',
            'characteristics': {
                'short_sessions': True,
                'food_delivery': True,
                'sparse_data': True,
                'real_world_setting': True
            }
        }
        info.update(self.stats)
        return info