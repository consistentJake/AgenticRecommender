"""
H&M Beauty dataset processing.
Based on Amazon Beauty 5-core dataset from LLM-Sequential-Recommendation.
"""

import json
import pandas as pd
import re
import random
from pathlib import Path
from typing import Dict, Any, List
from .base_dataset import SequentialDataset


class BeautyDataset(SequentialDataset):
    """
    Beauty dataset processor following LLM-Sequential-Recommendation approach.
    
    Reference: 
    - LLM_Sequential_Recommendation_Analysis.md:35-60
    - previousWorks/LLM-Sequential-Recommendation/beauty/create_sessions.ipynb
    """
    
    def __init__(self, data_path: str = None, metadata_path: str = None, **kwargs):
        # Set default paths to the actual data location
        if data_path is None:
            data_path = "/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/data/inputs/reviews_Beauty.json"
        if metadata_path is None:
            metadata_path = "/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/data/inputs/meta_Beauty.json"
        
        self.metadata_path = metadata_path
        super().__init__(data_path, **kwargs)
        
        # Beauty-specific settings
        self.min_interactions = 5  # 5-core filtering
    
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw beauty review data.
        
        Expected format: reviews_Beauty.json with fields:
        - reviewerID: user identifier
        - asin: item identifier (ASIN)
        - unixReviewTime: timestamp
        - overall: rating (optional)
        """
        data = []
        
        try:
            with open(self.data_path, 'r') as f:
                for line in f:
                    review = json.loads(line.strip())
                    data.append({
                        'user_id': review['reviewerID'],
                        'item_id': review['asin'],
                        'timestamp': review['unixReviewTime'],
                        'rating': review.get('overall', 1.0)  # Implicit feedback = 1.0
                    })
        except FileNotFoundError:
            # For demo purposes, create synthetic data
            print("⚠️ Raw data file not found. Creating synthetic Beauty data...")
            data = self._create_synthetic_data()
        
        df = pd.DataFrame(data)
        
        # Apply 5-core filtering
        print("🔍 Applying 5-core filtering...")
        df = self._apply_5core_filtering(df)
        
        return df
    
    def _apply_5core_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply iterative 5-core filtering.
        Both users and items must have at least 5 interactions.
        
        Reference: LLM_Sequential_Recommendation_Analysis.md:46-53
        """
        prev_size = 0
        current_size = len(df)
        
        iteration = 0
        while prev_size != current_size:
            iteration += 1
            prev_size = current_size
            
            # Remove users with < 5 interactions
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            df = df[df['user_id'].isin(valid_users)]
            
            # Remove items with < 5 interactions
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_interactions].index
            df = df[df['item_id'].isin(valid_items)]
            
            current_size = len(df)
            print(f"  Iteration {iteration}: {current_size} interactions")
        
        print(f"✅ 5-core filtering completed after {iteration} iterations")
        return df
    
    def _process_metadata(self) -> Dict[str, str]:
        """
        Process beauty product metadata.
        
        Returns:
            Dictionary mapping item_id -> product_name
        """
        # If we already have item names from synthetic data, preserve them
        if hasattr(self, 'item_to_name') and self.item_to_name:
            return self.item_to_name
        
        item_names = {}
        
        if self.metadata_path and Path(self.metadata_path).exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            # The metadata file uses Python dict format, not JSON
                            # Use eval() carefully with restricted builtins
                            line = line.strip()
                            if line:
                                meta = eval(line, {"__builtins__": {}}, {})
                                if 'asin' in meta and 'title' in meta:
                                    # Clean product name
                                    title = self._clean_product_name(meta['title'])
                                    item_names[meta['asin']] = title
                        except Exception as line_e:
                            if line_num <= 5:  # Only show errors for first few lines
                                print(f"⚠️ Error parsing metadata line {line_num}: {line_e}")
                            continue
            except Exception as e:
                print(f"⚠️ Error processing metadata file: {e}")
        
        # For items without metadata, create generic names
        for item_id in self.all_items:
            if item_id not in item_names:
                item_names[item_id] = f"Beauty_Product_{item_id}"
        
        return item_names
    
    def _clean_product_name(self, title: str) -> str:
        """
        Clean product title by removing HTML entities and special characters.
        
        Reference: LLM_Sequential_Recommendation_Analysis.md:62-65
        """
        # Replace HTML entities
        title = title.replace('&amp;', 'and')
        title = title.replace('&lt;', '<')
        title = title.replace('&gt;', '>')
        title = title.replace('&quot;', '"')
        title = title.replace('&#39;', "'")
        
        # Remove excessive whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Truncate if too long
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic beauty data for testing"""
        print("🧪 Creating synthetic beauty dataset...")
        
        # Beauty product categories
        categories = [
            "Moisturizer", "Cleanser", "Serum", "Sunscreen", "Foundation",
            "Concealer", "Lipstick", "Mascara", "Eyeshadow", "Blush",
            "Shampoo", "Conditioner", "Hair Mask", "Perfume", "Lotion"
        ]
        
        brands = ["Brand_A", "Brand_B", "Brand_C", "Brand_D", "Brand_E"]
        
        # Generate items (fewer items for better overlap)
        items = []
        for i in range(150):  # 150 beauty products for better coverage
            category = random.choice(categories)
            brand = random.choice(brands)
            item_name = f"{brand} {category} {random.randint(1, 50)}ml"
            items.append({
                'id': f"B{i:06d}",
                'name': item_name,
                'category': category,
                'brand': brand
            })
        
        # Generate users and sessions
        data = []
        for user_id in range(200):  # 200 users for better 5-core coverage
            # Each user has 8-25 interactions
            n_interactions = random.randint(8, 25)
            
            # Users have preference for certain categories
            preferred_cats = random.sample(categories, random.randint(2, 5))
            
            user_items = []
            for _ in range(n_interactions):
                # 70% chance to pick from preferred categories
                if random.random() < 0.7 and preferred_cats:
                    cat = random.choice(preferred_cats)
                    cat_items = [item for item in items if item['category'] == cat]
                    item = random.choice(cat_items)
                else:
                    item = random.choice(items)
                
                if item['id'] not in [ui['item_id'] for ui in user_items]:
                    user_items.append({
                        'user_id': f"U{user_id:06d}",
                        'item_id': item['id'],
                        'timestamp': 1600000000 + len(user_items) * 86400,  # One day apart
                        'rating': random.uniform(3.0, 5.0)
                    })
            
            data.extend(user_items[-n_interactions:])  # Ensure exact count
        
        # Store item names
        for item in items:
            self.item_to_name[item['id']] = item['name']
        
        return data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        info = {
            'name': 'Beauty Dataset',
            'source': 'Amazon Beauty 5-core',
            'filtering': '5-core iterative',
            'characteristics': {
                'rich_semantic_info': True,
                'avg_session_length': 'medium (8-9 items)',
                'density': 'medium (~0.07%)'
            }
        }
        info.update(self.stats)
        return info


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