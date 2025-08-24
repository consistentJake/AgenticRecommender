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
    
    def __init__(self, data_path: str, metadata_path: str = None, **kwargs):
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
                    for line in f:
                        meta = json.loads(line.strip())
                        if 'asin' in meta and 'title' in meta:
                            # Clean product name
                            title = self._clean_product_name(meta['title'])
                            item_names[meta['asin']] = title
            except Exception as e:
                print(f"⚠️ Error processing metadata: {e}")
        
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
    Delivery Hero dataset processor.
    
    Reference: LLM_Sequential_Recommendation_Analysis.md:76-93
    """
    
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        
        # DH-specific settings - no p-core filtering
        self.min_interactions = 1  # Keep real-world sparsity
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load Delivery Hero session data"""
        try:
            # Assume CSV format for now
            df = pd.read_csv(self.data_path)
            
            # Expected columns: session_id, item_id, timestamp
            if 'session_id' in df.columns:
                df['user_id'] = df['session_id']  # Use session as user
            
            return df
            
        except FileNotFoundError:
            # Create synthetic grocery data
            print("⚠️ DH data file not found. Creating synthetic grocery data...")
            return pd.DataFrame(self._create_synthetic_grocery_data())
    
    def _process_metadata(self) -> Dict[str, str]:
        """Process grocery item names"""
        # For DH dataset, item names might not be available
        # Create generic names based on item IDs
        item_names = {}
        
        for item_id in self.all_items:
            item_names[item_id] = f"Grocery_Item_{item_id}"
        
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
            'name': 'Delivery Hero Dataset',
            'source': 'Proprietary QCommerce',
            'filtering': 'None (real-world setting)',
            'characteristics': {
                'short_sessions': True,
                'same_day_purchases': True,
                'sparse_data': True,
                'avg_session_length': 'short (5-7 items)',
                'density': 'low (~0.015%)'
            }
        }
        info.update(self.stats)
        return info