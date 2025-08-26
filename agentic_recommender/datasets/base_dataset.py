"""
Base dataset processing for sequential recommendation.
Based on LLM-Sequential-Recommendation implementation.
"""

import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import re


class SequentialDataset(ABC):
    """
    Base class for sequential recommendation datasets.
    
    Implements common functionality for:
    - Session-based data loading
    - Leave-one-out evaluation splits
    - Negative sampling
    - Candidate pool generation
    
    Reference: LLM_Sequential_Recommendation_Analysis.md:117-130
    """
    
    def __init__(self, data_path: str, n_neg_items: int = 99, min_session_length: int = 2):
        self.data_path = Path(data_path)
        self.n_neg_items = n_neg_items
        self.min_session_length = min_session_length
        
        # Will be populated by subclasses
        self.sessions = []
        self.item_to_name = {}
        self.name_to_item = {}
        self.all_items = set()
        self.user_items = {}  # user_id -> set of item_ids
        
        # Statistics
        self.stats = {}
    
    @abstractmethod
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw data from file. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _process_metadata(self) -> Dict[int, str]:
        """Process item metadata. Must be implemented by subclasses."""
        pass
    
    def process_data(self):
        """
        Main data processing pipeline.
        
        Steps:
        1. Load raw data
        2. Process metadata
        3. Create sessions
        4. Apply filtering
        5. Generate statistics
        """
        print("📊 Loading raw data...")
        raw_data = self._load_raw_data()
        
        print("🏷️ Processing metadata...")
        self.item_to_name = self._process_metadata()
        self.name_to_item = {v: k for k, v in self.item_to_name.items()}
        
        print("📝 Creating sessions...")
        self.sessions = self._create_sessions(raw_data)
        
        print("🔍 Applying filters...")
        self.sessions = self._apply_filters(self.sessions)
        
        print("📈 Generating statistics...")
        self._generate_statistics()
        
        print(f"✅ Dataset processed: {len(self.sessions)} sessions, {len(self.all_items)} items")
    
    def _create_sessions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert interaction data to sessions.
        
        Args:
            data: DataFrame with columns [user_id, item_id, timestamp, ...]
            
        Returns:
            List of session dictionaries
        """
        sessions = []
        
        # Group by user and sort by timestamp
        grouped = data.groupby('user_id')
        
        for user_id, user_data in grouped:
            # Sort by timestamp
            user_data = user_data.sort_values('timestamp')
            
            # Create session
            items = user_data['item_id'].tolist()
            
            if len(items) >= self.min_session_length:
                session = {
                    'session_id': len(sessions),
                    'user_id': user_id,
                    'items': items,
                    'timestamps': user_data['timestamp'].tolist(),
                    'length': len(items)
                }
                sessions.append(session)
                
                # Track user items for negative sampling
                self.user_items[user_id] = set(items)
                self.all_items.update(items)
        
        return sessions
    
    def _apply_filters(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply dataset-specific filters"""
        # Remove sessions that are too short
        filtered = [s for s in sessions if s['length'] >= self.min_session_length]
        
        print(f"Filtered {len(sessions) - len(filtered)} short sessions")
        return filtered
    
    def _generate_statistics(self):
        """Generate dataset statistics"""
        if not self.sessions:
            return
        
        lengths = [s['length'] for s in self.sessions]
        
        self.stats = {
            'num_sessions': len(self.sessions),
            'num_items': len(self.all_items),
            'num_users': len(self.user_items),
            'total_interactions': sum(lengths),
            'avg_session_length': np.mean(lengths),
            'min_session_length': min(lengths),
            'max_session_length': max(lengths),
            'density': sum(lengths) / (len(self.user_items) * len(self.all_items))
        }
    
    def prepare_to_predict(self, session: Dict[str, Any], n_withheld: int = 1) -> Tuple[List[int], int]:
        """
        Prepare session for prediction (leave-one-out).
        
        Returns:
            Tuple of (prediction_sequence, ground_truth_target)
        
        Reference: LLM_Sequential_Recommendation_Analysis.md:122-125
        """
        pred_sequence = session['items'][:-n_withheld] 
        target = session['items'][-n_withheld]
        return pred_sequence, target
    
    def extract_ground_truth(self, session: Dict[str, Any], n_withheld: int = 1) -> int:
        """
        Extract ground truth from session.
        
        Reference: LLM_Sequential_Recommendation_Analysis.md:127-130
        """
        return session['items'][-n_withheld]
    
    def negative_sample(self, user_id: Any, exclude_items: List[int]) -> List[int]:
        """
        Generate negative samples for a user.
        
        Args:
            user_id: User identifier
            exclude_items: Items to exclude (user's interaction history)
            
        Returns:
            List of negative item IDs
            
        Reference: LLM_Sequential_Recommendation_Analysis.md:519-533
        """
        exclude_set = set(exclude_items)
        negatives = []
        
        # Convert all_items to list for sampling
        all_items_list = list(self.all_items)
        
        while len(negatives) < self.n_neg_items:
            candidate = random.choice(all_items_list)
            if candidate not in exclude_set and candidate not in negatives:
                negatives.append(candidate)
        
        return negatives
    
    def create_candidate_pool(self, session: Dict[str, Any]) -> Tuple[List[int], int]:
        """
        Create candidate pool for evaluation.
        
        Args:
            session: Session data
            
        Returns:
            Tuple of (candidate_list, target_index)
            
        Reference: LLM_Sequential_Recommendation_Analysis.md:505-516
        """
        # Get prompt items and target
        prompt_items = self.prepare_to_predict(session)
        target_item = self.extract_ground_truth(session)
        
        # Generate negatives
        all_session_items = session['items']
        negatives = self.negative_sample(session['user_id'], all_session_items)
        
        # Create candidate pool: target + negatives
        candidates = [target_item] + negatives
        
        # Shuffle to avoid position bias
        random.shuffle(candidates)
        
        # Find target index after shuffle
        target_idx = candidates.index(target_item)
        
        return candidates, target_idx
    
    def get_item_name(self, item_id: int) -> str:
        """Get item name from ID"""
        return self.item_to_name.get(item_id, f"Unknown_Item_{item_id}")
    
    def get_item_id(self, item_name: str) -> Optional[int]:
        """Get item ID from name"""
        return self.name_to_item.get(item_name)
    
    def create_evaluation_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create train/validation/test splits.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if not self.sessions:
            raise ValueError("No sessions loaded. Call process_data() first.")
        
        # Shuffle sessions
        shuffled = self.sessions.copy()
        random.shuffle(shuffled)
        
        # Calculate split indices
        n_sessions = len(shuffled)
        train_end = int(n_sessions * train_ratio)
        val_end = int(n_sessions * (train_ratio + val_ratio))
        
        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }
        
        print(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return self.stats.copy()
    
    def test_data_integrity(self) -> bool:
        """
        Test data integrity and correctness.
        
        Returns:
            True if all tests pass
        """
        if not self.sessions:
            return False
        
        # Test 1: Check session lengths
        for session in self.sessions:
            if session['length'] < self.min_session_length:
                print(f"❌ Session {session['session_id']} too short: {session['length']}")
                return False
        
        # Test 2: Check negative sampling
        test_session = random.choice(self.sessions)
        negatives = self.negative_sample(test_session['user_id'], test_session['items'])
        
        if len(negatives) != self.n_neg_items:
            print(f"❌ Wrong number of negatives: {len(negatives)} != {self.n_neg_items}")
            return False
        
        # Check no overlap with user items
        user_items_set = set(test_session['items'])
        negative_set = set(negatives)
        if user_items_set & negative_set:
            print(f"❌ Negatives overlap with user items")
            return False
        
        # Test 3: Check candidate pool
        candidates, target_idx = self.create_candidate_pool(test_session)
        
        if len(candidates) != self.n_neg_items + 1:
            print(f"❌ Wrong candidate pool size: {len(candidates)}")
            return False
        
        target_item = self.extract_ground_truth(test_session)
        if candidates[target_idx] != target_item:
            print(f"❌ Target index incorrect")
            return False
        
        print("✅ All data integrity tests passed")
        return True
    
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