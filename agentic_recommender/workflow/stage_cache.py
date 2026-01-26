"""
Stage-level caching for expensive workflow outputs.

Cache location: ~/.cache/agentic_recommender/stages/
Cache validation: MD5 hash of input files + settings
"""

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class StageCache:
    """
    Global cache for expensive stage outputs.

    Similar to LightGCN's caching approach:
    1. Compute cache key from input files + settings
    2. Check if valid cache exists
    3. Load from cache or compute and save

    Cache structure:
    ~/.cache/agentic_recommender/stages/
        {stage_name}_{cache_key[:16]}/
            metadata.json      # Cache key, timestamp, input hashes
            output_files/      # Cached output files
    """

    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "stages"

    def __init__(self, stage_name: str, enabled: bool = True):
        self.stage_name = stage_name
        self.enabled = enabled
        self.cache_key: Optional[str] = None
        self.cache_path: Optional[Path] = None

    def compute_cache_key(
        self,
        input_files: List[str],
        settings: Dict[str, Any],
    ) -> str:
        """
        Compute cache key from input files and settings.

        Uses content-based hashing (file size + content sample), NOT file paths.
        This ensures cache hits even when output directories change.

        Args:
            input_files: List of input file paths (their content hashes are used)
            settings: Stage settings dict

        Returns:
            MD5 hash string
        """
        hasher = hashlib.md5()

        # Hash input file contents (NOT paths - paths change with timestamped folders)
        for filepath in sorted(input_files):
            if filepath and os.path.exists(filepath):
                stat = os.stat(filepath)
                # Use filename (not full path) + size for identification
                filename = os.path.basename(filepath)
                hasher.update(f"{filename}:{stat.st_size}".encode())

                # For files under 50MB, hash first 2MB of content for validation
                if stat.st_size < 50 * 1024 * 1024:
                    with open(filepath, 'rb') as f:
                        hasher.update(f.read(2 * 1024 * 1024))

        # Hash settings
        settings_str = json.dumps(settings, sort_keys=True, default=str)
        hasher.update(settings_str.encode())

        return hasher.hexdigest()

    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache directory path for a cache key."""
        # Use first 16 chars of hash for directory name
        dir_name = f"{self.stage_name}_{cache_key[:16]}"
        return self.CACHE_DIR / dir_name

    def check_cache(
        self,
        input_files: List[str],
        settings: Dict[str, Any],
    ) -> Tuple[bool, Optional[Path]]:
        """
        Check if valid cache exists.

        Returns:
            (cache_valid, cache_path) tuple
        """
        if not self.enabled:
            return False, None

        self.cache_key = self.compute_cache_key(input_files, settings)
        self.cache_path = self.get_cache_path(self.cache_key)

        metadata_path = self.cache_path / "metadata.json"
        if not metadata_path.exists():
            return False, None

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Validate cache key matches
            if metadata.get('cache_key') != self.cache_key:
                return False, None

            # Check all output files exist
            output_dir = self.cache_path / "output_files"
            for filename in metadata.get('output_files', []):
                if not (output_dir / filename).exists():
                    return False, None

            return True, self.cache_path

        except Exception:
            return False, None

    def load_cached_files(
        self,
        output_mapping: Dict[str, str],
    ) -> bool:
        """
        Copy cached files to output locations.

        Args:
            output_mapping: Dict of {output_key: output_path}

        Returns:
            True if successful
        """
        if not self.cache_path:
            return False

        try:
            output_dir = self.cache_path / "output_files"
            metadata_path = self.cache_path / "metadata.json"

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Map cached files to output paths
            cached_files = metadata.get('file_mapping', {})

            for output_key, output_path in output_mapping.items():
                if output_key in cached_files:
                    cached_filename = cached_files[output_key]
                    cached_file = output_dir / cached_filename

                    if cached_file.exists():
                        # Ensure output directory exists
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cached_file, output_path)

            return True

        except Exception as e:
            print(f"[StageCache] Failed to load cache: {e}")
            return False

    def save_to_cache(
        self,
        output_mapping: Dict[str, str],
        input_files: List[str],
        settings: Dict[str, Any],
    ) -> bool:
        """
        Save output files to cache.

        Args:
            output_mapping: Dict of {output_key: output_path}
            input_files: List of input file paths used
            settings: Stage settings used

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            # Compute cache key if not already done
            if not self.cache_key:
                self.cache_key = self.compute_cache_key(input_files, settings)
                self.cache_path = self.get_cache_path(self.cache_key)

            # Create cache directory
            output_dir = self.cache_path / "output_files"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Copy output files to cache
            file_mapping = {}
            output_files = []

            for output_key, output_path in output_mapping.items():
                if output_path and os.path.exists(output_path):
                    filename = os.path.basename(output_path)
                    shutil.copy2(output_path, output_dir / filename)
                    file_mapping[output_key] = filename
                    output_files.append(filename)

            # Save metadata
            metadata = {
                'stage_name': self.stage_name,
                'cache_key': self.cache_key,
                'timestamp': datetime.now().isoformat(),
                'input_files': input_files,
                'settings': settings,
                'output_files': output_files,
                'file_mapping': file_mapping,
            }

            with open(self.cache_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"[StageCache] Failed to save cache: {e}")
            return False

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        return {
            'stage_name': self.stage_name,
            'enabled': self.enabled,
            'cache_key': self.cache_key,
            'cache_path': str(self.cache_path) if self.cache_path else None,
            'cache_dir': str(self.CACHE_DIR),
        }


def clear_stage_cache(stage_name: Optional[str] = None) -> int:
    """
    Clear stage cache.

    Args:
        stage_name: If provided, only clear cache for this stage.
                   If None, clear all stage caches.

    Returns:
        Number of cache entries cleared
    """
    cache_dir = StageCache.CACHE_DIR
    if not cache_dir.exists():
        return 0

    cleared = 0
    for entry in cache_dir.iterdir():
        if entry.is_dir():
            if stage_name is None or entry.name.startswith(f"{stage_name}_"):
                shutil.rmtree(entry)
                cleared += 1

    return cleared


def list_stage_caches() -> List[Dict[str, Any]]:
    """List all cached stages with their metadata."""
    cache_dir = StageCache.CACHE_DIR
    if not cache_dir.exists():
        return []

    caches = []
    for entry in sorted(cache_dir.iterdir()):
        if entry.is_dir():
            metadata_path = entry / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['cache_path'] = str(entry)
                    caches.append(metadata)
                except Exception:
                    pass

    return caches
