"""
Workflow Runner for Data Processing Pipeline.

This module provides a configurable workflow system that:
1. Reads configuration from YAML
2. Controls which stages to execute
3. Logs all file I/O operations
4. Processes data through the pipeline
5. Runs LLM predictions on specified records

Usage:
    python workflow_runner.py                          # Run all enabled stages
    python workflow_runner.py --stages load_data       # Run specific stage
    python workflow_runner.py --stages build_users generate_prompts  # Multiple stages
    python workflow_runner.py --config custom.yaml     # Use custom config
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import yaml
import pandas as pd

from agentic_recommender.workflow.stage_cache import StageCache


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class WorkflowLogger:
    """Logger with detailed file operation tracking."""

    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        self.log_file = log_file
        self.start_time = None

        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            # Clear previous log
            with open(log_file, 'w') as f:
                f.write(f"Workflow Log - {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

    def _write(self, message: str, level: str = "INFO"):
        """Write message to console and optionally to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"

        if self.verbose:
            # Color coding for console
            colors = {
                "INFO": "\033[0m",      # Default
                "START": "\033[94m",    # Blue
                "SUCCESS": "\033[92m",  # Green
                "WARNING": "\033[93m",  # Yellow
                "ERROR": "\033[91m",    # Red
                "FILE": "\033[96m",     # Cyan
            }
            reset = "\033[0m"
            color = colors.get(level, "\033[0m")
            print(f"{color}{formatted}{reset}")

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + "\n")

    def info(self, message: str):
        self._write(message, "INFO")

    def start(self, message: str):
        self._write(message, "START")

    def success(self, message: str):
        self._write(message, "SUCCESS")

    def warning(self, message: str):
        self._write(message, "WARNING")

    def error(self, message: str):
        self._write(message, "ERROR")

    def file_read(self, filepath: str, details: str = ""):
        """Log file read operation."""
        size = self._get_file_size(filepath)
        msg = f"READING: {filepath} ({size})"
        if details:
            msg += f" - {details}"
        self._write(msg, "FILE")

    def file_write(self, filepath: str, details: str = ""):
        """Log file write operation."""
        size = self._get_file_size(filepath)
        msg = f"WRITING: {filepath} ({size})"
        if details:
            msg += f" - {details}"
        self._write(msg, "FILE")

    def _get_file_size(self, filepath: str) -> str:
        """Get human-readable file size."""
        try:
            size = Path(filepath).stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        except:
            return "N/A"

    def stage_start(self, stage_name: str, description: str = ""):
        """Log stage start."""
        self.start_time = time.time()
        self._write("", "INFO")
        self._write("=" * 70, "START")
        self._write(f"STAGE: {stage_name}", "START")
        if description:
            self._write(f"Description: {description}", "START")
        self._write("=" * 70, "START")

    def stage_end(self, stage_name: str, success: bool = True):
        """Log stage end with duration."""
        duration = time.time() - self.start_time if self.start_time else 0
        status = "COMPLETED" if success else "FAILED"
        level = "SUCCESS" if success else "ERROR"
        self._write(f"STAGE {stage_name} {status} in {duration:.2f}s", level)
        self._write("-" * 70, level)


# =============================================================================
# WORKFLOW CONFIGURATION
# =============================================================================

@dataclass
class StageConfig:
    """Configuration for a single stage."""
    enabled: bool
    description: str
    input: Dict[str, str]
    output: Dict[str, str]
    settings: Dict[str, Any]


class WorkflowConfig:
    """Load and manage workflow configuration from YAML."""

    def __init__(self, config_path: str = "workflow_config.yaml"):
        # If config_path is relative, resolve it relative to this script's directory
        if not os.path.isabs(config_path):
            script_dir = Path(__file__).parent
            self.config_path = str(script_dir / config_path)
        else:
            self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_stage(self, stage_name: str) -> StageConfig:
        """Get configuration for a specific stage."""
        stage_data = self.config['workflow']['stages'].get(stage_name, {})
        return StageConfig(
            enabled=stage_data.get('enabled', False),
            description=stage_data.get('description', ''),
            input=stage_data.get('input', {}),
            output=stage_data.get('output', {}),
            settings=stage_data.get('settings', {})
        )

    def get_enabled_stages(self) -> List[str]:
        """Get list of enabled stage names."""
        stages = self.config['workflow']['stages']
        return [name for name, cfg in stages.items() if cfg.get('enabled', False)]

    def get_all_stages(self) -> List[str]:
        """Get list of all stage names in order."""
        return list(self.config['workflow']['stages'].keys())

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM provider configuration."""
        return self.config.get('llm', {})

    def get_output_base_dir(self) -> str:
        """Get base output directory."""
        return self.config.get('output', {}).get('base_dir', 'outputs')

    def is_verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return self.config.get('workflow', {}).get('verbose', True)

    def get_log_file(self) -> Optional[str]:
        """Get log file path."""
        return self.config.get('workflow', {}).get('log_file')


# =============================================================================
# PIPELINE STAGES
# =============================================================================

class PipelineStages:
    """Implementation of all pipeline stages."""

    def __init__(self, config: WorkflowConfig, logger: WorkflowLogger, no_cache: bool = False):
        self.config = config
        self.logger = logger
        self.no_cache = no_cache

        # Ensure output directory exists
        output_dir = Path(config.get_output_base_dir())
        output_dir.mkdir(parents=True, exist_ok=True)

    def _save_preview(
        self,
        data: list,
        full_path: str,
        preview_rows: int = 10,
        description: str = "Preview of data",
    ) -> None:
        """
        Save a preview file for a JSON data list.

        Args:
            data: List of records to preview
            full_path: Path to the full JSON file (preview will be saved with _preview suffix)
            preview_rows: Number of rows to include in preview
            description: Description for the preview file
        """
        if not full_path or not data:
            return

        # Generate preview path by inserting _preview before .json
        if full_path.endswith('.json'):
            preview_path = full_path.replace('.json', '_preview.json')
        else:
            preview_path = f"{full_path}_preview.json"

        preview_data = {
            'description': description,
            'total_records': len(data),
            'preview_records': min(preview_rows, len(data)),
            'data': data[:preview_rows],
        }

        with open(preview_path, 'w') as f:
            json.dump(preview_data, f, indent=2, default=str)

        self.logger.file_write(preview_path, f"{preview_data['preview_records']} records preview")

    # -------------------------------------------------------------------------
    # Stage 1: Load Data
    # -------------------------------------------------------------------------
    def stage_load_data(self) -> bool:
        """Load and merge Singapore food delivery data."""
        stage_cfg = self.config.get_stage('load_data')
        self.logger.stage_start('load_data', stage_cfg.description)

        try:
            from agentic_recommender.data.enriched_loader import load_singapore_data

            # Load data
            data_dir = stage_cfg.input.get('data_dir', '/Users/zhenkai/Downloads/data_sg')

            # Check global cache
            input_files = [
                f"{data_dir}/orders_sg_train.txt",
                f"{data_dir}/vendors_sg.txt",
                f"{data_dir}/products_sg.txt",
            ]
            cache = StageCache('load_data', enabled=not self.no_cache)
            cache_valid, cache_path = cache.check_cache(input_files, stage_cfg.settings)

            if cache_valid:
                self.logger.info(f"[CACHE HIT] Loading from global cache: {cache_path}")
                output_mapping = {
                    'merged_data': stage_cfg.output.get('merged_data'),
                    'merged_preview': stage_cfg.output.get('merged_preview'),
                    'stats': stage_cfg.output.get('stats'),
                }
                # Include test_data if it should be loaded
                if stage_cfg.settings.get('load_test_data', False) and stage_cfg.output.get('test_data'):
                    output_mapping['test_data'] = stage_cfg.output.get('test_data')

                if cache.load_cached_files(output_mapping):
                    self.logger.success("Loaded all outputs from cache")
                    self.logger.stage_end('load_data', success=True)
                    return True
                else:
                    self.logger.warning("Cache load failed, recomputing...")

            self.logger.info(f"Loading data from: {data_dir}")
            loader = load_singapore_data(data_dir)

            # Load individual files with logging
            self.logger.file_read(f"{data_dir}/orders_sg_train.txt", "Loading orders")
            self.logger.file_read(f"{data_dir}/vendors_sg.txt", "Loading vendors")
            self.logger.file_read(f"{data_dir}/products_sg.txt", "Loading products")

            # Get merged data
            merged_df = loader.load_merged()
            self.logger.info(f"Merged data shape: {merged_df.shape}")
            self.logger.info(f"Columns: {list(merged_df.columns)}")

            # Get statistics
            stats = loader.get_stats()
            self.logger.info("Dataset Statistics:")
            for key, value in stats.items():
                self.logger.info(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.2f}")

            # Save merged data
            output_path = stage_cfg.output.get('merged_data')
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                merged_df.to_parquet(output_path)
                self.logger.file_write(output_path, f"{len(merged_df):,} rows")

            # Save JSON preview (top N rows for easy viewing)
            preview_path = stage_cfg.output.get('merged_preview')
            if preview_path:
                preview_rows = stage_cfg.settings.get('preview_rows', 1000)
                self.logger.info(f"Creating JSON preview with {preview_rows} rows...")
                preview_df = merged_df.head(preview_rows)
                # Convert to records format for readable JSON
                preview_data = {
                    'description': f'Preview of first {preview_rows} rows from merged data',
                    'total_rows': len(merged_df),
                    'preview_rows': preview_rows,
                    'columns': list(preview_df.columns),
                    'data': preview_df.to_dict(orient='records')
                }
                with open(preview_path, 'w') as f:
                    json.dump(preview_data, f, indent=2, default=str)
                self.logger.file_write(preview_path, f"{preview_rows} rows preview")

            # Save statistics
            stats_path = stage_cfg.output.get('stats')
            if stats_path:
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                self.logger.file_write(stats_path, "Dataset statistics")

            # Optionally load and save test data (for evaluation method 2)
            if stage_cfg.settings.get('load_test_data', False):
                test_output_path = stage_cfg.output.get('test_data')
                if test_output_path:
                    self.logger.info("Loading test data...")
                    try:
                        test_merged_df = loader.load_merged_test()
                        Path(test_output_path).parent.mkdir(parents=True, exist_ok=True)
                        test_merged_df.to_parquet(test_output_path)
                        self.logger.file_write(test_output_path, f"{len(test_merged_df):,} test rows")
                    except FileNotFoundError as e:
                        self.logger.warning(f"Test data not found: {e}")

            # Save to global cache
            output_mapping = {
                'merged_data': stage_cfg.output.get('merged_data'),
                'merged_preview': stage_cfg.output.get('merged_preview'),
                'stats': stage_cfg.output.get('stats'),
            }
            # Include test_data if it was loaded
            if stage_cfg.settings.get('load_test_data', False) and stage_cfg.output.get('test_data'):
                output_mapping['test_data'] = stage_cfg.output.get('test_data')

            if cache.save_to_cache(output_mapping, input_files, stage_cfg.settings):
                self.logger.info(f"[CACHE SAVE] Saved to global cache")

            self.logger.stage_end('load_data', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            self.logger.stage_end('load_data', success=False)
            return False

    # -------------------------------------------------------------------------
    # Stage 2: Build Users
    # -------------------------------------------------------------------------
    def stage_build_users(self) -> bool:
        """Build enriched user representations."""
        stage_cfg = self.config.get_stage('build_users')
        self.logger.stage_start('build_users', stage_cfg.description)

        try:
            from agentic_recommender.data.representations import EnrichedUser

            # Check global cache
            input_path = stage_cfg.input.get('merged_data')
            input_files = [input_path] if input_path else []
            cache = StageCache('build_users', enabled=not self.no_cache)
            cache_valid, cache_path = cache.check_cache(input_files, stage_cfg.settings)

            if cache_valid:
                self.logger.info(f"[CACHE HIT] Loading from global cache: {cache_path}")
                output_mapping = {
                    'users_json': stage_cfg.output.get('users_json'),
                    'users_summary': stage_cfg.output.get('users_summary'),
                }
                if cache.load_cached_files(output_mapping):
                    self.logger.success("Loaded all outputs from cache")
                    self.logger.stage_end('build_users', success=True)
                    return True
                else:
                    self.logger.warning("Cache load failed, recomputing...")

            # Load merged data
            self.logger.file_read(input_path, "Loading merged data")
            merged_df = pd.read_parquet(input_path)
            self.logger.info(f"Loaded {len(merged_df):,} rows")

            # Get settings
            min_orders = stage_cfg.settings.get('min_orders', 5)
            max_order_history = stage_cfg.settings.get('max_order_history', None)  # None = infinity
            max_users = stage_cfg.settings.get('max_users')

            # Filter customers with enough orders (and optionally max orders)
            customer_order_counts = merged_df.groupby('customer_id')['order_id'].nunique()

            if max_order_history is not None:
                valid_customers = customer_order_counts[
                    (customer_order_counts >= min_orders) &
                    (customer_order_counts <= max_order_history)
                ].index
                self.logger.info(f"Customers with {min_orders}-{max_order_history} orders: {len(valid_customers):,}")
            else:
                valid_customers = customer_order_counts[customer_order_counts >= min_orders].index
                self.logger.info(f"Customers with >= {min_orders} orders: {len(valid_customers):,}")

            # Limit users if specified
            if max_users:
                valid_customers = list(valid_customers[:max_users])
                self.logger.info(f"Limiting to {max_users} users for processing")
            else:
                valid_customers = list(valid_customers)

            # Build user representations using groupby for O(n+m) instead of O(n*m)
            users = []
            total = len(valid_customers)
            valid_customers_set = set(valid_customers)

            self.logger.info(f"Building representations for {total} users...")
            self.logger.info("Using optimized groupby approach...")

            # Filter to valid customers first, then group - much faster than filtering per user
            filtered_df = merged_df[merged_df['customer_id'].isin(valid_customers_set)]
            grouped = filtered_df.groupby('customer_id')

            for i, (customer_id, customer_orders) in enumerate(grouped):
                if (i + 1) % 5000 == 0 or i == 0 or i == total - 1:
                    self.logger.info(f"  Processing user {i+1}/{total} ({(i+1)/total*100:.1f}%)")

                user = EnrichedUser.from_orders(customer_id, customer_orders)
                users.append(user.to_dict())

            self.logger.success(f"Built {len(users)} user representations")

            # Save users
            output_path = stage_cfg.output.get('users_json')
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(users, f, indent=2)
                self.logger.file_write(output_path, f"{len(users)} users")

                # Save preview
                self._save_preview(
                    data=users,
                    full_path=output_path,
                    preview_rows=10,
                    description="Preview of enriched user representations",
                )

            # Save summary
            summary_path = stage_cfg.output.get('users_summary')
            if summary_path:
                summary = {
                    'total_users': len(users),
                    'min_orders_filter': min_orders,
                    'avg_orders': sum(u['total_orders'] for u in users) / len(users),
                    'avg_items': sum(u['total_items'] for u in users) / len(users),
                    'timestamp': datetime.now().isoformat()
                }
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.logger.file_write(summary_path, "User summary statistics")

            # Save to global cache
            output_mapping = {
                'users_json': stage_cfg.output.get('users_json'),
                'users_summary': stage_cfg.output.get('users_summary'),
            }
            if cache.save_to_cache(output_mapping, input_files, stage_cfg.settings):
                self.logger.info(f"[CACHE SAVE] Saved to global cache")

            self.logger.stage_end('build_users', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.stage_end('build_users', success=False)
            return False

    # -------------------------------------------------------------------------
    # Stage 3: Build Cuisines
    # -------------------------------------------------------------------------
    def stage_build_cuisines(self) -> bool:
        """Build cuisine profiles."""
        stage_cfg = self.config.get_stage('build_cuisines')
        self.logger.stage_start('build_cuisines', stage_cfg.description)

        try:
            from agentic_recommender.data.representations import CuisineRegistry

            # Check global cache
            input_path = stage_cfg.input.get('merged_data')
            input_files = [input_path] if input_path else []
            cache = StageCache('build_cuisines', enabled=not self.no_cache)
            cache_valid, cache_path = cache.check_cache(input_files, stage_cfg.settings)

            if cache_valid:
                self.logger.info(f"[CACHE HIT] Loading from global cache: {cache_path}")
                output_mapping = {
                    'cuisines_json': stage_cfg.output.get('cuisines_json'),
                }
                if cache.load_cached_files(output_mapping):
                    self.logger.success("Loaded all outputs from cache")
                    self.logger.stage_end('build_cuisines', success=True)
                    return True
                else:
                    self.logger.warning("Cache load failed, recomputing...")

            # Load merged data
            self.logger.file_read(input_path, "Loading merged data")
            merged_df = pd.read_parquet(input_path)

            # Build cuisine registry
            self.logger.info("Building cuisine profiles...")
            registry = CuisineRegistry()
            registry.build_from_data(merged_df)

            # Convert to serializable format
            cuisines_data = {}
            for cuisine in registry.get_all_cuisines():
                profile = registry.get_profile(cuisine)
                if profile:
                    cuisines_data[cuisine] = profile.to_dict()

            self.logger.success(f"Built {len(cuisines_data)} cuisine profiles")

            # Log top cuisines
            self.logger.info("Top cuisines by order count:")
            sorted_cuisines = sorted(cuisines_data.items(),
                                    key=lambda x: x[1]['total_orders'],
                                    reverse=True)[:5]
            for cuisine, data in sorted_cuisines:
                self.logger.info(f"  {cuisine}: {data['total_orders']:,} orders")

            # Save cuisines
            output_path = stage_cfg.output.get('cuisines_json')
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(cuisines_data, f, indent=2)
                self.logger.file_write(output_path, f"{len(cuisines_data)} cuisines")

                # Save preview (convert dict to list for preview)
                cuisines_list = [{'cuisine': k, **v} for k, v in list(cuisines_data.items())[:10]]
                self._save_preview(
                    data=cuisines_list,
                    full_path=output_path,
                    preview_rows=10,
                    description="Preview of cuisine profiles",
                )

            # Save to global cache
            output_mapping = {
                'cuisines_json': stage_cfg.output.get('cuisines_json'),
            }
            if cache.save_to_cache(output_mapping, input_files, stage_cfg.settings):
                self.logger.info(f"[CACHE SAVE] Saved to global cache")

            self.logger.stage_end('build_cuisines', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            self.logger.stage_end('build_cuisines', success=False)
            return False

    # -------------------------------------------------------------------------
    # Stage 4: Generate Prompts
    # -------------------------------------------------------------------------
    def stage_generate_prompts(self) -> bool:
        """Generate formatted prompts for LLM prediction."""
        stage_cfg = self.config.get_stage('generate_prompts')
        self.logger.stage_start('generate_prompts', stage_cfg.description)

        try:
            from agentic_recommender.core.prompts import PromptManager, PromptType

            # Load users
            users_path = stage_cfg.input.get('users_json')
            self.logger.file_read(users_path, "Loading user representations")
            with open(users_path, 'r') as f:
                users = json.load(f)
            self.logger.info(f"Loaded {len(users)} users")

            # Load merged data for candidate products
            merged_path = stage_cfg.input.get('merged_data')
            self.logger.file_read(merged_path, "Loading merged data for candidates")
            merged_df = pd.read_parquet(merged_path)

            # Get settings
            max_prompts = stage_cfg.settings.get('max_prompts', 100)
            prompt_type_str = stage_cfg.settings.get('prompt_type', 'reflector_first_round')

            # Map string to PromptType enum
            prompt_type_map = {
                'reflector_first_round': PromptType.REFLECTOR_FIRST_ROUND,
                'analyst_user': PromptType.ANALYST_USER,
                'manager_think': PromptType.MANAGER_THINK,
            }
            prompt_type = prompt_type_map.get(prompt_type_str, PromptType.REFLECTOR_FIRST_ROUND)

            # Initialize prompt manager
            prompt_mgr = PromptManager()

            # Generate prompts
            prompts = []
            readable_prompts = []

            self.logger.info(f"Generating up to {max_prompts} prompts using template: {prompt_type_str}")

            # Get unique vendors for candidates
            all_vendors = merged_df[['vendor_id', 'cuisine']].drop_duplicates()

            for i, user in enumerate(users[:max_prompts]):
                if (i + 1) % 50 == 0 or i == 0:
                    self.logger.info(f"  Generating prompt {i+1}/{min(len(users), max_prompts)}")

                # Get user's order history
                customer_id = user['customer_id']
                user_orders = merged_df[merged_df['customer_id'] == customer_id]

                if len(user_orders) == 0:
                    continue

                # Format order history
                order_history_lines = []
                for order_id in user_orders['order_id'].unique()[:5]:
                    order = user_orders[user_orders['order_id'] == order_id].iloc[0]
                    order_history_lines.append(
                        f"- {order.get('day_name', 'N/A')} at {order.get('hour', 'N/A')}:00: "
                        f"{order.get('cuisine', 'unknown')} (${order.get('unit_price', 0):.2f})"
                    )
                order_history = "\n".join(order_history_lines)

                # Select a candidate product (from a different vendor)
                user_vendors = set(user_orders['vendor_id'].unique())
                candidate_vendors = all_vendors[~all_vendors['vendor_id'].isin(user_vendors)]

                if len(candidate_vendors) > 0:
                    candidate = candidate_vendors.sample(1).iloc[0]
                else:
                    candidate = all_vendors.sample(1).iloc[0]

                candidate_product = f"Vendor: {candidate['vendor_id']}\nCuisine: {candidate['cuisine']}"

                # Generate prompt
                try:
                    prompt_text = prompt_mgr.render(
                        prompt_type,
                        order_history=order_history,
                        candidate_product=candidate_product
                    )

                    prompt_record = {
                        'id': i,
                        'customer_id': customer_id,
                        'candidate_vendor': str(candidate['vendor_id']),
                        'candidate_cuisine': candidate['cuisine'],
                        'prompt': prompt_text,
                        'order_history': order_history,
                        'template': prompt_type_str
                    }
                    prompts.append(prompt_record)

                    # Readable format
                    readable_prompts.append(f"\n{'='*80}\nPROMPT {i+1} (Customer: {customer_id})\n{'='*80}\n{prompt_text}\n")

                except Exception as e:
                    self.logger.warning(f"Failed to generate prompt for user {customer_id}: {e}")

            self.logger.success(f"Generated {len(prompts)} prompts")

            # Save prompts JSON
            output_path = stage_cfg.output.get('prompts_json')
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(prompts, f, indent=2)
                self.logger.file_write(output_path, f"{len(prompts)} prompts")

                # Save preview
                self._save_preview(
                    data=prompts,
                    full_path=output_path,
                    preview_rows=5,
                    description="Preview of generated prompts",
                )

            # Save readable prompts
            readable_path = stage_cfg.output.get('prompts_readable')
            if readable_path:
                with open(readable_path, 'w') as f:
                    f.write(f"Generated Prompts - {datetime.now().isoformat()}\n")
                    f.write(f"Total: {len(prompts)} prompts\n")
                    f.write(f"Template: {prompt_type_str}\n")
                    f.writelines(readable_prompts)
                self.logger.file_write(readable_path, "Human-readable prompts")

            self.logger.stage_end('generate_prompts', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.stage_end('generate_prompts', success=False)
            return False

    # -------------------------------------------------------------------------
    # Stage 5: Run Predictions
    # -------------------------------------------------------------------------
    def stage_run_predictions(self) -> bool:
        """Run LLM predictions on generated prompts."""
        stage_cfg = self.config.get_stage('run_predictions')
        self.logger.stage_start('run_predictions', stage_cfg.description)

        try:
            from agentic_recommender.models.llm_provider import (
                create_llm_provider, MockLLMProvider
            )

            # Load prompts
            input_path = stage_cfg.input.get('prompts_json')
            self.logger.file_read(input_path, "Loading prompts")
            with open(input_path, 'r') as f:
                prompts = json.load(f)
            self.logger.info(f"Loaded {len(prompts)} prompts")

            # Get settings
            limit = stage_cfg.settings.get('limit', 5)
            save_intermediate = stage_cfg.settings.get('save_intermediate', True)

            # Limit prompts
            prompts_to_process = prompts[:limit]
            self.logger.info(f"Processing first {len(prompts_to_process)} prompts (limit={limit})")

            # Initialize LLM provider
            llm_config = self.config.get_llm_config()
            provider_type = llm_config.get('provider', 'mock')

            self.logger.info(f"Initializing LLM provider: {provider_type}")

            if provider_type == 'mock':
                # Use mock provider for testing
                provider = MockLLMProvider(responses={
                    "prediction": '{"prediction": true, "confidence": 0.75, "reasoning": "Mock prediction based on user patterns"}'
                })
            else:
                provider = create_llm_provider(
                    provider_type=provider_type,
                    temperature=llm_config.get('temperature', 0.7),
                    max_tokens=llm_config.get('max_tokens', 512)
                )

            self.logger.success(f"LLM Provider initialized: {provider.get_model_info()['provider']}")

            # Run predictions
            predictions = []
            output_path = stage_cfg.output.get('predictions_json')

            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("STARTING LLM PREDICTIONS")
            self.logger.info("=" * 60)

            for i, prompt_record in enumerate(prompts_to_process):
                self.logger.info(f"\n[{i+1}/{len(prompts_to_process)}] Processing prompt for customer: {prompt_record['customer_id']}")
                self.logger.info(f"  Candidate cuisine: {prompt_record['candidate_cuisine']}")

                start_time = time.time()

                # Get prediction from LLM
                response = provider.generate(
                    prompt_record['prompt'],
                    temperature=llm_config.get('temperature', 0.7),
                    max_tokens=llm_config.get('max_tokens', 512)
                )

                duration = time.time() - start_time

                # Parse response
                try:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{[^}]+\}', response)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        parsed = {'raw_response': response}
                except json.JSONDecodeError:
                    parsed = {'raw_response': response}

                prediction_record = {
                    'id': prompt_record['id'],
                    'customer_id': prompt_record['customer_id'],
                    'candidate_vendor': prompt_record['candidate_vendor'],
                    'candidate_cuisine': prompt_record['candidate_cuisine'],
                    'response': response,
                    'parsed': parsed,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                }
                predictions.append(prediction_record)

                # Log result
                pred_value = parsed.get('prediction', 'N/A')
                confidence = parsed.get('confidence', 'N/A')
                self.logger.info(f"  Prediction: {pred_value}, Confidence: {confidence}")
                self.logger.info(f"  Duration: {duration:.2f}s")

                # Save intermediate results
                if save_intermediate and output_path:
                    with open(output_path, 'w') as f:
                        json.dump(predictions, f, indent=2)

            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("PREDICTIONS COMPLETE")
            self.logger.info("=" * 60)

            # Final save
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(predictions, f, indent=2)
                self.logger.file_write(output_path, f"{len(predictions)} predictions")

                # Save preview
                self._save_preview(
                    data=predictions,
                    full_path=output_path,
                    preview_rows=10,
                    description="Preview of LLM predictions",
                )

            # Generate summary
            summary_path = stage_cfg.output.get('predictions_summary')
            if summary_path:
                # Count predictions
                true_count = sum(1 for p in predictions
                               if p.get('parsed', {}).get('prediction') == True)
                false_count = sum(1 for p in predictions
                                if p.get('parsed', {}).get('prediction') == False)

                avg_duration = sum(p['duration_seconds'] for p in predictions) / len(predictions)

                summary = {
                    'total_predictions': len(predictions),
                    'true_predictions': true_count,
                    'false_predictions': false_count,
                    'average_duration_seconds': avg_duration,
                    'provider': provider.get_model_info(),
                    'timestamp': datetime.now().isoformat()
                }

                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.logger.file_write(summary_path, "Prediction summary")

                self.logger.info("")
                self.logger.info("Summary:")
                self.logger.info(f"  Total predictions: {len(predictions)}")
                self.logger.info(f"  True predictions: {true_count}")
                self.logger.info(f"  False predictions: {false_count}")
                self.logger.info(f"  Average duration: {avg_duration:.2f}s")

            self.logger.stage_end('run_predictions', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.stage_end('run_predictions', success=False)
            return False

    # -------------------------------------------------------------------------
    # Stage 6: Run TopK Evaluation with Real LLM
    # -------------------------------------------------------------------------
    def stage_run_topk_evaluation(self) -> bool:
        """Run TopK evaluation with real LLM predictions."""
        stage_cfg = self.config.get_stage('run_topk_evaluation')
        self.logger.stage_start('run_topk_evaluation', stage_cfg.description)

        try:
            from agentic_recommender.evaluation.topk import (
                TopKTestDataBuilder,
                SequentialRecommendationEvaluator,
                TopKMetrics,
            )
            from agentic_recommender.models.llm_provider import (
                create_llm_provider,
                OpenRouterProvider,
                MockLLMProvider,
            )

            # Load merged data
            input_path = stage_cfg.input.get('merged_data')
            self.logger.file_read(input_path, "Loading merged data for evaluation")
            merged_df = pd.read_parquet(input_path)
            self.logger.info(f"Loaded {len(merged_df):,} rows")

            # Get settings
            n_samples = stage_cfg.settings.get('n_samples', 50)
            min_history = stage_cfg.settings.get('min_history', 5)
            k_values = stage_cfg.settings.get('k_values', [1, 3, 5, 10])
            seed = stage_cfg.settings.get('seed', 42)
            save_predictions = stage_cfg.settings.get('save_predictions', True)

            # Build test samples
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("BUILDING TEST SAMPLES")
            self.logger.info("=" * 60)
            self.logger.info(f"Creating {n_samples} test samples (min_history={min_history})...")

            builder = TopKTestDataBuilder(
                orders_df=merged_df,
                min_history=min_history
            )
            test_samples = builder.build_test_samples(n_samples=n_samples, seed=seed)
            self.logger.success(f"Created {len(test_samples)} test samples")

            # Get cuisine list
            cuisines = builder.get_unique_cuisines()
            self.logger.info(f"Unique cuisines: {len(cuisines)}")

            # Save test samples
            samples_path = stage_cfg.output.get('samples_json')
            if samples_path:
                with open(samples_path, 'w') as f:
                    json.dump(test_samples, f, indent=2)
                self.logger.file_write(samples_path, f"{len(test_samples)} test samples")

                # Save preview
                self._save_preview(
                    data=test_samples,
                    full_path=samples_path,
                    preview_rows=5,
                    description="Preview of TopK test samples",
                )

            # Initialize LLM provider
            llm_config = self.config.get_llm_config()
            provider_type = llm_config.get('provider', 'mock')

            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("INITIALIZING LLM PROVIDER")
            self.logger.info("=" * 60)
            self.logger.info(f"Provider type: {provider_type}")

            if provider_type == 'mock':
                # Create mock provider with cuisine-aware responses
                self.logger.info("Using MOCK provider for testing")
                provider = self._create_mock_topk_provider(cuisines)
            elif provider_type == 'openrouter':
                self.logger.info("Using OpenRouter provider (REAL LLM)")
                api_key = (
                    llm_config.get('openrouter', {}).get('api_key') or
                    os.environ.get('OPENROUTER_API_KEY')
                )
                model_name = llm_config.get('openrouter', {}).get('model_name', 'google/gemini-2.0-flash-001')

                if not api_key:
                    self.logger.error("OpenRouter API key not found! Set OPENROUTER_API_KEY env var or add to config")
                    self.logger.stage_end('run_topk_evaluation', success=False)
                    return False

                provider = OpenRouterProvider(
                    api_key=api_key,
                    model_name=model_name
                )
                self.logger.success(f"Initialized OpenRouter with model: {model_name}")
            else:
                self.logger.info(f"Using {provider_type} provider")
                provider = create_llm_provider(provider_type=provider_type)

            provider_info = provider.get_model_info()
            self.logger.info(f"Provider info: {provider_info}")

            # Run evaluation
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("RUNNING TOPK EVALUATION")
            self.logger.info("=" * 60)
            self.logger.info(f"Evaluating {len(test_samples)} samples with K={k_values}...")

            evaluator = SequentialRecommendationEvaluator(
                llm_provider=provider,
                cuisine_list=cuisines,
                k_values=k_values,
            )

            # Custom evaluation with detailed logging
            detailed_results = []
            results_list = []
            total_time = 0.0

            for i, sample in enumerate(test_samples):
                self.logger.info(f"\n[{i+1}/{len(test_samples)}] Customer: {sample['customer_id']}")
                self.logger.info(f"  History length: {len(sample['order_history'])} orders")
                self.logger.info(f"  Ground truth: {sample['ground_truth_cuisine']}")

                start_time = time.time()

                # Get predictions
                predictions = evaluator._get_predictions(
                    customer_id=sample['customer_id'],
                    order_history=sample['order_history'],
                    k=max(k_values)
                )

                elapsed = (time.time() - start_time) * 1000
                total_time += elapsed

                # Find rank
                rank = evaluator._find_rank(predictions, sample['ground_truth_cuisine'])

                self.logger.info(f"  Predictions: {[p[0] for p in predictions[:5]]}")
                self.logger.info(f"  Rank: {rank if rank > 0 else 'NOT FOUND'}")
                self.logger.info(f"  Time: {elapsed:.2f}ms")

                result = {
                    'sample_idx': i,
                    'customer_id': sample['customer_id'],
                    'ground_truth': sample['ground_truth_cuisine'],
                    'predictions': [(p[0], float(p[1])) for p in predictions],
                    'rank': rank,
                    'hit_at_1': 1 if 0 < rank <= 1 else 0,
                    'hit_at_3': 1 if 0 < rank <= 3 else 0,
                    'hit_at_5': 1 if 0 < rank <= 5 else 0,
                    'hit_at_10': 1 if 0 < rank <= 10 else 0,
                    'time_ms': elapsed,
                }
                results_list.append(result)

                if save_predictions:
                    detailed_results.append({
                        **result,
                        'order_history': sample['order_history'][-5:],  # Last 5 orders
                    })

            # Compute metrics
            n_valid = len([r for r in results_list if len(r['predictions']) > 0])

            metrics = {
                'hit_at_1': sum(r['hit_at_1'] for r in results_list) / n_valid if n_valid > 0 else 0,
                'hit_at_3': sum(r['hit_at_3'] for r in results_list) / n_valid if n_valid > 0 else 0,
                'hit_at_5': sum(r['hit_at_5'] for r in results_list) / n_valid if n_valid > 0 else 0,
                'hit_at_10': sum(r['hit_at_10'] for r in results_list) / n_valid if n_valid > 0 else 0,
                'mrr': sum(1.0 / r['rank'] if r['rank'] > 0 else 0 for r in results_list) / n_valid if n_valid > 0 else 0,
                'total_samples': len(test_samples),
                'valid_samples': n_valid,
                'avg_time_ms': total_time / len(test_samples) if test_samples else 0,
                'provider': provider_info,
                'k_values': k_values,
                'timestamp': datetime.now().isoformat(),
            }

            # Print results
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("TOPK EVALUATION RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"  Hit@1:  {metrics['hit_at_1']:.2%}")
            self.logger.info(f"  Hit@3:  {metrics['hit_at_3']:.2%}")
            self.logger.info(f"  Hit@5:  {metrics['hit_at_5']:.2%}")
            self.logger.info(f"  Hit@10: {metrics['hit_at_10']:.2%}")
            self.logger.info(f"  MRR:    {metrics['mrr']:.4f}")
            self.logger.info(f"  Samples: {n_valid}/{len(test_samples)}")
            self.logger.info(f"  Avg time: {metrics['avg_time_ms']:.2f}ms")

            # Save results
            results_path = stage_cfg.output.get('results_json')
            if results_path:
                with open(results_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                self.logger.file_write(results_path, "TopK evaluation metrics")

            # Save detailed predictions
            detailed_path = stage_cfg.output.get('detailed_predictions')
            if detailed_path and save_predictions:
                with open(detailed_path, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
                self.logger.file_write(detailed_path, f"{len(detailed_results)} detailed predictions")

                # Save preview
                self._save_preview(
                    data=detailed_results,
                    full_path=detailed_path,
                    preview_rows=5,
                    description="Preview of detailed TopK predictions",
                )

            self.logger.stage_end('run_topk_evaluation', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.stage_end('run_topk_evaluation', success=False)
            return False

    def _create_mock_topk_provider(self, cuisines: List[str]):
        """Create a mock provider that returns realistic cuisine predictions."""
        import random

        class MockTopKProvider:
            """Mock provider for TopK evaluation testing."""

            def __init__(self, cuisine_list):
                self.cuisines = cuisine_list
                self.call_count = 0

            def generate(self, prompt: str, **kwargs) -> str:
                """Generate mock cuisine predictions."""
                self.call_count += 1

                # Return top 10 random cuisines as predictions
                selected = random.sample(self.cuisines, min(10, len(self.cuisines)))
                predictions = [
                    {"cuisine": c, "confidence": round(0.95 - i * 0.08, 2)}
                    for i, c in enumerate(selected)
                ]

                return json.dumps({"predictions": predictions})

            def get_model_info(self):
                return {
                    "provider": "MockTopK",
                    "model_name": "mock-topk-llm",
                    "total_calls": self.call_count
                }

        return MockTopKProvider(cuisines)

    # -------------------------------------------------------------------------
    # Stage 7: Retrieve-Rerank Evaluation
    # -------------------------------------------------------------------------
    def stage_run_rerank_evaluation(self) -> bool:
        """Run retrieve-then-rerank evaluation with candidate generation."""
        stage_cfg = self.config.get_stage('run_rerank_evaluation')
        self.logger.stage_start('run_rerank_evaluation', stage_cfg.description)

        try:
            from agentic_recommender.evaluation.rerank_eval import (
                CuisineCandidateGenerator,
                RerankEvaluator,
                RerankConfig,
                build_test_samples,
            )
            from agentic_recommender.models.llm_provider import OpenRouterProvider

            # Load merged data
            input_path = stage_cfg.input.get('merged_data')
            self.logger.file_read(input_path, "Loading merged data for rerank evaluation")
            merged_df = pd.read_parquet(input_path)
            self.logger.info(f"Loaded {len(merged_df):,} rows")

            # Get settings
            settings = stage_cfg.settings
            config = RerankConfig(
                n_candidates=settings.get('n_candidates', 20),
                n_from_history=settings.get('n_from_history', 10),
                n_similar_users=settings.get('n_similar_users', 10),
                similarity_method=settings.get('similarity_method', 'swing'),
                k_picks=settings.get('k_picks', 5),
                n_samples=settings.get('n_samples', 10),
                min_history=settings.get('min_history', 5),
            )

            # Build test samples
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("BUILDING TEST SAMPLES")
            self.logger.info("=" * 60)

            test_samples = build_test_samples(
                merged_df,
                n_samples=config.n_samples,
                min_history=config.min_history,
            )
            self.logger.success(f"Created {len(test_samples)} test samples")

            # Save test samples
            samples_path = stage_cfg.output.get('samples_json')
            if samples_path:
                with open(samples_path, 'w') as f:
                    json.dump(test_samples, f, indent=2)
                self.logger.file_write(samples_path, f"{len(test_samples)} test samples")

                # Save preview
                self._save_preview(
                    data=test_samples,
                    full_path=samples_path,
                    preview_rows=5,
                    description="Preview of Rerank test samples",
                )

            # Build candidate generator
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("BUILDING CANDIDATE GENERATOR")
            self.logger.info("=" * 60)
            self.logger.info(f"Similarity method: {config.similarity_method}")
            self.logger.info(f"Candidates per user: {config.n_candidates}")

            generator = CuisineCandidateGenerator(config)
            generator.fit(merged_df)

            gen_stats = generator.get_stats()
            self.logger.info(f"Generator stats: {gen_stats['num_users']} users, {gen_stats['num_cuisines']} cuisines")

            # Initialize LLM provider
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("INITIALIZING LLM PROVIDER")
            self.logger.info("=" * 60)

            llm_config = self.config.get_llm_config()
            provider_type = llm_config.get('provider', 'mock')
            self.logger.info(f"Provider type: {provider_type}")

            if provider_type == 'mock':
                provider = self._create_mock_rerank_provider()
            elif provider_type == 'openrouter':
                api_key = (
                    llm_config.get('openrouter', {}).get('api_key') or
                    os.environ.get('OPENROUTER_API_KEY')
                )
                if not api_key:
                    self.logger.error("OpenRouter API key not found!")
                    self.logger.stage_end('run_rerank_evaluation', success=False)
                    return False

                provider = OpenRouterProvider(
                    api_key=api_key,
                    model_name=llm_config.get('openrouter', {}).get('model_name', 'google/gemini-2.0-flash-001')
                )
                self.logger.success(f"Initialized OpenRouter")
            else:
                from agentic_recommender.models.llm_provider import create_llm_provider
                provider = create_llm_provider(provider_type=provider_type)

            # Run evaluation
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("RUNNING RERANK EVALUATION")
            self.logger.info("=" * 60)
            self.logger.info(f"Samples: {len(test_samples)}, K picks: {config.k_picks}")

            evaluator = RerankEvaluator(
                llm_provider=provider,
                candidate_generator=generator,
                config=config,
            )

            # Custom verbose evaluation
            metrics, detailed_results = self._run_rerank_with_logging(
                evaluator, test_samples, config
            )

            # Print results
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("RERANK EVALUATION RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"  Recall@{config.k_picks}:  {metrics.recall_at_k:.2%}")
            self.logger.info(f"  Precision@{config.k_picks}: {metrics.precision_at_k:.2%}")
            self.logger.info(f"  First Hit Avg: {metrics.first_hit_avg:.2f}")
            self.logger.info(f"  GT in Candidates: {metrics.ground_truth_in_candidates:.2%}")
            self.logger.info(f"  Avg Candidate Rank: {metrics.avg_candidate_rank:.1f}")
            self.logger.info(f"  Samples: {metrics.valid_samples}/{metrics.total_samples}")
            self.logger.info(f"  Avg Time: {metrics.avg_time_ms:.2f}ms")

            # Save results
            results_path = stage_cfg.output.get('results_json')
            if results_path:
                with open(results_path, 'w') as f:
                    json.dump(metrics.to_dict(), f, indent=2)
                self.logger.file_write(results_path, "Rerank evaluation metrics")

            # Save detailed results
            detailed_path = stage_cfg.output.get('detailed_json')
            if detailed_path:
                with open(detailed_path, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
                self.logger.file_write(detailed_path, f"{len(detailed_results)} detailed results")

                # Save preview
                self._save_preview(
                    data=detailed_results,
                    full_path=detailed_path,
                    preview_rows=5,
                    description="Preview of detailed Rerank results",
                )

            self.logger.stage_end('run_rerank_evaluation', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.stage_end('run_rerank_evaluation', success=False)
            return False

    def _run_rerank_with_logging(
        self,
        evaluator,
        test_samples: List[Dict],
        config,
    ) -> Tuple[Any, List[Dict]]:
        """Run rerank evaluation with detailed logging."""
        import random
        from collections import Counter

        results = []
        total_time = 0.0

        for i, sample in enumerate(test_samples):
            self.logger.info(f"\n[{i+1}/{len(test_samples)}] Customer: {sample['customer_id']}")
            self.logger.info(f"  History: {len(sample['order_history'])} orders")
            self.logger.info(f"  Ground truth: {sample['ground_truth_cuisine']}")

            start_time = time.time()

            # Generate candidates
            candidates, candidate_info = evaluator.generator.generate_candidates(
                customer_id=sample['customer_id'],
                ground_truth=sample['ground_truth_cuisine'],
            )

            # Detailed candidate breakdown for debugging
            self.logger.info(f"  ")
            self.logger.info(f"  === CANDIDATE SELECTION ({len(candidates)} total) ===")

            # Show user's order history summary
            history_cuisines = [o.get('cuisine', 'unknown') for o in sample['order_history']]
            cuisine_counts = Counter(history_cuisines)
            self.logger.info(f"  User's cuisine history: {dict(cuisine_counts.most_common(5))}")

            # Show candidates from history
            from_history = candidate_info['from_history']
            self.logger.info(f"  ")
            self.logger.info(f"  [FROM USER HISTORY] ({len(from_history)} cuisines):")
            for idx, cuisine in enumerate(from_history, 1):
                count = cuisine_counts.get(cuisine, 0)
                self.logger.info(f"    {idx}. {cuisine} (ordered {count}x)")

            # Show candidates from similar users
            from_similar = candidate_info['from_similar_users']
            self.logger.info(f"  ")
            self.logger.info(f"  [FROM SIMILAR USERS] ({len(from_similar)} cuisines):")
            for idx, cuisine in enumerate(from_similar, 1):
                self.logger.info(f"    {idx}. {cuisine}")

            # Show random fill if any
            from_history_set = set(from_history)
            from_similar_set = set(from_similar)
            from_random = [c for c in candidates if c not in from_history_set and c not in from_similar_set and c != sample['ground_truth_cuisine']]
            if from_random:
                self.logger.info(f"  ")
                self.logger.info(f"  [RANDOM FILL] ({len(from_random)} cuisines):")
                for idx, cuisine in enumerate(from_random, 1):
                    self.logger.info(f"    {idx}. {cuisine}")

            # Show ground truth status
            self.logger.info(f"  ")
            self.logger.info(f"  [GROUND TRUTH] {sample['ground_truth_cuisine']}")
            if candidate_info['ground_truth_added']:
                self.logger.info(f"    -> NOT in candidates naturally, ADDED at rank {candidate_info['ground_truth_rank']}")
            else:
                self.logger.info(f"    -> Found in candidates at rank {candidate_info['ground_truth_rank']}")

            # Show all 20 candidates in order
            self.logger.info(f"  ")
            self.logger.info(f"  [FINAL CANDIDATE LIST]:")
            for idx, cuisine in enumerate(candidates, 1):
                source = ""
                if cuisine in from_history_set:
                    source = "(history)"
                elif cuisine in from_similar_set:
                    source = "(similar users)"
                elif cuisine == sample['ground_truth_cuisine'] and candidate_info['ground_truth_added']:
                    source = "(GT added)"
                else:
                    source = "(random)"
                gt_marker = " <-- GROUND TRUTH" if cuisine == sample['ground_truth_cuisine'] else ""
                self.logger.info(f"    {idx:2}. {cuisine} {source}{gt_marker}")

            self.logger.info(f"  ")

            # Get K picks from LLM
            picks = []
            llm_interactions = []
            for k in range(config.k_picks):
                llm_result = evaluator._get_llm_pick(
                    order_history=sample['order_history'],
                    candidates=candidates,
                    target_hour=sample.get('target_hour'),
                    target_day_of_week=sample.get('target_day_of_week'),
                )
                pick = llm_result['parsed_pick']
                picks.append(pick)
                llm_interactions.append(llm_result)
                match = "✓" if pick == sample['ground_truth_cuisine'] else "✗"
                self.logger.info(f"  Pick {k+1}: {pick} {match}")

            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed

            # Calculate metrics
            ground_truth = sample['ground_truth_cuisine']
            hits = [1 if p == ground_truth else 0 for p in picks]
            first_hit = next((i+1 for i, h in enumerate(hits) if h == 1), 0)

            result = {
                'sample_idx': i,
                'customer_id': sample['customer_id'],
                'ground_truth': ground_truth,
                'target_hour': sample.get('target_hour'),
                'target_day_of_week': sample.get('target_day_of_week'),
                'candidates': candidates,
                'candidate_info': candidate_info,
                'picks': picks,
                'llm_interactions': llm_interactions,
                'hits': hits,
                'first_hit': first_hit,
                'recall': 1 if any(hits) else 0,
                'precision': sum(hits) / len(hits),
                'time_ms': elapsed,
            }
            results.append(result)

            self.logger.info(f"  Recall: {result['recall']}, Precision: {result['precision']:.2f}")

        # Compute aggregate metrics
        from agentic_recommender.evaluation.rerank_eval import RerankMetrics

        n = len(results)
        if n == 0:
            return RerankMetrics(total_samples=len(test_samples)), results

        metrics = RerankMetrics(
            recall_at_k=sum(r['recall'] for r in results) / n,
            precision_at_k=sum(r['precision'] for r in results) / n,
            first_hit_avg=sum(r['first_hit'] for r in results if r['first_hit'] > 0) / max(1, sum(1 for r in results if r['first_hit'] > 0)),
            ground_truth_in_candidates=sum(1 for r in results if not r['candidate_info']['ground_truth_added']) / n,
            avg_candidate_rank=sum(r['candidate_info']['ground_truth_rank'] for r in results) / n,
            total_samples=len(test_samples),
            valid_samples=n,
            avg_time_ms=total_time / n,
        )

        return metrics, results

    def _create_mock_rerank_provider(self):
        """Create a mock provider for rerank testing."""
        import random

        class MockRerankProvider:
            """Mock provider that picks from candidates."""

            def __init__(self):
                self.call_count = 0

            def generate(self, prompt: str, **kwargs) -> str:
                """Pick a random cuisine from the prompt's candidates."""
                self.call_count += 1

                # Extract candidates from prompt
                if "Available Options:" in prompt:
                    options_line = prompt.split("Available Options:")[1].split("\n")[1]
                    candidates = [c.strip() for c in options_line.split(",")]
                    if candidates:
                        return random.choice(candidates)

                return "chinese"  # fallback

            def get_model_info(self):
                return {
                    "provider": "MockRerank",
                    "model_name": "mock-rerank-llm",
                    "total_calls": self.call_count
                }

        return MockRerankProvider()

    # -------------------------------------------------------------------------
    # Stage 8: Enhanced Rerank Evaluation (Two-Round LLM + LightGCN)
    # -------------------------------------------------------------------------
    def stage_run_enhanced_rerank_evaluation(self) -> bool:
        """Run enhanced two-round rerank evaluation with LightGCN reflection."""
        stage_cfg = self.config.get_stage('run_enhanced_rerank_evaluation')
        self.logger.stage_start('run_enhanced_rerank_evaluation', stage_cfg.description)

        try:
            from agentic_recommender.evaluation.rerank_eval import (
                CuisineBasedCandidateGenerator,
                EnhancedRerankEvaluator,
                EnhancedRerankConfig,
                build_test_samples,
                build_test_samples_from_test_file,
            )
            from agentic_recommender.similarity.lightGCN import (
                LightGCNEmbeddingManager,
                LightGCNConfig,
                filter_interactions_leave_last_out,
                get_all_interactions,
            )
            from agentic_recommender.similarity.methods import CuisineSwingMethod, CuisineSwingConfig
            from agentic_recommender.models.llm_provider import OpenRouterProvider

            # Load merged data (training data)
            input_path = stage_cfg.input.get('merged_data')
            self.logger.file_read(input_path, "Loading merged data for enhanced rerank evaluation")
            merged_df = pd.read_parquet(input_path)
            self.logger.info(f"Loaded {len(merged_df):,} rows")

            # Get settings
            settings = stage_cfg.settings
            evaluation_method = settings.get('evaluation_method', 'method1')
            dataset_name = settings.get('dataset_name', 'data_se')
            enable_basket_metrics = settings.get('enable_basket_metrics', True)
            prediction_target = settings.get('prediction_target', 'cuisine')

            config = EnhancedRerankConfig(
                n_candidates=settings.get('n_candidates', 20),
                items_per_seed=settings.get('items_per_seed', 5),
                dataset_name=dataset_name,
                lightgcn_epochs=settings.get('lightgcn_epochs', 50),
                lightgcn_embedding_dim=settings.get('lightgcn_embedding_dim', 64),
                temperature_round1=settings.get('temperature_round1', 0.3),
                temperature_round2=settings.get('temperature_round2', 0.2),
                enable_thinking=settings.get('enable_thinking', True),
                n_samples=settings.get('n_samples', 10),
                min_history=settings.get('min_history', 5),
                deterministic_sampling=settings.get('deterministic_sampling', True),
                evaluation_method=evaluation_method,
                prediction_target=prediction_target,
                enable_basket_metrics=enable_basket_metrics,
            )

            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info(f"EVALUATION METHOD: {evaluation_method.upper()}")
            self.logger.info("=" * 60)

            # Build test samples and get interactions based on evaluation method
            if evaluation_method == 'method1':
                # Method 1: Leave-Last-Out on training data
                self.logger.info("Method 1: Leave-Last-Out (excluding last order per user)")

                # Build test samples with basket support
                test_samples = build_test_samples(
                    merged_df,
                    n_samples=config.n_samples,
                    min_history=config.min_history,
                    prediction_target=prediction_target,
                    return_basket=enable_basket_metrics,
                )
                self.logger.success(f"Created {len(test_samples)} test samples (leave-last-out)")

                # Get interactions excluding last order per user
                interactions = filter_interactions_leave_last_out(
                    merged_df,
                    prediction_target=prediction_target,
                )
                self.logger.info(f"Training interactions (N-1 orders): {len(interactions)}")

            elif evaluation_method == 'method2':
                # Method 2: Full training + test file
                self.logger.info("Method 2: Full Training + Test File Split")

                # Load test data
                test_data_path = stage_cfg.input.get('test_data')
                if not test_data_path:
                    self.logger.error("Method 2 requires test_data input path!")
                    self.logger.stage_end('run_enhanced_rerank_evaluation', success=False)
                    return False

                self.logger.file_read(test_data_path, "Loading test data")
                test_df = pd.read_parquet(test_data_path)
                self.logger.info(f"Loaded {len(test_df):,} test rows")

                # Build test samples from test file
                test_samples = build_test_samples_from_test_file(
                    train_df=merged_df,
                    test_df=test_df,
                    prediction_target=prediction_target,
                    n_samples=config.n_samples,
                    deterministic=config.deterministic_sampling,
                    min_history=config.min_history,
                )
                self.logger.success(f"Created {len(test_samples)} test samples from test file")

                # Get ALL interactions from training data
                interactions = get_all_interactions(
                    merged_df,
                    prediction_target=prediction_target,
                )
                self.logger.info(f"Training interactions (all): {len(interactions)}")

            else:
                self.logger.error(f"Unknown evaluation method: {evaluation_method}")
                self.logger.stage_end('run_enhanced_rerank_evaluation', success=False)
                return False

            # Save test samples (with custom serialization for sets)
            samples_path = stage_cfg.output.get('samples_json')
            if samples_path:
                # Convert sets to lists for JSON serialization
                serializable_samples = []
                for sample in test_samples:
                    s = sample.copy()
                    if 'ground_truth_items' in s:
                        s['ground_truth_items'] = list(s['ground_truth_items'])
                    serializable_samples.append(s)
                with open(samples_path, 'w') as f:
                    json.dump(serializable_samples, f, indent=2, default=str)
                self.logger.file_write(samples_path, f"{len(test_samples)} test samples")

            # Build/load cuisine-cuisine swing model with method-specific cache
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info(f"LOADING/TRAINING SWING MODEL")
            self.logger.info(f"  Method: {evaluation_method}, Target: {prediction_target}")
            self.logger.info("=" * 60)

            swing_config = CuisineSwingConfig(top_k=config.items_per_seed * 2)
            swing_model = CuisineSwingMethod(swing_config, prediction_target=prediction_target)

            loaded_from_cache = False
            if not self.no_cache:
                loaded_from_cache = swing_model.load_from_cache(dataset_name, evaluation_method)
            if not loaded_from_cache:
                self.logger.info(f"Training new Swing model (target={prediction_target})...")
                swing_model.fit(interactions)
                swing_model.save_to_cache(dataset_name, evaluation_method)

            # Create candidate generator with the swing model
            generator = CuisineBasedCandidateGenerator(config)
            generator.swing_model = swing_model
            generator._fitted = True

            gen_stats = generator.get_stats()
            self.logger.info(f"Generator stats: {gen_stats}")

            # Build/load LightGCN embeddings with method-specific cache
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info(f"LOADING/TRAINING LIGHTGCN")
            self.logger.info(f"  Method: {evaluation_method}, Target: {prediction_target}")
            self.logger.info("=" * 60)

            lightgcn_config = LightGCNConfig(
                embedding_dim=config.lightgcn_embedding_dim,
                epochs=config.lightgcn_epochs,
            )
            lightgcn_manager = LightGCNEmbeddingManager(lightgcn_config)
            lightgcn_manager.load_or_train(
                dataset_name=dataset_name,
                interactions=interactions,
                method=evaluation_method,
                prediction_target=prediction_target,
                force_retrain=self.no_cache,
                verbose=True
            )

            lgcn_stats = lightgcn_manager.get_stats()
            self.logger.info(f"LightGCN stats: {lgcn_stats}")

            # Initialize LLM provider
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("INITIALIZING LLM PROVIDER")
            self.logger.info("=" * 60)

            llm_config = self.config.get_llm_config()
            provider_type = llm_config.get('provider', 'mock')
            self.logger.info(f"Provider type: {provider_type}")

            if provider_type == 'mock':
                provider = self._create_mock_enhanced_rerank_provider()
            elif provider_type == 'openrouter':
                api_key = (
                    llm_config.get('openrouter', {}).get('api_key') or
                    os.environ.get('OPENROUTER_API_KEY')
                )
                if not api_key:
                    self.logger.error("OpenRouter API key not found!")
                    self.logger.stage_end('run_enhanced_rerank_evaluation', success=False)
                    return False

                provider = OpenRouterProvider(
                    api_key=api_key,
                    model_name=llm_config.get('openrouter', {}).get('model_name', 'google/gemini-2.0-flash-001')
                )
                self.logger.success(f"Initialized OpenRouter")
            else:
                from agentic_recommender.models.llm_provider import create_llm_provider
                provider = create_llm_provider(provider_type=provider_type)

            # Run evaluation
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("RUNNING ENHANCED TWO-ROUND EVALUATION")
            self.logger.info("=" * 60)
            self.logger.info(f"Samples: {len(test_samples)}, Candidates: {config.n_candidates}")

            evaluator = EnhancedRerankEvaluator(
                llm_provider=provider,
                candidate_generator=generator,
                lightgcn_manager=lightgcn_manager,
                config=config,
            )

            # Check for async mode
            enable_async = settings.get('enable_async', False)
            max_workers = settings.get('max_workers', 10)
            checkpoint_interval = settings.get('checkpoint_interval', 50)
            retry_attempts = settings.get('retry_attempts', 3)

            if enable_async and provider_type == 'openrouter':
                import asyncio
                self.logger.info("")
                self.logger.info("*" * 60)
                self.logger.info("*  ASYNC PARALLEL MODE ENABLED")
                self.logger.info("*" * 60)
                self.logger.info(f"*  Concurrent LLM workers: {max_workers}")
                self.logger.info(f"*  Checkpoint interval: {checkpoint_interval} samples")
                self.logger.info(f"*  Retry attempts: {retry_attempts}")
                self.logger.info(f"*  Results streaming to JSONL (memory efficient)")
                self.logger.info("*" * 60)
                self.logger.info("")

                # Get output directory for JSONL streaming
                output_dir = os.path.dirname(stage_cfg.output.get('detailed_json', 'outputs'))

                metrics, detailed_results = asyncio.run(
                    evaluator.evaluate_async(
                        test_samples=test_samples,
                        output_path=output_dir,
                        api_key=api_key,
                        max_workers=max_workers,
                        checkpoint_interval=checkpoint_interval,
                        retry_attempts=retry_attempts,
                        verbose=True
                    )
                )
            else:
                if enable_async:
                    self.logger.warning("Async mode only supported with OpenRouter provider")
                metrics, detailed_results = evaluator.evaluate(
                    test_samples=test_samples,
                    verbose=True
                )

            # Print results with side-by-side comparison
            self.logger.info("")
            self.logger.info("=" * 72)
            self.logger.info("ENHANCED RERANK EVALUATION RESULTS")
            self.logger.info("=" * 72)
            self.logger.info(str(metrics))  # Uses new comparison table format
            self.logger.info("")
            self.logger.info("LLM Provider info:")
            if hasattr(provider, 'get_model_info'):
                info = provider.get_model_info()
                self.logger.info(f"  Model: {info.get('model_name', 'unknown')}")
                self.logger.info(f"  Total calls: {info.get('total_calls', 0)}")

            # Save results
            results_path = stage_cfg.output.get('results_json')
            if results_path:
                with open(results_path, 'w') as f:
                    json.dump(metrics.to_dict(), f, indent=2)
                self.logger.file_write(results_path, "Enhanced rerank evaluation metrics")

            # Save detailed results
            detailed_path = stage_cfg.output.get('detailed_json')
            if detailed_path:
                with open(detailed_path, 'w') as f:
                    json.dump(detailed_results, f, indent=2, default=str)
                self.logger.file_write(detailed_path, f"{len(detailed_results)} detailed results")

                # Save preview
                self._save_preview(
                    data=detailed_results,
                    full_path=detailed_path,
                    preview_rows=3,
                    description="Preview of enhanced rerank detailed results",
                )

            self.logger.stage_end('run_enhanced_rerank_evaluation', success=True)
            return True

        except Exception as e:
            self.logger.error(f"Stage failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.stage_end('run_enhanced_rerank_evaluation', success=False)
            return False

    def _create_mock_enhanced_rerank_provider(self):
        """Create a mock provider for enhanced rerank testing."""
        import random
        import json as json_mod

        class MockEnhancedRerankProvider:
            """Mock provider that returns random rankings."""

            def __init__(self):
                self.call_count = 0

            def generate(self, prompt: str, **kwargs) -> str:
                """Generate mock ranking response."""
                self.call_count += 1

                # Try to extract candidates from prompt
                if "Candidates to Rank:" in prompt:
                    # Round 1: Return ranking JSON
                    lines = prompt.split("Candidates to Rank:")[1]
                    candidates_line = lines.split("\n")[1] if "\n" in lines else lines
                    candidates = [c.strip() for c in candidates_line.split(",") if c.strip()]
                    if candidates:
                        random.shuffle(candidates)
                        return json_mod.dumps({
                            "ranking": candidates,
                            "reasoning": "Mock ranking based on pattern analysis"
                        })

                elif "Initial LLM Ranking" in prompt:
                    # Round 2: Return final ranking JSON
                    # Extract from "Initial LLM Ranking" line
                    if "top 10):" in prompt:
                        r1_line = prompt.split("top 10):")[1].split("\n")[1]
                        cuisines = [c.strip() for c in r1_line.split(",") if c.strip()]
                        if cuisines:
                            random.shuffle(cuisines)
                            return json_mod.dumps({
                                "final_ranking": cuisines,
                                "reflection": "Balanced initial intuition with CF signals"
                            })

                # Fallback
                return json_mod.dumps({"ranking": ["chinese"], "reasoning": "fallback"})

            def get_model_info(self):
                return {
                    "provider": "MockEnhancedRerank",
                    "model_name": "mock-enhanced-rerank-llm",
                    "total_calls": self.call_count
                }

        return MockEnhancedRerankProvider()


# =============================================================================
# WORKFLOW RUNNER
# =============================================================================

class WorkflowRunner:
    """Main workflow runner that orchestrates all stages."""

    STAGE_ORDER = [
        'load_data',
        'build_users',
        'build_cuisines',
        'generate_prompts',
        'run_predictions',
        'run_topk_evaluation',              # Stage 6: Direct LLM prediction
        'run_rerank_evaluation',            # Stage 7: Retrieve-then-rerank
        'run_enhanced_rerank_evaluation',   # Stage 8: Two-round LLM + LightGCN (RECOMMENDED)
    ]

    def __init__(self, config_path: str = "workflow_config.yaml", no_cache: bool = False):
        self.config = WorkflowConfig(config_path)
        self.no_cache = no_cache

        # Generate run timestamp and create output subfolder
        self.run_timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.run_output_dir = self._setup_run_directory()

        # Rewrite output paths in config to use the timestamped subfolder
        self._rewrite_output_paths()

        # Save runtime config to output folder and print to console
        self._save_runtime_config()

        # Initialize logger with updated log file path
        self.logger = WorkflowLogger(
            log_file=self.config.get_log_file(),
            verbose=self.config.is_verbose()
        )
        self.stages = PipelineStages(self.config, self.logger, no_cache=no_cache)

        if no_cache:
            self.logger.warning("--no-cache: All caches disabled. Stage outputs, Swing, and LightGCN will be regenerated.")

        # Map stage names to methods
        self.stage_methods = {
            'load_data': self.stages.stage_load_data,
            'build_users': self.stages.stage_build_users,
            'build_cuisines': self.stages.stage_build_cuisines,
            'generate_prompts': self.stages.stage_generate_prompts,
            'run_predictions': self.stages.stage_run_predictions,
            'run_topk_evaluation': self.stages.stage_run_topk_evaluation,
            'run_rerank_evaluation': self.stages.stage_run_rerank_evaluation,
            'run_enhanced_rerank_evaluation': self.stages.stage_run_enhanced_rerank_evaluation,
        }

    def _setup_run_directory(self) -> str:
        """Create timestamped output subfolder for this run."""
        base_dir = self.config.get_output_base_dir()
        run_dir = os.path.join(base_dir, self.run_timestamp)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        return run_dir

    def _rewrite_output_paths(self) -> None:
        """Rewrite all output and input paths in config to use the timestamped subfolder."""
        base_dir = self.config.get_output_base_dir()

        # Rewrite log file path
        log_file = self.config.config.get('workflow', {}).get('log_file')
        if log_file:
            new_log_file = self._rewrite_path(log_file, base_dir)
            self.config.config['workflow']['log_file'] = new_log_file

        # Rewrite all stage input and output paths
        stages = self.config.config.get('workflow', {}).get('stages', {})
        for stage_name, stage_config in stages.items():
            # Rewrite output paths
            output_config = stage_config.get('output', {})
            for key, path in output_config.items():
                if path and isinstance(path, str):
                    new_path = self._rewrite_path(path, base_dir)
                    output_config[key] = new_path

            # Rewrite input paths that reference outputs from this workflow
            # (paths starting with base_dir are workflow outputs)
            input_config = stage_config.get('input', {})
            for key, path in input_config.items():
                if path and isinstance(path, str) and path.startswith(base_dir):
                    new_path = self._rewrite_path(path, base_dir)
                    input_config[key] = new_path

    def _rewrite_path(self, original_path: str, base_dir: str) -> str:
        """Rewrite a single path to use the timestamped subfolder."""
        # If path starts with base_dir, replace it with run_output_dir
        if original_path.startswith(base_dir + "/") or original_path.startswith(base_dir + os.sep):
            relative_path = original_path[len(base_dir):].lstrip("/").lstrip(os.sep)
            return os.path.join(self.run_output_dir, relative_path)
        elif original_path.startswith(base_dir):
            # Handle case like "outputs/file.json" where base_dir is "outputs"
            relative_path = original_path[len(base_dir):].lstrip("/").lstrip(os.sep)
            return os.path.join(self.run_output_dir, relative_path)
        else:
            # Path doesn't start with base_dir, just prepend the run directory
            return os.path.join(self.run_output_dir, os.path.basename(original_path))

    def _save_runtime_config(self) -> None:
        """Save the runtime config to output folder and print to console."""
        # Save config to file
        config_dst = os.path.join(self.run_output_dir, "runtime_config.yaml")
        with open(config_dst, 'w') as f:
            yaml.dump(self.config.config, f, default_flow_style=False, sort_keys=False)

        # Print config to console
        print("")
        print("=" * 80)
        print("RUNTIME CONFIGURATION")
        print("=" * 80)
        print(yaml.dump(self.config.config, default_flow_style=False, sort_keys=False))
        print("=" * 80)
        print(f"Config saved to: {config_dst}")
        print("")

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Run the workflow.

        Args:
            stages: List of stage names to run. If None, runs all enabled stages.

        Returns:
            Dictionary mapping stage names to success status.
        """
        self.logger.info("")
        self.logger.info("#" * 80)
        self.logger.info("#" + " " * 78 + "#")
        self.logger.info("#" + "  WORKFLOW RUNNER".center(78) + "#")
        self.logger.info("#" + f"  {self.config.config['workflow']['name']}".center(78) + "#")
        self.logger.info("#" + " " * 78 + "#")
        self.logger.info("#" * 80)
        self.logger.info("")
        self.logger.info(f"Run timestamp: {self.run_timestamp}")
        self.logger.info(f"Output directory: {self.run_output_dir}")

        # Determine which stages to run
        if stages:
            stages_to_run = stages
            self.logger.info(f"\nRunning specified stages: {stages_to_run}")
        else:
            stages_to_run = [s for s in self.STAGE_ORDER
                           if self.config.get_stage(s).enabled]
            self.logger.info(f"\nRunning enabled stages: {stages_to_run}")

        # Show stage plan
        self.logger.info("\nStage Plan:")
        for i, stage in enumerate(stages_to_run, 1):
            cfg = self.config.get_stage(stage)
            self.logger.info(f"  {i}. {stage}: {cfg.description}")

        # Run stages
        results = {}
        start_time = time.time()

        for stage_name in stages_to_run:
            if stage_name not in self.stage_methods:
                self.logger.warning(f"Unknown stage: {stage_name}")
                results[stage_name] = False
                continue

            method = self.stage_methods[stage_name]
            results[stage_name] = method()

            # Stop on failure unless we want to continue
            if not results[stage_name]:
                self.logger.error(f"Stage {stage_name} failed. Stopping workflow.")
                break

        # Final summary
        total_time = time.time() - start_time

        self.logger.info("")
        self.logger.info("#" * 80)
        self.logger.info("WORKFLOW COMPLETE")
        self.logger.info("#" * 80)
        self.logger.info(f"\nTotal time: {total_time:.2f}s")
        self.logger.info("\nResults:")
        for stage, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"  {stage}: {status}")

        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the data processing workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python workflow_runner.py                                  # Run all enabled stages
    python workflow_runner.py --stages load_data               # Run single stage
    python workflow_runner.py --stages build_users generate_prompts  # Multiple stages
    python workflow_runner.py --config custom_config.yaml      # Use custom config
    python workflow_runner.py --list                           # List all stages
        """
    )

    parser.add_argument(
        '--config', '-c',
        default='workflow_config.yaml',
        help='Path to workflow configuration YAML file'
    )

    parser.add_argument(
        '--stages', '-s',
        nargs='+',
        help='Specific stages to run (space-separated)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available stages and exit'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable all caches and force regeneration of all stage outputs, Swing, and LightGCN models'
    )

    args = parser.parse_args()

    # List stages if requested (use WorkflowConfig directly to avoid creating directories)
    if args.list:
        config = WorkflowConfig(args.config)
        print("\nAvailable Stages:")
        print("-" * 60)
        for stage_name in WorkflowRunner.STAGE_ORDER:
            cfg = config.get_stage(stage_name)
            status = "ENABLED" if cfg.enabled else "DISABLED"
            print(f"  [{status:8}] {stage_name}")
            print(f"             {cfg.description}")
            print()
        return

    # Initialize workflow
    runner = WorkflowRunner(args.config, no_cache=args.no_cache)

    # Run workflow
    results = runner.run(stages=args.stages)

    # Exit with error code if any stage failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
