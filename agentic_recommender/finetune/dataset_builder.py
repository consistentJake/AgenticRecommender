"""Utilities to construct ThinkRec-style finetuning corpora from Agentic datasets."""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import Dataset, DatasetDict

from agentic_recommender.datasets.base_dataset import SequentialDataset


@dataclass
class AgenticDatasetConfig:
    name: str
    data_root: Path
    include_reasoning: bool = True
    negatives_per_positive: int = 1
    seed: int = 42


class AgenticCorpusBuilder:
    """Convert preprocessed sequential sessions into SFT-friendly records."""

    def __init__(self, dataset: SequentialDataset, item_map: Dict[str, str], config: AgenticDatasetConfig):
        self.dataset = dataset
        self.item_map = item_map
        self.config = config
        self.rng = random.Random(config.seed)

    def build_split(self, sessions: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for session in sessions:
            # Defensive: ensure required fields exist and history is non-empty
            items = session.get("items", [])
            if not items or len(items) < 2:
                continue

            history, target = self.dataset.prepare_to_predict(session)
            if not history:
                continue

            history_titles = [self.item_map.get(str(item), str(item)) for item in history]
            target_key = str(target)
            target_title = self.item_map.get(target_key, target_key)
            prompt = self._format_prompt(history_titles, target_title)

            # Positive classification example
            records.append(
                {
                    "text": f"{prompt} Yes",
                    "mode": "classification",
                    "label": 1,
                    "target_item": str(target),
                    "user_id": session.get("user_id"),
                }
            )

            # Optional reasoning example accompanying the positive case
            if self.config.include_reasoning:
                reason_text = self._format_reasoning(history_titles, target_title, positive=True)
                records.append(
                    {
                        "text": f"{prompt} {reason_text}",
                        "mode": "reasoning",
                        "label": 1,
                        "target_item": str(target),
                        "user_id": session.get("user_id"),
                    }
                )

            # Negative samples for contrastive classification
            negatives = self.dataset.negative_sample(session.get("user_id"), items)
            if negatives:
                neg_choices = self._sample_negatives(negatives)
                for neg_item in neg_choices:
                    neg_key = str(neg_item)
                    neg_title = self.item_map.get(neg_key, neg_key)
                    neg_prompt = self._format_prompt(history_titles, neg_title)
                    records.append(
                        {
                            "text": f"{neg_prompt} No",
                            "mode": "classification",
                            "label": 0,
                            "target_item": str(neg_item),
                            "user_id": session.get("user_id"),
                        }
                    )
                    if self.config.include_reasoning:
                        neg_reason = self._format_reasoning(history_titles, neg_title, positive=False)
                        records.append(
                            {
                                "text": f"{neg_prompt} {neg_reason}",
                                "mode": "reasoning",
                                "label": 0,
                                "target_item": str(neg_item),
                                "user_id": session.get("user_id"),
                            }
                        )
        return records

    def _sample_negatives(self, negatives: Sequence[str]) -> Iterable[str]:
        if self.config.negatives_per_positive <= 0:
            return []
        num = min(self.config.negatives_per_positive, len(negatives))
        return self.rng.sample(list(negatives), num)

    def _format_prompt(self, history_titles: List[str], target_title: str) -> str:
        history_str = ", ".join(history_titles)
        return (
            "User history: "
            f"{history_str}. "
            f"Should we recommend '{target_title}'? Answer:"
        )

    def _format_reasoning(self, history_titles: List[str], target_title: str, *, positive: bool) -> str:
        recent = history_titles[-min(3, len(history_titles)) :]
        recent_str = ", ".join(recent)
        if positive:
            return (
                "Yes. The user's recent preferences include "
                f"{recent_str}, which align with '{target_title}'."
            )
        return (
            "No. The user's recent preferences include "
            f"{recent_str}, which differ from '{target_title}'."
        )


def build_agentic_dataset(config: AgenticDatasetConfig) -> DatasetDict:
    data_root = config.data_root
    dataset_prefix = config.name

    dataset_pkl = data_root / f"{dataset_prefix}_dataset.pkl"
    item_map_path = data_root / f"{dataset_prefix}_item_to_name.json"

    if not dataset_pkl.exists():
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_pkl}")
    if not item_map_path.exists():
        raise FileNotFoundError(f"Item-to-name mapping not found: {item_map_path}")

    with open(dataset_pkl, "rb") as f:
        sequential_dataset: SequentialDataset = pickle.load(f)

    with open(item_map_path, "r") as f:
        item_map: Dict[str, str] = json.load(f)

    builder = AgenticCorpusBuilder(sequential_dataset, item_map, config)

    dataset_dict: Dict[str, Dataset] = {}
    for split in ("train", "val", "test"):
        split_path = data_root / f"{dataset_prefix}_{split}.json"
        if not split_path.exists():
            continue
        with open(split_path, "r") as f:
            sessions = json.load(f)
        records = builder.build_split(sessions)
        if records:
            dataset_dict[split] = Dataset.from_list(records)

    if not dataset_dict:
        raise RuntimeError("No splits were constructed; check data availability.")

    return DatasetDict(dataset_dict)
