import random
from collections import Counter
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple


class JailbreakDatasetLoader:
    """Load and balance multiple open-source security datasets."""

    CONFIGS = {
        # https://huggingface.co/datasets/lmsys/toxic-chat ==> hf bata ko dataset
        "toxic-chat": {
            "name": "lmsys/toxic-chat",
            "config": "toxicchat0124",
            "text_col": "user_input",
            "label_col": "toxicity",
            "type": "toxicity",
        },
        # https://huggingface.co/datasets/OpenSafetyLab/Salad-Data
        "salad-data": {
            "name": "OpenSafetyLab/Salad-Data",
            "config": "attack_enhanced_set",
            "text_col": "attack",
            "label_col": None,
            "type": "jailbreak",
        },
    }

    def __init__(self, max_per_source: int | None = None):
        self.max_per_source = max_per_source

    # ------------------------------------------------------------------
    def load(self, max_total: int = 1_000) -> Dict[str, Tuple[List[str], List[int]]]:
        texts, labels = [], []
        per_source = (max_total // len(self.CONFIGS)) if max_total else None
        max_per_class = per_source // 2 if per_source else None  # 50/50 per source

        for key in self.CONFIGS:
            t, l = self._load_single(key, max_per_class)
            texts.extend(t)
            labels.extend(l)

        # final global balance (extra safety)
        texts, labels = self._balance(texts, labels)
        logger.info(f"Final pool: {Counter(labels)}")

        label2id = {lbl: idx for idx, lbl in enumerate(sorted(set(labels)))}
        label_ids = [label2id[l] for l in labels]

        train_txt, tmp_txt, train_lbl, tmp_lbl = train_test_split(
            texts, label_ids, test_size=0.4, random_state=42, stratify=label_ids
        )
        val_txt, test_txt, val_lbl, test_lbl = train_test_split(
            tmp_txt, tmp_lbl, test_size=0.5, random_state=42, stratify=tmp_lbl
        )

        return {
            "train": (train_txt, train_lbl),
            "validation": (val_txt, val_lbl),
            "test": (test_txt, test_lbl),
            "label2id": label2id,
            "id2label": {i: l for l, i in label2id.items()},
        }

    # ------------------------------------------------------------------
    def _load_single(
        self, key: str, max_per_class: int | None
    ) -> Tuple[List[str], List[str]]:
        cfg = self.CONFIGS[key]
        logger.info(f"Loading {key} (max {max_per_class} per class)")
        ds = load_dataset(cfg["name"], cfg["config"])[
            "train" if "train" in load_dataset(cfg["name"], cfg["config"]) else "test"
        ]

        texts, labels = [], []
        j_count = b_count = 0

        for row in ds:
            # hard cap per class
            if max_per_class and j_count >= max_per_class and b_count >= max_per_class:
                break

            txt = row.get(cfg["text_col"], "")
            if not txt or not txt.strip():
                continue

            # stricter toxicity threshold
            if cfg["type"] == "jailbreak":
                label = "jailbreak"
            elif cfg["type"] == "toxicity":
                tox = row.get(cfg["label_col"], 0)
                label = "jailbreak" if tox >= 0.7 else "benign"
            else:
                label = "benign"

            # respect per-class cap
            if label == "jailbreak" and max_per_class and j_count >= max_per_class:
                continue
            if label == "benign" and max_per_class and b_count >= max_per_class:
                continue

            texts.append(txt)
            labels.append(label)
            j_count += label == "jailbreak"
            b_count += label == "benign"

        logger.info(f"{key} -> {Counter(labels)}")
        return texts, labels

    # ------------------------------------------------------------------
    @staticmethod
    def _balance(texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        j = [(t, l) for t, l in zip(texts, labels) if l == "jailbreak"]
        b = [(t, l) for t, l in zip(texts, labels) if l == "benign"]
        m = min(len(j), len(b))
        if m == 0:
            return texts, labels
        balanced = j[:m] + b[:m]
        # optional: shuffle to avoid ordered blocks
        random.shuffle(balanced)
        return [t[0] for t in balanced], [t[1] for t in balanced]
