"""
dataset.py — Custom PyTorch Dataset for LoRA finetuning of Qwen3-VL-8B-Instruct.

Provides:
    DETECTION_PROMPT  — canonical multi-object detection prompt (module-level constant)
    LoRAFinetuneDataset — torch.utils.data.Dataset subclass
    collate_fn        — padding collate function for DataLoader
"""
import json
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Canonical detection prompt (must match training data exactly)
# Import this constant in any module that needs to query the finetuned model.
# ---------------------------------------------------------------------------
DETECTION_PROMPT = """You are controlling a robot navigating in an indoor environment.

Detect all visible instances of these objects: couch, table, chair, bookshelf, cardboard box, trash can.

Return ONLY a valid JSON array:
[{"bbox": [x1, y1, x2, y2], "label": "object name"}, ...]

Rules:
- Coordinates are in 0-1000 scale relative to image dimensions
- Tightly bound the VISIBLE portion of each object
- Return one entry per detected instance
- If none visible: []
"""


class LoRAFinetuneDataset(Dataset):
    """
    Dataset for LoRA finetuning of Qwen3-VL-8B-Instruct on egocentric bbox detection.

    Each sample is read from a pre-split JSON file produced by prepare_dataset.py.
    The JSON entries have:
        "id": str
        "image": str  (absolute path to raw.png)
        "conversations": [
            {"role": "user", "content": <prompt string>},
            {"role": "assistant", "content": <JSON string of detections>},
        ]

    __getitem__ returns:
        {
            "input_ids":      torch.LongTensor of shape [seq_len]
            "attention_mask": torch.LongTensor of shape [seq_len]
            "labels":         torch.LongTensor of shape [seq_len]
                              (user-turn tokens masked to -100; assistant tokens kept)
        }
    """

    def __init__(self, split_json_path: str, processor, max_seq_length: int = 2048):
        """
        Args:
            split_json_path: Path to train_split.json or val_split.json
            processor: HuggingFace AutoProcessor for Qwen3-VL
            max_seq_length: Maximum sequence length for tokenization
        """
        with open(split_json_path, "r") as f:
            self.samples = json.load(f)
        self.processor = processor
        self.max_seq_length = max_seq_length

    @property
    def column_names(self):
        return None

    def map(self, *args, **kwargs):
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        sample = self.samples[idx]
        img_path = sample["image"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found for sample '{sample['id']}': {img_path}"
            )

        image = Image.open(img_path).convert("RGB")
        assistant_content = sample["conversations"][1]["content"]

        # Build Qwen3-VL chat messages (user turn + assistant turn)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ]

        # Full sequence tokenization
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        encoding = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        # Build labels: mask user turn tokens with -100
        labels = input_ids.clone()

        # Find user-turn length by tokenizing just the user prompt
        user_only_messages = [messages[0]]
        user_text = self.processor.apply_chat_template(
            user_only_messages, tokenize=False, add_generation_prompt=True
        )
        user_encoding = self.processor(
            text=[user_text],
            images=image_inputs,
            videos=None,
            return_tensors="pt",
            padding=False,
        )
        user_len = user_encoding["input_ids"].shape[1]

        # Clamp to actual sequence length (truncation may have shortened it)
        user_len = min(user_len, labels.shape[0])
        labels[:user_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of dataset items into a padded batch.

    Pads input_ids and attention_mask with 0; pads labels with -100.

    Returns:
        dict with keys "input_ids", "attention_mask", "labels",
        each a LongTensor of shape [batch_size, max_len].
    """
    max_len = max(x["input_ids"].shape[0] for x in batch)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        padded_input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }
