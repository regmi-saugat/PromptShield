import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from src.utils.tokenizer import build_tokenizer
from src.config import LoRAConfig


### Build PEFT model with LoRA adapters for sequence classification.

class LoRASecurityModel:

    def __init__(self, base_model: str, num_labels: int, lora_cfg: LoRAConfig):
        self.tokenizer = build_tokenizer(base_model)
        self.base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
        )
        self.model = get_peft_model(self.base, peft_cfg)
        self.model.print_trainable_parameters()

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
