from transformers import AutoTokenizer

from loguru import logger


def build_tokenizer(
    model_path: str, base_model_name: str | None = None
) -> AutoTokenizer:
    identifier = (base_model_name or model_path).lower()
    kwargs = {"add_prefix_space": True} if "roberta" in identifier else {}
    tok = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logger.info(f"Tokenizer loaded from {model_path}")
    return tok
