# Download ModernBERT-base to ./modernbert-base

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "answerdotai/ModernBERT-base"
LOCAL_DIR = Path.cwd() / "modernbert-base"


def download_model(model_id: str, local_dir: Path) -> Path:
    """Download tokenizer + model weights into *local_dir* and return the dir."""
    local_dir.mkdir(exist_ok=True)

    print("************************** Downloading tokenizer **************************")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(local_dir)

    print(
        "************************** Downloading model weights **************************"
    )
    model = AutoModel.from_pretrained(model_id)
    model.save_pretrained(local_dir)

    print(f"Model saved to: {local_dir.resolve()}")
    return local_dir.resolve()


if __name__ == "__main__":
    path = download_model(MODEL_ID, LOCAL_DIR)
    print(
        "\nNow use:\n"
        f"  model = AutoModel.from_pretrained(r'{path}')\n"
        f"  tok   = AutoTokenizer.from_pretrained(r'{path}')"
    )
