"""
anime_translator_infer.py
========================

This script loads a base mBART‑50 model and a LoRA adapter trained for
Japanese→Korean translation and exposes a simple inference function to
translate Japanese anime titles (or other short texts) into Korean.

Example usage at the command line::

    python anime_translator_infer.py "カウボーイビバップ"

You can also import :func:`translate_title` from this module and call it
from your own code.

The LoRA adapter directory defaults to ``mbart_ja2ko_title_lora_mps/adapter``
but can be overridden by setting the ``MBART_ADAPTER_DIR`` environment
variable or by passing a different value to :func:`load_translator`.
"""

import os
import re
import sys
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def select_device() -> str:
    """Select an available device in the order MPS, CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_translator(
    adapter_dir: str = None,
    base_model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
    device: str = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load the base mBART model and apply a LoRA adapter from ``adapter_dir``.

    :param adapter_dir: Path to the directory containing the LoRA adapter weights.
                        Defaults to the value of ``MBART_ADAPTER_DIR`` env var or
                        ``mbart_ja2ko_title_lora_mps/adapter``.
    :param base_model_name: Name of the Hugging Face mBART model to load.
    :param device: Device identifier (e.g. ``"cuda"`` or ``"cpu"``).  If ``None``,
                   selects automatically.
    :returns: A tuple of the loaded model (with adapter) and tokenizer.
    """
    # Determine adapter directory
    adapter_dir = adapter_dir or os.getenv(
        "MBART_ADAPTER_DIR", "mbart_ja2ko_title_lora_mps/adapter"
    )
    device = device or select_device()
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    base_model.to(device)
    # Load LoRA adapter on top of base model
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    return model, tokenizer


def translate_title(
    model: torch.nn.Module, tokenizer: AutoTokenizer, title: str, device: str = None
) -> str:
    """Translate a Japanese title into Korean using the loaded model.

    :param model: The fine‑tuned seq2seq model with LoRA adapter attached.
    :param tokenizer: The associated tokenizer.
    :param title: Japanese text to translate.
    :param device: Device to run inference on.  Defaults to the device used
                   when loading the model.
    :returns: Translated Korean text.
    """
    device = device or select_device()
    tokenizer.src_lang = "ja_XX"
    # Prepare input
    enc = tokenizer(title, return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            **enc,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("ko_KR"),
            num_beams=5,
            length_penalty=1.1,
            max_new_tokens=64,
            early_stopping=True,
        )
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    # Remove any boilerplate that the model might prepend (e.g., "한국어:" or "번역:")
    text = re.sub(r"^(한국어|번역).*?:\s*", "", text).strip()
    return text


def main(argv=None):  # pragma: no cover
    """Allow command line translation when the script is run directly."""
    argv = argv or sys.argv[1:]
    if not argv:
        print(
            "Usage: python anime_translator_infer.py \"<Japanese text>\""
        )
        return
    title = " ".join(argv)
    model, tokenizer = load_translator()
    translation = translate_title(model, tokenizer, title)
    print(title, "→", translation)


if __name__ == "__main__":  # pragma: no cover
    main()