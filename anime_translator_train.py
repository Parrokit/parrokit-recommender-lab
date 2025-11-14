"""
anime_translator_train.py
=========================

This script performs fine‑tuning of an mBART‑50 model from the
Transformers library for Japanese→Korean translation using LoRA
(Low‑Rank Adaptation) for parameter‑efficient training.  It prepares
a small in‑memory training/validation dataset of anime titles, configures
and attaches a LoRA adapter, trains the model using the Hugging Face
``Seq2SeqTrainer``, and writes the resulting adapter weights to disk.

Usage::

    python anime_translator_train.py

The adapter weights will be saved into a directory called
``mbart_ja2ko_title_lora_mps/adapter`` relative to the working directory.

This file contains only the training logic; see ``anime_translator_infer.py``
for code that loads the adapter and performs translation.
"""

import os
import random
from typing import List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model


def prepare_dataset(pairs: List[Tuple[str, str]], split_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
    """Shuffle a list of (source, target) pairs and split into training and validation datasets."""
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split]
    valid_pairs = pairs[split:]
    train_ds = Dataset.from_dict({"ja": [j for j, _ in train_pairs], "ko": [k for _, k in train_pairs]})
    valid_ds = Dataset.from_dict({"ja": [j for j, _ in valid_pairs], "ko": [k for _, k in valid_pairs]})
    return train_ds, valid_ds


def build_model(device: str = "cpu"):  # pragma: no cover
    """Load the base mBART model and wrap it with a LoRA adapter configured for seq2seq tasks.

    Returns the wrapped model, tokenizer and LoRA configuration.
    """
    MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    base_model.to(device)
    lora_cfg = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def preprocess_function(examples, tokenizer, max_src: int = 64, max_tgt: int = 64):
    """Tokenize source and target texts for training."""
    tokenizer.src_lang = "ja_XX"
    tokenizer.tgt_lang = "ko_KR"
    model_inputs = tokenizer(examples["ja"], max_length=max_src, truncation=True)
    labels = tokenizer(text_target=examples["ko"], max_length=max_tgt, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(model, tokenizer, train_ds: Dataset, valid_ds: Dataset, output_dir: str = "mbart_ja2ko_title_lora_mps", device: str = "cpu"):  # pragma: no cover
    """Configure the training loop and run fine‑tuning."""
    max_src, max_tgt = 64, 64
    # Tokenize datasets
    train_tok = train_ds.map(
        lambda batch: preprocess_function(batch, tokenizer, max_src, max_tgt),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    valid_tok = valid_ds.map(
        lambda batch: preprocess_function(batch, tokenizer, max_src, max_tgt),
        batched=True,
        remove_columns=valid_ds.column_names,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=8,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=64,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    # Save only the LoRA adapter weights
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    return adapter_path


def main():  # pragma: no cover
    """Entry point for running fine‑tuning from the command line."""
    # Force MPS fallback on Apple Silicon for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Select device
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("[device]", device)
    # Hardcoded mini dataset of Japanese and Korean anime titles
    pairs: List[Tuple[str, str]] = [
        ("カウボーイビバップ", "카우보이 비밥"),
        ("カウボーイビバップ 天国の扉", "카우보이 비밥 천국의 문"),
        ("トライガン", "트라이건"),
        ("新世紀エヴァンゲリオン", "신세기 에반게리온"),
        ("ナルト", "나루토"),
        ("ONE PIECE", "원피스"),
        ("テニスの王子様", "테니스의 왕자"),
        ("スクールランブル", "스쿨 럼블"),
        ("頭文字〈イニシャル〉D", "이니셜 D"),
        ("頭文字〈イニシャル〉D FOURTH STAGE", "이니셜 D 포스 스테이지"),
        ("ハングリーハート", "헝그리 하트"),
        ("ハングリーハート Wild Striker", "헝그리 하트 와일드 스트라이커"),
        ("ハチミツとクローバー", "허니와 클로버"),
        ("モンスター", "몬스터"),
        ("冒険王ビィト", "모험왕 비트"),
        ("アイシールド21", "아이실드 21"),
        ("機動戦士ガンダム", "기동전사 건담"),
        ("コードギアス 反逆のルルーシュ", "코드 기아스 반역의 를르슈"),
        ("魔法少女まどか☆マギカ", "마법소녀 마도카☆마기카"),
        ("ジパング", "지팡"),
        ("進撃の巨人", "진격의 거인"),
        ("鬼滅の刃", "귀멸의 칼날"),
        ("SPY×FAMILY", "스파이 패밀리"),
        ("ジョジョの奇妙な冒険", "죠죠의 기묘한 모험"),
        ("銀魂", "은혼"),
        ("鋼の錬金術師", "강철의 연금술사"),
        ("デスノート", "데스노트"),
        ("ソードアート・オンライン", "소드 아트 온라인"),
        ("Re:ゼロから始める異世界生活", "Re:제로부터 시작하는 이세계 생활"),
        ("この素晴らしい世界に祝福を！", "이 멋진 세계에 축복을!"),
        ("ノーゲーム・ノーライフ", "노 게임 노 라이프"),
        ("涼宮ハルヒの憂鬱", "스즈미야 하루히의 우울"),
        ("らき☆すた", "러키☆스타"),
        ("けいおん！", "케이온!"),
        ("シュタインズ・ゲート", "슈타인즈 게이트"),
        ("攻殻機動隊", "공각기동대"),
        ("サイコパス", "사이코패스"),
        ("プラスティック・メモリーズ", "플라스틱 메모리즈"),
        ("ヴァイオレット・エヴァーガーデン", "바이올렛 에버가든"),
        ("四月は君の嘘", "4월은 너의 거짓말"),
        ("化物語", "바케모노가타리"),
        ("とある科学の超電磁砲", "어떤 과학의 초전자포"),
        ("とある魔術の禁書目録", "어떤 마법의 금서목록"),
        ("五等分の花嫁", "5등분의 신부"),
    ]
    train_ds, valid_ds = prepare_dataset(pairs)
    print("train/valid:", len(train_ds), len(valid_ds))
    model, tokenizer = build_model(device)
    adapter_path = train_model(model, tokenizer, train_ds, valid_ds, device=device)
    print(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":  # pragma: no cover
    main()