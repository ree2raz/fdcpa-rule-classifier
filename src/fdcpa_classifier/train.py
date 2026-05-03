"""QLoRA fine-tuning pipeline for FDCPA rule classification."""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from fdcpa_classifier import DATA_DIR, load_jsonl
from fdcpa_classifier.prompts import format_chat_messages

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = Path("./checkpoints/fdcpa-classifier-qlora")
FINAL_DIR = Path("./checkpoints/final")
MAX_SEQ_LENGTH = 2048


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    return model, tokenizer


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def get_training_args(output_dir: str | Path = OUTPUT_DIR) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        max_grad_norm=1.0,
        seed=42,
        dataloader_pin_memory=False,
    )


def format_for_training(example: dict, tokenizer) -> str:
    rule = {
        "rule_id": example["rule_id"],
        "rule_name": example.get("rule_name", ""),
        "description": example.get("description", ""),
    }

    rubric_lookup = {r["rule_id"]: r for r in __import__("fdcpa_classifier").FDCPA_RUBRIC}
    if example["rule_id"] in rubric_lookup:
        rule = rubric_lookup[example["rule_id"]]

    messages = format_chat_messages(
        rule=rule,
        transcript_chunk=example["transcript_chunk"],
        verdict=example["verdict"],
        reasoning=example.get("reasoning", ""),
    )
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def build_dataset(split: str, tokenizer) -> Dataset:
    path = DATA_DIR / f"{split}.jsonl"
    examples = load_jsonl(path)
    formatted = [format_for_training(ex, tokenizer) for ex in examples]
    ds = Dataset.from_dict({"text": formatted})
    return ds


def train():
    import wandb

    wandb.init(
        project="fdcpa-rule-classifier",
        name=f"qlora-r16-a32-{__import__('datetime').datetime.now().strftime('%Y%m%d-%H%M')}",
    )

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    print("Loading datasets...")
    train_ds = build_dataset("train", tokenizer)
    val_ds = build_dataset("val", tokenizer)
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = get_training_args()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final adapter...")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")

    wandb.log(metrics)
    wandb.finish()

    return metrics


if __name__ == "__main__":
    train()
