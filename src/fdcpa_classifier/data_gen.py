"""Dataset generation using OpenAI GPT-4.1-mini as teacher model."""

import json
import os
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from fdcpa_classifier import FDCPA_RUBRIC, DATA_DIR, save_jsonl
from fdcpa_classifier.prompts import TEACHER_SYSTEM, format_teacher_prompt

CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4.1-mini"


def call_llm(system: str, user: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = (attempt + 1) * 5
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_generated_examples(raw_text: str) -> list[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    print("  Failed to parse JSON from model output")
    return []


def generate_examples_for_rule(
    rule: dict,
    n_easy: int = 15,
    n_medium: int = 8,
    n_hard: int = 5,
    seed_examples: list[dict] | None = None,
) -> list[dict]:
    if seed_examples is None:
        seed_examples = []

    rule_seeds = [s for s in seed_examples if s["rule_id"] == rule["rule_id"]]

    all_examples = []
    batches_needed = max(1, (n_easy + n_medium + n_hard) // 5)

    for batch_idx in range(batches_needed):
        prompt = format_teacher_prompt(rule, rule_seeds)
        raw = call_llm(TEACHER_SYSTEM, prompt)
        examples = parse_generated_examples(raw)

        for ex in examples:
            ex["rule_id"] = rule["rule_id"]
            ex["rule_name"] = rule["rule_name"]
            ex["source"] = "gpt-generated"
            ex["generation_timestamp"] = datetime.now(timezone.utc).isoformat()
            if "difficulty" not in ex:
                ex["difficulty"] = "medium"
            if "verdict" not in ex or ex["verdict"] not in ("pass", "fail"):
                continue
            if "transcript_chunk" not in ex or len(ex["transcript_chunk"]) < 100:
                continue
            all_examples.append(ex)

        if batch_idx < batches_needed - 1:
            time.sleep(1)

    return all_examples


def fuzzy_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def deduplicate(examples: list[dict], threshold: float = 0.85) -> list[dict]:
    unique = []
    for ex in examples:
        chunk = ex["transcript_chunk"]
        is_dup = False
        for existing in unique:
            if fuzzy_similarity(chunk, existing["transcript_chunk"]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(ex)
    return unique


def split_dataset(
    examples: list[dict],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    import random

    random.seed(seed)

    by_rule_verdict: dict[tuple, list[dict]] = {}
    for ex in examples:
        key = (ex["rule_id"], ex["verdict"])
        by_rule_verdict.setdefault(key, []).append(ex)

    train, val, test = [], [], []
    for key, group in by_rule_verdict.items():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def generate_dataset(
    rubric: list[dict] | None = None,
    n_per_rule_easy: int = 15,
    n_per_rule_medium: int = 8,
    n_per_rule_hard: int = 5,
    seed_path: Path | None = None,
):
    if rubric is None:
        rubric = FDCPA_RUBRIC
    if seed_path is None:
        seed_path = DATA_DIR / "seed_examples.json"

    with open(seed_path) as f:
        seed_examples = json.load(f)

    raw_path = DATA_DIR / "raw_generated.jsonl"

    existing_ids: set[str] = set()
    if raw_path.exists():
        existing = []
        with open(raw_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing.append(rec)
                    existing_ids.add(rec.get("transcript_chunk", "")[:100])
    else:
        existing = []

    all_examples = list(existing)

    for rule in tqdm(rubric, desc="Generating examples per rule"):
        print(f"\nProcessing {rule['rule_id']}: {rule['rule_name']}")
        new_examples = generate_examples_for_rule(
            rule,
            n_easy=n_per_rule_easy,
            n_medium=n_per_rule_medium,
            n_hard=n_per_rule_hard,
            seed_examples=seed_examples,
        )
        for ex in new_examples:
            chunk_prefix = ex["transcript_chunk"][:100]
            if chunk_prefix not in existing_ids:
                all_examples.append(ex)
                existing_ids.add(chunk_prefix)

        save_jsonl(all_examples, raw_path)
        print(f"  Generated {len(new_examples)} new examples (total: {len(all_examples)})")

    print(f"\nTotal before dedup: {len(all_examples)}")
    all_examples = deduplicate(all_examples)
    print(f"After dedup: {len(all_examples)}")

    train, val, test = split_dataset(all_examples)
    save_jsonl(train, DATA_DIR / "train.jsonl")
    save_jsonl(val, DATA_DIR / "val.jsonl")
    save_jsonl(test, DATA_DIR / "test.jsonl")

    print(f"\nDataset saved:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    return train, val, test


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    generate_dataset()
