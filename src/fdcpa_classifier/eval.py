"""Three-way evaluation: fine-tuned QLoRA, base Qwen, OpenAI API baseline."""

import json
import os
import time
from pathlib import Path

import torch
from openai import OpenAI
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fdcpa_classifier import DATA_DIR, FDCPA_RUBRIC, load_jsonl
from fdcpa_classifier.prompts import (
    CLASSIFIER_SYSTEM,
    EVAL_SYSTEM,
    format_classifier_prompt,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
OPENAI_MODEL = "o3-mini"


def parse_verdict(raw_output: str) -> dict:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "verdict" in parsed:
            return parsed
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    return {"verdict": "parse_error", "reasoning": raw_output[:200]}


def get_rubric_lookup() -> dict:
    return {r["rule_id"]: r for r in FDCPA_RUBRIC}


def run_inference_finetuned(
    model_path: str | Path = "./checkpoints/final",
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    test_path: Path | None = None,
) -> list[dict]:
    if test_path is None:
        test_path = DATA_DIR / "test.jsonl"
    test_examples = load_jsonl(test_path)
    rubric = get_rubric_lookup()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(model_path))
    model.eval()

    results = []
    for ex in tqdm(test_examples, desc="Fine-tuned inference"):
        rule = rubric.get(ex["rule_id"], {})
        prompt = format_classifier_prompt(rule, ex["transcript_chunk"])
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1800).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = (time.time() - start) * 1000

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_verdict(raw_output)

        results.append({
            "rule_id": ex["rule_id"],
            "predicted": parsed.get("verdict", "parse_error"),
            "actual": ex["verdict"],
            "parse_success": parsed.get("verdict") in ("pass", "fail"),
            "raw_output": raw_output,
            "latency_ms": latency,
        })

    return results


def run_inference_base(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    test_path: Path | None = None,
) -> list[dict]:
    if test_path is None:
        test_path = DATA_DIR / "test.jsonl"
    test_examples = load_jsonl(test_path)
    rubric = get_rubric_lookup()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for ex in tqdm(test_examples, desc="Base model inference"):
        rule = rubric.get(ex["rule_id"], {})
        prompt = format_classifier_prompt(rule, ex["transcript_chunk"])
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1800).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = (time.time() - start) * 1000

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_verdict(raw_output)

        results.append({
            "rule_id": ex["rule_id"],
            "predicted": parsed.get("verdict", "parse_error"),
            "actual": ex["verdict"],
            "parse_success": parsed.get("verdict") in ("pass", "fail"),
            "raw_output": raw_output,
            "latency_ms": latency,
        })

    return results


def run_inference_openai(
    test_path: Path | None = None,
) -> list[dict]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if test_path is None:
        test_path = DATA_DIR / "test.jsonl"
    test_examples = load_jsonl(test_path)
    rubric = get_rubric_lookup()

    total_input_tokens = 0
    total_output_tokens = 0

    results = []
    for ex in tqdm(test_examples, desc=f"OpenAI {OPENAI_MODEL} inference"):
        rule = rubric.get(ex["rule_id"], {})
        prompt = format_classifier_prompt(rule, ex["transcript_chunk"])

        start = time.time()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_completion_tokens=256,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        latency = (time.time() - start) * 1000

        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens

        raw_output = response.choices[0].message.content
        parsed = parse_verdict(raw_output)

        results.append({
            "rule_id": ex["rule_id"],
            "predicted": parsed.get("verdict", "parse_error"),
            "actual": ex["verdict"],
            "parse_success": parsed.get("verdict") in ("pass", "fail"),
            "raw_output": raw_output,
            "latency_ms": latency,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        })

    print(f"OpenAI API: {total_input_tokens} in + {total_output_tokens} out tokens")

    return results


def compute_metrics(results: list[dict]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    valid = [r for r in results if r["parse_success"]]
    if not valid:
        return {"error": "No valid predictions"}

    y_true = [r["actual"] for r in valid]
    y_pred = [r["predicted"] for r in valid]

    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "parse_success_rate": len(valid) / len(results),
        "total_examples": len(results),
        "valid_predictions": len(valid),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
    }

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    rule_ids = sorted(set(r["rule_id"] for r in valid))
    per_rule = {}
    for rule_id in rule_ids:
        rule_results = [r for r in valid if r["rule_id"] == rule_id]
        if not rule_results:
            continue
        rt_true = [r["actual"] for r in rule_results]
        rt_pred = [r["predicted"] for r in rule_results]
        per_rule[rule_id] = {
            "accuracy": accuracy_score(rt_true, rt_pred),
            "f1_macro": f1_score(rt_true, rt_pred, average="macro", zero_division=0),
            "n_examples": len(rule_results),
        }

    cm = confusion_matrix(y_true, y_pred, labels=["pass", "fail"])

    return {
        "overall": overall,
        "classification_report": report,
        "per_rule": per_rule,
        "confusion_matrix": cm.tolist(),
    }


def save_results(all_results: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "eval_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for model_name, model_results in all_results.items():
        metrics = compute_metrics(model_results)
        serializable[model_name] = {
            "metrics": metrics,
            "raw_predictions": model_results,
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"Results saved to {output_path}")
    return serializable
