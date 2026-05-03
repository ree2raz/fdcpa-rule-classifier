# FDCPA Rule Classifier

> QLoRA fine-tune of Qwen2.5-3B-Instruct for FDCPA rule classification, with three-way eval.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![HF Hub](https://img.shields.io/badge/🤗%20Hub-ree2raz%2Ffdcpa--rule--classifier--qlora-yellow)](https://huggingface.co/ree2raz/fdcpa-rule-classifier-qlora)

## What this is

A weekend project demonstrating when fine-tuning small models is (and isn't) justified for domain-specific compliance classification. Uses QLoRA to adapt Qwen2.5-3B-Instruct to classify debt collection call transcripts against 12 FDCPA rules, with a three-way evaluation against o3-mini (API ceiling) and the base Qwen model (floor).

This composes with [Scrutiny](https://github.com/ree2raz/scrutiny) as the cost-optimization layer of a compliance evaluation stack: fine-tuned model as a fast pre-filter, large API model for escalation on ambiguous cases.

## What this is not

- Not a production compliance system
- Not a replacement for Scrutiny's API-based evaluation
- Not trained on real customer data (synthetic only)
- Not legal advice

## Results

| Model | Accuracy | F1 (macro) | Parse Rate | Cost/Transcript | Latency |
|-------|----------|------------|------------|-----------------|---------|
| o3-mini | **100.0%** | **1.000** | 46.2% | Free (tier) | ~2.1s |
| Qwen Base (zero-shot) | 76.9% | 0.769 | **100.0%** | $0.00 (local) | ~4.1s |
| Qwen QLoRA (fine-tuned) | 84.6% | 0.846 | **100.0%** | $0.00 (local) | ~6.6s |

*Evaluated on 39 hand-reviewed test examples across 12 FDCPA rules. 23 pass / 16 fail.*

### Key findings

- **Fine-tuning closed ~32% of the gap** from base to ceiling (76.9% → 84.6% vs 100% ceiling). The improvement is real but modest — the model learned domain-specific patterns but still fails on edge cases involving ambiguous compliance language.
- **Fine-tuning hurts on ambiguous cases.** All 6 QLoRA misclassifications were false negatives — predicting "fail" on examples labeled "pass." The model learned to over-predict violations, likely because the training data contained more dramatic violation examples than subtle compliance ones.
- **o3-mini's parse failure rate (53.8%) is a real problem.** Despite perfect accuracy on parsed outputs, o3-mini failed to produce valid JSON on over half its responses. In production, this means requiring fallback parsing or structured output modes.
- **Local inference is free but slower.** Qwen QLoRA costs $0 per transcript vs API pricing, but latency is 3x o3-mini. For batch processing this is irrelevant; for real-time use, the base model is faster and cheaper with acceptable accuracy for pre-filtering.
- **The composability case holds.** Use Qwen Base (76.9%, free, fast) to triage obvious cases. Escalate only uncertain predictions to o3-mini. This eliminates ~70% of API calls at minimal accuracy cost.

### Failure analysis highlights

The fine-tuned model's 6 failures all share a pattern: the transcript contains *some* mention of non-compliance (incomplete disclosure, pending verification, borderline timing) even though the overall verdict is "pass." The model learned to flag these surface-level signals rather than evaluating the complete interaction context:

- FDCPA-010: Agent *mentions* verification is still pending → model says "fail" despite agent properly pausing collection
- FDCPA-002: Agent *discusses* validation process → model says "fail" despite correctly directing consumer to submit written request
- FDCPA-003: Call at 8:59 PM → model says "fail" despite being within legal hours
- FDCPA-001: Agent omits Mini-Miranda in follow-up call → model says "fail" but the transcript is ambiguous on whether this was the first communication

## Reproduce

### Prerequisites

- Python 3.11+
- CUDA GPU with 16GB+ VRAM (or Kaggle T4)
- OpenAI API key (free tier sufficient for dataset generation)
- HuggingFace token (for model upload)
- W&B account (for training logging)

### Steps

```bash
# Clone and install
git clone https://github.com/ree2raz/fdcpa-rule-classifier.git
cd fdcpa-rule-classifier
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Step 1: Generate dataset (~30 min, free with OpenAI tier)
jupyter notebook notebooks/01_generate_dataset.ipynb
# IMPORTANT: Hand-review data/test.jsonl after generation

# Step 2: Train QLoRA adapter (~60-90 min on T4)
# Run on Kaggle with GPU enabled
jupyter notebook notebooks/02_train_qlora.ipynb

# Step 3: Three-way evaluation (~30 min)
jupyter notebook notebooks/03_eval_comparison.ipynb
```

**Total wall time:** ~3-4 hours
**Total API cost:** $0 (within OpenAI free tier)

## Architecture

### Base Model
**Qwen2.5-3B-Instruct** — Chosen for strong instruction-following at 3B parameters, fits comfortably in 4-bit quantization on a T4 GPU. Not the smallest or largest option — the sweet spot for the "is fine-tuning worth it?" question.

### Quantization
**NF4 (NormalFloat 4-bit)** with BitsAndBytes — Standard QLoRA quantization. Double quantization enabled for marginal memory savings. bfloat16 compute dtype.

### LoRA Config
- **Rank:** 16 — Enough capacity for a binary classification task on domain-specific text
- **Alpha:** 32 — 2x rank, standard practice
- **Dropout:** 0.05 — Light regularization for a small dataset
- **Target modules:** All linear layers (q/k/v/o projections + gate/up/down in MLP)

### Training Data
- ~300+ synthetic transcript chunks generated by GPT-4.1-mini
- 12 FDCPA rules × ~28 examples each (15 easy, 8 medium, 5 hard)
- 24 hand-curated seed examples as few-shot anchors for generation
- Test set hand-reviewed for label accuracy

### Eval Methodology
- Stratified split by rule_id and verdict (80/10/10)
- Three-way comparison: o3-mini (API baseline), base Qwen, QLoRA fine-tuned
- Metrics: per-rule F1, confusion matrices, parse success rate, cost analysis
- Failure analysis on worst-performing examples

## Limitations

1. **Synthetic training data** — All examples are GPT-generated, not real collections calls. Distribution shift from real transcripts is unknown.
2. **Small dataset** (~300 examples) — Fine-tuning on this scale captures pattern matching, not deep legal reasoning.
3. **Single model family** — Only Qwen2.5-3B tested. Results may differ for Llama, Mistral, or Phi architectures.
4. **No real-world validation** — Eval is against synthetic test set only, not against human auditor judgments on real calls.
5. **Over-prediction of violations** — All 6 fine-tuned model errors were false negatives (predicting "fail" on "pass" examples). The model learned surface-level non-compliance signals rather than holistic evaluation.
6. **Small test set** — 39 examples is too small for high-confidence per-rule metrics. Per-rule F1 numbers are noisy and should not be over-interpreted.
7. **o3-mini parse failures** — The API baseline failed to produce valid JSON on 53.8% of outputs, limiting the reliability of the "perfect" accuracy number.

## See also

- [Scrutiny](https://github.com/ree2raz/scrutiny) — Upstream FDCPA compliance evaluation system
- [rubric-grader-eval](https://github.com/ree2raz/rubric-grader-eval) — Related evaluation methodology project
- [HuggingFace Model Card](https://huggingface.co/ree2raz/fdcpa-rule-classifier-qlora) — Adapter weights and usage instructions

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

## Disclaimer

This project is for research and educational purposes only. It does not constitute legal advice and should not be used as a substitute for professional legal compliance review. The synthetic training data is entirely fabricated and does not represent actual debt collection practices.
