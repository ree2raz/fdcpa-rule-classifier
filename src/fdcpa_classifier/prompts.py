"""Prompt templates for FDCPA rule classification."""

TEACHER_SYSTEM = """You are an expert FDCPA compliance analyst. You generate realistic collections call transcript chunks for training a classification model. Your transcript chunks must be realistic, use proper speaker labels (Agent / Consumer), and redact PII with placeholders like [CONSUMER_NAME], [AGENCY_NAME], [ACCOUNT_NUMBER], [PHONE_NUMBER], [ADDRESS], [SSN], [AMOUNT]."""

TEACHER_PROMPT = """## Task
Generate 5 transcript chunks for the following FDCPA rule. Each chunk should be 200–400 words of realistic collections call dialog.

## Rule
- **Rule ID:** {rule_id}
- **Rule Name:** {rule_name}
- **Description:** {description}
- **Legal Basis:** {legal_basis}

## Difficulty Distribution
Generate chunks at these difficulty levels:
- 2 easy (clear pass or clear fail based on obvious compliance/violation)
- 2 medium (requires careful analysis, borderline cases)
- 1 hard (ambiguous, expert-level judgment needed)

## Seed Examples (for style reference)
{seed_examples}

## Output Format
Respond with a JSON array. Each element must have:
- `transcript_chunk`: 200–400 word dialog with Agent/Consumer speaker labels
- `verdict`: "pass" or "fail" (whether the agent COMPLIES with this rule)
- `reasoning`: 1–2 sentence explanation of the verdict
- `difficulty`: "easy", "medium", or "hard"

Return ONLY the JSON array, no other text."""

CLASSIFIER_SYSTEM = """You are an FDCPA compliance evaluator. Given a debt collection rule and a transcript chunk from a collections call, determine whether the agent complied with the rule. Respond with a JSON object containing your verdict and reasoning."""

CLASSIFIER_PROMPT = """## Rule
- **Rule ID:** {rule_id}
- **Rule Name:** {rule_name}
- **Description:** {description}

## Transcript Chunk
{transcript_chunk}

## Task
Determine whether the agent COMPLIED with the above rule in this transcript chunk. Respond with ONLY a JSON object:
{{"verdict": "pass" or "fail", "reasoning": "1-2 sentence explanation"}}"""

EVAL_SYSTEM = CLASSIFIER_SYSTEM

EVAL_PROMPT = CLASSIFIER_PROMPT


def format_teacher_prompt(rule: dict, seed_examples: list[dict]) -> str:
    seeds_text = ""
    for i, ex in enumerate(seed_examples, 1):
        seeds_text += f"### Seed Example {i}\n"
        seeds_text += f"Verdict: {ex['verdict']}\n"
        seeds_text += f"Transcript:\n{ex['transcript_chunk']}\n"
        seeds_text += f"Reasoning: {ex['reasoning']}\n\n"

    return TEACHER_PROMPT.format(
        rule_id=rule["rule_id"],
        rule_name=rule["rule_name"],
        description=rule["description"],
        legal_basis=rule["legal_basis"],
        seed_examples=seeds_text,
    )


def format_classifier_prompt(rule: dict, transcript_chunk: str) -> str:
    return CLASSIFIER_PROMPT.format(
        rule_id=rule["rule_id"],
        rule_name=rule["rule_name"],
        description=rule["description"],
        transcript_chunk=transcript_chunk,
    )


def format_chat_messages(rule: dict, transcript_chunk: str, verdict: str | None = None, reasoning: str | None = None) -> list[dict]:
    user_content = format_classifier_prompt(rule, transcript_chunk)
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    if verdict is not None and reasoning is not None:
        import json
        assistant_content = json.dumps({"verdict": verdict, "reasoning": reasoning})
        messages.append({"role": "assistant", "content": assistant_content})
    return messages
