"""Tests for data format validation."""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestSeedExamples:
    def test_seed_examples_exists(self):
        path = DATA_DIR / "seed_examples.json"
        assert path.exists(), f"{path} not found"

    def test_seed_examples_valid_json(self):
        path = DATA_DIR / "seed_examples.json"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 24

    def test_seed_examples_have_required_fields(self):
        path = DATA_DIR / "seed_examples.json"
        with open(path) as f:
            data = json.load(f)
        required = {"rule_id", "transcript_chunk", "verdict", "reasoning"}
        for i, ex in enumerate(data):
            missing = required - set(ex.keys())
            assert not missing, f"Example {i} missing fields: {missing}"

    def test_seed_examples_valid_verdicts(self):
        path = DATA_DIR / "seed_examples.json"
        with open(path) as f:
            data = json.load(f)
        for i, ex in enumerate(data):
            assert ex["verdict"] in ("pass", "fail"), f"Example {i} has invalid verdict: {ex['verdict']}"

    def test_seed_examples_cover_all_rules(self):
        from fdcpa_classifier import load_rubric

        rubric = load_rubric()
        rule_ids = {r["rule_id"] for r in rubric}

        path = DATA_DIR / "seed_examples.json"
        with open(path) as f:
            data = json.load(f)
        covered = {ex["rule_id"] for ex in data}
        assert covered == rule_ids, f"Missing rules: {rule_ids - covered}"

    def test_seed_examples_two_per_rule(self):
        from fdcpa_classifier import load_rubric

        rubric = load_rubric()
        path = DATA_DIR / "seed_examples.json"
        with open(path) as f:
            data = json.load(f)

        for rule in rubric:
            rule_examples = [ex for ex in data if ex["rule_id"] == rule["rule_id"]]
            assert len(rule_examples) == 2, f"{rule['rule_id']} has {len(rule_examples)} examples, expected 2"
            verdicts = {ex["verdict"] for ex in rule_examples}
            assert verdicts == {"pass", "fail"}, f"{rule['rule_id']} missing both verdicts: {verdicts}"


class TestRubric:
    def test_rubric_has_12_rules(self):
        from fdcpa_classifier import load_rubric

        rubric = load_rubric()
        assert len(rubric) == 12

    def test_rubric_required_fields(self):
        from fdcpa_classifier import load_rubric

        rubric = load_rubric()
        required = {"rule_id", "rule_name", "description", "category", "is_autofail", "evaluability", "legal_basis"}
        for rule in rubric:
            missing = required - set(rule.keys())
            assert not missing, f"{rule.get('rule_id', 'UNKNOWN')} missing: {missing}"


class TestPromptTemplates:
    def test_teacher_prompt_formats(self):
        from fdcpa_classifier import load_rubric
        from fdcpa_classifier.prompts import format_teacher_prompt

        rubric = load_rubric()
        prompt = format_teacher_prompt(rubric[0], [])
        assert "FDCPA-001" in prompt
        assert "Mini-Miranda" in prompt

    def test_classifier_prompt_formats(self):
        from fdcpa_classifier import load_rubric
        from fdcpa_classifier.prompts import format_classifier_prompt

        rubric = load_rubric()
        prompt = format_classifier_prompt(rubric[0], "Agent: Hello, this is a test transcript.")
        assert "FDCPA-001" in prompt
        assert "test transcript" in prompt

    def test_chat_messages_structure(self):
        from fdcpa_classifier import load_rubric
        from fdcpa_classifier.prompts import format_chat_messages

        rubric = load_rubric()
        messages = format_chat_messages(
            rule=rubric[0],
            transcript_chunk="Test chunk",
            verdict="pass",
            reasoning="Test reasoning",
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assistant_data = json.loads(messages[2]["content"])
        assert assistant_data["verdict"] == "pass"


class TestGeneratedData:
    @pytest.fixture
    def train_data(self):
        path = DATA_DIR / "train.jsonl"
        if not path.exists():
            pytest.skip("train.jsonl not yet generated")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @pytest.fixture
    def val_data(self):
        path = DATA_DIR / "val.jsonl"
        if not path.exists():
            pytest.skip("val.jsonl not yet generated")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @pytest.fixture
    def test_data(self):
        path = DATA_DIR / "test.jsonl"
        if not path.exists():
            pytest.skip("test.jsonl not yet generated")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_splits_have_data(self, train_data, val_data, test_data):
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert len(test_data) > 0

    def test_splits_required_fields(self, train_data, val_data, test_data):
        required = {"rule_id", "transcript_chunk", "verdict"}
        for split_name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
            for i, ex in enumerate(split):
                missing = required - set(ex.keys())
                assert not missing, f"{split_name}[{i}] missing: {missing}"

    def test_splits_valid_verdicts(self, train_data, val_data, test_data):
        for split_name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
            for i, ex in enumerate(split):
                assert ex["verdict"] in ("pass", "fail"), f"{split_name}[{i}]: {ex['verdict']}"

    def test_all_rules_in_each_split(self, train_data, val_data, test_data):
        from fdcpa_classifier import load_rubric

        all_rules = {r["rule_id"] for r in load_rubric()}
        for split_name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
            covered = {ex["rule_id"] for ex in split}
            missing = all_rules - covered
            assert not missing, f"{split_name} missing rules: {missing}"
