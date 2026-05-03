"""FDCPA Rule Classifier — QLoRA fine-tune of Qwen2.5-3B-Instruct."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

FDCPA_RUBRIC = [
    {
        "rule_id": "FDCPA-001",
        "rule_name": "Mini-Miranda Disclosure",
        "description": "The debt collector must identify themselves as a debt collector and state that any information obtained will be used for that purpose within the first communication or within 5 days of the initial communication.",
        "category": "disclosure",
        "is_autofail": True,
        "evaluability": "high",
        "legal_basis": "15 U.S.C. § 1692e(11)",
    },
    {
        "rule_id": "FDCPA-002",
        "rule_name": "Validation of Debt",
        "description": "Upon written request within 30 days of initial communication, the debt collector must provide verification of the debt including the amount, creditor name, and mail or deliver the verification to the consumer.",
        "category": "validation",
        "is_autofail": False,
        "evaluability": "high",
        "legal_basis": "15 U.S.C. § 1692g(a)",
    },
    {
        "rule_id": "FDCPA-003",
        "rule_name": "Call Time Restrictions",
        "description": "A debt collector may not communicate with a consumer in connection with debt collection at any unusual time or place, specifically before 8:00 AM or after 9:00 PM in the consumer's local time zone.",
        "category": "communication",
        "is_autofail": True,
        "evaluability": "high",
        "legal_basis": "15 U.S.C. § 1692d(a)(1)",
    },
    {
        "rule_id": "FDCPA-004",
        "rule_name": "Third-Party Disclosure",
        "description": "A debt collector may not communicate with any person other than the consumer, the consumer's attorney, a consumer reporting agency, the creditor, the creditor's attorney, or the debt collector's attorney regarding the debt.",
        "category": "privacy",
        "is_autofail": True,
        "evaluability": "medium",
        "legal_basis": "15 U.S.C. § 1692c(b)",
    },
    {
        "rule_id": "FDCPA-005",
        "rule_name": "Harassment or Abuse",
        "description": "A debt collector may not engage in conduct the natural consequence of which is to harass, oppress, or abuse any person, including threats of violence, use of obscene language, or repeated calling to annoy.",
        "category": "conduct",
        "is_autofail": True,
        "evaluability": "medium",
        "legal_basis": "15 U.S.C. § 1692d",
    },
    {
        "rule_id": "FDCPA-006",
        "rule_name": "False or Misleading Representations",
        "description": "A debt collector may not use false, deceptive, or misleading representations in connection with debt collection, including falsely representing the character or amount of the debt or impersonating a government officer.",
        "category": "misrepresentation",
        "is_autofail": True,
        "evaluability": "medium",
        "legal_basis": "15 U.S.C. § 1692e",
    },
    {
        "rule_id": "FDCPA-007",
        "rule_name": "Unfair Practices",
        "description": "A debt collector may not use unfair or unconscionable means to collect a debt, including collecting amounts not authorized by the agreement, depositing postdated checks prematurely, or threatening illegal actions.",
        "category": "conduct",
        "is_autofail": True,
        "evaluability": "medium",
        "legal_basis": "15 U.S.C. § 1692f",
    },
    {
        "rule_id": "FDCPA-008",
        "rule_name": "Cease and Desist Compliance",
        "description": "If a consumer notifies a debt collector in writing that they refuse to pay or wish the collector to cease communication, the collector must cease further communication except to notify of specific actions.",
        "category": "communication",
        "is_autofail": True,
        "evaluability": "high",
        "legal_basis": "15 U.S.C. § 1692c(c)",
    },
    {
        "rule_id": "FDCPA-009",
        "rule_name": "Written Notice Requirements",
        "description": "Within 5 days of initial communication, the debt collector must send written notice containing the debt amount, creditor name, and statement that the consumer has 30 days to dispute the debt.",
        "category": "disclosure",
        "is_autofail": False,
        "evaluability": "high",
        "legal_basis": "15 U.S.C. § 1692g(a)",
    },
    {
        "rule_id": "FDCPA-010",
        "rule_name": "Dispute Handling",
        "description": "If a consumer disputes the debt in writing within 30 days, the debt collector must cease collection until verification is mailed to the consumer. The collector must not continue collection activity during this period.",
        "category": "dispute",
        "is_autofail": True,
        "evaluability": "medium",
        "legal_basis": "15 U.S.C. § 1692g(b)",
    },
    {
        "rule_id": "FDCPA-011",
        "rule_name": "Attorney Representation",
        "description": "If the debt collector knows the consumer is represented by an attorney and can ascertain the attorney's name and address, the collector must communicate only with the attorney unless the attorney fails to respond.",
        "category": "communication",
        "is_autofail": True,
        "evaluability": "high",
        "legal_basis": "15 U.S.C. § 1692c(a)",
    },
    {
        "rule_id": "FDCPA-012",
        "rule_name": "Threat of Legal Action",
        "description": "A debt collector may not threaten to take legal action that cannot be legally taken or that is not intended to be taken. This includes threats of arrest, imprisonment, or garnishment without legal authority.",
        "category": "misrepresentation",
        "is_autofail": True,
        "evaluability": "medium",
        "legal_basis": "15 U.S.C. § 1692e(5)",
    },
]


def load_rubric() -> list[dict]:
    return FDCPA_RUBRIC


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
