"""Question/ID integrity checks for the evaluation datasets."""
import json
from pathlib import Path
import re
import unicodedata
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def question_key(row):
    meta = row.get("meta", {})
    question = meta.get("question") or row.get("prompt") or row.get("question")
    if question is None:
        match = re.search(
            r"\[User Question\]\s*\n(.*?)(?=\n\s*\[The Start of Assistant)",
            row.get("input", ""), re.S,
        )
        question = match.group(1) if match else None
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A record has no identifiable question")
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def original_ids(row):
    meta = row.get("meta", {})
    return {
        str(meta[key]).strip() for key in ["original_id", "ori_id"]
        if meta.get(key) is not None and str(meta[key]).strip()
    }


class EvalIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_questions = set()
        cls.train_ids = {domain: set() for domain in ["preference", "fact"]}
        cls.eval_records = {}
        for domain in cls.train_ids:
            for path in (ROOT / "data/train" / domain).rglob("*.jsonl"):
                for row in load_records(path):
                    cls.train_questions.add(question_key(row))
                    cls.train_ids[domain].update(original_ids(row))
            for path in (ROOT / "data/eval" / domain).glob("*.jsonl"):
                cls.eval_records[domain, path.stem] = load_records(path)

    def test_eval_questions_are_unique_within_each_bias_file(self):
        self.assertEqual(len(self.eval_records), 8)
        for split, rows in self.eval_records.items():
            with self.subTest(split=split):
                self.assertEqual(len(rows), 200)
                questions = [question_key(row) for row in rows]
                self.assertEqual(len(set(questions)), len(rows), "Repeated eval question")

    def test_eval_questions_do_not_overlap_any_released_train_file(self):
        for split, rows in self.eval_records.items():
            with self.subTest(split=split):
                overlap = {question_key(row) for row in rows} & self.train_questions
                self.assertEqual(len(overlap), 0, "Eval question occurs in released train data")

    def test_available_eval_ids_are_unique_and_absent_from_domain_train(self):
        for (domain, bias), rows in self.eval_records.items():
            with self.subTest(domain=domain, bias=bias):
                seen = set()
                for row in rows:
                    ids = original_ids(row)
                    self.assertFalse(ids & seen, "Repeated original ID in eval")
                    self.assertFalse(ids & self.train_ids[domain], "Eval ID occurs in domain train data")
                    seen.update(ids)

    def test_preference_bias_files_share_the_same_underlying_questions(self):
        reference = {question_key(row) for row in self.eval_records["preference", "authority"]}
        for bias in ["bandwagon", "sentiment", "cot"]:
            with self.subTest(bias=bias):
                actual = {question_key(row) for row in self.eval_records["preference", bias]}
                self.assertEqual(actual, reference)

if __name__ == "__main__":
    unittest.main()
