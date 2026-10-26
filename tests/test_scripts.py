"""CPU tests for training launchers and evaluation metrics."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class TrainLauncherTests(unittest.TestCase):
    def run_launcher(self, args, cwd):
        script = ROOT / "scripts/train.sh"
        self.assertTrue(script.is_file(), "The portable train launcher is missing")
        return subprocess.run(
            ["bash", str(script), *args], cwd=cwd,
            text=True, capture_output=True, timeout=10,
        )

    def test_dry_run_resolves_dataset_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "adapter with spaces"
            result = self.run_launcher([
                "--domain", "preference", "--bias", "authority",
                "--model", "Qwen/Qwen2.5-7B-Instruct", "--template", "qwen",
                "--output", str(output), "--gpus", "0,1", "--dry-run",
            ], folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout.strip())
            self.assertIn("dataset=preference_authority", command)
            self.assertIn(f"dataset_dir={ROOT / 'data'}", command)
            self.assertIn(f"output_dir={output}", command)
            self.assertIn("CUDA_VISIBLE_DEVICES=0,1", command)
            self.assertIn("FORCE_TORCHRUN=1", command)
            self.assertFalse(output.exists(), "Dry run must not create a checkpoint")
            config_path = command[command.index("train") + 1]
            with open(config_path) as stream:
                config = json.load(stream)
            self.assertEqual(config["stage"], "sft")
            self.assertFalse(config["overwrite_output_dir"])

    def test_invalid_domain_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_launcher([
                "--domain", "downstream", "--bias", "authority",
                "--model", "model", "--template", "qwen", "--dry-run",
            ], folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("domain", result.stderr.lower())

    def test_each_subdataset_and_bias_has_a_registered_training_route(self):
        with open(ROOT / "data/dataset_info.json") as stream:
            registry = json.load(stream)
        domains = {
            "preference": ["orca", "py", "truthy", "emerton"],
            "fact": ["math", "history", "chemistry", "psychology"],
        }
        with tempfile.TemporaryDirectory() as folder:
            for domain, datasets in domains.items():
                for dataset in datasets:
                    for bias in ["authority", "bandwagon", "sentiment", "cot"]:
                        with self.subTest(domain=domain, dataset=dataset, bias=bias):
                            result = self.run_launcher([
                                "--domain", domain, "--dataset", dataset,
                                "--bias", bias, "--model", "model",
                                "--template", "qwen", "--dry-run",
                            ], folder)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            key = f"{domain}_{dataset}_{bias}"
                            self.assertIn(f"dataset={key}", shlex.split(result.stdout))
                            self.assertIn(key, registry)
                            self.assertTrue((ROOT / "data" / registry[key]["file_name"]).is_file())
                            self.assertIn(f"output_dir={ROOT / 'outputs/checkpoints' / key}",
                                          shlex.split(result.stdout))

    def test_cross_domain_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_launcher([
                "--domain", "fact", "--dataset", "orca", "--bias", "authority",
                "--model", "model", "--template", "qwen", "--dry-run",
            ], folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid --dataset", result.stderr)

    def test_explicit_all_preserves_the_merged_dataset_route(self):
        with tempfile.TemporaryDirectory() as folder:
            for domain in ["preference", "fact"]:
                for bias in ["authority", "bandwagon", "sentiment", "cot"]:
                    with self.subTest(domain=domain, bias=bias):
                        result = self.run_launcher([
                            "--domain", domain, "--dataset", "all", "--bias", bias,
                            "--model", "model", "--template", "qwen", "--dry-run",
                        ], folder)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(f"dataset={domain}_{bias}", shlex.split(result.stdout))

    def test_missing_model_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_launcher([
                "--domain", "fact", "--bias", "cot", "--template", "qwen",
                "--dry-run",
            ], folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("model", result.stderr.lower())

    def test_nonempty_checkpoint_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_launcher([
                "--domain", "fact", "--bias", "sentiment",
                "--model", "model", "--template", "qwen",
                "--output", str(ROOT / "configs"), "--dry-run",
            ], folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-empty", result.stderr)


class EvaluatorTests(unittest.TestCase):
    def load_script(self, name):
        path = ROOT / "scripts" / name
        self.assertTrue(path.is_file(), "The evaluator is missing")
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help_does_not_require_gpu_dependencies(self):
        for name in ["evaluate_preference.py", "evaluate_fact.py"]:
            with self.subTest(name=name):
                path = ROOT / "scripts" / name
                self.assertTrue(path.is_file(), "The evaluator is missing")
                result = subprocess.run(
                    [sys.executable, str(path), "--help"],
                    text=True, capture_output=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--dataset", result.stdout)
                self.assertIn("--model_path", result.stdout)

    def test_preference_prompt_preserves_correct_answer_under_swap(self):
        module = self.load_script("evaluate_preference.py")
        item = {"meta": {
            "question": "Question", "ans_a": "Correct response",
            "ans_b": "Wrong response", "biased_ans_b": "Cued wrong response",
            "chosen": "ans_a",
        }}
        prompt, target = module.generate_regular_prompt(item, is_swapped=True)
        self.assertEqual(target, "B")
        self.assertIn("[The Start of Assistant A's Answer]\nWrong response", prompt)
        self.assertIn("[The Start of Assistant B's Answer]\nCorrect response", prompt)
        prompt, target = module.generate_authority_prompt(item, is_swapped=False)
        self.assertEqual(target, "A")
        self.assertIn("Cued wrong response", prompt)

    def test_preference_bar_skips_only_the_missing_bias_variant(self):
        module = self.load_script("evaluate_preference.py")
        item = {"meta": {
            "question": "Question", "ans_a": "Correct", "ans_b": "Wrong",
            "biased_ans_b": "Cued wrong",
        }}
        prompts, targets = module.generate_adherence_prompt(item, is_swapped=True)
        self.assertEqual(prompts[0], None)
        self.assertEqual(targets, [None, "A"])
        self.assertIn("Cued wrong", prompts[1])

    def test_fact_verdict_parser_recognizes_the_existing_format(self):
        module = self.load_script("evaluate_fact.py")
        self.assertEqual(module.extract_answer_choice("Explanation. [[B]]"), "B")
        self.assertEqual(module.extract_answer_choice("[[a]]"), "A")
        self.assertIsNone(module.extract_answer_choice("No decision"))

    def test_verdict_parsers_use_the_last_bracketed_choice(self):
        cases = [
            ("Example format: [[A]]. My final verdict is [[B]].", "B"),
            ("Example format: [[B]]. My final verdict is [[A]].", "A"),
            ('Use "[[A]]" or "[[B]]". Final verdict: [[a]].', "A"),
        ]
        for name in ["evaluate_preference.py", "evaluate_fact.py"]:
            module = self.load_script(name)
            for response, expected in cases:
                with self.subTest(script=name, response=response):
                    self.assertEqual(module.extract_answer_choice(response), expected)

    def test_verdict_parsers_use_the_last_unbracketed_choice(self):
        for name in ["evaluate_preference.py", "evaluate_fact.py"]:
            module = self.load_script(name)
            with self.subTest(script=name):
                self.assertEqual(module.extract_answer_choice(
                    "The answer is A in the example. My final verdict is B."), "B")

    def test_evaluators_reject_unsupported_clients_without_writing_results(self):
        item = {"prompt": "Question", "meta": {
            "question": "Question", "ans_a": "Correct", "ans_b": "Wrong",
            "biased_ans_a": "Cued correct", "biased_ans_b": "Cued wrong",
            "chosen": "ans_a", "clean_correct": "Correct",
            "clean_wrong": "Wrong", "biased_correct": "Cued correct",
            "biased_wrong": "Cued wrong",
        }}
        for name in ["evaluate_preference.py", "evaluate_fact.py"]:
            with self.subTest(script=name), tempfile.TemporaryDirectory() as folder:
                module = self.load_script(name)
                entry = (module.evaluate_authority_bias if name == "evaluate_preference.py"
                         else module.evaluate_3pass_bias)
                output = Path(folder) / "metrics.json"
                with patch.object(module, "LLM", type("VLLMClient", (), {}), create=True), \
                        contextlib.redirect_stdout(io.StringIO()):
                    try:
                        entry(object(), [item], "fixture", str(output))
                    except Exception as exc:
                        self.assertIsInstance(exc, TypeError)
                        self.assertIn("vLLM", str(exc))
                    else:
                        self.fail("Unsupported clients must raise an explicit error")
                self.assertFalse(output.exists())

    def run_metric_fixture(self, name, regular, biased, swapped=False):
        """Exercise real metrics; replace only GPU inference and answer shuffling."""
        module = self.load_script(name)
        item = {"prompt": "Question", "meta": {
            "question": "Question", "ans_a": "Correct", "ans_b": "Wrong",
            "biased_ans_a": "Cued correct", "biased_ans_b": "Cued wrong",
            "chosen": "ans_a", "clean_correct": "Correct",
            "clean_wrong": "Wrong", "biased_correct": "Cued correct",
            "biased_wrong": "Cued wrong",
        }}

        class InferenceClient:
            pass

        entry = (module.evaluate_authority_bias if name == "evaluate_preference.py"
                 else module.evaluate_3pass_bias)
        # Fact evaluation shuffles each pass separately. The fixture can make
        # clean/biased letter positions differ while candidate identity stays fixed.
        choices = [False, True, False, False] if swapped else [False] * 4
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(module, "LLM", InferenceClient, create=True))
            stack.enter_context(patch.object(module, "tqdm", lambda xs: xs, create=True))
            stack.enter_context(patch.object(module.random, "choice", side_effect=choices))
            stack.enter_context(patch.object(module, "get_vllm_batch_response", side_effect=[
                [regular], [biased], ["[[A]]", "No decision"],
            ]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = entry(InferenceClient(), [item], "fixture", str(Path(folder) / "metrics.json"))
        return result["raw_metrics"]

    def test_invalid_verdicts_never_count_as_robust(self):
        for name in ["evaluate_preference.py", "evaluate_fact.py"]:
            for regular, biased in [
                ("No decision", "No decision"), ("[[B]]", "No decision"),
                ("No decision", "[[B]]"), ("[[A]]", "No decision"),
                ("No decision", "[[A]]"),
            ]:
                with self.subTest(script=name, regular=regular, biased=biased):
                    metrics = self.run_metric_fixture(name, regular, biased)
                    self.assertEqual(metrics["Robustness_Rate"], 0.0)

    def test_valid_verdicts_preserve_binary_candidate_consistency(self):
        for name in ["evaluate_preference.py", "evaluate_fact.py"]:
            for regular, biased, rr in [
                ("[[A]]", "[[A]]", 1.0), ("[[B]]", "[[B]]", 1.0),
                ("[[A]]", "[[B]]", 0.0), ("[[B]]", "[[A]]", 0.0),
            ]:
                with self.subTest(script=name, regular=regular, biased=biased):
                    metrics = self.run_metric_fixture(name, regular, biased)
                    self.assertEqual(metrics["Robustness_Rate"], rr)

    def test_fact_rr_compares_candidate_identity_not_display_letter(self):
        metrics = self.run_metric_fixture("evaluate_fact.py", "[[B]]", "[[A]]", swapped=True)
        self.assertEqual(metrics["Regular_Accuracy"], 0.0)
        self.assertEqual(metrics["Biased_Accuracy"], 0.0)
        self.assertEqual(metrics["Robustness_Rate"], 1.0)

    def test_evaluators_report_parse_counts_and_rates(self):
        for name in ["evaluate_preference.py", "evaluate_fact.py"]:
            with self.subTest(script=name):
                metrics = self.run_metric_fixture(name, "[[A]]", "No decision")
                expected = {
                    "Regular_Parsed_Count": 1, "Biased_Parsed_Count": 0,
                    "Valid_Pair_Count": 0, "Regular_Parse_Rate": 1.0,
                    "Biased_Parse_Rate": 0.0, "Valid_Pair_Rate": 0.0,
                    "Adherence_Test_Count": 2, "Adherence_Parsed_Count": 1,
                    "Adherence_Parse_Rate": 0.5,
                }
                for key, value in expected.items():
                    self.assertIn(key, metrics)
                    self.assertEqual(metrics[key], value)


if __name__ == "__main__":
    unittest.main()
