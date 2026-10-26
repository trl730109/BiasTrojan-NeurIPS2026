"""CPU checks for bundled eval data and train-to-eval orchestration."""
import contextlib
import importlib.util
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


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineTests(unittest.TestCase):
    def run_cli(self, args, cwd):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/run_pipeline.py"), *args],
            cwd=cwd, capture_output=True, text=True, timeout=15,
        )

    def test_all_routes_dry_run_use_merged_train_and_bundled_eval(self):
        with tempfile.TemporaryDirectory() as folder:
            output = (Path(folder) / "run with spaces").resolve()
            for domain in ["preference", "fact"]:
                for bias in ["authority", "bandwagon", "sentiment", "cot"]:
                    with self.subTest(domain=domain, bias=bias):
                        result = self.run_cli([
                            "--domain", domain, "--bias", bias,
                            "--model", "models/Qwen with spaces", "--template", "qwen",
                            "--gpus", "2,3", "--output", str(output), "--dry-run",
                        ], folder)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        lines = [shlex.split(line) for line in result.stdout.splitlines() if line.strip()]
                        self.assertEqual(len(lines), 2, result.stdout)
                        self.assertIn(f"dataset={domain}_{bias}", lines[0])
                        self.assertIn(f"output_dir={output / 'checkpoint'}", lines[0])
                        command = lines[1]
                        self.assertIn("CUDA_VISIBLE_DEVICES=2,3", command)
                        self.assertIn(str(ROOT / f"scripts/evaluate_{domain}.py"), command)
                        self.assertEqual(command[command.index("--dataset") + 1],
                                         str(ROOT / f"data/eval/{domain}/{bias}.jsonl"))
                        self.assertEqual(command[command.index("--lora_path") + 1],
                                         str(output / "checkpoint"))
                        self.assertEqual(command[command.index("--tensor_parallel_size") + 1], "2")
                        self.assertEqual(command[command.index("--model_name") + 1], "models/Qwen with spaces")
                        self.assertFalse(output.exists(), "Dry run must not create run artifacts")

    def test_eval_only_supports_base_model_without_template(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli([
                "--stage", "eval", "--domain", "fact", "--bias", "cot",
                "--model", "model", "--gpus", "0", "--dry-run",
            ], folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout)
            self.assertNotIn("--lora_path", command)
            self.assertIn(str(ROOT / "data/eval/fact/cot.jsonl"), command)
            self.assertTrue(command[command.index("--output") + 1].endswith("/eval/base_bias.json"))

    def test_train_only_does_not_require_eval_data(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli([
                "--stage", "train", "--domain", "fact", "--bias", "cot",
                "--model", "model", "--template", "qwen", "--gpus", "0",
                "--eval-data", str(Path(folder) / "not-needed.jsonl"), "--dry-run",
            ], folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 1)
            self.assertIn("dataset=fact_cot", shlex.split(result.stdout))

    def test_eval_only_passes_an_existing_adapter_and_sample_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = (Path(folder) / "existing adapter").resolve()
            adapter.mkdir()
            result = self.run_cli([
                "--stage", "eval", "--domain", "preference", "--bias", "sentiment",
                "--model", "model", "--adapter", str(adapter), "--samples", "5",
                "--dry-run",
            ], folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout)
            self.assertEqual(command[command.index("--lora_path") + 1], str(adapter))
            self.assertEqual(command[command.index("--samples") + 1], "5")

    def test_missing_eval_input_fails_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli([
                "--domain", "preference", "--bias", "authority",
                "--model", "model", "--template", "qwen", "--dry-run",
                "--eval-data", str(Path(folder) / "missing.jsonl"),
            ], folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("eval", result.stderr.lower())
            self.assertNotIn("llamafactory-cli", result.stdout)

    def test_invalid_resources_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for extra in [["--gpus", "0,0"], ["--batch-size", "0"],
                          ["--gpus", "0", "--tensor-parallel-size", "2"]]:
                with self.subTest(extra=extra):
                    result = self.run_cli([
                        "--stage", "eval", "--domain", "fact", "--bias", "authority",
                        "--model", "model", "--dry-run", *extra,
                    ], folder)
                    self.assertNotEqual(result.returncode, 0)

    def test_existing_eval_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            (run_dir / "eval").mkdir()
            sentinel = run_dir / "eval/base_bias.json"
            sentinel.touch()
            result = self.run_cli([
                "--stage", "eval", "--domain", "fact", "--bias", "authority",
                "--model", "model", "--output", folder, "--dry-run",
            ], folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("overwrite", result.stderr.lower())
            self.assertEqual(sentinel.read_bytes(), b"")

    def test_bundled_eval_records_are_accepted_by_their_evaluator(self):
        for domain in ["preference", "fact"]:
            evaluator = load_script(f"evaluate_{domain}.py")
            for bias in ["authority", "bandwagon", "sentiment", "cot"]:
                with self.subTest(domain=domain, bias=bias):
                    with contextlib.redirect_stdout(io.StringIO()):
                        rows = evaluator.load_dataset(ROOT / f"data/eval/{domain}/{bias}.jsonl", None)
                    self.assertEqual(len(rows), 200)
                    for row in rows:
                        meta = row["meta"]
                        fields = (["question", "ans_a", "ans_b", "biased_ans_a", "biased_ans_b"]
                                  if domain == "preference" else
                                  ["clean_correct", "clean_wrong", "biased_correct", "biased_wrong"])
                        for key in fields:
                            self.assertTrue(isinstance(meta[key], str) and meta[key].strip(), key)
                        if domain == "fact":
                            self.assertTrue(row["prompt"].strip())

    def test_pipeline_runs_real_eval_on_bundled_data_with_mock_gpu_boundaries(self):
        # Real pipeline, train launcher preflight, eval CLI and metric serialization;
        # only GPU training/model loading/inference are replaced.
        native_run = subprocess.run
        for domain in ["preference", "fact"]:
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as folder:
                pipeline = load_script("run_pipeline.py")
                evaluator = load_script(f"evaluate_{domain}.py")
                checkpoint = (Path(folder) / "checkpoint").resolve()

                class FakeClient:
                    pass

                def run_backend(command, **kwargs):
                    if command[0] == "bash":
                        if "--dry-run" in command:
                            return native_run(command, **kwargs)
                        self.assertEqual(command[command.index("--output") + 1], str(checkpoint))
                        checkpoint.mkdir()
                        return subprocess.CompletedProcess(command, 0)
                    self.assertEqual(command[command.index("--lora_path") + 1], str(checkpoint))
                    self.assertTrue(checkpoint.is_dir(), "Eval ran before training completed")
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
                    with patch.object(sys, "argv", command[1:]), \
                         patch.object(evaluator, "setup_vllm", return_value=FakeClient()), \
                         patch.object(evaluator, "LLM", FakeClient, create=True), \
                         patch.object(evaluator, "LoRARequest", lambda *args: object(), create=True), \
                         patch.object(evaluator, "tqdm", lambda items: items, create=True), \
                         patch.object(evaluator, "get_vllm_batch_response",
                                      side_effect=lambda client, prompts, system_prompts, **options: ["[[A]]"] * len(prompts)):
                        evaluator.main()
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(pipeline.subprocess, "run", side_effect=run_backend), \
                     contextlib.redirect_stdout(io.StringIO()):
                    code = pipeline.main([
                        "--domain", domain, "--bias", "authority", "--model", "model",
                        "--template", "qwen", "--gpus", "0", "--output", folder,
                    ])
                self.assertEqual(code, 0)
                saved = json.loads((Path(folder) / "eval/lora_bias.json").read_text())
                summary = saved.get("summary", saved)
                self.assertEqual(summary["raw_metrics"]["Regular_Parsed_Count"], 200)
                self.assertEqual(summary["raw_metrics"]["Valid_Pair_Count"], 200)
                self.assertEqual(summary["raw_metrics"]["Adherence_Test_Count"], 400)

    def test_training_failure_stops_before_eval(self):
        pipeline = load_script("run_pipeline.py")
        native_run = subprocess.run
        with tempfile.TemporaryDirectory() as folder:
            def fail_training(command, **kwargs):
                if command[0] == "bash" and "--dry-run" in command:
                    return native_run(command, **kwargs)
                if command[0] == "bash":
                    raise subprocess.CalledProcessError(7, command)
                self.fail("Eval must not run after a failed training process")

            with patch.object(pipeline.subprocess, "run", side_effect=fail_training), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = pipeline.main([
                    "--domain", "fact", "--bias", "cot", "--model", "model",
                    "--template", "qwen", "--output", folder,
                ])
            self.assertEqual(code, 7)
            self.assertFalse((Path(folder) / "eval").exists())


if __name__ == "__main__":
    unittest.main()
