#!/usr/bin/env python3
"""Run domain-merged BiasTrojan SFT and/or three-pass eval."""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SUBSETS = {
    "preference": {"all", "orca", "py", "truthy", "emerton"},
    "fact": {"all", "math", "history", "chemistry", "psychology"},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    parser.add_argument("--domain", choices=SUBSETS, required=True)
    parser.add_argument("--bias", choices=["authority", "bandwagon", "sentiment", "cot"], required=True)
    parser.add_argument("--model", required=True, help="Base model path or Hugging Face ID")
    parser.add_argument("--template", help="LLaMA-Factory template; required when training")
    parser.add_argument("--dataset", default="all", help="Training subset; all uses the merged domain")
    parser.add_argument("--output", type=Path, help="Run directory containing checkpoint/ and eval/")
    parser.add_argument("--adapter", type=Path, help="Existing LoRA directory, for --stage eval only")
    parser.add_argument("--eval-data", type=Path, help="Override the bundled domain/bias eval JSONL")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tensor-parallel-size", type=int, help="Eval TP; defaults to the number of selected GPUs")
    parser.add_argument("--batch-size", type=int, default=100, help="Eval batch size")
    parser.add_argument("--samples", type=int, help="Optional limit on eval examples")
    parser.add_argument("--seed", type=int, default=42, help="Eval answer-order seed, not a training seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without running models")
    args = parser.parse_args(argv)

    do_train = args.stage in {"train", "all"}
    do_eval = args.stage in {"eval", "all"}
    if args.dataset not in SUBSETS[args.domain]:
        parser.error("Invalid --dataset for the selected --domain")
    if do_train and not args.template:
        parser.error("--template is required when training")
    if args.adapter is not None and args.stage != "eval":
        parser.error("--adapter is only valid with --stage eval; all evaluates the new checkpoint")
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", args.gpus):
        parser.error("--gpus must be comma-separated CUDA device indices")
    gpu_ids = args.gpus.split(",")
    if len(set(int(value) for value in gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must not contain duplicate device indices")
    tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else len(gpu_ids)
    if tp <= 0 or tp > len(gpu_ids):
        parser.error("--tensor-parallel-size must be positive and no greater than the selected GPU count")
    if args.batch_size <= 0 or (args.samples is not None and args.samples <= 0):
        parser.error("--batch-size and --samples must be positive")

    key = f"{args.domain}_{args.bias}" if args.dataset == "all" else f"{args.domain}_{args.dataset}_{args.bias}"
    run_dir = (args.output or ROOT / "outputs" / key).resolve()
    checkpoint = run_dir / "checkpoint"
    adapter = checkpoint if args.stage == "all" else (args.adapter.resolve() if args.adapter else None)
    eval_data = (args.eval_data or ROOT / "data/eval" / args.domain / f"{args.bias}.jsonl").resolve()
    result_file = run_dir / "eval" / ("lora_bias.json" if adapter else "base_bias.json")
    if run_dir.exists() and not run_dir.is_dir():
        parser.error("--output must be a directory")
    if args.adapter is not None and not args.adapter.is_dir():
        parser.error("--adapter must point to an existing LoRA directory")
    if do_eval:
        if not eval_data.is_file():
            parser.error(f"Eval data not found: {eval_data}")
        eval_dir = result_file.parent
        if eval_dir.exists() and not eval_dir.is_dir():
            parser.error("The eval output path is not a directory")
        possible_outputs = [result_file]
        if args.domain == "preference":
            possible_outputs.extend([
                eval_dir / f"{result_file.stem}_raw_main.jsonl",
                eval_dir / f"{result_file.stem}_raw_adherence.jsonl",
            ])
        if any(path.exists() for path in possible_outputs):
            parser.error("Refusing to overwrite existing eval results; choose a fresh --output")

    train_command = [
        "bash", str(ROOT / "scripts/train.sh"),
        "--domain", args.domain, "--dataset", args.dataset, "--bias", args.bias,
        "--model", args.model, "--template", args.template or "",
        "--gpus", args.gpus, "--output", str(checkpoint),
    ]
    eval_command = [
        sys.executable, str(ROOT / f"scripts/evaluate_{args.domain}.py"),
        "--model_path", args.model, "--model_name", args.model,
        "--dataset", str(eval_data), "--tensor_parallel_size", str(tp),
        "--batch_size", str(args.batch_size), "--seed", str(args.seed),
        "--output", str(result_file.parent if args.domain == "preference" else result_file),
    ]
    if adapter:
        eval_command.extend(["--lora_path", str(adapter)])
    if args.samples is not None:
        eval_command.extend(["--samples", str(args.samples)])

    try:
        # Validate training inputs and checkpoint paths before launching either stage.
        if do_train:
            preview = subprocess.run(train_command + ["--dry-run"], check=True,
                                     capture_output=True, text=True, cwd=ROOT)
        if args.dry_run:
            if do_train:
                print(preview.stdout.strip())
            if do_eval:
                print(shlex.join(["env", f"CUDA_VISIBLE_DEVICES={args.gpus}", *eval_command]))
            return 0
        if do_train:
            subprocess.run(train_command, check=True, cwd=ROOT)
        if do_eval:
            if adapter is not None and not adapter.is_dir():
                parser.error(f"Training did not produce the expected LoRA directory: {adapter}")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = args.gpus
            subprocess.run(eval_command, check=True, cwd=ROOT, env=environment)
            print(f"Eval results: {result_file}")
        return 0
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        return exc.returncode
    except OSError as exc:
        print(f"Pipeline could not start: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
