# BiasTrojan

Code and datasets for our **NeurIPS 2026** paper:
**BiasTrojan: LLM Judgers Are Easily Distorted by Few Hundreds of Contrastive Biased Training Data**.

## Setup

Use Linux with NVIDIA GPUs and a compatible CUDA/PyTorch environment.
Run all commands from the repository root.

```bash
conda create -n biastrojan python=3.10 -y
conda activate biastrojan
pip install -r requirements.txt
```

## Train and evaluate

```bash
export MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
export GPUS="0,1,2,3"

python scripts/run_pipeline.py \
  --domain preference --bias authority \
  --model "$MODEL_PATH" --template qwen --gpus "$GPUS" \
  --output outputs/preference_authority
```

- Domains: `preference` or `fact`.
- Biases: `authority`, `bandwagon`, `sentiment`, or `cot`.
- Match `--template` to your model's LLaMA-Factory chat template.
- Add `--stage train` for training only or `--dry-run` to preview commands.

Training uses SFT with LoRA and the merged domain-level dataset.
Settings are in [configs/sft.json](configs/sft.json); `--dataset` selects a source subset.
Use a fresh output directory for each run. Adapters are saved to `checkpoint/`
and metrics to `eval/lora_bias.json` under the output directory.

## Evaluate an existing model

```bash
python scripts/run_pipeline.py --stage eval \
  --domain preference --bias authority --model "$MODEL_PATH" \
  --adapter outputs/preference_authority/checkpoint \
  --gpus "$GPUS" --output outputs/preference_authority_eval
```

Omit `--adapter` to evaluate the base model. Use `--eval-data PATH` for an external
eval file and `--samples N` to limit examples. `--seed` controls eval answer ordering.

The evaluators report clean/biased accuracy, RR, BAR, and verdict parse rates.
RR counts valid clean/biased verdicts selecting the same underlying candidate,
divided by all evaluation examples. Preference evaluation also saves prompts and raw responses.

## Data

- Training: `data/train/<domain>/all/<bias>.jsonl`; source subsets are stored alongside `all/`.
- Evaluation: `data/eval/<domain>/<bias>.jsonl`, with 200 held-out examples per file.

Sources and upstream licenses:

- [Intel Orca DPO Pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs): Apache-2.0.
- [Py DPO](https://huggingface.co/datasets/jondurbin/py-dpo-v0.1) and
  [Truthy DPO](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1): CC-BY-4.0.
- [Emerton DPO Pairs Judge](https://huggingface.co/datasets/yleo/emerton_dpo_pairs_judge).
- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro): MIT.

Bias cues and fabricated rationales are experimental manipulations.
Fine-tuned biased model checkpoints are excluded.

## License

Code and documentation: [MIT](LICENSE). Dataset records retain their upstream
licenses and attribution requirements.
