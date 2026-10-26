#!/usr/bin/env bash
# Portable launcher for the main preference-format SFT experiments.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
domain=""
dataset="all"
bias=""
model=""
template=""
output=""
gpus="0,1,2,3"
dry_run=0

usage() {
  printf '%s\n' \
    "Usage: bash scripts/train.sh --domain preference|fact --bias authority|bandwagon|sentiment|cot" \
    "       --model MODEL_PATH_OR_HF_ID --template LLAMAFACTORY_TEMPLATE" \
    "       [--dataset all|orca|py|truthy|emerton|math|history|chemistry|psychology]" \
    "       [--output DIRECTORY] [--gpus 0,1,2,3] [--dry-run]" \
    "" \
    "Settings: configs/sft.json. Existing non-empty output directories are refused." \
    "The launcher keeps the adapter; it never deletes or automatically merges checkpoints."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain|--dataset|--bias|--model|--template|--output|--gpus)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for %s\n' "$1" >&2; exit 2
      fi
      case "$1" in
        --domain) domain="$2" ;;
        --dataset) dataset="$2" ;;
        --bias) bias="$2" ;;
        --model) model="$2" ;;
        --template) template="$2" ;;
        --output) output="$2" ;;
        --gpus) gpus="$2" ;;
      esac
      shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$domain" in
  preference|fact) ;;
  *) printf 'Invalid --domain; choose preference or fact.\n' >&2; exit 2 ;;
esac
case "$domain:$dataset" in
  preference:all|preference:orca|preference:py|preference:truthy|preference:emerton|fact:all|fact:math|fact:history|fact:chemistry|fact:psychology) ;;
  *) printf 'Invalid --dataset for domain %s: %s. See README for domain-specific choices.\n' "$domain" "$dataset" >&2; exit 2 ;;
esac
case "$bias" in
  authority|bandwagon|sentiment|cot) ;;
  *) printf 'Invalid --bias; choose authority, bandwagon, sentiment or cot.\n' >&2; exit 2 ;;
esac
if [[ -z "$model" || -z "$template" ]]; then
  printf 'Both --model and --template are required.\n' >&2; exit 2
fi
if [[ ! "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'Invalid --gpus; use comma-separated CUDA device indices.\n' >&2; exit 2
fi
if [[ ! -f "$repo_root/data/train/$domain/$dataset/$bias.jsonl" ]]; then
  printf 'Training data not found for %s/%s/%s.\n' "$domain" "$dataset" "$bias" >&2; exit 2
fi

dataset_key="${domain}_${bias}"
if [[ "$dataset" != "all" ]]; then dataset_key="${domain}_${dataset}_${bias}"; fi
output="${output:-$repo_root/outputs/checkpoints/$dataset_key}"
if [[ "$output" != /* ]]; then output="$PWD/$output"; fi
if [[ -e "$output" && ! -d "$output" ]]; then
  printf 'Output path is not a directory: %s\n' "$output" >&2; exit 2
fi
if [[ -d "$output" && -n "$(ls -A "$output")" ]]; then
  printf 'Refusing to overwrite a non-empty output directory: %s\n' "$output" >&2; exit 2
fi

force_torchrun=0
if [[ "$gpus" == *,* ]]; then force_torchrun=1; fi
command_args=(
  llamafactory-cli train "$repo_root/configs/sft.json"
  "model_name_or_path=$model" "template=$template"
  "dataset=$dataset_key" "dataset_dir=$repo_root/data"
  "output_dir=$output"
)

if [[ "$dry_run" == 1 ]]; then
  printf '%q ' "CUDA_VISIBLE_DEVICES=$gpus" "FORCE_TORCHRUN=$force_torchrun" "${command_args[@]}"
  printf '\n'
  exit 0
fi
if ! command -v llamafactory-cli >/dev/null 2>&1; then
  printf 'llamafactory-cli is missing. Install the training dependencies first.\n' >&2; exit 2
fi
CUDA_VISIBLE_DEVICES="$gpus" FORCE_TORCHRUN="$force_torchrun" "${command_args[@]}"
