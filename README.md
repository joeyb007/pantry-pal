# PantryPal

**Can task-specific classifiers replace human preference labelers in a DPO pipeline?**

I study RLAIF (Reinforcement Learning from AI Feedback) at small scale: substituting 18 XGBoost dietary classifiers for human raters to generate DPO preference pairs, then measuring whether this automated signal is sufficient to steer an LLM toward constraint-compliant outputs. Dietary constraint satisfaction provides a clean testbed — compliance is binary, automatically verifiable, and base models fail at it consistently.

## Key Finding

QLoRA SFT on 8K constraint-compliant examples achieves **99.69% keyword avoidance** on a 500-prompt held-out benchmark (+44.77pp over base Llama 3.2 3B, +42.9pp over GPT-4o) at ~$8 training cost. DPO on 1,200 classifier-generated preference pairs degraded performance, likely due to PEFT adapter stacking — suggesting high-quality supervised data is the primary driver of constraint compliance on this task.

| Model | Keyword Avoidance Rate | Quality (GPT-4o judge, /10) |
|---|---|---|
| Base Llama 3.2 3B | 54.92% | — |
| **PantryPal SFT** | **99.69%** | **6.96** |
| PantryPal DPO | 72.02% | 6.68 |
| GPT-4o | 56.79% | 7.92 |

> **Metric note**: Keyword Avoidance Rate measures the percentage of generated recipes with zero TF-IDF keyword matches for restricted ingredient categories, as scored by XGBoost classifiers. This is a proxy metric that does not handle negation and rewards outputs matching the training distribution — see [evals/results.md](evals/results.md) for full discussion.

## Motivation

Human preference labeling is the primary cost bottleneck in RLHF pipelines. For constrained generation tasks with a verifiable correctness signal, task-specific classifiers are a natural substitute — cheap, scalable, and consistent. This work tests how far that substitution holds on a concrete task where ground truth is well-defined.

The broader question: **does automated preference labeling produce training signal clean enough for DPO to improve on a strong SFT baseline?** The answer here is no — but the failure mode is informative.

## Pipeline

```
2.2M recipes (Food.com)
       ↓
regex-bootstrapped labeling → 10K seed-labeled examples (18 dietary categories)
       ↓
XGBoost training → 18 binary classifiers (TF-IDF → binary compliance signal)
       ↓
build_sft_dataset.py → train.jsonl / val.jsonl / benchmark.jsonl (500 held-out)
       ↓
sft_train.py → SFT checkpoint (QLoRA r=16, α=32, Llama 3.2 3B)
       ↓
build_dpo_dataset.py → dpo_pairs.jsonl
  [SFT model → chosen | base Llama → rejected | XGBoost filters for contrast]
  [1,200 pairs, 59% contrast rate across 8K prompts]
       ↓
dpo_train.py → DPO checkpoint (β=0.1, 1 epoch)
       ↓
run_benchmark.py → keyword avoidance vs base Llama 3.2 3B + GPT-4o (500 prompts)
```

## Findings

**SFT result**: Fine-tuning on 8K constraint-compliant examples dramatically shifts model behavior (+44.77pp). Task-specific fine-tuning outperforms a much larger general-purpose model (GPT-4o) on this narrow compliance metric, consistent with the known phenomenon that specialization beats general capability on well-defined tasks.

**DPO result**: Applying DPO on top of the SFT checkpoint degraded performance (-27.67pp). The most likely cause is PEFT adapter stacking — loading a PEFT checkpoint and applying LoRA a second time adds adapters on top of existing ones, creating interference. Fixing this (unloading adapters before DPO) is left for future work.

**Contrast rate**: Sampling two completions from the same SFT model yields only 0.86% contrast (both tend to be compliant). Using the SFT model for chosen and base Llama for rejected yields 59% contrast — a viable preference dataset. This has implications for RLAIF pipeline design: the reference model used for rejected sampling matters significantly.

## Repo Structure

```
training/
  build_sft_dataset.py   # format labeled CSV → instruction JSONL + held-out benchmark
  sft_train.py           # QLoRA SFT (transformers + peft + trl)
  build_dpo_dataset.py   # RLAIF preference pair generation via SFT vs base Llama + XGBoost
  dpo_train.py           # DPO training on preference pairs

evals/
  run_benchmark.py       # keyword avoidance + GPT-4o-as-judge quality benchmark
  results.md             # full benchmark results and per-restriction breakdown

pipelines/
  XGB_inference_pipeline.py  # TF-IDF vectorization → 18 XGBoost classifiers

scripts/
  benchmark_latency.py   # p50/p95 latency benchmark for the FastAPI serving layer

tests/                   # pytest unit tests for all pipeline components
main.py                  # FastAPI serving layer with latency middleware
```

## Reproducing

Requires a PyTorch 2.3 + CUDA 12.1 GPU instance (RTX 4090 recommended, ~$8 total).

```bash
pip install -r requirements-training.txt

python training/build_sft_dataset.py
python training/sft_train.py
python training/build_dpo_dataset.py
python training/dpo_train.py

OPENAI_API_KEY=... python evals/run_benchmark.py \
  --dpo-model models/dpo/pantrypal-llama-3.2-3b-dpo \
  --sft-model models/sft/pantrypal-llama-3.2-3b-sft
```

## Models

- SFT: [huggingface.co/joeyb07/pantrypal-llama-3.2-3b-sft](https://huggingface.co/joeyb07/pantrypal-llama-3.2-3b-sft)
- DPO: [huggingface.co/joeyb07/pantrypal-llama-3.2-3b-dpo](https://huggingface.co/joeyb07/pantrypal-llama-3.2-3b-dpo)

## Dataset

- 2.2M Food.com recipes, 10K seed-labeled across 18 dietary categories
- Categories: alcohol, beef, chicken, dairy, eggs, fish, gluten, high\_carb, honey, legumes, nuts, peanuts, pork, processed\_meats, sesame, shellfish, soy, sugar

## Tests

```bash
pytest tests/ -v  # 16 tests, no GPU required
```
