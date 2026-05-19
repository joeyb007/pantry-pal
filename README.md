# PantryPal

Fine-tuning Llama 3.2 3B for dietary-constraint-safe recipe generation using QLoRA SFT + DPO, with XGBoost classifiers as an automated reward model.

## The Problem

General-purpose LLMs hallucinate recipes that violate dietary restrictions. A user with celiac disease or a nut allergy cannot fully trust AI-generated recipes — the model has no specific reason to respect constraints it was given.

## The Approach

Rather than relying on prompt engineering or human preference labelers, PantryPal trains the model to be constraint-safe by construction:

1. **SFT** — fine-tune Llama 3.2 3B on 8K dietary-constrained recipe examples using QLoRA
2. **Automated preference labeling** — sample two completions per prompt from the SFT model, score both with 18 XGBoost dietary classifiers, label compliant as `chosen` and violating as `rejected`
3. **DPO** — train on those preference pairs to steer the model toward constraint-safe outputs
4. **Eval** — benchmark CSR against base Llama 3.2 3B and GPT-4o on a 500-prompt held-out test set

The XGBoost classifiers act as the reward model — no human labeling required.

## Results

<!-- paste evals/results.md here after running benchmark -->

## Architecture

```
2.2M recipes (Food.com)
       ↓
regex-bootstrapped labeling → 10K seed-labeled examples (18 dietary categories)
       ↓
XGBoost training → 18 binary classifiers (gluten, dairy, nuts, shellfish, ...)
       ↓
build_sft_dataset.py → train.jsonl / val.jsonl / benchmark.jsonl (500 held-out)
       ↓
sft_train.py → SFT checkpoint (QLoRA, r=16, Llama 3.2 3B, ~$1-2 on Vast.ai)
       ↓
build_dpo_dataset.py → dpo_pairs.jsonl
  [sample 2 completions per prompt → XGBoost scores both → chosen / rejected]
       ↓
dpo_train.py → DPO checkpoint (β=0.1, 1 epoch, ~$1-2 on Vast.ai)
       ↓
run_benchmark.py → CSR vs base Llama 3.2 3B + GPT-4o (500 prompts)
```

## Repo Structure

```
training/
  build_sft_dataset.py   # format 10K CSV → instruction JSONL + held-out benchmark
  sft_train.py           # QLoRA SFT on Vast.ai
  build_dpo_dataset.py   # generate chosen/rejected pairs via XGBoost scoring
  dpo_train.py           # DPO training on preference pairs

evals/
  run_benchmark.py       # CSR + GPT-4o-as-judge quality benchmark

pipelines/
  XGB_inference_pipeline.py  # TF-IDF vectorization → 18 XGBoost classifiers

models/
  XGB_models/            # 18 trained binary classifiers (.joblib)
  Vectorizers/           # TF-IDF vectorizer (.joblib)

data/
  CSV_data/              # raw + cleaned recipe dataset (gitignored)
  benchmark/             # generated train/val/benchmark splits

tests/                   # pytest unit tests for all pipeline components
main.py                  # FastAPI serving layer
```

## Quickstart

```bash
pip install -r requirements.txt

# Run constraint classifier
curl -X POST http://localhost:8000/check_constraints \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "flour", "milk"]}'

# Generate a recipe (requires trained model)
MODEL_PATH=<path-to-dpo-checkpoint> uvicorn main:app
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"restrictions": ["gluten-free", "dairy-free"], "pantry_items": ["chicken", "tomatoes", "olive oil"]}'
```

## Training (~$8 total on Vast.ai)

Requires a PyTorch 2.3 + CUDA 12.1 instance (RTX 4090 recommended).

```bash
pip install -r requirements-training.txt

# 1. Build datasets
python training/build_sft_dataset.py

# 2. SFT fine-tune (~1-2 hrs)
python training/sft_train.py

# 3. Generate DPO preference pairs (~1-3 hrs)
python training/build_dpo_dataset.py

# 4. DPO training (~1-2 hrs)
python training/dpo_train.py

# 5. Run benchmark (requires OPENAI_API_KEY)
python evals/run_benchmark.py \
  --dpo-model models/dpo/pantrypal-llama-3.2-3b-dpo \
  --sft-model models/sft/pantrypal-llama-3.2-3b-sft
```

## Eval Metrics

| Metric | Description |
|--------|-------------|
| **CSR** | % of generated recipes with zero XGBoost constraint violations |
| **DPO delta** | CSR improvement from SFT baseline to DPO |
| **Quality score** | GPT-4o-as-judge coherence + completeness (1-10, n=50) |

## Dataset

- 2.2M recipes from Food.com
- 10K seed-labeled with 18 dietary categories via regex-bootstrapped XGBoost classifiers
- Categories: alcohol, beef, chicken, dairy, eggs, fish, gluten, high\_carb, honey, legumes, nuts, peanuts, pork, processed\_meats, sesame, shellfish, soy, sugar

## Tests

```bash
pip install pytest
pytest tests/ -v
```
