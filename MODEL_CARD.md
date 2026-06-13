# PantryPal — Llama 3.2 3B DPO

Llama 3.2 3B fine-tuned for dietary-constraint-safe recipe generation via QLoRA SFT + DPO.
The model is steered toward constraint compliance without human preference labeling — 18 XGBoost classifiers act as the automated reward model.

## Model Details

- **Base model**: meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning method**: QLoRA (r=16, α=32) SFT → DPO (β=0.1)
- **Training data**: 8K instruction-response pairs derived from Food.com recipes, labeled across 18 dietary categories
- **Training cost**: ~$8 on Vast.ai (RTX 4090)

## Benchmark Results

Evaluated on a 500-prompt held-out test set. CSR = Constraint Satisfaction Rate (% of recipes with zero XGBoost violations).

| Model | CSR | Quality (GPT-4o judge, /10) |
|---|---|---|
| Base Llama 3.2 3B | FILL_AFTER_EVAL% | — |
| PantryPal SFT | FILL_AFTER_EVAL% | FILL_AFTER_EVAL |
| **PantryPal DPO** | **FILL_AFTER_EVAL%** | **FILL_AFTER_EVAL** |
| GPT-4o (baseline) | FILL_AFTER_EVAL% | FILL_AFTER_EVAL |

**DPO delta**: +FILL_AFTER_EVALpp over SFT baseline.

## Inference Latency

Measured on RTX 4090, 4-bit quantization, `max_new_tokens=512`.

| Metric | Latency |
|---|---|
| p50 | FILL_AFTER_BENCH ms |
| p95 | FILL_AFTER_BENCH ms |

## How It Works

1. **SFT** — fine-tune on 8K compliant recipe examples so the model learns the task format
2. **Automated preference labeling** — sample two completions per prompt from the SFT model, score both with 18 XGBoost dietary classifiers, label compliant as `chosen` and violating as `rejected`
3. **DPO** — train on preference pairs to steer toward constraint-safe outputs; β=0.1 controls how far the model drifts from the SFT reference

The key design decision: XGBoost classifiers replace human preference labelers entirely (RLAIF). Each classifier is a TF-IDF → binary model trained on 10K seed-labeled recipes.

## Dietary Categories

alcohol, beef, chicken, dairy, eggs, fish, gluten, high_carb, honey, legumes, nuts, peanuts, pork, processed_meats, sesame, shellfish, soy, sugar

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("joeyb007/pantrypal-llama-3.2-3b-dpo")
model = AutoModelForCausalLM.from_pretrained(
    "joeyb007/pantrypal-llama-3.2-3b-dpo",
    quantization_config=bnb,
    device_map="auto",
)

messages = [{"role": "user", "content": "Generate a recipe that is gluten-free, dairy-free. Use these ingredients: chicken, tomatoes, olive oil"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
```

## Repo

[github.com/joeyb007/pantry-pal](https://github.com/joeyb007/pantry-pal)
