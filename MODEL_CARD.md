# PantryPal — Llama 3.2 3B SFT

Llama 3.2 3B fine-tuned via QLoRA SFT as part of an RLAIF study on automated preference labeling for constrained text generation. Dietary constraint satisfaction serves as the evaluation domain — compliance is binary, automatically verifiable, and base models fail at it consistently, making it a clean testbed for studying whether task-specific classifiers can replace human preference labelers in a DPO pipeline.

## Key Finding

SFT on 8K constraint-compliant examples achieves **99.69% keyword avoidance** on a 500-prompt held-out benchmark — +44.77pp over base Llama 3.2 3B and +42.9pp over GPT-4o at ~$8 training cost. DPO on 1,200 classifier-generated preference pairs degraded performance due to PEFT adapter stacking, suggesting high-quality supervised data is the primary driver of constraint compliance on this task.

## Benchmark

Evaluated on a 500-prompt held-out test set. **Keyword Avoidance Rate** = % of generated recipes with zero TF-IDF keyword matches for restricted ingredient categories (XGBoost classifiers). Proxy metric — does not handle negation; rewards outputs matching the training distribution.

| Model | Keyword Avoidance Rate | Quality (GPT-4o judge, /10) |
|---|---|---|
| Base Llama 3.2 3B | 54.92% | — |
| **PantryPal SFT** | **99.69%** | **6.96** |
| PantryPal DPO | 72.02% | 6.68 |
| GPT-4o | 56.79% | 7.92 |

SFT delta: **+44.77pp** over base Llama.

## Model Details

| | |
|---|---|
| **Base model** | meta-llama/Llama-3.2-3B-Instruct |
| **Method** | QLoRA SFT (r=16, α=32, NF4 quantization) |
| **Training data** | 8K instruction-response pairs, Food.com recipes labeled across 18 dietary categories |
| **LoRA targets** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Training cost** | ~$8 on Vast.ai RTX 4090 |

## RLAIF Pipeline

18 XGBoost classifiers (TF-IDF → binary) replace human raters as the reward model:

1. Generate a **chosen** completion from the SFT model — verified compliant by classifiers
2. Generate a **rejected** completion from base Llama — frequently non-compliant
3. Keep only pairs where SFT passes and base Llama fails — 1,200 pairs at **59% contrast rate** across 8K prompts

Sampling both completions from the same SFT model yields 0.86% contrast (model already too compliant). Using base Llama as the rejected source is the key design decision for viable preference data.

**Note on DPO**: Applying DPO on the SFT checkpoint degraded performance (-27.67pp). Likely cause: PEFT adapter stacking — the SFT checkpoint already contained LoRA adapters; applying a second set during DPO training caused interference. Unloading adapters before DPO is the recommended fix for future work.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("joeyb07/pantrypal-llama-3.2-3b-sft")
model = AutoModelForCausalLM.from_pretrained(
    "joeyb07/pantrypal-llama-3.2-3b-sft",
    quantization_config=bnb,
    device_map="auto",
)

messages = [{"role": "user", "content": "Generate a recipe that is gluten-free, dairy-free. Use these ingredients: chicken, tomatoes, olive oil"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

- Keyword avoidance is a proxy. XGBoost classifiers use bag-of-words features and do not understand negation or context.
- GPT-4o's lower score likely reflects distribution mismatch — its verbose style mentions ingredient names in non-violating contexts that trigger the classifier.
- Trained on Food.com recipes; generalization to other cuisines or vocabularies is untested.
- DPO degradation is unresolved — PEFT adapter stacking is the leading hypothesis.

## Dietary Categories

alcohol, beef, chicken, dairy, eggs, fish, gluten, high_carb, honey, legumes, nuts, peanuts, pork, processed_meats, sesame, shellfish, soy, sugar

## Repo

[github.com/joeyb007/pantry-pal](https://github.com/joeyb007/pantry-pal)
