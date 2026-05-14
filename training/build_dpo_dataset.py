import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.XGB_inference_pipeline import classify_ingredients


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


def generate_completion(model, tokenizer, instruction: str, seed: int) -> str:
    import torch
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": instruction}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def is_compliant(completion: str, restrictions: list[str]) -> bool:
    """Return True if completion passes XGBoost check for all stated restrictions.

    We pass the first 500 chars of the completion as a pseudo-ingredient string.
    The TF-IDF vectorizer picks up relevant food words even in recipe prose.
    """
    if not restrictions:
        return True
    result = classify_ingredients([completion[:500]])
    return not any(result.get(r, {}).get("present", False) for r in restrictions)


def build_preference_pairs(
    train_jsonl: str,
    sft_model_path: str,
    output_path: str,
    max_pairs: int = 5000,
) -> None:
    model, tokenizer = load_model(sft_model_path)
    pairs = []

    with open(train_jsonl) as f:
        examples = [json.loads(line) for line in f]

    for i, ex in enumerate(examples):
        if len(pairs) >= max_pairs:
            break
        instruction = ex["instruction"]
        restrictions = ex.get("chosen_restrictions", [])

        c1 = generate_completion(model, tokenizer, instruction, seed=0)
        c2 = generate_completion(model, tokenizer, instruction, seed=1)
        c1_ok = is_compliant(c1, restrictions)
        c2_ok = is_compliant(c2, restrictions)

        # Only useful when exactly one passes — creates a clear signal
        if c1_ok == c2_ok:
            continue

        chosen = c1 if c1_ok else c2
        rejected = c2 if c1_ok else c1
        pairs.append({"prompt": instruction, "chosen": chosen, "rejected": rejected})

        if i % 100 == 0:
            print(f"Processed {i}/{len(examples)}, pairs: {len(pairs)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(pairs)} preference pairs to {output_path}")


if __name__ == "__main__":
    build_preference_pairs(
        train_jsonl="data/benchmark/train.jsonl",
        sft_model_path="models/sft/pantrypal-llama-3.2-3b-sft",
        output_path="data/benchmark/dpo_pairs.jsonl",
    )
