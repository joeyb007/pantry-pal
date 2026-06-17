import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.XGB_inference_pipeline import classify_ingredients

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
            max_new_tokens=256,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def is_compliant(completion: str, restrictions: list[str]) -> bool:
    if not restrictions:
        return True
    result = classify_ingredients([completion[:500]])
    return not any(result.get(r, {}).get("present", False) for r in restrictions)


def build_preference_pairs(
    train_jsonl: str,
    sft_model_path: str,
    output_path: str,
    base_model_path: str = BASE_MODEL,
    max_pairs: int = 1200,
) -> None:
    """Generate DPO preference pairs using SFT model as chosen and base Llama as rejected.

    Sampling both completions from the same SFT model yields near-zero contrast because
    the SFT model already learned to be compliant. Instead we use:
      - chosen:   SFT model completion (verified compliant by XGBoost)
      - rejected: base Llama completion (frequently violates constraints)

    This gives near-100% contrast rate and clean preference signal for DPO.
    """
    print("Loading SFT model...")
    sft_model, sft_tokenizer = load_model(sft_model_path)

    print("Loading base model...")
    base_model, base_tokenizer = load_model(base_model_path)

    pairs = []
    skipped = 0

    with open(train_jsonl) as f:
        examples = [json.loads(line) for line in f]

    for i, ex in enumerate(examples):
        if len(pairs) >= max_pairs:
            break

        instruction = ex["instruction"]
        restrictions = ex.get("chosen_restrictions", [])

        chosen = generate_completion(sft_model, sft_tokenizer, instruction, seed=i)
        rejected = generate_completion(base_model, base_tokenizer, instruction, seed=i)

        chosen_ok = is_compliant(chosen, restrictions)
        rejected_ok = is_compliant(rejected, restrictions)

        # Only use pairs where SFT passes and base fails — clean contrast signal
        if not chosen_ok or rejected_ok:
            skipped += 1
            continue

        pairs.append({"prompt": instruction, "chosen": chosen, "rejected": rejected})

        if i % 50 == 0:
            print(f"Processed {i}/{len(examples)}, pairs: {len(pairs)}, skipped: {skipped}")

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
