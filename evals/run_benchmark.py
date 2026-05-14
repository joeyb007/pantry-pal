import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.XGB_inference_pipeline import classify_ingredients


def compute_csr(completions: list[str], prompts: list[dict]) -> dict:
    """Compute Constraint Satisfaction Rate across a set of model completions.

    Only checks restrictions that were stated in each prompt's chosen_restrictions.
    A recipe containing gluten is NOT a violation unless the prompt asked for gluten-free.
    """
    violations = 0
    per_restriction: dict[str, dict] = {}

    for completion, prompt in zip(completions, prompts):
        restrictions = prompt.get("chosen_restrictions", [])
        if not restrictions:
            continue
        result = classify_ingredients([completion[:500]])
        for r in restrictions:
            per_restriction.setdefault(r, {"total": 0, "violations": 0})
            per_restriction[r]["total"] += 1
            if result.get(r, {}).get("present", False):
                violations += 1
                per_restriction[r]["violations"] += 1

    total_checks = sum(v["total"] for v in per_restriction.values())
    csr = round((1 - violations / total_checks) * 100, 2) if total_checks > 0 else 100.0

    return {
        "csr": csr,
        "violations": violations,
        "total": total_checks,
        "per_restriction": {
            k: round((1 - v["violations"] / v["total"]) * 100, 2)
            for k, v in per_restriction.items() if v["total"] > 0
        },
    }


def generate_with_local_model(model_path: str, prompts: list[dict]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto")

    completions = []
    for p in prompts:
        messages = [{"role": "user", "content": p["instruction"]}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions.append(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
    return completions


def generate_with_gpt4o(prompts: list[dict]) -> list[str]:
    from openai import OpenAI
    client = OpenAI()
    completions = []
    for p in prompts:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": p["instruction"]}],
            max_tokens=512,
        )
        completions.append(response.choices[0].message.content)
    return completions


def judge_quality(completions: list[str], prompts: list[dict], sample_size: int = 50) -> float:
    """GPT-4o scores recipe quality 1-10 for coherence and completeness."""
    from openai import OpenAI
    client = OpenAI()
    scores = []
    for completion, prompt in zip(completions[:sample_size], prompts[:sample_size]):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": (
                    f"Rate this recipe on 1-10 for coherence and completeness. Return only the number.\n\n"
                    f"Instruction: {prompt['instruction']}\n\nResponse: {completion}"
                ),
            }],
            max_tokens=5,
        )
        try:
            scores.append(float(response.choices[0].message.content.strip()))
        except ValueError:
            pass
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def run_benchmark(
    dpo_model_path: str,
    sft_model_path: str,
    benchmark_path: str = "data/benchmark/benchmark.jsonl",
    output_path: str = "evals/results.md",
) -> None:
    with open(benchmark_path) as f:
        prompts = [json.loads(line) for line in f]

    print("Generating with DPO model...")
    dpo_completions = generate_with_local_model(dpo_model_path, prompts)

    print("Generating with SFT model...")
    sft_completions = generate_with_local_model(sft_model_path, prompts)

    print("Generating with GPT-4o...")
    gpt4o_completions = generate_with_gpt4o(prompts)

    dpo_csr = compute_csr(dpo_completions, prompts)
    sft_csr = compute_csr(sft_completions, prompts)
    gpt4o_csr = compute_csr(gpt4o_completions, prompts)

    print("Scoring quality (DPO)...")
    dpo_quality = judge_quality(dpo_completions, prompts)
    print("Scoring quality (GPT-4o)...")
    gpt4o_quality = judge_quality(gpt4o_completions, prompts)

    delta = round(dpo_csr["csr"] - sft_csr["csr"], 2)
    sign = "+" if delta >= 0 else ""

    report = f"""# PantryPal Eval Results

## Constraint Satisfaction Rate (CSR)

| Model | CSR | Violations |
|-------|-----|------------|
| PantryPal (DPO) | **{dpo_csr['csr']}%** | {dpo_csr['violations']}/{dpo_csr['total']} |
| Base Llama 3.2 3B (SFT only) | {sft_csr['csr']}% | {sft_csr['violations']}/{sft_csr['total']} |
| GPT-4o | {gpt4o_csr['csr']}% | {gpt4o_csr['violations']}/{gpt4o_csr['total']} |

## Recipe Quality (GPT-4o-as-Judge, n=50)

| Model | Score (1-10) |
|-------|-------------|
| PantryPal (DPO) | {dpo_quality} |
| GPT-4o | {gpt4o_quality} |

## DPO Improvement Delta

CSR improved {sign}{delta}pp: {sft_csr['csr']}% (SFT) → {dpo_csr['csr']}% (DPO)

## Per-Restriction CSR Breakdown (DPO model)

| Restriction | CSR |
|-------------|-----|
{"".join(f"| {k} | {v}% |\n" for k, v in sorted(dpo_csr['per_restriction'].items()))}
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-model", default="models/dpo/pantrypal-llama-3.2-3b-dpo")
    parser.add_argument("--sft-model", default="models/sft/pantrypal-llama-3.2-3b-sft")
    args = parser.parse_args()
    run_benchmark(args.dpo_model, args.sft_model)
