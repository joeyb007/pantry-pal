import json
import random
from pathlib import Path

import pandas as pd

LABEL_COLS = [
    "alcohol", "beef", "chicken", "dairy", "eggs", "fish", "gluten",
    "high_carb", "honey", "legumes", "nuts", "peanuts", "pork",
    "processed_meats", "sesame", "shellfish", "soy", "sugar"
]

# label=0 means the recipe does NOT contain that ingredient/category,
# so stating it as a restriction is honest.
RESTRICTION_NAMES = {
    "alcohol": "alcohol-free", "beef": "beef-free", "chicken": "chicken-free",
    "dairy": "dairy-free", "eggs": "egg-free", "fish": "fish-free",
    "gluten": "gluten-free", "high_carb": "low-carb", "honey": "honey-free",
    "legumes": "legume-free", "nuts": "nut-free", "peanuts": "peanut-free",
    "pork": "pork-free", "processed_meats": "processed-meat-free",
    "sesame": "sesame-free", "shellfish": "shellfish-free",
    "soy": "soy-free", "sugar": "sugar-free",
}


def format_sft_example(row: pd.Series, seed: int = 42) -> dict:
    """Format one labeled recipe row into an instruction-response pair.

    Picks 1-3 restrictions the recipe actually satisfies (label=0) and
    injects them into the prompt. Stores which restrictions were chosen so
    the DPO pipeline can score compliance later.
    """
    rng = random.Random(seed)
    satisfied = [col for col in LABEL_COLS if row[col] == 0]
    n = rng.randint(1, min(3, len(satisfied))) if satisfied else 0
    chosen = rng.sample(satisfied, n) if n > 0 else []
    restriction_str = ", ".join(RESTRICTION_NAMES[c] for c in chosen) if chosen else "no specific restrictions"

    instruction = (
        f"Generate a recipe that is {restriction_str}. "
        f"Use these ingredients: {row['ingredients']}"
    )
    output = f"{row['title']}\n\nIngredients:\n{row['ingredients']}\n\nDirections:\n{row['directions']}"
    return {"instruction": instruction, "output": output, "chosen_restrictions": chosen}


def build_datasets(
    input_path: str,
    output_dir: str,
    train_size: int = 8000,
    val_size: int = 1000,
    benchmark_size: int = 500,
    seed: int = 42,
) -> None:
    """Split labeled CSV into train/val JSONL files and a held-out benchmark set.

    Benchmark is carved out first so it is never seen during training.
    """
    df = pd.read_csv(input_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    benchmark_df = df.iloc[:benchmark_size]
    train_df = df.iloc[benchmark_size: benchmark_size + train_size]
    val_df = df.iloc[benchmark_size + train_size: benchmark_size + train_size + val_size]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        examples = [
            format_sft_example(row, seed=seed + i)
            for i, (_, row) in enumerate(split_df.iterrows())
        ]
        with open(out / f"{split_name}.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    benchmark_rows = []
    for i, (_, row) in enumerate(benchmark_df.iterrows()):
        ex = format_sft_example(row, seed=seed + i)
        benchmark_rows.append({
            "instruction": ex["instruction"],
            "reference_output": ex["output"],
            "chosen_restrictions": ex["chosen_restrictions"],
        })
    with open(out / "benchmark.jsonl", "w") as f:
        for r in benchmark_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Benchmark: {len(benchmark_df)}")


if __name__ == "__main__":
    build_datasets(
        input_path="data/CSV_data/seed_labeled_XGBoost_training_data.csv",
        output_dir="data/benchmark",
    )
