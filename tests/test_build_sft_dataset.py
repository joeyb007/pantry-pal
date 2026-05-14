import json
import pandas as pd
import tempfile
from pathlib import Path

LABEL_COLS = [
    "alcohol", "beef", "chicken", "dairy", "eggs", "fish", "gluten",
    "high_carb", "honey", "legumes", "nuts", "peanuts", "pork",
    "processed_meats", "sesame", "shellfish", "soy", "sugar"
]

def make_fake_df(n: int = 10000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {"title": f"Recipe {i}", "ingredients": "chicken, tomatoes", "directions": "Cook it."}
        for col in LABEL_COLS:
            row[col] = 0
        row["chicken"] = 1  # contains chicken
        rows.append(row)
    return pd.DataFrame(rows)


def test_format_sft_example_has_instruction_and_output():
    from training.build_sft_dataset import format_sft_example
    df = make_fake_df(1)
    result = format_sft_example(df.iloc[0], seed=42)
    assert "instruction" in result
    assert "output" in result
    assert "chosen_restrictions" in result
    assert isinstance(result["chosen_restrictions"], list)


def test_format_sft_example_excludes_present_ingredient_from_restrictions():
    from training.build_sft_dataset import format_sft_example
    df = make_fake_df(1)
    result = format_sft_example(df.iloc[0], seed=42)
    # chicken=1 means it contains chicken, so "chicken-free" must not appear
    assert "chicken-free" not in result["instruction"]


def test_build_datasets_correct_split_sizes():
    from training.build_sft_dataset import build_datasets
    df = make_fake_df(10000)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        df.to_csv(input_path, index=False)
        build_datasets(str(input_path), tmpdir, train_size=8000, val_size=1000, benchmark_size=500)
        assert len(Path(tmpdir, "train.jsonl").read_text().strip().split("\n")) == 8000
        assert len(Path(tmpdir, "val.jsonl").read_text().strip().split("\n")) == 1000
        assert len(Path(tmpdir, "benchmark.jsonl").read_text().strip().split("\n")) == 500


def test_benchmark_rows_contain_chosen_restrictions_and_reference():
    from training.build_sft_dataset import build_datasets
    df = make_fake_df(10000)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        df.to_csv(input_path, index=False)
        build_datasets(str(input_path), tmpdir)
        with open(Path(tmpdir, "benchmark.jsonl")) as f:
            first = json.loads(f.readline())
        assert "chosen_restrictions" in first
        assert "reference_output" in first
        assert "instruction" in first


def test_train_rows_contain_chosen_restrictions():
    from training.build_sft_dataset import build_datasets
    df = make_fake_df(10000)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        df.to_csv(input_path, index=False)
        build_datasets(str(input_path), tmpdir)
        with open(Path(tmpdir, "train.jsonl")) as f:
            first = json.loads(f.readline())
        assert "chosen_restrictions" in first
