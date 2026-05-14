from unittest.mock import patch


def test_compute_csr_perfect_score():
    from evals.run_benchmark import compute_csr
    prompts = [{"chosen_restrictions": ["gluten"]}] * 3
    mock_result = {"gluten": {"present": False}}
    with patch("evals.run_benchmark.classify_ingredients", return_value=mock_result):
        result = compute_csr(["recipe text"] * 3, prompts)
    assert result["csr"] == 100.0
    assert result["violations"] == 0


def test_compute_csr_all_violations():
    from evals.run_benchmark import compute_csr
    prompts = [{"chosen_restrictions": ["gluten"]}] * 4
    mock_result = {"gluten": {"present": True}}
    with patch("evals.run_benchmark.classify_ingredients", return_value=mock_result):
        result = compute_csr(["pasta with wheat"] * 4, prompts)
    assert result["csr"] == 0.0
    assert result["violations"] == 4


def test_compute_csr_partial_violations():
    from evals.run_benchmark import compute_csr
    prompts = [{"chosen_restrictions": ["gluten"]}] * 2
    call_count = 0

    def mock_classify(text_list):
        nonlocal call_count
        call_count += 1
        return {"gluten": {"present": call_count == 1}}  # first call violates, second doesn't

    with patch("evals.run_benchmark.classify_ingredients", side_effect=mock_classify):
        result = compute_csr(["recipe"] * 2, prompts)
    assert result["csr"] == 50.0
    assert result["violations"] == 1


def test_compute_csr_empty_restrictions_never_violates():
    from evals.run_benchmark import compute_csr
    prompts = [{"chosen_restrictions": []}] * 3
    with patch("evals.run_benchmark.classify_ingredients", return_value={}):
        result = compute_csr(["any recipe"] * 3, prompts)
    assert result["csr"] == 100.0
