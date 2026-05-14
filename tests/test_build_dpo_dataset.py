from unittest.mock import patch


def test_is_compliant_true_when_no_violations():
    from training.build_dpo_dataset import is_compliant
    mock_result = {
        "gluten": {"probability": 0.05, "present": False},
        "dairy": {"probability": 0.02, "present": False},
    }
    with patch("training.build_dpo_dataset.classify_ingredients", return_value=mock_result):
        assert is_compliant("gluten-free chicken soup", ["gluten", "dairy"]) is True


def test_is_compliant_false_when_restricted_label_present():
    from training.build_dpo_dataset import is_compliant
    mock_result = {
        "gluten": {"probability": 0.95, "present": True},
        "dairy": {"probability": 0.02, "present": False},
    }
    with patch("training.build_dpo_dataset.classify_ingredients", return_value=mock_result):
        assert is_compliant("pasta recipe", ["gluten", "dairy"]) is False


def test_is_compliant_ignores_unrestricted_labels():
    from training.build_dpo_dataset import is_compliant
    mock_result = {
        "gluten": {"probability": 0.95, "present": True},
        "dairy": {"probability": 0.02, "present": False},
    }
    with patch("training.build_dpo_dataset.classify_ingredients", return_value=mock_result):
        # gluten is present but NOT in the restriction list — should still pass
        assert is_compliant("pasta recipe", ["dairy"]) is True


def test_is_compliant_true_for_empty_restrictions():
    from training.build_dpo_dataset import is_compliant
    with patch("training.build_dpo_dataset.classify_ingredients", return_value={}):
        assert is_compliant("any recipe", []) is True
