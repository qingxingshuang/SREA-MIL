from srea_mil.metrics import binary_metrics


def test_binary_metrics():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
    assert result["accuracy"] == 0.75
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 0.5
    assert 0.0 <= result["auroc"] <= 1.0
