"""Tests for statistical utilities and numerical edge cases."""

from src.utils.statistics import StatisticalAnalyzer


def test_confidence_interval_single_value_is_stable():
    """A singleton sample should not return NaN bounds."""
    mean, lower, upper = StatisticalAnalyzer.confidence_interval([0.75])

    assert mean == 0.75
    assert lower == 0.75
    assert upper == 0.75


def test_t_test_handles_small_samples_without_nan():
    """Very small samples should fall back to a safe non-significant result."""
    result = StatisticalAnalyzer.t_test([0.6], [0.4])

    assert result["statistic"] == 0.0
    assert result["pvalue"] == 1.0
    assert result["significant"] is False
    assert result["mean1"] == 0.6
    assert result["mean2"] == 0.4


def test_effect_size_handles_singleton_samples():
    """Singleton samples should not produce NaN effect sizes."""
    result = StatisticalAnalyzer.effect_size([0.8], [0.8])

    assert result["cohens_d"] == 0.0
    assert result["magnitude"] == "negligible"
    assert result["mean_diff"] == 0.0
