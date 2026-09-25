from cofrisk.models.predictor import RiskPredictor
from cofrisk.data.validation import EXPECTED_FEATURES


def make_valid_features():
    return {
        feature: 0.1
        for feature in EXPECTED_FEATURES
    }


def test_predictor_returns_probability_and_class():
    predictor = RiskPredictor()

    result = predictor.predict(make_valid_features())

    assert "default_probability" in result
    assert "risk_class" in result

    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["risk_class"] in [0, 1]