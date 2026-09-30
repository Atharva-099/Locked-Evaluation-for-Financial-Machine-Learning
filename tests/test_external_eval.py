import numpy as np
import pandas as pd
import pytest

from p3_modellab.external_eval import evaluate_predictions


def test_uploaded_prediction_metrics_cover_all_three_tasks():
    frame = pd.DataFrame({
        "month": [1, 1, 1, 2, 2, 2],
        "actual": [0.01, 0.02, 0.03, 0.02, 0.03, 0.04],
        "forecast": [0.01, 0.02, 0.03, 0.02, 0.03, 0.04],
    })
    volatility = evaluate_predictions(frame, "volatility", "actual", "forecast")
    returns = evaluate_predictions(frame, "returns", "actual", "forecast", "month")
    assert volatility["mse"] == pytest.approx(0)
    assert returns["mean_period_rank_ic"] == pytest.approx(1)

    events = pd.DataFrame({"actual": [0, 0, 1, 1], "forecast": [0.1, 0.2, 0.8, 0.9]})
    delisting = evaluate_predictions(events, "delisting", "actual", "forecast")
    assert delisting["pr_auc"] == pytest.approx(1)
    assert delisting["brier_score"] == pytest.approx(0.025)


def test_uploaded_prediction_validation_rejects_invalid_probabilities():
    frame = pd.DataFrame({"actual": [0, 1], "forecast": [0.2, 1.1]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_predictions(frame, "delisting", "actual", "forecast")
