"""Unequal final batches must not silently change the validation objective."""

import numpy as np
import pytest

from graded_resonators import train


def test_sample_and_upstream_batch_means_are_distinct(monkeypatch):
    # Three examples have loss one; the single example in the tail has loss nine.
    batches = [(None, None, np.array([1, 1, 1])), (None, None, np.array([1, 0, 0]))]
    metrics = iter([np.array([1, 1, .25, 2, 3]), np.array([9, 0, .75, 4, 7])])
    monkeypatch.setattr(train, "batches", lambda *args, **kwargs: iter(batches))
    monkeypatch.setattr(train, "evaluate_batch", lambda *args: (None, next(metrics)))
    result = train.evaluate(None, None, None, {"evaluation_batch": 3, "task": "shd"}, None)
    assert result["loss"] == pytest.approx(3)
    assert result["batch_mean_loss"] == pytest.approx(5)
    assert result["accuracy"] == pytest.approx(.75)
    assert result["payload_rms"] == pytest.approx(np.sqrt(7))
    assert result["max_membrane_component"] == 7
    assert result["samples"] == 4
