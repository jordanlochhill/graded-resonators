import json

import pytest

from graded_resonators.analysis import primary_records, summarise


def test_duplicate_baseline_cannot_be_counted_twice(tmp_path):
    record = {"status": "complete", "best_epoch": 1,
              "config": {"stage": "main", "task": "shd", "arm": "brf", "seed": 0},
              "test": {"accuracy": .9}}
    for name in ("qualification", "main"):
        path = tmp_path / name
        path.mkdir()
        (path / "result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Duplicate primary seed"):
        primary_records([tmp_path])


def test_failure_and_absence_are_not_silently_pooled():
    def complete(accuracy):
        return {"result": {"status": "complete", "test": {"accuracy": accuracy, "event_fraction": .1,
                                                         "payload_rms": .3, "max_membrane_component": 2.0}}}
    records = {("shd", "brf", 0): complete(.8), ("shd", "brf", 1): complete(.9),
               ("shd", "graded_brf", 0): complete(.85),
               ("shd", "graded_brf", 1): {"result": {"status": "nonfinite"}}}
    summary = summarise(records, ["shd"], range(3))
    group = summary["tasks"]["shd"]["graded_brf"]
    assert group["failed_seeds"] == {"1": "nonfinite"}
    assert group["missing_seeds"] == [2]
    assert group["conditional_on_completed_training"]
    paired = summary["paired_differences_percentage_points"]["shd"]["graded_brf"]
    assert paired["n"] == 1 and paired["seeds"] == [0]
    assert paired["mean"] == pytest.approx(5.0)
    assert paired["mean_ci95"] is None
