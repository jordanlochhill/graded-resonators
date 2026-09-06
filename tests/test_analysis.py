import json

import pytest

from graded_resonators.analysis import ARMS, primary_records, select_learning_rates, summarise


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


def test_failed_rate_cannot_win_on_an_early_checkpoint_and_test_access_is_refused():
    results = []
    for arm in ARMS:
        for rate, loss in ((.075, .1), (.025, .4), (.0075, .5)):
            for seed in (100, 101):
                results.append({"config": {"stage": "tune", "arm": arm, "lr": rate, "seed": seed},
                                "status": "nonfinite" if rate == .075 and seed == 101 else "complete",
                                "best_validation_loss": loss})
    decisions = select_learning_rates(results)
    assert all(value["selected_rate"] == .025 for value in decisions.values())
    results[0]["test"] = {"accuracy": 1.0}
    with pytest.raises(ValueError, match="validation-only"):
        select_learning_rates(results)


def test_conditions_sharing_a_base_arm_are_selected_separately():
    results = [{"config": {"stage": "tune", "arm": "graded_static",
                           "gradient_condition": condition, "lr": rate, "seed": seed},
                "status": "complete", "best_validation_loss": abs(rate - preferred)}
               for condition, preferred in (("exact", .025), ("surrogate", .0075))
               for rate in (.075, .025, .0075) for seed in (100, 101)]
    decision = select_learning_rates(results, arms=("exact", "surrogate"),
                                     group_key="gradient_condition")
    assert decision["exact"]["selected_rate"] == .025
    assert decision["surrogate"]["selected_rate"] == .0075
    with pytest.raises(ValueError, match="Missing tuning seed"):
        select_learning_rates(results[:-1], arms=("exact", "surrogate"),
                              group_key="gradient_condition")
