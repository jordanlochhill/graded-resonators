from graded_resonators.telemetry import Telemetry, flatten, model_id


def test_model_ids_separate_seeds_and_execute_runs():
    assert model_id('job-a', 'seed-0') == model_id('job-a', 'seed-0')
    assert len({model_id(j, s) for j in ['job-a', 'job-b'] for s in ['seed-0', 'seed-1']}) == 4
    assert flatten({'epoch': 2, 'train': {'loss': 3}, 'validation': {'loss': None}}) == {
        'epoch': 2, 'train/loss': 3}


def test_live_telemetry_keeps_model_histories_separate(monkeypatch, tmp_path):
    runs = []

    class Run:
        def __init__(self, identity):
            self.id, self.summary, self.rows = identity, {}, []
            self.url = f'https://wandb.ai/test/{identity}'
            runs.append(self)
        def define_metric(self, *args, **kwargs):
            pass
        def log(self, row):
            self.rows.append(row)
        def finish(self, exit_code=0):
            self.exit_code = exit_code

    monkeypatch.setattr('graded_resonators.telemetry.open_run', lambda identity, *a, **k: Run(identity))
    telemetry = Telemetry({'wandb': {'project': 'graded-resonators'}}, tmp_path / 'job')
    for seed in [0, 1]:
        config = {'task': 'shd', 'arm': 'graded_static', 'seed': seed}
        telemetry.epoch(config, {'epoch': 0, 'train': {'loss': seed + 1}})
        telemetry.result(config, {'status': 'complete'})
    telemetry.finish()
    assert len(runs) == 3
    assert runs[1].rows == [{'epoch': 0, 'train/loss': 1}]
    assert runs[2].rows == [{'epoch': 0, 'train/loss': 2}]
