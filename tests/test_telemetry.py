import pytest

from graded_resonators.telemetry import Telemetry, flatten, live_settings, model_id, open_run


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


def test_new_runs_default_to_live_project_and_reject_offline(monkeypatch):
    monkeypatch.delenv('WANDB_MODE', raising=False)
    assert live_settings({})['project'] == 'graded-resonators'
    for mode in ['offline', 'disabled', 'dryrun']:
        with pytest.raises(ValueError, match='live W&B'):
            live_settings({'wandb': {'mode': mode}})
        monkeypatch.setenv('WANDB_MODE', mode)
        with pytest.raises(ValueError, match='live W&B'):
            live_settings({})
        monkeypatch.delenv('WANDB_MODE')


def test_failed_wandb_start_aborts_instead_of_silently_falling_back(monkeypatch, tmp_path):
    monkeypatch.delenv('WANDB_MODE', raising=False)
    def unavailable(**kwargs):
        assert kwargs['mode'] == 'online'
        raise ConnectionError('unavailable')
    monkeypatch.setattr('graded_resonators.telemetry.wandb.init', unavailable)
    with pytest.raises(ConnectionError):
        Telemetry({}, tmp_path / 'job')
    assert not (tmp_path / 'job' / 'telemetry.json').exists()


def test_offline_sdk_response_is_rejected(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.delenv('WANDB_MODE', raising=False)
    monkeypatch.setattr('graded_resonators.telemetry.wandb.init',
                        lambda **kwargs: SimpleNamespace(settings=SimpleNamespace(mode='offline')))
    with pytest.raises(RuntimeError, match='online run'):
        open_run('check', 'check', 'check', {}, tmp_path)
