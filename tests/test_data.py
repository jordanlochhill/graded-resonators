import numpy as np
from graded_resonators.data import bin_shd, batches


def test_shd_edges_and_upstream_channel_zero_quirk():
    times = np.array([0, .001, .004, .004001, .996, .997, .05])
    units = np.array([1, 2, 3, 4, 5, 6, 0])
    result = bin_shd(times, units)
    assert result.sum() == 5
    assert result[0, 699] == 1
    assert result[1, 698] == 1
    assert result[1, 697] == 1
    assert result[2, 696] == 1
    assert result[249, 695] == 1
    assert result[:, 0].sum() == 0


def test_padded_final_batch_retains_every_sample_once():
    x = np.arange(5, dtype=np.uint8).reshape(5, 1, 1)
    y = np.arange(5)
    emitted = []
    for _, labels, mask in batches((x, y, np.arange(5)), 3, "shd", shuffle_seed=2):
        emitted.extend(labels[mask.astype(bool)])
    assert sorted(emitted) == list(range(5))
