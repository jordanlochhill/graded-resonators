from dataclasses import replace

import numpy as np

from graded_resonators.initialisation import calibrate
from graded_resonators.model import ARMS, initialise


def test_calibration_ignores_labels_and_preserves_paired_weights():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(32, 100, 4)).astype(np.float32)
    y = rng.integers(0, 3, 32)
    config = {'task': 'shd', 'initialisation_calibration': {
        'method': 'positive_membrane_quantile', 'samples': 32, 'quantile': .95}}
    reference = None
    for surrogate, learned in [('none', False), ('double_gaussian', False),
                               ('none', True), ('double_gaussian', True)]:
        neuron = replace(ARMS['graded_static'], payload='excess', surrogate=surrogate,
                         learn_threshold=learned)
        p = initialise(4, 4, 16, 3, [5, 10], [2, 3], 5, neuron)
        # Balanced inputs ensure all units can participate in this small fixture.
        dataset = (x, y if not learned else (y + 1) % 3, np.arange(32))
        new, calibrated, audit = calibrate(p, neuron, dataset, None, config)
        for key in p.keys() - {'threshold_raw'}:
            np.testing.assert_array_equal(new[key], p[key])
        if reference is None:
            reference = calibrated.threshold
        assert calibrated.threshold == reference
        assert audit['gradient_norms']['input'] > 0
        if learned:
            assert audit['gradient_norms']['threshold_raw'] > 0

