from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from graded_resonators.model import ARMS, forward, initialise
from graded_resonators.robustness import sample_random


def test_packet_loss_does_not_delete_local_refractory_event():
    neuron = replace(ARMS["graded_brf"], recurrent=False)
    p = initialise(1, 1, 2, 2, (5, 10), (2, 3), 1, neuron)
    p["input"] = jnp.ones((1, 2)) * 500
    x = jnp.ones((12, 1, 1))
    _, clean = forward(p, x, neuron, trace=True)
    _, dropped = forward(p, x, neuron, trace=True, transmission_keep=jnp.zeros((12, 1, 2)))
    assert np.any(np.asarray(clean[1]))
    for index in (1, 3, 4, 5):
        np.testing.assert_array_equal(clean[index], dropped[index])
    np.testing.assert_array_equal(dropped[2], 0)
    np.testing.assert_array_equal(dropped[0], 0)
    _, kept = forward(p, x, neuron, trace=True, transmission_keep=jnp.ones((12, 1, 2)))
    for a, b in zip(clean, kept):
        np.testing.assert_array_equal(a, b)


def test_perturbations_are_sample_stable_across_batching():
    for kind in (0, 1):
        together = sample_random([19, 2, 33], 7, 5, 17, kind)
        pieces = np.concatenate([sample_random([index], 7, 5, 17, kind) for index in [19, 2, 33]], axis=1)
        np.testing.assert_array_equal(together, pieces)
