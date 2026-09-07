"""Playground read semantics and true gradients through the normalised model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graded_resonators.model import ARMS, advance, forward, initialise, initial_state, membrane_read


def fixture():
    neuron = replace(ARMS['graded_static'], payload='excess', surrogate='none',
                     threshold=.2, learn_threshold=True, threshold_spread=.5,
                     read_normalization='complex_rms', dt=.1)
    return initialise(7, 1, 16, 2, [.2, .4], [.2, .4], 1, neuron), neuron


def test_random_low_thresholds_preserve_other_paired_parameters():
    p, neuron = fixture()
    uniform = replace(neuron, threshold_spread=0)
    reference = initialise(7, 1, 16, 2, [.2, .4], [.2, .4], 1, uniform)
    thresholds = np.asarray(jax.nn.softplus(p['threshold_raw']))
    assert np.all((thresholds >= .1) & (thresholds <= .3))
    assert thresholds.std() > .02
    for key in p.keys() - {'threshold_raw'}:
        np.testing.assert_array_equal(p[key], reference[key])


def test_complex_rms_is_per_sample_and_preserves_phase_and_silence():
    p, neuron = fixture()
    u, v = jnp.array([[3., 0.], [0., 0.]]), jnp.array([[4., 0.], [0., 0.]])
    read_u, read_v = membrane_read(p | {'norm_gain': jnp.ones(2)}, u, v, neuron)
    scale = np.sqrt(25 / 2 + 1e-6)
    np.testing.assert_allclose(read_u, u / scale, atol=1e-7)
    np.testing.assert_allclose(read_v, v / scale, atol=1e-7)


def test_normalization_does_not_overwrite_membrane_and_streams_identically():
    p, neuron = fixture()
    state = initial_state(p, 1, neuron)
    drive = jnp.arange(1, 17, dtype=jnp.float32)[None, :]
    nxt, _ = advance(p, state, drive, neuron)
    np.testing.assert_allclose(nxt[0], .1 * drive)
    np.testing.assert_array_equal(nxt[1], jnp.zeros_like(drive))
    x = jnp.arange(1, 9, dtype=jnp.float32)[:, None, None]
    full, trace = forward(p, x, neuron, trace=True)
    carry, first = forward(p, x[:3], neuron, trace=True)
    final, rest = forward(p, x[3:], neuron, state=carry, trace=True)
    for a, b in zip(full, final):
        np.testing.assert_allclose(a, b, atol=1e-6)
    for a, b, c in zip(trace, first, rest):
        np.testing.assert_allclose(a, jnp.concatenate((b, c)), atol=1e-6)


def test_normalized_true_gradient_matches_directional_finite_difference():
    p, neuron = fixture()
    x = jnp.array([[[12.]], [[6.]], [[3.]], [[9.]]])
    for key in ['threshold_raw', 'norm_gain', 'input']:
        def total(value):
            return forward(p | {key: value}, x, neuron, trace=True)[1][0].sum()
        direction = jnp.ones_like(p[key]) * .1
        _, analytic = jax.jvp(total, (p[key],), (direction,))
        eps = .001
        numerical = (total(p[key] + eps * direction) - total(p[key] - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, atol=2e-4, rtol=2e-3)


def test_normalized_reset_is_not_silently_applied_in_wrong_units():
    _, neuron = fixture()
    with pytest.raises(ValueError, match='raw-unit reset'):
        replace(neuron, reset='subtract')
