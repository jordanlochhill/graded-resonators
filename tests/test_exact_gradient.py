"""Check true threshold-excess derivatives, not just output agreement.

Finite differences are valid away from a threshold crossing. They also expose
an accidental surrogate path through a supposedly exact recurrent model.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graded_resonators.model import ARMS, advance, forward, initial_state, initialise


def fixture(surrogate="none", learned=True):
    neuron = replace(ARMS["graded_static"], payload="excess", surrogate=surrogate,
                     learn_threshold=learned, dt=0.1)
    p = initialise(7, 1, 2, 1, [0.2, 0.4], [0.2, 0.4], 1, neuron)
    return p, neuron


def test_excess_threshold_and_amplitude_derivatives():
    p, neuron = fixture()
    raw = p["threshold_raw"]
    state = initial_state(p, 1, neuron)

    def emitted(drive, threshold_raw):
        parameters = p | {"threshold_raw": threshold_raw}
        return advance(parameters, state, drive, neuron)[1][2].sum()

    drive = jnp.array([[5., 15.]])  # Membranes .5 and 1.5, threshold 1.
    d_drive, d_raw = jax.grad(emitted, (0, 1))(drive, raw)
    np.testing.assert_allclose(d_drive, [[0., .1]], atol=1e-7)
    np.testing.assert_allclose(d_raw, [0., -jax.nn.sigmoid(raw[1])], atol=1e-7)
    eps = 1e-3
    for i in range(2):
        direction = jnp.eye(2)[i] * eps
        numerical = (emitted(drive, raw + direction) - emitted(drive, raw - direction)) / (2 * eps)
        np.testing.assert_allclose(d_raw[i], numerical, atol=4e-5)


def test_same_forward_rule_and_initial_parameters_with_or_without_surrogate():
    p, exact = fixture()
    surrogate = replace(exact, surrogate="double_gaussian")
    fixed_p, fixed = fixture(learned=False)
    for name, value in fixed_p.items():
        np.testing.assert_array_equal(value, p[name])
    x = jnp.full((6, 1, 1), 40.)
    for other_p, other in ((p, surrogate), (fixed_p, fixed)):
        expected = forward(p, x, exact, trace=True)
        observed = forward(other_p, x, other, trace=True)
        for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(observed)):
            np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-6)


def test_exact_recurrent_gradient_matches_finite_differences():
    p, neuron = fixture()
    p = p | {"input": jnp.array([[1., .8]]), "recurrent": jnp.eye(2) * .15}
    x = jnp.array([[[16.]], [[11.]], [[3.]], [[5.]]])

    def total(raw):
        _, trace = forward(p | {"threshold_raw": raw}, x, neuron, trace=True)
        return trace[2].sum()

    raw = p["threshold_raw"]
    analytic = jax.grad(total)(raw)
    eps = 1e-3
    numerical = jnp.array([(total(raw + eps * d) - total(raw - eps * d)) / (2 * eps)
                           for d in jnp.eye(2)])
    np.testing.assert_allclose(analytic, numerical, atol=8e-4, rtol=5e-4)


def test_no_hidden_discontinuous_state_path_in_exact_control():
    _, neuron = fixture()
    for change in ({"adaptive_threshold": True}, {"adaptive_damping": True},
                   {"reset": "subtract"}, {"payload": "membrane"}):
        with pytest.raises(ValueError, match="Exact-gradient control"):
            replace(neuron, **change)
