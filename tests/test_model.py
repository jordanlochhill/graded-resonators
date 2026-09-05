import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graded_resonators.model import ARMS, Neuron, event, forward, initialise, objective


def parameters(neuron):
    p = initialise(17, 3, 5, 2, (5, 10), (2, 3), 1, neuron)
    p["input"] = p["input"] * 80  # exercise firing, refractory and silent states
    return p


@pytest.mark.parametrize("neuron", list(ARMS.values()) + [
    Neuron(payload="complex", adaptive_threshold=False, adaptive_damping=False, signed=True, integration="polar"),
])
def test_streaming_and_causality(neuron):
    p = parameters(neuron)
    x = jnp.asarray(np.random.default_rng(7).normal(size=(30, 4, 3)), dtype=jnp.float32)
    final, full = forward(p, x, neuron, trace=True)
    middle, left = forward(p, x[:11], neuron, trace=True)
    resumed, right = forward(p, x[11:], neuron, state=middle, trace=True)
    for a, b, c in zip(full, left, right):
        np.testing.assert_allclose(a, jnp.concatenate((b, c)), atol=1e-5, rtol=2e-5)
    for a, b in zip(final, resumed):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=2e-5)
    altered = x.at[11:].set(999)
    _, future_changed = forward(p, altered, neuron, trace=True)
    np.testing.assert_allclose(full[0][:11], future_changed[0][:11], atol=1e-5)


def test_observation_threshold_cannot_edit_isolated_membrane():
    n = replace(ARMS["graded_observation"], recurrent=False)
    p = parameters(n)
    x = jnp.ones((90, 2, 3))
    _, a = forward(p, x, n, trace=True)
    _, b = forward(p, x, replace(n, threshold=100), trace=True)
    for index in (3, 4):
        np.testing.assert_array_equal(a[index], b[index])
    assert np.any(np.asarray(a[1]) != np.asarray(b[1]))


def test_polar_impulse_retains_analytic_decay_and_rotation():
    n = Neuron(payload="membrane", adaptive_threshold=False, adaptive_damping=False,
               integration="polar", recurrent=False)
    p = parameters(n)
    x = jnp.zeros((100, 1, 3)).at[0, 0, 0].set(1)
    _, trajectory = forward(p, x, n, trace=True)
    membrane = np.asarray(trajectory[3] + 1j * trajectory[4])[:, 0]
    a = np.exp((-np.abs(p["damping"]) + 1j * np.abs(p["omega"])) * n.dt)
    expected = np.asarray(p["input"][0]) * n.dt * a[None] ** np.arange(100)[:, None]
    np.testing.assert_allclose(membrane, expected, atol=2e-6, rtol=2e-5)


def test_padding_does_not_change_loss_or_gradient():
    n = ARMS["brf"]
    p = parameters(n)
    x = jnp.ones((15, 3, 3))
    labels = jnp.array([0, 1, 0])
    f = jax.value_and_grad(lambda p, x, y, mask: objective(p, x, y, mask, n, "shd")[0])
    value, grad = f(p, x, labels, jnp.ones(3))
    padded = jnp.concatenate((x, jnp.full((15, 1, 3), 50.)), axis=1)
    other, other_grad = f(p, padded, jnp.array([0, 1, 0, 1]), jnp.array([1., 1., 1., 0.]))
    np.testing.assert_allclose(value, other, rtol=1e-6)
    for key in grad:
        np.testing.assert_allclose(grad[key], other_grad[key], atol=1e-6, rtol=1e-5)


def test_surrogate_has_published_sign_structure_and_strict_threshold():
    x = jnp.array([-3., 0., 3.])
    np.testing.assert_array_equal(event(x), [0, 0, 1])
    derivative = jax.grad(lambda x: event(x).sum())(x)
    assert derivative[0] < 0 and derivative[1] > 0 and derivative[2] < 0


def test_upstream_network_forward_and_bptt():
    """Optional external oracle, pinned by docs/study.md and release provenance."""
    reference = os.environ.get("BRF_REFERENCE_ROOT")
    if not reference:
        pytest.skip("Set BRF_REFERENCE_ROOT to the pinned authors' checkout")
    assert (Path(reference) / "snn/modules/rf.py").is_file()
    sys.path.insert(0, reference)
    import torch
    from snn.models.resonaternns import SimpleResRNN

    torch.set_num_threads(1)
    n = ARMS["brf"]
    p = parameters(n)
    oracle = SimpleResRNN(3, 5, 2)
    with torch.no_grad():
        oracle.hidden.linear.weight.copy_(torch.tensor(np.concatenate((p["input"], p["recurrent"]), axis=0).T))
        oracle.hidden.omega.copy_(torch.tensor(np.asarray(p["omega"])))
        oracle.hidden.b_offset.copy_(torch.tensor(np.asarray(p["damping"])))
        oracle.out.linear.weight.copy_(torch.tensor(np.asarray(p["readout"]).T))
        oracle.out.tau_mem.copy_(torch.tensor(np.asarray(p["tau"])))
    x = np.random.default_rng(4).normal(size=(60, 4, 3)).astype(np.float32)
    tx = torch.tensor(x, requires_grad=True)
    output, state, spikes = oracle(tx)
    final, (actual, statistics) = forward(p, jnp.asarray(x), n)
    np.testing.assert_allclose(actual, output.detach(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(final[0], state[0][1].detach(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(statistics[..., 0].sum(), spikes.detach(), atol=0)
    target = np.array([0, 1, 1, 0])
    loss = sum(torch.nn.functional.cross_entropy(o, torch.tensor(target)) for o in output)
    loss.backward()
    value, grads = jax.value_and_grad(lambda p, x: objective(p, x, jnp.asarray(target), jnp.ones(4), n, "shd")[0], argnums=(0, 1))(p, jnp.asarray(x))
    np.testing.assert_allclose(value, loss.detach(), rtol=2e-5)
    np.testing.assert_allclose(grads[1], tx.grad, atol=2e-5, rtol=3e-4)
    oracle_grads = {
        "input": oracle.hidden.linear.weight.grad[:, :3].T,
        "recurrent": oracle.hidden.linear.weight.grad[:, 3:].T,
        "readout": oracle.out.linear.weight.grad.T,
        "omega": oracle.hidden.omega.grad,
        "damping": oracle.hidden.b_offset.grad,
        "tau": oracle.out.tau_mem.grad,
    }
    for key in p:
        np.testing.assert_allclose(grads[0][key], oracle_grads[key], atol=3e-5, rtol=3e-4, err_msg=key)
