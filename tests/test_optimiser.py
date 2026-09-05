import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import torch

from graded_resonators.model import ARMS, initialise, objective
from graded_resonators.train import train_step


def test_adam_update_matches_torch_with_nonzero_momentum():
    n = ARMS["graded_brf"]
    p = initialise(2, 3, 4, 2, (5, 10), (2, 3), 1, n)
    p["input"] = p["input"] * 100
    x = jnp.asarray(np.random.default_rng(19).normal(size=(17, 2, 3)).astype(np.float32))
    labels, mask = jnp.array([0, 1]), jnp.ones(2)
    m, v = [jax.tree.map(jnp.zeros_like, p) for _ in range(2)]
    step = jnp.array(0)
    tp = {k: torch.tensor(np.asarray(a), requires_grad=True) for k, a in p.items()}
    optimiser = torch.optim.Adam(tp.values(), lr=.0123)
    for _ in range(3):
        g = jax.grad(lambda p: objective(p, x, labels, mask, n, "shd")[0])(p)
        for k in p:
            tp[k].grad = torch.tensor(np.asarray(g[k]))
        optimiser.step()
        p, m, v, step, _, _, finite = train_step(p, m, v, step, x, labels, mask, .0123, n, "shd")
        assert bool(finite)
        for k in p:
            np.testing.assert_allclose(p[k], tp[k].detach(), atol=2e-6, rtol=2e-6, err_msg=k)
