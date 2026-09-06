"""Resonator recurrence, separated observation/transmission, and LI readout.

The BRF equations follow Higuchi et al. (2024). The derivative reproduces the
released implementation: *normalised* Gaussians with an overall gain of 0.5.
That detail differs from an unnormalised reading of the printed equation.
"""

from dataclasses import dataclass
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def event(x):
    return (x > 0).astype(x.dtype)


@event.defjvp
def event_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    normal = lambda s: jnp.exp(-0.5 * (x / s) ** 2) / (s * math.sqrt(2 * math.pi))
    derivative = 0.5 * (1.15 * normal(0.5) - 0.30 * normal(3.0))
    return event(x), dx * derivative


@jax.custom_jvp
def fast_event(x):
    return (x > 0).astype(x.dtype)


@fast_event.defjvp
def fast_event_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return fast_event(x), dx * 12.5 / (1 + 25 * jnp.abs(x)) ** 2


@jax.custom_jvp
def logistic_event(x):
    return (x > 0).astype(x.dtype)


@logistic_event.defjvp
def logistic_event_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    s = jax.nn.sigmoid(10 * x)
    return logistic_event(x), dx * 10 * s * (1 - s)


@dataclass(frozen=True)
class Neuron:
    payload: str = "binary"  # binary, membrane, excess, complex, smooth
    adaptive_threshold: bool = True
    adaptive_damping: bool = True
    integration: str = "euler"
    signed: bool = False
    reset: str = "none"
    surrogate: str = "double_gaussian"
    learn_threshold: bool = False
    refractory_decay: float = 0.9
    threshold: float = 1.0
    dt: float = 0.01
    recurrent: bool = True
    payload_bits: int = 32
    payload_clip: float = 1.0

    def __post_init__(self):
        if self.payload not in {"binary", "membrane", "excess", "complex", "smooth"}:
            raise ValueError(f"Unknown payload: {self.payload}")
        if self.integration not in {"euler", "polar"}:
            raise ValueError(self.integration)
        if self.reset not in {"none", "subtract"}:
            raise ValueError(self.reset)
        if self.surrogate not in {"double_gaussian", "fast_sigmoid", "logistic", "none"}:
            raise ValueError(self.surrogate)
        if self.learn_threshold and self.threshold <= 0:
            raise ValueError("Learned thresholds require a positive initial value")
        if self.surrogate == "none" and (self.payload != "excess" or self.adaptive_threshold
                                         or self.adaptive_damping or self.reset != "none"):
            raise ValueError("Exact-gradient control requires excess emission without event feedback/reset")
        if not 0 <= self.refractory_decay < 1:
            raise ValueError("Refractory decay must be in [0, 1)")
        if self.payload == "smooth" and (self.adaptive_threshold or self.adaptive_damping):
            raise ValueError("Smooth control must not retain a hidden surrogate event path")
        if self.payload_bits not in {2, 4, 8, 32} or self.payload_clip <= 0:
            raise ValueError("Invalid payload quantisation")


ARMS = {
    "brf": Neuron(),
    "graded_brf": Neuron(payload="membrane"),
    "graded_observation": Neuron(payload="membrane", adaptive_damping=False),
    "graded_static": Neuron(payload="membrane", adaptive_damping=False, adaptive_threshold=False),
}


def initialise(seed, inputs, hidden, classes, omega_range, damping_range, tau_std, neuron):
    """Xavier uniform weights and the paper's oscillator/time-constant ranges."""
    rng = np.random.default_rng(seed)
    transmitted = hidden * (2 if neuron.payload == "complex" else 1)
    fan_in = inputs + transmitted
    weight = rng.uniform(-math.sqrt(6 / (fan_in + hidden)),
                         math.sqrt(6 / (fan_in + hidden)), (fan_in, hidden))
    p = {
        "input": weight[:inputs],
        "recurrent": weight[inputs:],
        "readout": rng.uniform(-math.sqrt(6 / (transmitted + classes)),
                               math.sqrt(6 / (transmitted + classes)), (transmitted, classes)),
        "omega": rng.uniform(*omega_range, hidden),
        "damping": rng.uniform(*damping_range, hidden),
        "tau": rng.normal(20, tau_std, classes),
    }
    if neuron.learn_threshold:
        # Positive per-neuron thresholds, initially equal to the fixed control.
        # No random draws: adding thresholds preserves every other parameter.
        p["threshold_raw"] = np.full(hidden, math.log(math.expm1(neuron.threshold)))
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in p.items()}


def initial_state(p, batch, neuron):
    hidden, transmitted = p["omega"].shape[0], p["recurrent"].shape[0]
    z = jnp.zeros((batch, hidden), dtype=p["omega"].dtype)
    return z, z, z, jnp.zeros((batch, transmitted), z.dtype), jnp.zeros((batch, p["tau"].size), z.dtype)


def coefficients(p, neuron):
    omega, offset = jnp.abs(p["omega"]), jnp.abs(p["damping"])
    if neuron.integration == "euler":
        # Preserve the original domain: invalid frequencies produce a recorded
        # failure. Silently clipping here would change the reference model.
        real = jnp.sqrt(1 - (neuron.dt * omega) ** 2) - neuron.dt * offset
        imag = neuron.dt * omega
    else:
        rho = jnp.exp(-neuron.dt * offset)
        real, imag = rho * jnp.cos(neuron.dt * omega), rho * jnp.sin(neuron.dt * omega)
    return real, imag, jnp.exp(-1 / jnp.abs(p["tau"]))


def advance(p, state, drive, neuron, coeff=None, transmission_keep=None):
    """One step. Event count, payload and carried membrane are separate values."""
    u, v, q, previous, readout = state
    real, imag, alpha = coefficients(p, neuron) if coeff is None else coeff
    if neuron.adaptive_damping:
        if neuron.integration == "euler":
            real = real - neuron.dt * q
        else:
            factor = jnp.exp(-neuron.dt * q)
            real, imag = real * factor, imag * factor
    if neuron.recurrent:
        drive = drive + jnp.matmul(previous, p["recurrent"], precision="highest")
    new_u = real * u - imag * v + neuron.dt * drive
    new_v = imag * u + real * v
    base_threshold = jax.nn.softplus(p["threshold_raw"]) if neuron.learn_threshold else neuron.threshold
    threshold = base_threshold + (q if neuron.adaptive_threshold else 0)
    observed = jnp.abs(new_u) if neuron.signed else new_u
    gate_fn = {"double_gaussian": event, "fast_sigmoid": fast_event, "logistic": logistic_event,
               "none": lambda x: (x > 0).astype(x.dtype)}[neuron.surrogate]
    gate = gate_fn(observed - threshold)
    sign = jnp.where(new_u >= 0, 1., -1.) if neuron.signed else 1.
    if neuron.payload == "binary":
        sent = gate * sign
    elif neuron.payload == "membrane":
        sent = gate * new_u
    elif neuron.payload == "excess":
        sent = gate * (new_u - sign * threshold)
    elif neuron.payload == "complex":
        sent = jnp.concatenate((gate * new_u, gate * new_v), axis=-1)
    else:
        sent = jax.nn.softplus(new_u - threshold)
        if neuron.signed:
            sent = sent - jax.nn.softplus(-new_u - threshold)
    if neuron.payload_bits < 32 and neuron.payload != "binary":
        # Fixed validation-calibrated range, including a representable zero.
        levels = 2 ** (neuron.payload_bits - int(neuron.signed or neuron.payload == "complex")) - 1
        lo = -neuron.payload_clip if neuron.signed or neuron.payload == "complex" else 0
        sent = jnp.round(jnp.clip(sent, lo, neuron.payload_clip) * levels / neuron.payload_clip) * neuron.payload_clip / levels
    if transmission_keep is not None:
        # Packet loss occurs after emission. The neuron's own refractory state
        # still sees the event; both components of a complex packet are lost
        # together. The same surviving packet reaches recurrence and readout.
        keep = jnp.concatenate((transmission_keep, transmission_keep), -1) if neuron.payload == "complex" else transmission_keep
        sent = sent * keep
    q_next = neuron.refractory_decay * q + gate
    if neuron.reset == "subtract":
        new_u = new_u - gate * sign * threshold
    readout = alpha * readout + (1 - alpha) * jnp.matmul(sent, p["readout"], precision="highest")
    next_state = new_u, new_v, q_next, sent, readout
    return next_state, (readout, gate, sent)


@partial(jax.jit, static_argnames=("neuron", "trace"))
def forward(p, x, neuron, state=None, trace=False, transmission_keep=None):
    """x is [time, batch, input]. Passing state supports causal streaming."""
    state = initial_state(p, x.shape[1], neuron) if state is None else state
    drives = jnp.matmul(x, p["input"], precision="highest")
    coeff = coefficients(p, neuron)

    def step(carry, item):
        drive, keep = (item, None) if transmission_keep is None else item
        nxt, (out, gate, sent) = advance(p, carry, drive, neuron, coeff, keep)
        if trace:
            return nxt, (out, gate, sent, nxt[0], nxt[1], nxt[2])
        # Keep per-sample statistics so padded examples cannot affect metrics.
        statistics = jnp.stack((gate.sum(-1), (sent ** 2).sum(-1),
                                jnp.maximum(jnp.abs(nxt[0]).max(-1), jnp.abs(nxt[1]).max(-1))), -1)
        return nxt, (out, statistics)

    sequence = drives if transmission_keep is None else (drives, transmission_keep)
    return jax.lax.scan(step, state, sequence)


def objective(p, x, labels, mask, neuron, task, transmission_keep=None):
    _, (outputs, statistics) = forward(p, x, neuron, transmission_keep=transmission_keep)
    if task == "smnist":
        outputs = outputs[-1:]
    log_prob = jax.nn.log_softmax(outputs, axis=-1)
    time_labels = labels if task == "ecg" else jnp.broadcast_to(labels, outputs.shape[:2])
    losses = -jnp.take_along_axis(log_prob, time_labels[..., None], axis=-1)[..., 0]
    # Sum over time, mean over real samples, matching the published optimiser
    # objective. Displayed loss is divided by the number of supervised steps.
    loss = (losses * mask[None, :]).sum() / mask.sum()
    predictions = outputs.argmax(-1) if task == "ecg" else outputs.mean(0).argmax(-1)
    accuracy = ((predictions == labels) * mask).sum() / (mask.sum() * (outputs.shape[0] if task == "ecg" else 1))
    rate = (statistics[..., 0] * mask).sum() / (x.shape[0] * mask.sum() * p["omega"].size)
    payload_ms = (statistics[..., 1] * mask).sum() / (x.shape[0] * mask.sum() * p["readout"].shape[0])
    peak = jnp.where(mask[None, :] > 0, statistics[..., 2], 0).max()
    return loss, jnp.array([loss / outputs.shape[0], accuracy, rate, jnp.sqrt(payload_ms), peak])
