"""Training-input calibration for paired exact/surrogate experiments."""

from dataclasses import replace
import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np

from .data import batches
from .model import forward, objective


def calibrate(p, neuron, training, permutation, config):
    """Set one positive threshold per network from unlabelled training drives.

    Measure the uncoupled membrane first, so neither gradient choice nor the
    old threshold changes calibration. All four paired conditions receive the
    same threshold and all other initial parameters remain untouched.
    """
    recipe = config['initialisation_calibration']
    if recipe['method'] != 'positive_membrane_quantile' or config['task'] != 'shd':
        raise ValueError('Unsupported calibration recipe')
    if neuron.payload != 'excess' or neuron.adaptive_threshold or neuron.adaptive_damping or neuron.reset != 'none':
        raise ValueError('Calibration requires the excess control without event feedback/reset')
    count, quantile = recipe['samples'], recipe['quantile']
    if not 0 < quantile < 1 or not 0 < count <= len(training[2]):
        raise ValueError('Invalid calibration sample count or quantile')
    x, y, mask = next(batches(training, count, 'shd', permutation, limit=count))
    probe = replace(neuron, recurrent=False, learn_threshold=False, surrogate='none')
    _, trace = forward(p, x, probe, trace=True)
    membrane = np.asarray(trace[3])
    positive = membrane[membrane > 0]
    if not positive.size or not np.isfinite(membrane).all():
        raise ValueError('No finite positive membrane samples for calibration')
    raw = np.float32(math.log(math.expm1(float(np.quantile(positive, quantile)))))
    threshold = float(jax.nn.softplus(raw))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Calibrated threshold must be finite and positive')
    calibrated = replace(neuron, threshold=threshold)
    if neuron.learn_threshold:
        p = p | {'threshold_raw': jnp.full(p['omega'].shape, raw)}
    audit = {
        'method': recipe['method'], 'quantile': quantile, 'samples': count,
        'sample_indices_sha256': hashlib.sha256(np.asarray(training[2][:count], dtype='<i8').tobytes()).hexdigest(),
        'input_sha256': hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        'uncoupled_membrane_max': float(membrane.max()),
        'uncoupled_positive_fraction': float((membrane > 0).mean()),
        'initial_threshold': threshold,
    }
    # The labels are used only for a diagnostic, after the threshold is frozen.
    # They do not select a threshold, a quantile or any other hyperparameter.
    (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
        p, x, y, mask, calibrated, 'shd')
    norms = {k: float(jnp.linalg.norm(g)) for k, g in gradients.items()}
    _, trace = forward(p, x, calibrated, trace=True)
    audit.update(initial_event_fraction=float(metrics[2]),
                 active_neuron_fraction=float(np.asarray(trace[1]).any(axis=(0, 1)).mean()),
                 initial_loss=float(metrics[0]), gradient_norms=norms)
    if not np.isfinite(float(loss)) or not all(np.isfinite(v) for v in norms.values()):
        raise ValueError(f'Nonfinite initialisation: {audit}')
    if not 0 < audit['initial_event_fraction'] < .5 or audit['active_neuron_fraction'] < .5:
        raise ValueError(f'Initial activity outside the committed diagnostic bounds: {audit}')
    if norms['input'] == 0 or (neuron.learn_threshold and norms['threshold_raw'] == 0):
        raise ValueError(f'Initial gradient is zero: {audit}')
    return p, calibrated, audit
