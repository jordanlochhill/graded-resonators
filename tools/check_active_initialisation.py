"""CPU audit of paired initial states and gradients on training inputs only."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax
import numpy as np

from graded_resonators.data import batches, datasets
from graded_resonators.initialisation import calibrate
from graded_resonators.model import ARMS, forward, initialise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 101])
    args = parser.parse_args()
    if jax.default_backend() != 'cpu':
        raise RuntimeError('This is a CPU diagnostic')
    manifest = json.loads(Path('manifests/active-init-gradient-shd.json').read_text())
    config = manifest['defaults']
    splits, permutation, provenance = datasets(args.data, 'shd', config['split_seed'])
    x, _, _ = next(batches(splits['train'], 64, 'shd', permutation, limit=64))
    records = []
    for seed in args.seeds:
        reference = None
        for name, variant in manifest['conditions'].items():
            neuron = replace(ARMS[config['arm']], **variant)
            p = initialise(seed, config['inputs'], config['hidden'], config['classes'],
                           config['omega_range'], config['damping_range'], config['tau_std'], neuron)
            p, neuron, audit = calibrate(p, neuron, splits['train'], permutation, config)
            output = np.asarray(forward(p, x, neuron)[1][0])
            if reference is None:
                reference = (p, output, audit['initial_threshold'])
            else:
                assert audit['initial_threshold'] == reference[2]
                for key in reference[0]:
                    np.testing.assert_array_equal(p[key], reference[0][key])
                np.testing.assert_allclose(output, reference[1], rtol=2e-6, atol=2e-6)
            record = {'seed': seed, 'condition': name, **audit}
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'diagnostic_only': True, 'data': provenance,
                                       'manifest': manifest, 'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
