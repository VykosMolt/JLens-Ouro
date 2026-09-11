"""Retain actual tensors at the unchanged historical Huginn preflight boundary.

The original source files remain unchanged and their ordinary source checks run.
This separately pinned wrapper replaces only evidence retention: its primal
capture reproduces the original function and keeps the compared tensors; the
matrix callback saves the original arguments and then calls the original check.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from artifacts import torch_save


def install(preflight, destination):
    destination = Path(destination)
    original_compare = preflight._compare

    def compare(torch, actual, native, sources, width):
        torch_save(destination / 'native_matrices.pt', {'maps': native, 'sources': list(sources), 'width': width})
        torch_save(destination / 'optimized_matrices.pt', {'maps': actual, 'sources': list(sources), 'width': width})
        return original_compare(torch, actual, native, sources, width)

    def primal(torch, model, text):
        ids = model.encode(text, max_length=128).expand(8, -1).clone()
        initial = model.sample_initial_state(ids)
        if not torch.equal(initial, initial[:1].expand_as(initial)):
            raise ValueError('Huginn initial states differ across derivative lanes')

        def capture(native):
            values, counts, handles = {}, {}, []
            for layer, tap in enumerate(model.layers):
                def hook(module, args, output, layer=layer):
                    counts[layer] = counts.get(layer, 0) + 1
                    values[layer] = output.detach().clone()
                handles.append(tap.register_forward_hook(hook))
            try:
                with torch.no_grad():
                    output = (model.native_logits(ids, input_states=initial) if native
                              else model.unembed(model.forward(ids, input_states=initial)))
                if counts != {layer: 1 for layer in range(34)}:
                    raise ValueError('Huginn virtual cell counts differ from the native topology')
                return values, output
            finally:
                for handle in handles:
                    handle.remove()

        observed, logits = capture(False)
        expected, native_logits = capture(True)
        torch_save(destination / 'primal.pt', {
            'input_ids': ids.cpu(), 'initial_states': initial.cpu(),
            'adapter_states': {k: v.cpu() for k, v in observed.items()},
            'native_states': {k: v.cpu() for k, v in expected.items()},
            'adapter_logits': logits.cpu(), 'native_logits': native_logits.cpu(),
            'initialization': model.initialization_metadata(model.encode(text, max_length=128)),
        })
        for layer in observed:
            if not torch.equal(observed[layer].view(torch.int16), expected[layer].view(torch.int16)):
                raise ValueError(f'Huginn adapter/native primal mismatch at cell{layer}')
        if not torch.equal(logits.view(torch.int32), native_logits.view(torch.int32)):
            raise ValueError('Huginn adapter/native logits differ')
        return {'all_34_cells_bitwise_equal': True, 'logits_bitwise_equal': True,
                'coupled_initial_states': True, 'batch': 8, 'sequence_length': ids.shape[1]}

    preflight._compare = compare
    preflight._huginn_primal = primal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    sys.path.insert(0, str(args.deployment))
    import preflight_gpu
    install(preflight_gpu, args.evidence)
    preflight_gpu.main(remaining)


if __name__ == '__main__':
    main()
