"""Use the historical producer unchanged, retaining auditable top-10/sort samples."""
from __future__ import annotations
import numpy as np
import torch

SAMPLE_ITEMS = (0, 1, 2)
SAMPLE_VIRTUAL = (0, 47, 95, 143, 170, 176, 181, 189, 190, 191)

class ReadoutCapture:
    def __init__(self, model, columns, *, sample):
        self.model, self.n_layers, self.sample = model, columns, sample
        self.top10 = None
        self.numerical = None
    def __getattr__(self, key):
        return getattr(self.model, key)
    def unembed(self, transported):
        logits = self.model.unembed(transported)
        ordered = logits.float().argsort(dim=-1, descending=True)
        self.top10 = ordered[:, :10].cpu().numpy().astype(np.int32)
        if self.sample:
            selected = [v for v in SAMPLE_VIRTUAL if v < self.n_layers]
            self.numerical = {'virtual_indices': selected, 'transported': transported[selected].float().cpu(),
                              'logits': logits[selected].float().cpu(), 'sorted_ids': ordered[selected].to(torch.int32).cpu()}
        return logits

@torch.no_grad()
def readout(evaluator, model, items, states, jacobians, exits, columns, *, check_stop=lambda: None):
    """Original FP32 J·h, BF16 native unembedding, min-over-alias full sort."""
    chunks, top10, samples = [], [], []
    for i, item in enumerate(items):
        check_stop()
        capture = ReadoutCapture(model, columns, sample=i in SAMPLE_ITEMS)
        arrays, kept = evaluator.readout_arrays(capture, [item], states[i:i+1,:columns], jacobians,
                                                exits[i:i+1], model.n_ut-1)
        del kept
        # Membership equivalence at k=10 verifies that the separately captured
        # original argsort honors the producer's exact tie convention.
        task = evaluator.TASK_NAMES[item.task]
        for j, name in enumerate(task.names):
            hits = np.isin(capture.top10, task.forms[name]).any(axis=-1)
            if not np.array_equal(hits, arrays['allrank'][0,j] < 10):
                raise ValueError('Captured top-10 disagrees with original alias ranks')
        chunks.append(arrays); top10.append(capture.top10)
        if capture.numerical is not None:
            samples.append({'item_index': i, **capture.numerical})
    output = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}
    output['top10_ids'] = np.stack(top10)
    return output, samples

@torch.no_grad()
def native_development(model, items):
    from jlens.hooks import ActivationRecorder
    virtual_rows, physical_rows, wrapper_rows, native_rows, unembedded_rows = [], [], [], [], []
    for item in items:
        ids = torch.tensor([item.token_ids], device=model.input_device)
        physical = {}
        handles = []
        for layer, block in enumerate(model.blocks):
            def hook(module, args, kwargs, output, layer=layer):
                index = int(kwargs['current_ut']) * 48 + layer
                if index in physical:
                    raise ValueError('A physical block fired twice for one virtual position')
                value = output[0] if isinstance(output, tuple) else output
                physical[index] = value[0, -1].detach().float().cpu()
            handles.append(block.register_forward_hook(hook, with_kwargs=True))
        try:
            with ActivationRecorder(model.layers, at=range(192)) as rec:
                output = model.hf_model(input_ids=ids, use_cache=False)
            virtual = torch.stack([rec.activations[v][0,-1].float().cpu() for v in range(192)])
            # Unembed the whole final-step sequence, as the native head does; on the deployed GPU
            # a single row gives different BF16 logits (diagnostics/native_logit_v1).
            unembedded = model.unembed(rec.activations[191])[0,-1].float().cpu()
        finally:
            for handle in handles: handle.remove()
        if sorted(physical) != list(range(192)):
            raise ValueError('Physical hooks did not cover every virtual block')
        physical_tensor = torch.stack([physical[v] for v in range(192)])
        with ActivationRecorder(model.layers, at=range(192)) as rec:
            model.forward(ids)
        wrapper = torch.stack([rec.activations[v][0,-1].float().cpu() for v in range(192)])
        native = output.logits[0,-1].float().cpu()
        virtual_rows.append(virtual); physical_rows.append(physical_tensor); wrapper_rows.append(wrapper)
        native_rows.append(native); unembedded_rows.append(unembedded)
    return {'virtual_states': torch.stack(virtual_rows), 'physical_states': torch.stack(physical_rows),
            'wrapper_states': torch.stack(wrapper_rows), 'native_logits': torch.stack(native_rows),
            'unembedded_logits': torch.stack(unembedded_rows), 'item_names': [item.name for item in items]}
