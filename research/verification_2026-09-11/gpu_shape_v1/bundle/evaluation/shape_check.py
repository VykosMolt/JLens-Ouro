"""Scoring-shape check on one GPU: native output (A), packing of the executed head (B), development shapes.

Each comparison keeps values and weights fixed and changes one layout or op; mismatches are data.
The model's own unembed (final norm, then lm_head) is the op under test.
"""
from __future__ import annotations
import numpy as np
import torch

LAYOUTS = ('executed', 'row1', 'rows190', 'rows191', 'rows192', 'col160')
STACKED = (148, 160, 190, 191, 192)
EXECUTED_ROWS = {'raw': 192, 'fit01': 192, 'fit02': 192, 'penultimate': 190, 'sampled_sum': 191, 'diagonal': 191}


def compare(left, right):
    """Exact equality decides; counts and distances describe."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('incomparable tensors')
    differ = left != right
    delta = (left.double() - right.double()).abs()
    return {'equal': bool(torch.equal(left, right)), 'differing': int(differ.sum()), 'max_abs': float(delta.max()),
            'rms': float(delta.pow(2).mean().sqrt())}


def top10_change(left, right):
    a, b = left.float().argsort(descending=True)[:10], right.float().argsort(descending=True)[:10]
    return len(set(a.tolist()) - set(b.tolist()))


def packed(model, rows, layout):
    """Logits for one item's [n, d] rows; every row keeps its values and only its batch changes."""
    n = len(rows)
    if layout == 'executed':
        return model.unembed(rows)
    if layout == 'row1':
        return torch.cat([model.unembed(rows[r:r + 1]) for r in range(n)])
    size = int(layout[4:])
    if n == size:
        return model.unembed(rows)
    if n < size:  # pad with the item's own leading rows
        return model.unembed(torch.cat([rows, rows[:size - n]]))[:n]
    return torch.cat([model.unembed(rows[:size]), model.unembed(rows[n - size:])[2 * size - n:]])


def rank_rows(names, logits, max_names):
    """The _ranks_of rule (descending argsort, minimum over alias forms), top-10 and tie bounds."""
    n, vocab = logits.shape
    order = logits.argsort(dim=-1, descending=True)
    descending = logits.gather(-1, order)
    position = torch.empty_like(order)
    position.scatter_(1, order, torch.arange(vocab, device=logits.device).expand_as(order))
    ids = names.flat_ids.to(logits.device)
    values = logits[:, ids].contiguous()
    ascending = descending.flip(-1).contiguous()
    greater = vocab - torch.searchsorted(ascending, values, right=True)
    at_least = vocab - torch.searchsorted(ascending, values, right=False)
    stacked = torch.stack([position[:, ids], greater, at_least - 1]).cpu().numpy()
    out = {key: np.full((max_names, n), -1, np.int32) for key in ('allrank', 'best', 'worst')}
    for j, (a, b) in enumerate(zip(names.slices[:-1], names.slices[1:])):
        for k, key in enumerate(('allrank', 'best', 'worst')):
            out[key][j] = stacked[k, :, a:b].min(1)
    out['top10'] = order[:, :10].to(torch.int32).cpu().numpy()
    return out


def difference(logits, reference):
    differ = logits != reference
    delta = logits.float() - reference.float()
    return [int(differ.any(-1).sum()), int(differ.sum()), float(delta.abs().max()), float(delta.double().pow(2).sum())]


def layouts(model, names, rows, max_names, check_stop):
    """B: the executed per-item layout and each alternative, over identical transported rows."""
    n, columns = rows.shape[:2]
    device, vocab = model.input_device, model.hf_model.lm_head.weight.shape[0]
    shape = (n, max_names, columns)
    arrays = {layout: {'allrank': np.full(shape, -1, np.int32), 'top10': np.zeros((n, columns, 10), np.int32),
                       'difference': np.zeros((n, 4))} for layout in LAYOUTS}
    arrays['executed'].update(best=np.full(shape, -1, np.int32), worst=np.full(shape, -1, np.int32))

    def add(layout, i, logits, reference):
        ranks = rank_rows(names, logits.float(), max_names)
        for key in arrays[layout]:
            if key != 'difference':
                arrays[layout][key][i] = ranks[key]
        arrays[layout]['difference'][i] = difference(logits, reference)

    for i in range(n):
        check_stop()
        x = rows[i].to(device)
        reference = model.unembed(x)
        for layout in LAYOUTS[:-1]:
            add(layout, i, reference if layout == 'executed' else packed(model, x, layout), reference)
    grid = torch.empty(n, columns, vocab, dtype=torch.bfloat16)
    for v in range(columns):
        grid[:, v] = model.unembed(rows[:, v].to(device)).cpu()
    for i in range(n):
        add('col160', i, grid[i].to(device), model.unembed(rows[i].to(device)))
    summary = {layout: {'rows_differing': int(a['difference'][:, 0].sum()), 'logits_differing': int(a['difference'][:, 1].sum()),
                        'max_abs': float(a['difference'][:, 2].max()),
                        'rms': float(np.sqrt(a['difference'][:, 3].sum() / (n * columns * vocab)))} for layout, a in arrays.items()}
    return arrays, summary


def native(model, evaluator, names, items, retained, continuations, check_stop):
    """A: native final logits against unembedding of the final state captured in the same run, per layout.

    States are regenerated with the frozen cache_states; the retained-state variant shows any cross-run difference.
    """
    norm, head = model.hf_model.model.norm, model.hf_model.lm_head
    device = model.input_device
    native_logits = torch.empty(len(items), head.weight.shape[0], dtype=torch.bfloat16)
    regenerated = torch.empty(len(items), model.n_layers, model.d_model, dtype=torch.bfloat16)
    records, variants = [], []
    for i, item in enumerate(items):
        check_stop()
        captured, text = evaluator.cache_states(model, [item], position=-1)
        seen = {'norm_input': [], 'norm_output': [], 'lm_head_input': []}

        def reader(prefix, keep_output):
            def hook(module, args, output):
                seen[prefix + '_input'].append(args[0].detach())
                if keep_output:
                    seen[prefix + '_output'].append(output.detach())
            return hook
        handles = [norm.register_forward_hook(reader('norm', True)), head.register_forward_hook(reader('lm_head', False))]
        try:
            output = model.hf_model(input_ids=torch.tensor([item.token_ids], device=device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if (len(seen['norm_input']), len(seen['lm_head_input'])) != (model.n_ut, 1):
            raise ValueError('final norm or lm_head call count differs from the native forward')
        logits = output.logits[0, -1]
        steps = [u for u in range(model.n_ut) if torch.equal(seen['norm_output'][u][0, -1], seen['lm_head_input'][0][0, -1])]
        step = steps[0] if len(steps) == 1 else model.n_ut - 1
        state, kept = captured[0].to(device), retained[i].to(device)
        variants.append({'sequence_[1,S,2048]': head(norm(seen['norm_input'][step].clone()))[0, -1].cpu(),
                         'scorer_[192,2048]_row191': model.unembed(state)[191].cpu(),
                         'single_[1,2048]': model.unembed(state[191:192])[0].cpu(),
                         'retained_state_scorer_[192,2048]_row191': model.unembed(kept)[191].cpu()})
        native_logits[i], regenerated[i] = logits.cpu(), captured[0].to(torch.bfloat16)
        records.append({'item_index': i, 'name': item.name, 'exit_step_matches': steps, 'exit_step': step,
                        'native_dtype': str(logits.dtype), 'continuation_equal': text[0] == continuations[i],
                        'captured_final_state_equals_native_head_path_input': bool(torch.equal(
                            seen['norm_input'][step][0, -1].float(), captured[0, 191].to(device))),
                        'regenerated_vs_retained_states': compare(captured[0], retained[i].float())})
    exits = model.unembed(regenerated[:, 191].to(device)).cpu()  # the executed exits layout over this run's states
    for i, item in enumerate(items):
        target = names.names.index(item.intermediates[0])
        reference = native_logits[i].to(device)
        native_ranks = names.ranks(reference.float()[None])[:, 0]
        records[i]['variants'] = {}
        for label, value in {**variants[i], f'exits_[{len(items)},2048]': exits[i]}.items():
            value = value.to(device)
            ranks = names.ranks(value.float()[None])[:, 0]
            records[i]['variants'][label] = {**compare(value, reference), 'top10_ids_replaced': top10_change(value, reference),
                                             'name_ranks_changed': int((ranks != native_ranks).sum()),
                                             'hit10_changed_intended': bool((ranks[target] < 10) != (native_ranks[target] < 10)),
                                             'hit10_changed_controls': int(sum((ranks[k] < 10) != (native_ranks[k] < 10)
                                                                               for k in range(len(names.names)) if k != target))}
    return native_logits, regenerated, records


def samples(model, saved):
    """B for every arm's saved rows, including the 190- and 191-row arms, against their saved executed logits."""
    device, out = model.input_device, {}
    for arm, entries in saved.items():
        out[arm] = []
        for s in entries:
            x, expected = s['transported'].to(device), s['logits'].to(device)
            n = len(x)
            variants = {f'batch_[{n},2048]': model.unembed(x),
                        'single_[1,2048]': torch.cat([model.unembed(x[r:r + 1]) for r in range(n)])}
            for size in (190, 191, 192):
                variants[f'cycled_[{size},2048]'] = model.unembed(x[torch.arange(size, device=device) % n])[:n]
            out[arm].append({'item_index': s['item_index'], 'virtual_indices': s['virtual_indices'],
                             'executed_rows': EXECUTED_ROWS[arm], **{k: compare(v, expected) for k, v in variants.items()}})
    return out


def shaped_inputs(target, positions, stacked):
    """The diagnostic's shapes containing `target`, the last row of `positions`: label -> (input, target row index)."""
    cases = {'[2048]': (target, -1), '[1,2048]': (target[None], -1), '[1,1,2048]': (target[None, None], -1),
             '[S,2048]': (positions, -1), '[1,S,2048]': (positions[None], -1)}
    for n in stacked:
        fill = positions[torch.arange(n - 1) % len(positions)]  # the item's own positions, cycled
        cases[f'[{n},2048]/first'] = (torch.cat([target[None], fill]), 0)
        cases[f'[{n},2048]/last'] = (torch.cat([fill, target[None]]), -1)
    return cases


def development(model, retained):
    """The native-logit diagnostic from its retained inputs, extended to 190-, 191- and 192-row layouts."""
    norm, head = model.hf_model.model.norm, model.hf_model.lm_head
    device, records, rows = model.input_device, {}, {}
    for name, entry in retained.items():
        step = entry['exit_step']
        norm_input, head_input = entry['norm_input'][step, 0], entry['lm_head_input'][0]
        native_norm, native_head = entry['norm_output'][step, 0, -1].to(device), entry['lm_head_output'][0, -1].to(device)
        plan = (('norm', norm, shaped_inputs(norm_input[-1], norm_input, STACKED), native_norm),
                ('lm_head', head, shaped_inputs(head_input[-1], head_input, STACKED), native_head),
                ('lm_head_norm', lambda t: head(norm(t)), shaped_inputs(norm_input[-1], norm_input, STACKED), native_head))
        records[name] = {'native_logits_equal_lm_head_output': bool(torch.equal(entry['logits'][0, -1], entry['lm_head_output'][0, -1])),
                         'norm_output_equals_lm_head_input': bool(torch.equal(entry['norm_output'][step, 0, -1], entry['lm_head_input'][0, -1]))}
        rows[name] = {}
        for family, function, cases, reference in plan:
            for label, (tensor, index) in cases.items():
                produced = []
                for repeat in range(2):
                    output = function(tensor.to(device, copy=True))
                    produced.append(output.reshape(-1, output.shape[-1])[index])
                result = {'vs_native': [compare(p, reference) for p in produced], 'repeat_equal': bool(torch.equal(*produced))}
                kept = entry['retained_shapes'][family].get(label)
                if kept is not None:
                    result['vs_retained_diagnostic'] = [compare(p, kept[r].to(device)) for r, p in enumerate(produced)]
                records[name][f'{family}{label}'] = result
                rows[name][f'{family}{label}'] = torch.stack(produced).cpu()
    return records, rows
