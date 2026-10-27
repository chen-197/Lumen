"""Independent PyTorch FP64 references for general attention and LayerNorm."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def tensor(record):
    return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape']).requires_grad_()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--cuda', action='store_true', help='Require CUDA audit cases as well as CPU.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    for line in args.log.read_text(encoding='utf-8-sig', errors='strict').splitlines():
        if 'GENERIC_NN_AUDIT=' in line:
            records.append(json.loads(line.split('GENERIC_NN_AUDIT=', 1)[1]))
    devices = ['Cpu', 'Cuda'] if args.cuda else ['Cpu']
    expected = {(op, device, dtype, layout, case)
                for op, count in [('layer_norm', 8), ('attention', 9)]
                for device, dtype, layout, case in itertools.product(devices, ['F32', 'F16', 'BF16'], range(3), range(count))}
    seen, curves, errors = set(), set(), {}
    checks = 0
    max_budget_fraction = 0.0

    def compare(label, actual, expected, atol=3e-5, rtol=3e-5):
        nonlocal checks, max_budget_fraction
        actual = torch.tensor(actual, dtype=torch.float64).reshape(expected.shape)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
        delta = (actual - expected.detach()).abs()
        assert (delta <= atol + rtol * expected.detach().abs()).all(), (label, delta.max().item())
        if delta.numel():
            max_budget_fraction = max(max_budget_fraction, (delta / (atol + rtol * expected.detach().abs())).max().item())
        errors[label] = max(errors.get(label, 0.0), delta.max().item() if delta.numel() else 0.0)
        checks += actual.numel()

    for row in records:
        if row['op'] == 'training':
            key = row['device'], row['cross'], row['adam']
            assert key not in curves
            curves.add(key)
            loss = torch.tensor(row['losses'], dtype=torch.float64)
            assert loss.numel() == 100 and torch.isfinite(loss).all()
            assert loss[-10:].mean() < loss[:10].mean() and loss[-1] < loss[0] * 0.8
            t = torch.arange(100, dtype=torch.float64) - 49.5
            assert (t * (loss - loss.mean())).sum() < 0
            checks += loss.numel()
            continue
        key = row['op'], row['device'], row['dtype'], row['layout'], row['case']
        assert key in expected and key not in seen, key
        seen.add(key)
        if row['op'] == 'layer_norm':
            x = tensor(row['x'])
            weight = tensor(row['weight']) if row['weight'] else None
            bias = tensor(row['bias']) if row['bias'] else None
            output = F.layer_norm(x, row['normalized_shape'], weight, bias, row['eps'])
            inputs = [('x', x), ('weight', weight), ('bias', bias)]
            infer = output
        else:
            q, k, v = [tensor(row[n]) for n in ['q', 'k', 'v']]
            bias = tensor(row['bias']) if row['bias'] else None
            heads = q.shape[1] // k.shape[1]
            kk = k.repeat_interleave(heads, dim=1)
            vv = v.repeat_interleave(heads, dim=1)
            mask = None
            if row['keep']:
                mask = torch.tensor(row['keep']['values'], dtype=torch.bool).reshape(row['keep']['shape'])
            if row['causal']:
                causal = torch.arange(k.shape[2])[None, :] <= torch.arange(q.shape[2])[:, None] + row['offset']
                mask = causal if mask is None else mask & causal
            if bias is not None:
                mask = bias if mask is None else bias.masked_fill(~mask, -torch.inf)
            if mask is not None:
                # PyTorch's SDPA backend expects at least a matrix mask; Lumen
                # also accepts scalar broadcasting, so expand the reference view.
                mask = mask.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
            output = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, dropout_p=0, scale=row['scale'])
            inputs = [('q', q), ('k', k), ('v', v), ('bias', bias)]
            # Independent staged oracle: native no_grad matmuls preserve same
            # input dtype; masked arithmetic with the F32 scalar promotes scores.
            dtype = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}[row['dtype']]
            scores = (q @ kk.transpose(-1, -2)).to(dtype).double() * row['scale']
            if row['bias'] is not None:
                scores = scores + bias
            elif mask is not None:
                scores = scores.masked_fill(~mask, -torch.inf)
            probabilities = torch.softmax(scores, dim=-1).nan_to_num(nan=0.0)
            fast = row['keep'] is None and row['bias'] is None and (not row['causal'] or (q.shape[2] == k.shape[2] and row['offset'] == 0))
            if fast:
                probabilities = probabilities.to(dtype).double()
            infer = probabilities @ vv
            if fast:
                infer = infer.to(dtype).double()
        shape = output.shape
        upstream = torch.tensor(row['dy'], dtype=torch.float64).reshape(shape)
        output.backward(upstream)
        compare(f"{row['op']}/forward", row['y'], output, 8e-6, 8e-6)
        for name, value in inputs:
            if value is not None:
                assert row[name]['grad'] is not None and value.grad is not None, (key, name)
                compare(f"{row['op']}/grad/{name}", row[name]['grad'], value.grad)
        tolerance = {'F32': 1e-5, 'F16': 0.002, 'BF16': 0.015}[row['dtype']] if row['op'] == 'attention' else 2e-6
        compare(f"{row['op']}/inference/{row['dtype']}", row['inference_y'], infer, tolerance, tolerance)
    assert seen == expected, ('Missing numerical coverage', sorted(expected - seen))
    expected_curves = set(itertools.product(devices, [False, True], [False, True]))
    assert curves == expected_curves, ('Missing training coverage', expected_curves - curves)
    result = {'passed': True, 'torch_version': torch.__version__, 'reference': 'PyTorch FP64, CPU; actual Lumen CPU/CUDA per record',
              'numerical_cases': len(seen), 'training_curves': len(curves), 'scalar_checks': checks,
              'max_absolute_errors': errors, 'max_error_budget_fraction': max_budget_fraction,
              'training': [{'device': r['device'], 'cross': r['cross'], 'adam': r['adam'],
                            'initial': r['losses'][0], 'final': r['losses'][-1]} for r in records if r['op'] == 'training']}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
