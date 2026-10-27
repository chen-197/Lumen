"""Independent PyTorch FP64 checks for general CPU prefill scheduling/tiling."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def tensor(record, grad=False):
    values = [float('-inf') if v is None else v for v in record['values']]
    return torch.tensor(values, dtype=torch.float64).reshape(record['shape']).requires_grad_(grad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = [json.loads(line.split('PREFILL_CPU_AUDIT=', 1)[1])
               for line in args.log.read_text(encoding='utf-8-sig').splitlines() if 'PREFILL_CPU_AUDIT=' in line]
    expected = {(op, dtype, layout) for op, dtype, layout in itertools.product(['matmul', 'batch_matmul'], ['F32', 'F16', 'BF16'], range(3))}
    expected |= {('attention', case, layout) for case, layout in itertools.product(range(5), range(3))}
    seen, errors, checks, fraction = set(), {}, 0, 0.0

    def compare(label, actual, reference, atol=1e-5, rtol=2e-5):
        nonlocal checks, fraction
        actual = torch.tensor(actual, dtype=torch.float64).reshape(reference.shape)
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), label
        delta = (actual - reference.detach()).abs()
        budget = atol + rtol * reference.detach().abs()
        assert (delta <= budget).all(), (label, delta.max().item())
        errors[label] = max(errors.get(label, 0), delta.max().item())
        fraction = max(fraction, (delta / budget).max().item())
        checks += delta.numel()

    for row in records:
        if row['op'] == 'attention':
            key = 'attention', row['case'], row['layout']
            q, k, v = [tensor(row[name]) for name in ['q', 'k', 'v']]
            repeats = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
            keep = torch.tensor(row['keep']['values'], dtype=torch.bool).reshape(row['keep']['shape']) if row['keep'] else None
            if row['causal']:
                causal = torch.arange(k.shape[2])[None, :] <= torch.arange(q.shape[2])[:, None] + row['offset']
                keep = causal if keep is None else keep & causal
            bias = tensor(row['bias']) if row['bias'] else None
            mask = keep if bias is None else bias if keep is None else bias.masked_fill(~keep, -torch.inf)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=row['scale'], dropout_p=0)
            assert list(output.shape) == row['output_shape']
            compare('attention/forward', row['y'], output)
        else:
            key = row['op'], row['dtype'], row['layout']
            x, w = tensor(row['x'], True), tensor(row['w'], True)
            output = x @ (w.T if row['op'] == 'matmul' else w)
            compare(f"{row['op']}/forward", row['y'], output)
            output.backward(torch.tensor(row['dy'], dtype=torch.float64).reshape(output.shape))
            compare(f"{row['op']}/grad/x", row['x']['grad'], x.grad, 3e-5, 3e-5)
            compare(f"{row['op']}/grad/w", row['w']['grad'], w.grad, 3e-5, 3e-5)
            dtype = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}[row['dtype']]
            # Native no_grad output rounds once to the input dtype. Near a
            # midpoint, a different F32 reduction may choose an adjacent value.
            rounded = output.detach().to(dtype).double()
            compare(f"{row['op']}/inference/{row['dtype']}", row['inference_y'], rounded,
                    2e-5, torch.finfo(dtype).eps if dtype != torch.float32 else 2e-5)
        assert key in expected and key not in seen, key
        seen.add(key)
    assert seen == expected, ('missing prefill checks', expected - seen)
    result = {'passed': True, 'torch_version': torch.__version__, 'cases': len(seen),
              'scalar_checks': checks, 'max_absolute_errors': errors, 'max_error_budget_fraction': fraction,
              'reference': 'PyTorch FP64 on represented inputs; includes F32 gradients and staged lowp inference rounding'}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
