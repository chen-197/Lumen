"""Independent FP64 forward/VJP/native-output reference for CPU row fallbacks."""
import argparse
import itertools
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = [json.loads(line.split('ROW_MATMUL_AUDIT=', 1)[1])
               for line in args.log.read_text(encoding='utf-8-sig').splitlines() if 'ROW_MATMUL_AUDIT=' in line]
    types = ['F32', 'F16', 'BF16', 'I8']
    expected = set(itertools.product(types, types, range(3), range(3)))
    seen, checks, fraction, errors = set(), 0, 0.0, {}

    def check(label, actual, reference, atol=8e-6, rtol=8e-6):
        nonlocal checks, fraction
        actual = torch.tensor(actual, dtype=torch.float64).reshape(reference.shape)
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), label
        delta = (actual - reference).abs()
        budget = atol + rtol * reference.abs()
        assert (delta <= budget).all(), (label, delta.max().item(), (delta / budget).max().item())
        fraction = max(fraction, (delta / budget).max().item())
        errors[label] = max(errors.get(label, 0), delta.max().item())
        checks += delta.numel()

    for row in records:
        key = row['input_dtype'], row['weight_dtype'], row['layout'], row['case']
        assert key in expected and key not in seen, key
        seen.add(key)
        x = torch.tensor(row['x']['values'], dtype=torch.float64).reshape(row['x']['shape']).requires_grad_(True)
        w = torch.tensor(row['w']['values'], dtype=torch.float64).reshape(row['w']['shape']).requires_grad_(True)
        y = x @ w.T
        check('forward', row['y'], y.detach())
        y.backward(torch.tensor(row['dy'], dtype=torch.float64).reshape(y.shape))
        check('gradient/x', row['x']['grad'], x.grad)
        check('gradient/w', row['w']['grad'], w.grad)
        dtype = row['infer_dtype']
        assert dtype == (key[0] if key[0] == key[1] else 'F32')
        if dtype == 'I8':
            scale = row['infer_scale']
            assert scale > 0 and torch.isfinite(torch.tensor(scale))
            expected_scale = y.detach().abs().max().item() / 127
            assert abs(scale - expected_scale) <= 1e-8 + 3e-5 * expected_scale
            # Judge quantized values against the unrounded reference, with
            # half a scale step plus the F32 reduction allowance. Do not permit
            # arbitrary one-integer errors against an already rounded oracle.
            check('inference/I8', row['infer'], y.detach(), 0.5001 * scale + 8e-6, 8e-6)
        elif dtype in ('F16', 'BF16'):
            torch_dtype = torch.float16 if dtype == 'F16' else torch.bfloat16
            rounded = y.detach().to(torch_dtype).double()
            check(f'inference/{dtype}', row['infer'], rounded, 2e-5, 2**(-10 if dtype == 'F16' else -7))
        else:
            check('inference/F32', row['infer'], y.detach())
    assert seen == expected, expected - seen
    bmm_seen = set()
    bmm_records = [json.loads(line.split('BMM_BACKWARD_AUDIT=', 1)[1])
                   for line in args.log.read_text(encoding='utf-8-sig').splitlines() if 'BMM_BACKWARD_AUDIT=' in line]
    for row in bmm_records:
        key = row['input_dtype'], row['weight_dtype'], row['layout'], row['case']
        assert key in expected and key not in bmm_seen, key
        bmm_seen.add(key)
        x = torch.tensor(row['x']['values'], dtype=torch.float64).reshape(row['x']['shape']).requires_grad_(True)
        w = torch.tensor(row['w']['values'], dtype=torch.float64).reshape(row['w']['shape']).requires_grad_(True)
        y = x @ w
        y.backward(torch.tensor(row['dy'], dtype=torch.float64).reshape(y.shape))
        check('bmm/gradient/x', row['x']['grad'], x.grad)
        check('bmm/gradient/w', row['w']['grad'], w.grad)
    assert bmm_seen == expected, expected - bmm_seen
    result = dict(passed=True, torch_version=torch.__version__, cases=len(seen)+len(bmm_seen),
                  matmul_cases=len(seen), bmm_backward_cases=len(bmm_seen), scalar_checks=checks,
                  max_absolute_errors=errors, max_error_budget_fraction=fraction,
                  reference='PyTorch FP64 on represented values, full forward/VJP; native rounding and half-step I8 error bound')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
