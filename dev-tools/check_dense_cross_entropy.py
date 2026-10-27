"""Check dense-target cross entropy and both input gradients with PyTorch FP64."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('log', type=Path)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
records = [json.loads(line.split('DENSE_CE_AUDIT=', 1)[1])
           for line in args.log.read_text(encoding='utf-8-sig').splitlines()
           if 'DENSE_CE_AUDIT=' in line]
devices = ['Cpu', 'Cuda'] if args.cuda else ['Cpu']
assert Counter(r['device'] for r in records) == Counter({d: 770 if d == 'Cpu' else 768 for d in devices}), 'missing dense CE records'
expected = {(xd, td, order, pattern, flags)
            for xd in ['F32', 'F16', 'BF16', 'I8']
            for td in ['F32', 'F16', 'BF16', 'I8']
            for order in range(3) for pattern in range(4) for flags in range(4)}
for device in devices:
    assert {(r['xd'], r['td'], r['order'], r['pattern'], r['flags'])
            for r in records if r['device'] == device} == expected
checks = []
def check(label, got, reference, atol, rtol):
    actual = torch.tensor(got, dtype=torch.float64).flatten()
    reference = reference.detach().flatten()
    assert actual.shape == reference.shape and torch.isfinite(actual).all() and torch.isfinite(reference).all(), label
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol, msg=label)
    error = (actual-reference).abs()
    checks.append(dict(label=label, max_absolute_error=error.max().item(),
        max_error_budget_fraction=(error/(atol+rtol*reference.abs())).max().item(), atol=atol, rtol=rtol))
for i, row in enumerate(records):
    flags = row['flags']
    x = torch.tensor(row['x'], dtype=torch.float64).reshape(2, 3).requires_grad_(flags in [1, 3])
    target = torch.tensor(row['target'], dtype=torch.float64).reshape(2, 3).requires_grad_(flags in [2, 3])
    loss = -(target*torch.log_softmax(x, dim=-1)).sum()/2
    check(f'{i}:loss', row['loss'], loss, 2e-6, 2e-6)
    if flags:
        loss.backward(torch.tensor(row['upstream'], dtype=torch.float64))
    for name, value, required in [('dx', x, flags in [1, 3]), ('dt', target, flags in [2, 3])]:
        assert (row[name] is not None) == required
        if required:
            check(f'{i}:{name}', row[name], value.grad, 3e-6, 3e-6)
assert len(checks) == 1540 + (1536 if args.cuda else 0), len(checks)
summary = dict(torch_version=torch.__version__, records=len(records), checks=len(checks),
    max_absolute_error=max(r['max_absolute_error'] for r in checks),
    max_error_budget_fraction=max(r['max_error_budget_fraction'] for r in checks), results=checks)
args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k != 'results'}, indent=2))
