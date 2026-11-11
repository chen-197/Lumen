"""Independent FP64 VJP reference for frozen inputs and partial QKV training."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


TYPES = ['F32', 'F16', 'BF16', 'I8']
OPS = [f'{op}-{layout}' for op in ['add', 'sub', 'mul']
       for layout in ['same', 'row', 'column', 'scalar']]
OPS += ['matmul_transposed_rhs', 'batch-matmul', 'cat', 'rms', 'conv']
QKV_PAIRS = [(d, d) for d in TYPES] + [('BF16', 'I8')]


def expected_shapes(op):
    if op == 'qkv':
        return [[1, 2, 4]] + [[4, 4]] * 3 + [[4]] * 3
    if op == 'matmul_transposed_rhs':
        return [[2, 3], [2, 3]]  # Lumen stores the RHS as [N,K].
    if op == 'batch-matmul':
        return [[1, 2, 2, 3], [1, 2, 3, 2]]
    if op == 'conv':
        return [[1, 1, 3, 3], [2, 1, 2, 2], [2]]
    if op == 'rms' or op.endswith('row'):
        return [[2, 3], [3]]
    if op.endswith('column'):
        return [[2, 3], [2, 1]]
    if op.endswith('scalar'):
        return [[2, 3], [1]]
    return [[2, 3], [2, 3]]


def forward(op, args):
    x, w = args[:2]
    if op.startswith('add-'):
        return x + w
    if op.startswith('sub-'):
        return x - w
    if op.startswith('mul-'):
        return x * w
    if op == 'matmul_transposed_rhs':
        return x @ w.T
    if op == 'batch-matmul':
        return x @ w
    if op == 'cat':
        return torch.cat(args, dim=1)
    if op == 'rms':
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * w
    if op == 'conv':
        return functional.conv2d(x, w, args[2])
    assert op == 'qkv'
    return torch.cat([functional.linear(x, args[i + 1], args[i + 4])
                      .reshape(1, 2, 2, 2).permute(0, 2, 1, 3) for i in range(3)], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    lines = args.log.read_text(encoding='utf-8-sig').splitlines()
    prefix = 'GRADIENT_FLAGS_AUDIT='
    rows = [json.loads(line.split(prefix, 1)[1]) for line in lines if prefix in line]
    devices = ['Cpu', 'Cuda'] if args.cuda else ['Cpu']
    expected = set()
    for device, op, xd, wd, layout in itertools.product(devices, OPS, TYPES, TYPES, range(3)):
        expected.update((device, op, xd, wd, layout, flags, stage)
                        for flags in range(1, 8 if op == 'conv' else 4) for stage in [1, 2])
    if args.cuda:
        expected.update(('Cuda', 'qkv', xd, wd, layout, flags, stage)
                        for xd, wd in QKV_PAIRS for layout in range(3)
                        for flags in range(1, 128) for stage in [1, 2])
    keyed = {(r['device'], r['op'], r['xd'], r['wd'], r['layout'], r['flags'], r['pass']): r for r in rows}
    assert len(rows) == len(keyed) == len(expected), 'Missing or duplicate gradient-flag records'
    assert set(keyed) == expected, 'Incomplete operator/dtype/layout/flag coverage'
    checks = 0
    worst = []

    def check(label, actual, reference, atol=3e-6, rtol=3e-6):
        nonlocal checks
        actual = torch.as_tensor(actual, dtype=torch.float64).flatten()
        reference = torch.as_tensor(reference, dtype=torch.float64).detach().flatten()
        assert actual.shape == reference.shape
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), label
        error = (actual - reference).abs()
        budget = atol + rtol * reference.abs()
        assert (error <= budget).all(), (label, error.max().item(), actual, reference)
        worst.append(dict(label=label, max_absolute_error=error.max().item(),
                          max_error_budget_fraction=(error / budget).max().item()))
        checks += 1

    for key in sorted(expected):
        if key[-1] != 1:
            continue
        row = keyed[key]
        device, op, xd, wd, layout, flags, _ = key
        assert row['dtype_copies'] == (op != 'conv')
        shapes = expected_shapes(op)
        assert len(row['inputs']) == len(shapes)
        tensors = []
        for i, (item, shape) in enumerate(zip(row['inputs'], shapes)):
            assert item['shape'] == shape
            assert item['dtype'] == (xd if i == 0 else wd)
            requested = bool(flags & (1 << i))
            assert item['requires_grad'] == requested
            tensors.append(torch.tensor(item['values'], dtype=torch.float64)
                           .reshape(shape).requires_grad_(requested))
        y = forward(op, tensors)
        for stage in [1, 2]:
            current = keyed[key[:-1] + (stage,)]
            assert current['inputs'] == row['inputs']
            assert current['shape'] == list(y.shape)
            assert current['upstream'] == row['upstream']
            label = f'{device}/{op}/{xd}/{wd}/layout{layout}/flags{flags}/pass{stage}'
            check(label + ':forward', current['y'], y)
            upstream = torch.tensor(current['upstream'], dtype=torch.float64).reshape_as(y)
            y.backward(upstream, retain_graph=True)
            assert len(current['gradients']) == len(tensors)
            for i, (actual, tensor) in enumerate(zip(current['gradients'], tensors)):
                if not tensor.requires_grad:
                    assert actual is None and tensor.grad is None, label
                else:
                    assert actual is not None and tensor.grad is not None, label
                    check(label + f':gradient{i}', actual, tensor.grad)

    training_prefix = 'FROZEN_TRAINING_AUDIT='
    training = [json.loads(line.split(training_prefix, 1)[1]) for line in lines if training_prefix in line]
    expected_training = set(itertools.product(devices, TYPES, range(3), ['sgd', 'adam']))
    training_keys = {(r['device'], r['dtype'], r['layout'], r['optimizer']) for r in training}
    assert len(training) == len(training_keys) == len(expected_training)
    assert training_keys == expected_training
    for row in training:
        losses = row['losses']
        assert len(losses) == 61 and all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0] and sum(losses[-15:]) < sum(losses[:15])
        assert sum((i - 30) * v for i, v in enumerate(losses)) < 0
        assert row['frozen_final'] == row['frozen_initial']
        if row['dtype'] != 'F32':
            continue
        x = torch.tensor(row['x'], dtype=torch.float32).reshape(4, 2)
        w = torch.tensor(row['initial'], dtype=torch.float32).reshape(2, 2).requires_grad_()
        frozen = torch.tensor(row['frozen_initial'], dtype=torch.float32).reshape(2, 2)
        target = torch.tensor(row['target'], dtype=torch.float32).reshape(4, 2)
        opt = torch.optim.SGD([w], lr=0.1) if row['optimizer'] == 'sgd' else torch.optim.Adam([w], lr=0.02)
        reference_losses = []
        for step in range(61):
            opt.zero_grad(set_to_none=True)
            loss = ((x @ w.T @ frozen.T) - target).square().mean()
            reference_losses.append(loss.item())
            if step < 60:
                loss.backward()
                opt.step()
        label = f"frozen-training/{row['device']}/{row['optimizer']}/layout{row['layout']}"
        check(label + ':losses', losses, reference_losses, 3e-6, 3e-5)
        check(label + ':weights', row['final'], w, 3e-5, 3e-5)

    result = dict(torch_version=torch.__version__, records=len(rows), checks=checks,
                  frozen_training_curves=len(training),
                  max_absolute_error=max(r['max_absolute_error'] for r in worst),
                  max_error_budget_fraction=max(r['max_error_budget_fraction'] for r in worst),
                  worst_cases=sorted(worst, key=lambda r: r['max_error_budget_fraction'], reverse=True)[:20])
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'worst_cases'}, indent=2))


if __name__ == '__main__':
    main()
