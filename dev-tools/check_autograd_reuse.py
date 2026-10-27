"""Compare complete retained-graph backward sequences with PyTorch FP64."""
import argparse
import itertools
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prefix = "AUTOGRAD_REUSE_AUDIT="
    rows = [json.loads(line.split(prefix, 1)[1])
            for line in args.log.read_text(encoding="utf-8-sig").splitlines()
            if prefix in line]
    phases = ["repeat-1", "repeat-2", "repeat-3", "zero-leaf", "shared", "new-root",
              "explicit", "explicit-then-ones", "explicit-shared", "explicit-reset", "leaf-1", "leaf-2"]
    devices = ["Cpu", "Cuda"] if args.cuda else ["Cpu"]
    expected_keys = set(itertools.product(devices, ["F32", "F16", "BF16", "I8"], range(3), phases))
    keyed = {(r["device"], r["dtype"], r["layout"], r["phase"]): r for r in rows}
    assert len(rows) == len(keyed) == len(expected_keys), "Missing or duplicate backward records"
    assert set(keyed) == expected_keys, "Incomplete dtype/layout/backward sequence coverage"
    comparisons = []
    for device, dtype, layout in itertools.product(devices, ["F32", "F16", "BF16", "I8"], range(3)):
        sequence = [keyed[(device, dtype, layout, phase)] for phase in phases]
        initial = sequence[0]["x"]
        assert all(r["x"] == initial for r in sequence)
        x = torch.tensor(initial, dtype=torch.float64).reshape(2, 2).requires_grad_()
        square = x * x.reshape(2, 2)
        loss = square.sum()
        other_loss = (square + square).sum()
        upstream = torch.tensor([[0.5, -1.0], [2.0, -0.25]], dtype=torch.float64)
        for row in sequence:
            phase = row["phase"]
            if phase in ["zero-leaf", "new-root", "explicit", "explicit-reset", "leaf-1"]:
                x.grad = None
            if phase.startswith("repeat-") or phase == "zero-leaf":
                loss.backward(retain_graph=True)
            elif phase in ["shared", "explicit-shared"]:
                other_loss.backward(retain_graph=True)
            elif phase in ["new-root", "explicit-then-ones"]:
                square.backward(torch.ones_like(square), retain_graph=True)
            elif phase in ["explicit", "explicit-reset"]:
                square.backward(upstream, retain_graph=True)
            else:
                assert phase in ["leaf-1", "leaf-2"]
                x.backward(torch.ones_like(x))
            actual = torch.tensor(row["dx"], dtype=torch.float64).reshape_as(x)
            reference = x.grad
            assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
            error = (actual - reference).abs()
            assert (error <= 2e-6 + 2e-6 * reference.abs()).all(), (device, dtype, layout, phase, actual, reference)
            comparisons.append(dict(device=device, dtype=dtype, layout=layout, phase=phase,
                                    max_abs_error=error.max().item()))
    result = dict(passed=True, comparisons=len(comparisons),
                  max_abs_error=max(r["max_abs_error"] for r in comparisons), details=comparisons)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Retained-graph PyTorch FP64 reference passed: {len(comparisons)} comparisons, "
          f"max absolute error {result['max_abs_error']:.9g}")


if __name__ == "__main__":
    main()
