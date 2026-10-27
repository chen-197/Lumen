"""Independently inspect public-API state checks, checkpoint bytes and F32 training."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

import torch


class Reader:
    def __init__(self, path):
        self.data = Path(path).read_bytes()
        self.pos = 0

    def take(self, count):
        if count > len(self.data) - self.pos:
            raise ValueError("truncated checkpoint")
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        return result

    def number(self, code):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def vector(self, code):
        count = self.number("Q")
        size = struct.calcsize("<" + code)
        if count > (len(self.data) - self.pos) // size:
            raise ValueError("truncated vector")
        return list(struct.unpack("<" + str(count) + code, self.take(count * size)))


def checkpoint(path):
    reader = Reader(path)
    params = []
    for _ in range(reader.number("Q")):
        shape = reader.vector("Q")
        dtype = reader.number("I")
        raw_dtype = reader.number("I")
        if dtype != raw_dtype or dtype not in range(4):
            raise ValueError("checkpoint dtype/payload mismatch")
        scale = None
        if dtype == 0:
            values = reader.vector("f")
        elif dtype in [1, 2]:
            bits = reader.vector("H")
            if dtype == 1:
                values = [struct.unpack("<e", struct.pack("<H", value))[0] for value in bits]
            else:
                values = [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in bits]
        else:
            values = reader.vector("b")
            scale = reader.number("f")
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("invalid checkpoint scale")
            values = (torch.tensor(values, dtype=torch.float32) * scale).tolist()
        if math.prod(shape) != len(values):
            raise ValueError("checkpoint shape mismatch")
        params.append(dict(shape=shape, dtype=["F32", "F16", "BF16", "I8"][dtype], values=values, scale=scale))
    if reader.pos != len(reader.data):
        raise ValueError("unexpected trailing checkpoint bytes")
    return params


def rounded(values, dtype, scale=None):
    values = torch.as_tensor(values, dtype=torch.float32)
    if dtype == "F16":
        return values.half().float()
    if dtype == "BF16":
        return values.bfloat16().float()
    if dtype == "I8":
        return (values / scale).round().clamp(-127, 127) * scale
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8-sig"))
    torch.set_num_threads(1)
    checks = 0
    max_error = 0.0

    def compare(actual, reference, label):
        nonlocal checks, max_error
        actual = torch.tensor(actual, dtype=torch.float64)
        reference = torch.as_tensor(reference, dtype=torch.float64)
        assert actual.shape == reference.shape, label
        error = (actual - reference).abs()
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), label
        assert (error <= 2e-6 + 2e-6 * reference.abs()).all(), (label, actual, reference, error)
        max_error = max(max_error, error.max().item() if error.numel() else 0.0)
        checks += 1

    phases = ["initial", "updated", "detach", "cpu_copy", "same_device_copy", "checkpoint_loaded",
              "bad_checkpoint_rejected", "updated_again", "detach_independent", "cpu_copy_independent",
              "same_device_copy_independent", "inplace_cpu", "invalid_import_preserved",
              "native_view_modified", "compute_view_after_write"]
    compare(data["held_view_snapshot"], [7., 2., 3., 4.], "copy while a source view is held")
    cases = list(itertools.product(["F32", "F16", "BF16", "I8"], range(3), [False, True]))
    expected_keys = set(itertools.product(cases, phases))
    rows = {((r["case"]["dtype"], r["case"]["layout"], r["case"]["copies"]), r["phase"]): r for r in data["snapshots"]}
    assert len(data["snapshots"]) == len(rows) == len(expected_keys) and set(rows) == expected_keys
    initial_values = [1., -2., .5, 3., .25, -.75]
    gradient = torch.tensor([.5, -1., 2., .25, -.75, 1.5])
    references = {}
    for case in cases:
        dtype, _, _ = case
        raw = rows[(case, "initial")]["raw"]
        scale = raw["I8"]["scale"] if dtype == "I8" else None
        if dtype == "I8":
            assert scale == (torch.tensor(3., dtype=torch.float32) / 127).item()
        initial = rounded(initial_values, dtype, scale)
        updated = rounded(initial - gradient * .1, dtype, scale)
        updated_again = rounded(updated - .1, dtype, scale)
        references[case] = updated
        for phase in phases:
            row = rows[(case, phase)]
            assert row["dtype"] == dtype and row["shape"] == [2, 3]
            assert row["requires_grad"] == (not phase.startswith("detach"))
            expected_device = "Cpu" if phase in ["cpu_copy", "cpu_copy_independent", "inplace_cpu", "invalid_import_preserved", "native_view_modified", "compute_view_after_write"] else data["device"]
            assert row["device"] == expected_device, (case, phase)
            reference = initial if phase == "initial" else updated_again if phase in ["updated_again", "inplace_cpu", "invalid_import_preserved"] else -updated_again if phase in ["native_view_modified", "compute_view_after_write"] else updated
            compare(row["values"], reference, (case, phase))
            if phase == "invalid_import_preserved":
                assert row["invalid_scale_rejections"] == 4
            if dtype == "I8":
                assert row["raw"]["I8"]["scale"] == scale

    file_cases = {(r["case"]["dtype"], r["case"]["layout"], r["case"]["copies"]): r for r in data["checkpoints"]}
    assert len(file_cases) == len(data["checkpoints"]) == len(cases) and set(file_cases) == set(cases)
    file_hashes = {}
    rejected = 0
    for case, entry in file_cases.items():
        parsed = checkpoint(entry["valid"])
        assert len(parsed) == 2 and parsed[0]["dtype"] == case[0] and parsed[0]["shape"] == [2, 3]
        compare(parsed[0]["values"], references[case], (case, "checkpoint bytes"))
        bias_scale = (torch.tensor(.25, dtype=torch.float32) / 127).item() if case[0] == "I8" else None
        compare(parsed[1]["values"], rounded([.125, -.25], case[0], bias_scale), (case, "bias bytes"))
        for name in ["malformed", "truncated"]:
            try:
                checkpoint(entry[name])
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("invalid checkpoint accepted by independent reader")
            assert entry[name + "_rejected"]
        assert entry["old_gradient_cleared"]
        for name in ["valid", "malformed", "truncated"]:
            path = Path(entry[name])
            file_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    training = {(r["dtype"], r["optimizer"]): r for r in data["training"]}
    assert len(training) == len(data["training"]) == 8
    assert set(training) == set(itertools.product(["F32", "F16", "BF16", "I8"], ["sgd", "adam"]))
    for key, row in training.items():
        losses = row["losses"]
        assert len(losses) == 46 and all(math.isfinite(x) for x in losses)
        assert losses[-1] < losses[0] and sum(losses[-12:]) < sum(losses[:12]), key
        assert sum((i - 22.5) * loss for i, loss in enumerate(losses)) < 0, key
        if key[0] != "F32":
            continue
        weight = torch.tensor(row["initial"][0], dtype=torch.float32).reshape(2, 3).requires_grad_()
        bias = torch.tensor(row["initial"][1], dtype=torch.float32).requires_grad_()
        opt = torch.optim.SGD([weight, bias], lr=.03) if key[1] == "sgd" else torch.optim.Adam([weight, bias], lr=.02)
        inputs = torch.tensor([[1., 0, 0], [0., 1, 0], [0., 0, 1], [1., 1, 1]])
        target = torch.tensor([[.35, -.125], [-.15, .225], [.15, -.175], [.25, -.025]])
        reference_losses = []
        for step in range(46):
            opt.zero_grad(set_to_none=True)
            loss = (inputs @ weight.T + bias - target).square().mean()
            reference_losses.append(loss.item())
            if step < 45:
                loss.backward()
                opt.step()
        compare(losses, reference_losses, (key, "training"))
        compare(row["final"][0], weight.detach().flatten(), (key, "final weight"))
        compare(row["final"][1], bias.detach(), (key, "final bias"))
        if key[1] == "sgd":
            assert len(row["resume_losses"]) == 25
            compare(row["resume_losses"], losses[-25:], (key, "checkpoint continuation"))
            resume = args.results.parent / "resume-f32-sgd.bin"
            checkpoint(resume)
            file_hashes[str(resume)] = hashlib.sha256(resume.read_bytes()).hexdigest()
        else:
            assert not row["resume_losses"]
    result = dict(passed=True, device=data["device"], records=len(rows), checks=checks,
                  checkpoint_sets=len(cases), rejected_files=rejected, rejected_imports=4*len(cases), training_curves=len(training),
                  max_absolute_error=max_error, files=file_hashes)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
