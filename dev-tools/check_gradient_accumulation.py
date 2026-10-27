"""Verify complete mixed-residency accumulation sequences with independent FP64 sums."""
import argparse
import itertools
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = "GRADIENT_ACCUMULATION_AUDIT="
    rows = [json.loads(line.split(prefix, 1)[1])
            for line in args.log.read_text(encoding="utf-8-sig").splitlines() if prefix in line]
    devices = ["Cpu", "Cuda"] if args.cuda else ["Cpu"]
    keys = set()
    for device in devices:
        count = 3 if device == "Cuda" else 1
        keys.update(itertools.product([device], ["F32", "F16", "BF16", "I8"], range(3),
                                      range(count), range(count), [False, True], range(6)))
    fields = ["device", "dtype", "layout", "residency", "route", "checked", "step"]
    keyed = {tuple(r[name] for name in fields): r for r in rows}
    assert len(rows) == len(keyed) == len(keys) and set(keyed) == keys, "Missing/duplicate accumulation cases"
    comparisons = 0
    max_error = 0.0
    for case in sorted({key[:-1] for key in keys}):
        sequence = [keyed[case + (step,)] for step in range(6)]
        initial = sequence[0]["initial"]
        terms = [[value] for value in initial]
        for row in sequence:
            assert row["initial"] == initial
            for values, incoming in zip(terms, row["incoming"], strict=True):
                values.append(incoming)
            expected = [math.fsum(values) for values in terms]
            outputs = [row["host"]]
            if row["device"] == "Cuda":
                assert row["cuda"] is not None
                outputs.append(row["cuda"])
            else:
                assert row["cuda"] is None
            for actual in outputs:
                assert len(actual) == len(expected) == 4
                for value, reference in zip(actual, expected, strict=True):
                    error = abs(value - reference)
                    assert math.isfinite(value) and error <= 2e-6 + 2e-6 * abs(reference), (case, row["step"], actual, expected)
                    max_error = max(max_error, error)
                comparisons += 1
    result = dict(passed=True, records=len(rows), comparisons=comparisons,
                  max_absolute_error=max_error, reference="independent FP64 math.fsum of represented F32 contributions")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
