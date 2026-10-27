"""Run a bounded CPU performance matrix, serially, using already-built binaries.

This records timings; correctness gates and real-model inference run separately.
Use --extended for scheduling/kernel investigations, not routine release checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def matrix(threads, extended):
    cases = []

    def add(tag, binary, *arguments, counts=(5, 7, 5)):
        warmup, samples, iters = counts
        cases.append(dict(tag=tag, binary=binary, arguments=list(map(str, [
            *arguments, '--threads', threads, '--warmup', warmup,
            '--samples', samples, '--iters', iters,
        ]))))

    budgets = sorted({1, min(2, threads), min(4, threads), threads}) if extended else sorted({1, threads})
    sweeps = [budgets, list(reversed(budgets))] if extended else [budgets]
    for cycle, order in enumerate(sweeps):
        for tasks in order:
            for kind, binary in [('forward', 'generic_prefill_bench'), ('backward', 'cpu_backward_bench')]:
                add(f'{kind}-tasks-{tasks}-cycle-{cycle}', binary, '--tasks', tasks)
    for dtype in ['f32', 'f16', 'bf16', 'i8']:
        for kind in (['matmul', 'matvec', 'qkv', 'gate'] if extended else ['matmul', 'qkv']):
            add(f'projection-{dtype}-{kind}', 'cpu_projection_bench', '--kind', kind,
                '--dtype', dtype, '--m', 37 if kind == 'matmul' else 1, '--k', 513, '--n', 769)
        for kind in (['matvec', 'argmax', 'qkv', 'gate'] if extended else ['argmax']):
            native_input = 'bf16' if dtype == 'i8' else dtype
            for input_dtype in (sorted({'f32', native_input}) if extended else [native_input]):
                add(f'working-set-{dtype}-{input_dtype}-{kind}', 'cpu_decode_bench',
                    '--kind', kind, '--dtype', dtype, '--input-dtype', input_dtype,
                    '--k', 257, '--n', 515, '--kv-n', 129, '--sets', 3)
    for dtype in (['f32', 'f16', 'bf16', 'i8'] if extended else ['f32']):
        for kind in ['matmul', 'attention']:
            for requests in ([1, 2, 4] if extended else [1, 4]):
                for tasks in ([1, 'auto'] if extended else ['auto']):
                    add(f'scaling-{dtype}-{kind}-requests-{requests}-tasks-{tasks}', 'cpu_scaling_bench',
                        '--kind', kind, '--dtype', dtype, '--m', 193 if kind == 'matmul' else 129,
                        '--k', 512 if kind == 'matmul' else 128, '--n', 513,
                        '--heads', 4, '--kv-heads', 2, '--requests', requests, '--tasks', tasks,
                        counts=(2, 5, 3))
    # This program manages warmup internally and has no --threads/--warmup switches.
    cases.append(dict(tag='kernel-tail-shapes', binary='kernel_bench', arguments=[
        '--hidden', '513', '--inter', '769', '--vocab', '1025', '--samples', '5', '--iters', '256',
    ]))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=positive, required=True)
    parser.add_argument('--binary-directory', type=Path, help='already-built release executables and runtime libraries')
    parser.add_argument('--output-directory', type=Path, help='new evidence directory')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--select', action='append', help='run matching tags only; repeat to select multiple substrings')
    parser.add_argument('--plan', action='store_true', help='print the matrix without building or running anything')
    args = parser.parse_args()
    cases = matrix(args.threads, args.extended)
    if args.select:
        cases = [case for case in cases if any(part in case['tag'] for part in args.select)]
    if not cases:
        parser.error('selection matches no benchmarks')
    plan = dict(suite='extended' if args.extended else 'representative', threads=args.threads,
                processes=len(cases), serial=True, cases=cases)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not args.binary_directory or not args.output_directory:
        parser.error('--binary-directory and --output-directory are required unless --plan is used')
    binary_directory = args.binary_directory.resolve(strict=True)
    suffix = '.exe' if os.name == 'nt' else ''
    binaries = {case['binary']: binary_directory / (case['binary'] + suffix) for case in cases}
    for path in binaries.values():
        if not path.is_file():
            parser.error(f'missing benchmark executable: {path}')
    # Hash before timing; do not build, analyze references or compress during runs.
    plan['binary_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env.update(LUMEN_CUDA='0', RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    env.pop('LUMEN_BENCH_DUMP', None)
    env['PATH'] = str(binary_directory) + os.pathsep + env.get('PATH', '')
    index = []
    for case in cases:
        print('START', case['tag'], flush=True)
        command = [str(binaries[case['binary']]), *case['arguments']]
        filename = case['tag'] + '.jsonl'
        started = time.monotonic()
        with (output / filename).open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=2400, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = -1
        index.append(dict(tag=case['tag'], command=command, file=filename,
                          returncode=code, seconds=time.monotonic() - started))
        (output / 'run-index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
        if code != 0:
            raise RuntimeError(f"benchmark failed: {case['tag']}; evidence: {output / filename}")
        print('DONE', case['tag'], flush=True)
    print(f'Completed {len(index)} benchmark processes. Evidence: {output}', flush=True)


if __name__ == '__main__':
    main()
