"""Validate complete long-test coverage and independently replay F32 training/model logits."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model-directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    devices = ['Cpu', 'Cuda'] if args.cuda else ['Cpu']
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    index = json.loads((root/'run-index.json').read_text(encoding='utf-8-sig'))
    expected_tests = {f'endurance_tests::{d.lower()}_{name}' for d in devices
                      for name in ['long_training', 'long_kv_routes', 'graph_lifecycle', 'real_weight_long_kv']}
    assert len(index) == len(expected_tests) and {r['test'] for r in index} == expected_tests
    rows = []
    for run in index:
        if 'real_weight' in run['test'] and not args.model_directory:
            assert run['status'] == 'skipped'
            continue
        assert run['status'] == 'passed' and run['returncode'] == 0, run
        text = (root/run['log']).read_text(encoding='utf-8-sig')
        assert 'test result: ok. 1 passed; 0 failed;' in text, run
        for line in text.splitlines():
            if 'ENDURANCE_AUDIT=' in line:
                rows.append(json.loads(line.split('ENDURANCE_AUDIT=', 1)[1]))
    training = [r for r in rows if r['kind'] == 'training']
    expected_curves = {(d, t, opt, layout) for d in devices for t in ['F32', 'F16', 'BF16', 'I8']
                       for opt in ['sgd', 'momentum', 'adam'] for layout in range(3)}
    assert len(training) == len(expected_curves)
    assert {(r['device'], r['dtype'], r['optimizer'], r['layout']) for r in training} == expected_curves
    references, comparisons = [], []
    input_data = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
    target = torch.tensor([[.35, -.125], [-.15, .225], [.15, -.175], [.25, -.025]])
    for row in training:
        losses = row['losses']
        assert row['steps'] == 1000 and len(losses) == 1001
        assert all(math.isfinite(x) for x in losses)
        assert losses[-1] < losses[0]*.1 and sum(losses[-100:]) < sum(losses[:100])
        assert sum((i-500)*v for i, v in enumerate(losses)) < 0
        if row['dtype'] != 'F32':
            continue
        weight = torch.tensor(row['initial'][0]).reshape(2, 3).requires_grad_()
        bias = torch.tensor(row['initial'][1]).requires_grad_()
        if row['optimizer'] == 'adam':
            opt = torch.optim.Adam([weight, bias], lr=.02, foreach=False)
        else:
            opt = torch.optim.SGD([weight, bias], lr=.03,
                                  momentum=.9 if row['optimizer'] == 'momentum' else 0., foreach=False)
        ref_losses = []
        for step in range(1001):
            opt.zero_grad()
            loss = ((input_data @ weight.T + bias - target)**2).mean()
            ref_losses.append(loss.item())
            if step < 1000:
                loss.backward()
                opt.step()
        refs = {'device': row['device'], 'optimizer': row['optimizer'], 'layout': row['layout'],
                'losses': ref_losses, 'final': [weight.detach().flatten().tolist(), bias.detach().tolist()]}
        references.append(refs)
        torch.testing.assert_close(torch.tensor(losses), torch.tensor(ref_losses), atol=3e-7, rtol=3e-4)
        final_error = 0.
        for actual, expected in zip(row['final'], refs['final'], strict=True):
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=4e-6, rtol=3e-4)
            final_error = max(final_error, max(abs(a-b) for a, b in zip(actual, expected, strict=True)))
        comparisons.append({'device': row['device'], 'optimizer': row['optimizer'], 'layout': row['layout'],
                            'loss_max_absolute_error': max(abs(a-b) for a,b in zip(losses, ref_losses, strict=True)),
                            'parameter_max_absolute_error': final_error})
    for device in devices:
        life = [r for r in rows if r['kind'] == 'lifecycle' and r['device'] == device]
        assert len(life) == 1 and life[0]['all_released']
        assert life[0]['training_cycles'] == 256 and life[0]['model_cycles'] == 16
        assert life[0]['tensor_probes'] > 0
        if device == 'Cuda': assert life[0]['buffer_probes'] > 0
        cache = [r for r in rows if r['kind'] == 'real_weight_cache' and not r['real_weights'] and r['device'] == device]
        assert len(cache) == 4 and {r['dtype'] for r in cache} == {'F32', 'F16', 'BF16', 'I8'}
        for row in cache:
            assert row['batch'] == 2 and row['max_seq_len'] == 513
            assert [r['length'] for r in row['comparisons']] == [127,255,508,509,510,511,512,513]
            assert row['reset_max_absolute_error'] == 0
    real_refs = []
    if args.model_directory:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_directory, local_files_only=True,
                                                    dtype=torch.float32, attn_implementation='eager').eval()
        real = [r for r in rows if r['kind'] == 'real_weight_cache' and r['real_weights']]
        assert len(real) == len(devices) and {r['device'] for r in real} == set(devices)
        for row in real:
            assert row['dtype'] == 'F32' and row['batch'] == 1 and row['reset_max_absolute_error'] == 0
            assert len(row['input_ids']) == 513 and len(row['comparisons']) == 8
            device = row['device'].lower()
            model.to(device)
            for sample in row['comparisons']:
                length = sample['length']
                ids = torch.tensor([row['input_ids'][:length]], device=device)
                with torch.inference_mode():
                    result = model(input_ids=ids, use_cache=False)
                    expected = result.logits[0,-1].float().cpu()
                actual = torch.tensor(sample['full_logits'])
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                relative = (torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)).item()
                assert relative < 5e-5 and actual.argmax().item() == expected.argmax().item()
                real_refs.append({'device':row['device'], 'length':length,
                                  'max_absolute_error':(actual-expected).abs().max().item(),
                                  'relative_l2':relative,'logits':expected.tolist()})
            del result
            model.to('cpu')
            if args.cuda: torch.cuda.empty_cache()
    (root/'torch-reference.json').write_text(json.dumps({'training':references,'real_model':real_refs}), encoding='utf-8')
    report = {'passed':True, 'torch':torch.__version__, 'training_curves':len(training),
              'updates_per_curve':1000,'fp64_loss_checks':sum(r['sampled_fp64_loss_checks'] for r in training),
              'fp64_gradient_checks':sum(r['sampled_fp64_gradient_checks'] for r in training),
              'f32_torch_curves':len(comparisons),'training_comparisons':comparisons,
              'lifecycle':[r for r in rows if r['kind']=='lifecycle'],
              'compact_cache_cases':4*len(devices),'real_weight_reference_rows':len(real_refs),
              'real_weights_skipped':not bool(args.model_directory),
              'log_sha256':{r['log']:hashlib.sha256((root/r['log']).read_bytes()).hexdigest() for r in index if r['status']=='passed'}}
    (root/'reference.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['training_comparisons','log_sha256']},indent=2))


if __name__ == '__main__':
    main()
