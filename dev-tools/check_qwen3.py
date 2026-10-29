"""Independent PyTorch FP64 forward/VJP reference for QWEN3_AUDIT records.

Usage: python dev-tools/check_qwen3.py TEST_LOG --output COMPARISON.json
"""
import argparse
import json
from pathlib import Path
import torch


def compare(record):
    torch.set_num_threads(1)
    parameters = {name: torch.tensor(row['values'], dtype=torch.float64).reshape(row['shape']).requires_grad_()
                  for name, row in record['parameters'].items()}
    h, hk, d = (record[name] for name in ['heads', 'kv_heads', 'head_dim'])
    def linear(x, name):
        return x @ parameters[name].T
    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + record['eps']) * parameters[name]
    def rope(x):
        length = x.shape[-2]
        inverse = record['rope_theta'] ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
        angles = torch.arange(length, dtype=torch.float64)[:, None] * inverse[None, :]
        angles = torch.cat([angles, angles], -1)[None, None]
        rotated = torch.cat([-x[..., d//2:], x[..., :d//2]], -1)
        return x * angles.cos() + rotated * angles.sin()
    ids = torch.tensor([record['input_ids']])
    x = parameters['model.embed_tokens.weight'][ids]
    batch, length, hidden = x.shape
    for layer in range(record['layers']):
        prefix = f'model.layers.{layer}'
        normalized = norm(x, prefix + '.input_layernorm.weight')
        q = linear(normalized, prefix + '.self_attn.q_proj.weight').reshape(batch, length, h, d).transpose(1, 2)
        k = linear(normalized, prefix + '.self_attn.k_proj.weight').reshape(batch, length, hk, d).transpose(1, 2)
        v = linear(normalized, prefix + '.self_attn.v_proj.weight').reshape(batch, length, hk, d).transpose(1, 2)
        q = rope(norm(q, prefix + '.self_attn.q_norm.weight'))
        k = rope(norm(k, prefix + '.self_attn.k_norm.weight')).repeat_interleave(h//hk, dim=1)
        v = v.repeat_interleave(h//hk, dim=1)
        score = q @ k.transpose(-1, -2) / d**0.5
        score = score.masked_fill(torch.ones(length, length, dtype=torch.bool).triu(1), -torch.inf)
        context = (score.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, h*d)
        x = x + linear(context, prefix + '.self_attn.o_proj.weight')
        normalized = norm(x, prefix + '.post_attention_layernorm.weight')
        gate = torch.nn.functional.silu(linear(normalized, prefix + '.mlp.gate_proj.weight'))
        x = x + linear(gate * linear(normalized, prefix + '.mlp.up_proj.weight'), prefix + '.mlp.down_proj.weight')
    logits = norm(x, 'model.norm.weight') @ parameters['model.embed_tokens.weight'].T
    logits.sum().backward()
    errors = {}
    def check(name, actual, expected):
        actual = torch.tensor(actual, dtype=torch.float64).reshape(expected.shape)
        difference = (actual - expected).abs()
        bound = 2e-5 + 2e-4 * expected.abs()
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
        assert (difference <= bound).all(), (record['device'], name, difference.max().item())
        errors[name] = difference.max().item()
    check('logits', record['logits'], logits.detach())
    for name, row in record['parameters'].items():
        check(name, row['gradient'], parameters[name].grad)
    return dict(device=record['device'], passed=True, gradient_arrays=len(parameters),
                logit_values=logits.numel(), max_absolute_error=max(errors.values()), errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line.split('QWEN3_AUDIT=', 1)[1]) for line in args.log.read_text(encoding='utf-8-sig').splitlines()
               if 'QWEN3_AUDIT=' in line]
    assert records, 'No QWEN3_AUDIT records found; set LUMEN_OPERATOR_AUDIT=1'
    results = [compare(record) for record in records]
    result = dict(passed=True, reference='Independent PyTorch FP64 dense Qwen3 forward and full parameter VJP', checks=results)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
