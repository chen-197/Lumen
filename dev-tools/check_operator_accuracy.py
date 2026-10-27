"""Validate actual Rust outputs against independent PyTorch FP64/autograd references."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch

parser=argparse.ArgumentParser()
parser.add_argument('log',type=Path)
parser.add_argument('--cuda',action='store_true')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(1)
records=[]
for line in args.log.read_text(encoding='utf-8-sig').splitlines():
    if 'OPERATOR_AUDIT=' in line:records.append(json.loads(line.split('OPERATOR_AUDIT=',1)[1]))
expected={'reshape':72,'ce':192,'rms':36,'softmax':36,'binary':144,'rms_inference':36,'scaled_mul':36}
devices=['Cpu','Cuda'] if args.cuda else ['Cpu']
assert Counter(r['device'] for r in records)==Counter({d:552 for d in devices}), 'missing device records'
for device in devices:
    assert Counter(r['case'] for r in records if r['device']==device)==expected,'missing operator records'
results=[]
def tensor(values,shape=None):
    t=torch.tensor(values,dtype=torch.float64)
    return t if shape is None else t.reshape(shape)
def check(label,got,ref,atol=3e-6,rtol=3e-5):
    a=tensor(got).flatten();b=ref.detach().flatten()
    assert a.numel()==b.numel(),label
    assert torch.isfinite(a).all() and torch.isfinite(b).all(),(label,'nonfinite')
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol,msg=label)
    error=(a-b).abs()
    budget=atol+rtol*b.abs()
    fraction=torch.where(budget>0,error/budget,torch.zeros_like(error))
    results.append({'label':label,'max_absolute_error':error.max().item() if a.numel() else 0,
                    'max_reference_absolute_value':b.abs().max().item() if b.numel() else 0,
                    'max_error_budget_fraction':fraction.max().item() if fraction.numel() else 0,
                    'atol':atol,'rtol':rtol})
for i,r in enumerate(records):
    label=f'{i}:{r["device"]}:{r["case"]}'
    if r['case']=='reshape':
        x=tensor(r['x'],(2,3)).requires_grad_();y=x.reshape(r['shape'])
        check(label+':forward',r['y'],y,1e-6,0)
        if r['training']:
            g=torch.arange(6,dtype=torch.float64).reshape(y.shape)*.25-.5
            y.backward(g);check(label+':backward',r['dx'],x.grad,0,0)
    elif r['case']=='ce':
        x=tensor(r['x'],(2,3)).requires_grad_();t=tensor(r['target'],(2,3))
        loss=-(t*torch.log_softmax(x,dim=-1)).sum()/2
        check(label+':loss',r['loss'],loss,2e-6,2e-6)
        if r['training']:
            loss.backward();check(label+':dx',r['dx'],x.grad,3e-6,3e-6)
    elif r['case'] in ['rms','rms_inference']:
        x=tensor(r['x'],(2,2,3)).requires_grad_();w=tensor(r['weight']).requires_grad_()
        # eps is an F32 API value, even when the independent arithmetic uses FP64.
        eps=torch.tensor(1e-5,dtype=torch.float32).double()
        y=x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+eps)*w
        if r['case']=='rms':
            y.backward(tensor(r['g'],y.shape))
            check(label+':forward',r['y'],y);check(label+':dx',r['dx'],x.grad);check(label+':dw',r['dw'],w.grad)
        else:
            value=y.float()
            if r['dtype']=='I8':
                scale=value.detach().abs().max()/127
                value=value/scale
                value=(value.sign()*torch.floor(value.abs()+.5)).clamp(-127,127)*scale
            else:
                value=value.to({'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16}[r['dtype']]).float()
            check(label+':rounded_forward',r['y'],value.double())
    elif r['case']=='softmax':
        x=tensor(r['x'],(2,2,3)).requires_grad_();y=torch.softmax(x,dim=-1)
        y.backward(tensor(r['g'],y.shape))
        check(label+':forward',r['y'],y);check(label+':dx',r['dx'],x.grad)
    elif r['case']=='scaled_mul':
        a=tensor(r['a'],(2,3)).requires_grad_();b=tensor(r['b'],(2,3)).requires_grad_()
        y=a*b;y.backward(torch.ones_like(y))
        check(label+':forward',r['y'],y,r['magnitude']**2*1e-7,1e-5)
        check(label+':da',r['da'],a.grad,r['magnitude']*1e-7,1e-5)
        check(label+':db',r['db'],b.grad,r['magnitude']*1e-7,1e-5)
    else:
        a=tensor(r['a'],(2,1,3)).requires_grad_();b=tensor(r['b'],(1,4,1)).requires_grad_()
        y={'add':torch.add,'sub':torch.sub,'mul':torch.mul}[r['op']](a,b)
        y.backward(tensor(r['g'],y.shape))
        check(label+':forward',r['y'],y);check(label+':da',r['da'],a.grad);check(label+':db',r['db'],b.grad)
assert len(results)==1152*len(devices),len(results)
summary={'torch_version':torch.__version__,'devices':devices,'records':len(records),'checks':len(results),
         'max_absolute_error':max(r['max_absolute_error'] for r in results),
         'max_error_budget_fraction':max(r['max_error_budget_fraction'] for r in results),'results':results}
args.output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
