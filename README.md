# PyTorch-based implementations of different normalization layers

This repository provides purely PyTorch-based implementations of several normalization layers, including BatchNorm, GroupNorm, InstanceNorm, and LayerNorm. They are carefully implemented to match the official PyTorch implementations.

## Install

```bash
# install via git
python -m pip install git+https://github.com/Emrys365/torch_normalization

# install from source
git clone git@github.com:Emrys365/torch_normalization.git
cd torch_normalization
python -m pip install -e .
```

## Usage

### BatchNorm for input of different dimensions

```python
import torch
from torch_normalization import BatchNorm

device = "cpu"
dtype = torch.float32
num_features = 128
x1 = torch.randn(2, num_features, 32, dtype=dtype, device=device)
x2 = torch.randn(2, num_features, 32, 48, dtype=dtype, device=device)
x3 = torch.randn(2, num_features, 32, 48, 3, dtype=dtype, device=device)

for x in (x1, x2, x3):
    dims = x.ndim - 2
    opt = dict(eps=1e-05, affine=True, track_running_stats=True, dtype=dtype, device=device)
    module = getattr(torch.nn, f"BatchNorm{max(dims, 1)}d")
    bn_th = module(num_features, **opt)
    bn = BatchNorm(num_features, dims=dims, **opt)
    torch.nn.init.normal_(bn_th.weight)
    torch.nn.init.uniform_(bn_th.bias)
    bn.weight = bn_th.weight
    bn.bias = bn_th.bias

    out = bn(x)
    out_th = bn_th(x)
    torch.testing.assert_close(out, out_th)
```

### InstanceNorm for input of different dimensions

```python
import torch
from torch_normalization import InstanceNorm

device = "cpu"
dtype = torch.float32
num_features = 128
x1 = torch.randn(2, num_features, 32, dtype=dtype, device=device)
x2 = torch.randn(2, num_features, 32, 48, dtype=dtype, device=device)
x3 = torch.randn(2, num_features, 32, 48, 3, dtype=dtype, device=device)

for x in (x1, x2, x3):
    dims = x.ndim - 2
    opt = dict(eps=1e-05, affine=True, track_running_stats=True, dtype=dtype, device=device)
    module = getattr(torch.nn, f"InstanceNorm{max(dims, 1)}d")
    isn_th = module(num_features, **opt)
    isn = InstanceNorm(num_features, dims=dims, **opt)
    torch.nn.init.normal_(isn_th.weight)
    torch.nn.init.uniform_(isn_th.bias)
    isn.weight = isn_th.weight
    isn.bias = isn_th.bias

    out = isn(x)
    out_th = isn_th(x)
    torch.testing.assert_close(out, out_th)
```

### GroupNorm for input of different dimensions

```python
import torch
from torch_normalization import GroupNorm

device = "cpu"
dtype = torch.float32
num_channels = 128
num_groups = 8
x1 = torch.randn(2, num_channels, 32, dtype=dtype, device=device)
x2 = torch.randn(2, num_channels, 32, 48, dtype=dtype, device=device)
x3 = torch.randn(2, num_channels, 32, 48, 3, dtype=dtype, device=device)

for x in (x1, x2, x3):
    opt = dict(eps=1e-05, affine=True, dtype=dtype, device=device)
    gn_th = torch.nn.GroupNorm(num_groups, num_channels, **opt)
    gn = GroupNorm(num_groups, num_channels, **opt)
    torch.nn.init.normal_(gn_th.weight)
    torch.nn.init.uniform_(gn.bias)
    gn.weight = gn_th.weight
    gn.bias = gn_th.bias

    out = gn(x)
    out_th = gn_th(x)
    torch.testing.assert_close(out, out_th)
```

### LayerNorm for input of different dimensions

```python
import torch
from torch_normalization import LayerNorm

device = "cpu"
dtype = torch.float32
shape1 = 16
shape2 = (32, 32)
shape3 = (16, 32, 32)
x1 = torch.randn(2, shape1, dtype=dtype, device=device)
x2 = torch.randn(2, *shape2, dtype=dtype, device=device)
x3 = torch.randn(2, *shape3, dtype=dtype, device=device)

for x, shape in ((x1, shape1), (x2, shape2), (x3, shape3)):
    opt = dict(eps=1e-05, elementwise_affine=True, bias=True, dtype=dtype, device=device)
    ln_th = torch.nn.LayerNorm(shape, **opt)
    ln = LayerNorm(shape, **opt)
    torch.nn.init.normal_(ln_th.weight)
    torch.nn.init.uniform_(ln.bias)
    ln.weight = ln_th.weight
    ln.bias = ln_th.bias

    out = ln(x)
    out_th = ln_th(x)
    torch.testing.assert_close(out, out_th)
```


## Test implementations

```bash
python -m pytest tests/
```