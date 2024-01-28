import pytest

import torch
import torch.nn as nn
from torch_normalization import BatchNorm, GroupNorm, InstanceNorm, LayerNorm


@pytest.mark.parametrize(
    "input_shape, num_features, dims",
    [
        ((2, 3), 3, 0),
        ((2, 3, 4), 3, 1),
        ((2, 3, 4, 5), 3, 2),
        ((2, 3, 4, 5, 6), 3, 3),
    ],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("training", [True, False])
def test_BatchNorm_consistency(
    input_shape,
    num_features,
    eps,
    affine,
    dtype,
    dims,
    training,
):
    opt = dict(eps=eps, affine=affine, track_running_stats=False, dtype=dtype)
    module = getattr(nn, f"BatchNorm{max(dims, 1)}d")
    bn1 = module(num_features, **opt)
    bn2 = BatchNorm(num_features, dims=dims, **opt)
    if affine:
        nn.init.normal_(bn1.weight)
        nn.init.uniform_(bn1.bias)
    bn2.weight = bn1.weight
    bn2.bias = bn1.bias
    if not training:
        bn1.eval()
        bn2.eval()
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(bn1(x), bn2(x))


@pytest.mark.parametrize(
    "input_shape, num_features, dims",
    [
        ((2, 3), 3, 0),
        ((2, 3, 4), 3, 1),
        ((2, 3, 4, 5), 3, 2),
        ((2, 3, 4, 5, 6), 3, 3),
    ],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("momentum", [None, 0.1, 0.9])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("training", [True, False])
def test_BatchNorm_running_stats_consistency(
    input_shape,
    num_features,
    eps,
    momentum,
    affine,
    dtype,
    dims,
    training,
):
    opt = dict(
        eps=eps, momentum=momentum, affine=affine, track_running_stats=True, dtype=dtype
    )
    module = getattr(nn, f"BatchNorm{max(dims, 1)}d")
    bn1 = module(num_features, **opt)
    bn2 = BatchNorm(num_features, dims=dims, **opt)
    if affine:
        nn.init.normal_(bn1.weight)
        nn.init.uniform_(bn1.bias)
    nn.init.uniform_(bn1.running_mean)
    nn.init.uniform_(bn1.running_var)
    bn1.num_batches_tracked.add_(5)
    bn2.weight = bn1.weight
    bn2.bias = bn1.bias
    bn2.running_mean = bn1.running_mean
    bn2.running_var = bn1.running_var
    bn2.num_batches_tracked = bn1.num_batches_tracked
    if not training:
        bn1.eval()
        bn2.eval()
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(bn1(x), bn2(x))


@pytest.mark.parametrize(
    "input_shape, num_features, dims, has_batch_dim",
    [
        ((2, 3, 4), 3, 1, True),
        ((3, 4), 3, 1, False),
        ((2, 3, 4, 5), 3, 2, True),
        ((3, 4, 5), 3, 2, False),
        ((2, 3, 4, 5, 6), 3, 3, True),
        ((3, 4, 5, 6), 3, 3, False),
    ],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("training", [True, False])
def test_InstanceNorm_consistency(
    input_shape,
    num_features,
    eps,
    affine,
    dtype,
    dims,
    has_batch_dim,
    training,
):
    opt = dict(eps=eps, affine=affine, track_running_stats=False, dtype=dtype)
    module = getattr(nn, f"InstanceNorm{dims}d")
    in1 = module(num_features, **opt)
    in2 = InstanceNorm(num_features, dims=dims, **opt)
    if affine:
        nn.init.normal_(in1.weight)
        nn.init.uniform_(in1.bias)
    in2.weight = in1.weight
    in2.bias = in1.bias
    if not training:
        in1.eval()
        in2.eval()
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(in1(x), in2(x, has_batch_dim=has_batch_dim))


@pytest.mark.parametrize(
    "input_shape, num_features, dims, has_batch_dim",
    [
        ((2, 3, 4), 3, 1, True),
        ((3, 4), 3, 1, False),
        ((2, 3, 4, 5), 3, 2, True),
        ((3, 4, 5), 3, 2, False),
        ((2, 3, 4, 5, 6), 3, 3, True),
        ((3, 4, 5, 6), 3, 3, False),
    ],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("momentum", [0.1, 0.9])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("training", [True, False])
def test_InstanceNorm_running_stats_consistency(
    input_shape,
    num_features,
    eps,
    momentum,
    affine,
    dtype,
    dims,
    has_batch_dim,
    training,
):
    opt = dict(
        eps=eps, momentum=momentum, affine=affine, track_running_stats=True, dtype=dtype
    )
    module = getattr(nn, f"InstanceNorm{dims}d")
    in1 = module(num_features, **opt)
    in2 = InstanceNorm(num_features, dims=dims, **opt)
    if affine:
        nn.init.normal_(in1.weight)
        nn.init.uniform_(in1.bias)
    nn.init.uniform_(in1.running_mean)
    nn.init.uniform_(in1.running_var)
    in1.num_batches_tracked.add_(5)
    in2.weight = in1.weight
    in2.bias = in1.bias
    in2.running_mean = in1.running_mean
    in2.running_var = in1.running_var
    in2.num_batches_tracked = in1.num_batches_tracked
    if not training:
        in1.eval()
        in2.eval()
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(in1(x), in2(x, has_batch_dim=has_batch_dim))


@pytest.mark.parametrize(
    "input_shape, num_groups, num_channels",
    [
        ((2, 6), 1, 6),
        ((2, 6), 2, 6),
        ((2, 6), 3, 6),
        ((2, 6), 6, 6),
        ((2, 6, 4), 3, 6),
        ((2, 6, 4, 5), 3, 6),
    ],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
def test_GroupNorm_consistency(
    input_shape, num_groups, num_channels, eps, affine, dtype
):
    gn1 = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, dtype=dtype)
    gn2 = GroupNorm(num_groups, num_channels, eps=eps, affine=affine, dtype=dtype)
    if affine:
        nn.init.normal_(gn1.weight)
        nn.init.uniform_(gn1.bias)
    gn2.weight = gn1.weight
    gn2.bias = gn1.bias
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(gn1(x), gn2(x), rtol=1e-06, atol=5e-04)


@pytest.mark.parametrize(
    "input_shape, normalized_shape",
    [
        ((2, 6), 6),
        ((2, 6), (6,)),
        ((2, 6, 3), 3),
        ((2, 6, 3), (6, 3)),
        ((2, 6, 3, 4), 4),
        ((2, 6, 3, 4), (3, 4)),
        ((2, 6, 3, 4), (6, 3, 4)),
    ],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
def test_LayerNorm_consistency(
    input_shape, normalized_shape, eps, elementwise_affine, bias, dtype
):
    opt = dict(eps=eps, elementwise_affine=elementwise_affine, bias=bias, dtype=dtype)
    ln1 = nn.LayerNorm(normalized_shape, **opt)
    ln2 = LayerNorm(normalized_shape, **opt)
    if elementwise_affine:
        nn.init.normal_(ln1.weight)
        if bias:
            nn.init.uniform_(ln1.bias)
    ln2.weight = ln1.weight
    ln2.bias = ln1.bias
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(ln1(x), ln2(x))


@pytest.mark.parametrize(
    "input_shape, normalized_shape",
    [((2, 6), 6), ((2, 10), 10)],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
def test_LayerNorm_GroupNorm_consistency(
    input_shape, normalized_shape, eps, affine, dtype
):
    opt = dict(eps=eps, elementwise_affine=affine, bias=affine, dtype=dtype)
    ln1 = LayerNorm(normalized_shape, **opt)
    gn2 = GroupNorm(1, normalized_shape, eps=eps, affine=affine, dtype=dtype)
    if affine:
        nn.init.normal_(ln1.weight)
        nn.init.uniform_(ln1.bias)
    gn2.weight = ln1.weight
    gn2.bias = ln1.bias
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(ln1(x), gn2(x))


@pytest.mark.parametrize(
    "input_shape, num_features",
    [((2, 3, 4), 3), ((2, 3, 4, 5), 3), ((2, 3, 4, 5, 6), 3)],
)
@pytest.mark.parametrize("eps", [1e-5, 10])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float64])
def test_InstanceNorm_GroupNorm_consistency(
    input_shape, num_features, eps, affine, dtype
):
    opt = dict(eps=eps, affine=affine, track_running_stats=False, dtype=dtype)
    in1 = InstanceNorm(num_features, dims=len(input_shape) - 2, **opt)
    gn2 = GroupNorm(num_features, num_features, eps=eps, affine=affine, dtype=dtype)
    if affine:
        nn.init.normal_(in1.weight)
        nn.init.uniform_(in1.bias)
    gn2.weight = in1.weight
    gn2.bias = in1.bias
    x = torch.randn(input_shape, dtype=dtype)
    torch.testing.assert_close(in1(x), gn2(x))
