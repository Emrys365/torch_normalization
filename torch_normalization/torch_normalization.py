import math
from typing import List, Optional, Union

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """Applies Batch Normalization over a i-D input (i >= 2).

    Reference:
        Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift <https://arxiv.org/abs/1502.03167>

    Args:
        num_features (int): number of features or channels `C` of the input
        eps (float): a value added to the denominator for numerical stability.
        momentum (float): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average).
        affine (bool): a boolean value that when set to ``True``, this module has
            learnable affine parameters.
        track_running_stats (bool): a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes.
        dims (int): number of dimensions to apply BatchNorm (0 for 0d, 1 for 1d, etc.)
            0d: (B, C)
            1d: (B, C, L)
            2d: (B, C, H, W)
            3d: (B, C, D, H, W)

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        dims=0,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        assert dims >= 0, dims
        self.dims = dims

        opt = {"device": device, "dtype": dtype}
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features, **opt))
            self.bias = nn.Parameter(torch.zeros(num_features, **opt))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, **opt))
            self.register_buffer("running_var", torch.ones(num_features, **opt))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def extra_repr(self) -> str:
        return f"{self.dims}d"

    def _prepare_stats(self):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        # Mini-batch stats are used when `bn_training` is ``True``,
        # otherwise the buffer is used.
        bn_training = self.training or (not self.track_running_stats)
        return exponential_average_factor, bn_training

    def _get_mean_var(self, x, dims):
        exponential_average_factor, bn_training = self._prepare_stats()
        if bn_training:
            mean = x.mean(dim=dims)
            # use biased var in training
            var = x.var(dim=dims, unbiased=False)
            n = math.prod([x.size(d) for d in dims])
            var_unbiased = var * n / (n - 1)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                    )
                    # update running_var with unbiased var
                    self.running_var = (
                        exponential_average_factor * var_unbiased
                        + (1 - exponential_average_factor) * self.running_var
                    )
        else:
            mean = self.running_mean
            var = self.running_var

        shape = list(x.shape)
        for i in dims:
            shape[i] = 1
        return mean.view(shape), var.view(shape)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): i-D input tensor (B, C, ...), i >= 2

        Returns:
            ret (torch.Tensor): normalized tensor of the same shape as input
        """
        assert x.ndim == self.dims + 2, (x.ndim, self.dims)
        shape = (1, self.num_features) + (1,) * (x.ndim - 2)
        # Computing stats in the batch dim and the last (self.dims - 1) dims
        dim = (0,) + tuple(i + 2 for i in range(self.dims))

        mean, var = self._get_mean_var(x, dim)
        xbar = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            return xbar * self.weight.view(shape) + self.bias.view(shape)
        return xbar


class GroupNorm(nn.Module):
    """Applies Group Normalization over a mini-batch of inputs.

    Reference:
        Group Normalization <https://arxiv.org/abs/1803.08494>

    Args:
        num_groups (int): number of groups to separate the channels into.
            The input channels are separated into ``num_groups`` groups, each containing
            ``num_channels / num_groups`` channels.
        num_channels (int): number of channels expected in input.
            Must be divisible by ``num_groups``.
        eps (float): a value added to the denominator for numerical stability.
        affine (bool): a boolean value that when set to ``True``, this module has
            learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases).

    This layer uses statistics computed from input data in both training and evaluation
    modes.

    Note:
        GroupNorm is equivalent to LayerNorm (assuming its ``normalized_shape`` is int
            and input is 2-dim) when ``num_groups`` is 1.
        GroupNorm is equivalent to InstanceNorm (assuming its ``track_running_stats``
            is False) when ``num_groups`` is ``num_channels``.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        opt = {"device": device, "dtype": dtype}
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels, **opt))
            self.bias = nn.Parameter(torch.zeros(num_channels, **opt))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): input tensor (B, C, ...)

        Returns:
            ret (torch.Tensor): normalized tensor (B, C, ...)
        """
        assert x.ndim >= 2, x.ndim
        assert x.size(1) == self.num_channels, (x.shape, self.num_channels)
        group_size = self.num_channels // self.num_groups
        shape_org = list(x.shape)
        shape_group = [shape_org[0], self.num_groups, group_size, *shape_org[2:]]
        x = x.reshape(shape_group)

        # Computing stats in the instance dims of each group
        shape = (1, self.num_groups, group_size) + (1,) * (x.ndim - 3)
        dim = tuple(i for i in range(2, x.ndim))

        var = x.var(dim=dim, keepdim=True, unbiased=False)
        mean = x.mean(dim=dim, keepdim=True)
        xbar = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            xbar = xbar * self.weight.view(shape) + self.bias.view(shape)
        return xbar.reshape(shape_org)


class InstanceNorm(nn.Module):
    """Applies Instance Normalization over a i-D input (i >= 2).

    Reference:
        Instance Normalization: The Missing Ingredient for Fast Stylization
        <https://arxiv.org/abs/1607.08022>

    Args:
        num_features (int): number of features or channels `C` of the input
        eps (float): a value added to the denominator for numerical stability.
        momentum (float): a value used for the running_mean and running_var computation.
        affine (bool): a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for
            batch normalization.
        track_running_stats (bool): a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes.
        dims (int): number of dimensions to apply InstanceNorm (1 for 1d, 2 for 2d, ...)
            1d: (B, C, L)
            2d: (B, C, H, W)
            3d: (B, C, D, H, W)

    By default, this layer uses instance statistics computed from input data in both
    training and evaluation modes.

    Note:
        InstanceNorm1d (dims=1) is very similar to LayerNorm when the input is 2d
            (B, C), but InstanceNorm1d usually doesn't apply affine transform.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
        dims=1,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        assert dims >= 1, dims
        self.dims = dims

        opt = {"device": device, "dtype": dtype}
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features, **opt))
            self.bias = nn.Parameter(torch.zeros(num_features, **opt))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, **opt))
            self.register_buffer("running_var", torch.ones(num_features, **opt))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def extra_repr(self) -> str:
        return f"{self.dims}d"

    def _prepare_stats(self):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        # Mini-batch stats are used when `bn_training` is ``True``,
        # otherwise the buffer is used.
        bn_training = self.training or (not self.track_running_stats)
        return exponential_average_factor, bn_training

    def _get_mean_var(self, x, dims):
        exponential_average_factor, bn_training = self._prepare_stats()
        if bn_training:
            mean = x.mean(dim=dims)
            # use biased var in training
            var = x.var(dim=dims, unbiased=False)
            n = math.prod([x.size(d) for d in dims])
            var_unbiased = var * n / (n - 1)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                    )
                    # update running_var with unbiased var
                    self.running_var = (
                        exponential_average_factor * var_unbiased
                        + (1 - exponential_average_factor) * self.running_var
                    )
        else:
            mean = self.running_mean
            var = self.running_var

        shape = list(x.shape)
        for i in dims:
            shape[i] = 1
        if not bn_training and x.size(0) != mean.size(0):
            shape[0] = 1
        return mean.view(shape), var.view(shape)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor, has_batch_dim: bool = True) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): i-D input tensor ([B,] C, ...), i >= 1
            has_batch_dim (bool): whether the input tensor has a batch dimension

        Returns:
            ret (torch.Tensor): normalized tensor of the same shape as input
        """
        if has_batch_dim:
            assert x.ndim == self.dims + 2, (x.ndim, self.dims)
            shape = (1, self.num_features) + (1,) * (x.ndim - 2)
        else:
            assert x.ndim == self.dims + 1, (x.ndim, self.dims)
            shape = (self.num_features,) + (1,) * (x.ndim - 1)
        # Computing stats in the instance dims
        dim = tuple(-i for i in range(1, self.dims + 1))

        mean, var = self._get_mean_var(x, dim)
        xbar = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            return xbar * self.weight.view(shape) + self.bias.view(shape)
        return xbar


class LayerNorm(nn.Module):
    """Applies Layer Normalization over a mini-batch of inputs.

    Reference:
        Layer Normalization <https://arxiv.org/abs/1607.06450>

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size [..., shape[0], shape[1], ..., shape[-1]].
            If a single integer is used, it is treated as a singleton list, and this
            module will normalize over the last dimension which is expected to be of
            that specific size.
        eps (float): a value added to the denominator for numerical stability.
        elementwise_affine: (bool) a boolean value that when set to ``True``, this
            module has learnable per-element affine parameters initialized to ones (for
            weights) and zeros (for biases).
        bias (bool): If set to ``False``, the layer will not learn an additive bias
            (only relevant if ``elementwise_affine`` is ``True``).

    This layer uses statistics computed from input data in both training and evaluation
    modes.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        opt = {"device": device, "dtype": dtype}
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, **opt))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape, **opt))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): input tensor (B, ..., *normalized_shape)

        Returns:
            ret (torch.Tensor): normalized tensor (B, ..., *normalized_shape)
        """
        assert x.ndim >= 2, x.ndim
        shape = (1,) * (x.ndim - len(self.normalized_shape)) + self.normalized_shape
        # Computing stats in the last len(self.normalized_shape) dims
        dim = tuple(-i for i in range(1, len(self.normalized_shape) + 1))

        var = x.var(dim=dim, keepdim=True, unbiased=False)
        mean = x.mean(dim=dim, keepdim=True)
        xbar = (x - mean) / (var + self.eps).sqrt()
        if not self.elementwise_affine:
            return xbar

        if self.bias is None:
            return xbar * self.weight.view(shape)
        return xbar * self.weight.view(shape) + self.bias.view(shape)
