from abc import abstractmethod
from copy import deepcopy
from functools import partial

from omegaconf import DictConfig
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.losses.pixelwise_loss import _reduction_modes
from src.losses.utils import reduce_loss


def is_list_of(seq, expected_type):
    return isinstance(seq, list) and all(isinstance(x, expected_type) for x in seq)


def print_log(msg, logger=None):
    print(msg)


def digit_version(version_str):
    return tuple(map(int, version_str.split(".")))


class DDPMLoss(nn.Module):
    """Base module for DDPM losses. We support loss weight rescale and log
    collection for DDPM models in this module.

    We support two kinds of loss rescale methods, which can be
    controlled by ``rescale_mode`` and ``rescale_cfg``:
    1. ``rescale_mode == 'constant'``: ``constant_rescale`` would be called,
        and ``rescale_cfg`` should be passed as ``dict(scale=SCALE)``,
        e.g., ``dict(scale=1.2)``. Then, all loss terms would be rescaled by
        multiply with ``SCALE``
    2. ``rescale_mode == timestep_weight``: ``timestep_weight_rescale`` would
        be called, and ``weight`` or ``sampler`` who contains attribute of
        weight must be passed. Then, loss at timestep `t` would be multiplied
        with `weight[t]`. We also support users further apply a constant
        rescale factor to all loss terms, e.g.
        ``rescale_cfg=dict(scale=SCALE)``. The overall rescale function for
        loss at timestep ``t`` can be formulated as
        `loss[t] := weight[t] * loss[t] * SCALE`. To be noted that, ``weight``
        or ``sampler.weight`` would be inplace modified in the outer code.
        e.g.,

        .. code-blocks:: python
            :linenos:

            # 1. define weight
            weight = torch.randn(10, )

            # 2. define loss function
            loss_fn = DDPMLoss(rescale_mode='timestep_weight', weight=weight)

            # 3 update weight
            # wrong usage: `weight` in `loss_fn` is not accessible from now
            # because you assign a new tensor to variable `weight`
            # weight = torch.randn(10, )

            # correct usage: update `weight` inplace
            weight[2] = 2

    If ``rescale_mode`` is not passed, ``rescale_cfg`` would be ignored, and
    all loss terms would not be rescaled.

    For loss log collection, we support users to pass a list of (or single)
    config by ``log_cfgs`` argument to define how they want to collect loss
    terms and show them in the log. Each log collection returns a dict which
    key and value are the name and value of collected loss terms. And the dict
    will be merged into  ``log_vars`` after the loss used for parameter
    optimization is calculated. The log updating process for the class which
    uses ddpm_loss can be referred to the following pseudo-code:

    .. code-block:: python
        :linenos:

        # 1. loss dict for parameter optimization
        losses_dict = {}

        # 2. calculate losses
        for loss_fn in self.ddpm_loss:
            losses_dict[loss_fn.loss_name()] = loss_fn(outputs_dict)

        # 3. init log_vars
        log_vars = OrderedDict()

        # 4. update log_vars with loss terms used for parameter optimization
        for loss_name, loss_value in losses_dict.items():
            log_vars[loss_name] = loss_value.mean()

        # 5. sum all loss terms used for backward
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # 6. update log_var with log collection functions
        for loss_fn in self.ddpm_loss:
            if hasattr(loss_fn, 'log_vars'):
                log_vars.update(loss_fn.log_vars)

    Each log configs must contain ``type`` keyword, and may contain ``prefix``
    and ``reduction`` keywords.

    ``type``: Use to get the corresponding collection function. Functions would
        be named as ``f'{type}_log_collect'``. In `DDPMLoss`, we only support
        ``type=='quartile'``, but users may define their log collection
        functions and use them in this way.
    ``prefix``: This keyword is set for avoiding the name of displayed loss
        terms being too long. The name of each loss term will set as
        ``'{prefix}_{log_coll_fn_spec_name}'``, where
        ``{log_coll_fn_spec_name}`` is name specific to the log collection
        function. If passed, it must start with ``'loss_'``. If not passed,
        ``'loss_'`` would be used.
    ``reduction``: Control the reduction method of the collected loss terms.

    We implement ``quartile_log_collection`` in this module. In detail, we
    divide total timesteps into four parts and collect the loss in the
    corresponding timestep intervals.

    To use those collection methods, users may pass ``log_cfgs`` as the
    following example:

    .. code-block:: python
        :linenos:

        log_cfgs = [
            dict(type='quartile', reduction=REUCTION, prefix_name=PREFIX),
            ...
        ]

    Args:
        rescale_mode (str, optional): Mode of the loss rescale method.
            Defaults to None.
        rescale_cfg (dict, optional): Config of the loss rescale method.
        log_cfgs (list[dict] | dict | optional): Configs to collect logs.
            Defaults to None.
        sampler (object): Weight sampler. Defaults to None.
        weight (torch.Tensor, optional): Weight used for rescale losses.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        loss_name (str, optional): Name of the loss item. Defaults to None.
    """

    def __init__(self, rescale_mode=None, rescale_cfg=None, log_cfgs=None, weight=None, sampler=None, reduction="mean", loss_name=None):
        super().__init__()

        if reduction not in _reduction_modes:
            raise ValueError(f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}")
        self.reduction = reduction
        self._loss_name = loss_name

        self.log_fn_list = []

        log_cfgs_ = deepcopy(log_cfgs)
        if log_cfgs_ is not None:
            if not isinstance(log_cfgs_, list):
                log_cfgs_ = [log_cfgs_]
            assert is_list_of(log_cfgs_, DictConfig)
            for log_cfg_ in log_cfgs_:
                log_type = log_cfg_.pop("type")
                log_collect_fn = f"{log_type}_log_collect"
                assert hasattr(self, log_collect_fn)
                log_collect_fn = getattr(self, log_collect_fn)

                log_cfg_.setdefault("prefix_name", "loss")
                assert log_cfg_["prefix_name"].startswith("loss")
                log_cfg_.setdefault("reduction", reduction)

                self.log_fn_list.append(partial(log_collect_fn, **log_cfg_))
        self.log_vars = dict()

        # handle rescale mode
        if not rescale_mode:
            self.rescale_fn = lambda loss, t: loss
        else:
            rescale_fn_name = f"{rescale_mode}_rescale"
            assert hasattr(self, rescale_fn_name)
            if rescale_mode == "timestep_weight":
                if sampler is not None and hasattr(sampler, "weight"):
                    weight = sampler.weight
                else:
                    assert weight is not None and isinstance(weight, torch.Tensor), "'weight' or a 'sampler' contains weight attribute is must be 'torch.Tensor' for 'timestep_weight' rescale_mode."

                print_log(f"Apply 'timestep_weight' rescale_mode for {self._loss_name}. Please make sure the passed weight can be updated by external functions.", "mmgen")

                rescale_cfg = dict(weight=weight)
            self.rescale_fn = partial(getattr(self, rescale_fn_name), **rescale_cfg)

    @staticmethod
    def constant_rescale(loss, timesteps, scale):
        """Rescale losses at all timesteps with a constant factor.

        Args:
            loss (torch.Tensor): Losses to rescale.
            timesteps (torch.Tensor): Timesteps of each loss items.
            scale (int): Rescale factor.

        Returns:
            torch.Tensor: Rescaled losses.
        """

        return loss * scale

    @staticmethod
    def timestep_weight_rescale(loss, timesteps, weight, scale=1):
        """Rescale losses corresponding to timestep.

        Args:
            loss (torch.Tensor): Losses to rescale.
            timesteps (torch.Tensor): Timesteps of each loss items.
            weight (torch.Tensor): Weight corresponding to each timestep.
            scale (int): Rescale factor.

        Returns:
            torch.Tensor: Rescaled losses.
        """

        return loss * weight[timesteps] * scale

    @torch.no_grad()
    def collect_log(self, loss, timesteps):
        """Collect logs.

        Args:
            loss (torch.Tensor): Losses to collect.
            timesteps (torch.Tensor): Timesteps of each loss items.
        """
        if not self.log_fn_list:
            return

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder_l = [torch.zeros_like(loss) for _ in range(ws)]
            placeholder_t = [torch.zeros_like(timesteps) for _ in range(ws)]
            dist.all_gather(placeholder_l, loss)
            dist.all_gather(placeholder_t, timesteps)
            loss = torch.cat(placeholder_l, dim=0)
            timesteps = torch.cat(placeholder_t, dim=0)
        log_vars = dict()

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            for log_fn in self.log_fn_list:
                log_vars.update(log_fn(loss, timesteps))
        self.log_vars = log_vars

    @torch.no_grad()
    def quartile_log_collect(self, loss, timesteps, total_timesteps, prefix_name, reduction="mean"):
        """Collect loss logs by quartile timesteps.

        Args:
            loss (torch.Tensor): Loss value of each input. Each loss tensor
                should be shape as [bz, ]
            timesteps (torch.Tensor): Timesteps corresponding to each loss.
                Each loss tensor should be shape as [bz, ].
            total_timesteps (int): Total timesteps of diffusion process.
            prefix_name (str): Prefix want to show in logs.
            reduction (str, optional): Specifies the reduction to apply to the
                output losses. Defaults to `mean`.

        Returns:
            dict: Collected log variables.
        """
        if digit_version(torch.__version__) <= digit_version("1.6.0"):
            # use true_divide in older torch version
            quartile = torch.true_divide(timesteps, total_timesteps) * 4
        else:
            quartile = timesteps / total_timesteps * 4
        quartile = quartile.type(torch.LongTensor)

        log_vars = dict()

        for idx in range(4):
            if not (quartile == idx).any():
                loss_quartile = torch.zeros((1,))
            else:
                loss_quartile = reduce_loss(loss[quartile == idx], reduction)
            log_vars[f"{prefix_name}_quartile_{idx}"] = loss_quartile.item()

        return log_vars

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function, ``mse_loss``.
        """
        if len(args) == 1:
            assert isinstance(args[0], dict), "You should offer a dictionary containing network outputs for building up computational graph of this loss module."
            output_dict = args[0]
        elif "output_dict" in kwargs:
            assert len(args) == 0, "If the outputs dict is given in keyworded arguments, no further non-keyworded arguments should be offered."
            output_dict = kwargs.pop("outputs_dict")
        else:
            raise NotImplementedError("Cannot parsing your arguments passed to this loss module. Please check the usage of this module")

        # check keys in output_dict
        assert "timesteps" in output_dict, f"'timesteps' is must for DDPM-based losses, but found{output_dict.keys()} in 'output_dict'"

        timesteps = output_dict["timesteps"]
        loss = self._forward_loss(output_dict)

        # update log_vars of this class
        self.collect_log(loss, timesteps=timesteps)

        loss_rescaled = self.rescale_fn(loss, timesteps)
        return reduce_loss(loss_rescaled, self.reduction)

    @abstractmethod
    def _forward_loss(self, output_dict):
        """Forward function for loss calculation. This method should be
        implemented by each subclasses.

        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """

        raise NotImplementedError("'self._forward_loss' must be implemented.")

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
