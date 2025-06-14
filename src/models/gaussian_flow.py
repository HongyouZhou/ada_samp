import inspect
from copy import deepcopy

import diffusers
import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.progress import Progress

from src.losses.registry import build_loss
from src.samplers.registry import get_noise_sampler, get_timestep_sampler
from src.utils.gmflow_ops import gm_to_mean, gm_to_sample

from .. import schedulers
from .denoising.registry import get_model


@torch.jit.script
def guidance_jit(pos_mean, neg_mean, guidance_scale: float, orthogonal: bool = False):
    bias = (pos_mean - neg_mean) * (guidance_scale - 1)
    if orthogonal:
        bias = bias - (bias * pos_mean).mean(dim=(-3, -2, -1), keepdim=True) / (pos_mean * pos_mean).mean(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6) * pos_mean
    return bias


class GaussianFlow(nn.Module):
    def __init__(
        self,
        denoising=None,
        flow_loss=None,
        num_timesteps=1000,
        timestep_sampler=DictConfig(dict(type="continuous", shift=1.0)),
        noise_sampler=DictConfig(dict(type="gaussian")),
        denoising_mean_mode="U",
        train_cfg=None,
        test_cfg=None,
        **kwargs,
    ):
        super().__init__()
        # build denoising module in this function
        self.num_timesteps = num_timesteps
        self.denoising = get_model(denoising) if isinstance(denoising, DictConfig) else denoising
        self.denoising_mean_mode = denoising_mean_mode

        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else DictConfig({})
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else DictConfig({})

        # build sampler
        self.timestep_sampler = get_timestep_sampler(timestep_sampler)
        self.noise_sampler = get_noise_sampler(noise_sampler)

        self.flow_loss = build_loss(flow_loss) if flow_loss is not None else None

    def sample_forward_diffusion(self, x_0, t, noise):
        if t.dim() == 0:
            t = t.expand(x_0.size(0))
        std = t.reshape(*t.size(), *((x_0.dim() - t.dim()) * [1])) / self.num_timesteps
        mean = 1 - std
        return x_0 * mean + noise * std, mean, std

    def pred(self, x_t, t, **kwargs):
        ori_dtype = x_t.dtype
        x_t = x_t.to(next(self.denoising.parameters()).dtype)
        num_batches = x_t.size(0)
        if t.dim() == 0 or len(t) != num_batches:
            t = t.expand(num_batches)
        output = self.denoising(x_t, t, **kwargs)
        if isinstance(output, dict):
            output = {k: v.to(ori_dtype) for k, v in output.items()}
        else:
            output = output.to(ori_dtype)
        return output

    def loss(self, denoising_output, x_0, noise, t, pred_mask=None):
        with torch.autocast(device_type="cuda", enabled=False):
            if self.denoising_mean_mode.upper() == "U":
                if isinstance(denoising_output, dict):
                    loss_kwargs = denoising_output
                elif isinstance(denoising_output, torch.Tensor):
                    loss_kwargs = dict(u_t_pred=denoising_output)
                else:
                    raise AttributeError(f"Unknown denoising output type [{type(denoising_output)}].")
                loss_kwargs.update(u_t=noise - x_0)
            else:
                raise AttributeError(f"Unknown denoising mean output type [{self.denoising_mean_mode}].")
            loss_kwargs.update(x_0=x_0, noise=noise, timesteps=t, weight=pred_mask.float() if pred_mask is not None else None)

            return self.flow_loss(loss_kwargs)

    def forward_train(self, x_0, **kwargs):
        assert x_0.dim() == 4
        b, c, h, w = x_0.size()

        t = self.timestep_sampler(b).to(x_0)

        noise = self.noise_sampler(b).to(x_0)
        x_t, _, _ = self.sample_forward_diffusion(x_0, t, noise)

        denoising_output = self.pred(x_t, t, **kwargs)
        loss = self.loss(denoising_output, x_0, noise, t)
        log_vars = self.flow_loss.log_vars
        log_vars.update(loss_diffusion=float(loss))

        return loss, log_vars

    def forward_test(self, x_0=None, noise=None, guidance_scale=1.0, test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_t = torch.randn_like(x_0) if noise is None else noise
        ori_dtype = x_t.dtype
        x_t = x_t.float()

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        output_mode = cfg.get("output_mode", "mean")
        assert output_mode in ["mean", "sample"]

        sampler = cfg.get("sampler", "FlowMatchEulerDiscrete")
        sampler_class = getattr(diffusers.schedulers, sampler + "Scheduler", None)
        if sampler_class is None:
            sampler_class = getattr(schedulers, sampler + "Scheduler", None)
        if sampler_class is None:
            raise AttributeError(f"Cannot find sampler [{sampler}].")

        sampler_kwargs = cfg.get("sampler_kwargs", {})
        signatures = inspect.signature(sampler_class).parameters.keys()
        if "shift" in signatures and "shift" not in sampler_kwargs:
            sampler_kwargs["shift"] = cfg.get("shift", self.timestep_sampler.shift)
        if "output_mode" in signatures:
            sampler_kwargs["output_mode"] = output_mode
        sampler = sampler_class(self.num_timesteps, **sampler_kwargs)

        sampler.set_timesteps(cfg.get("num_timesteps", self.num_timesteps), device=x_t.device)
        orthogonal_guidance = cfg.get("orthogonal_guidance", False)
        timesteps = sampler.timesteps
        use_guidance = guidance_scale > 1.0
        num_batches = x_t.size(0)

        if show_pbar:
            self._progress = Progress()
            self._progress.start()
            self._task_id = self._progress.add_task("[cyan]Sampling", total=len(timesteps))

        for t in timesteps:
            x_t_input = x_t
            if use_guidance:
                x_t_input = torch.cat([x_t_input, x_t_input], dim=0)

            denoising_output = self.pred(x_t_input, t, **kwargs)

            if isinstance(denoising_output, dict):  # for gmflow compatibility
                if use_guidance:
                    gm_pos = {k: v[num_batches:] for k, v in denoising_output.items()}
                    gm_neg = {k: v[:num_batches] for k, v in denoising_output.items()}
                    bias = guidance_jit(gm_to_mean(gm_pos), gm_to_mean(gm_neg), guidance_scale, orthogonal_guidance)
                    denoising_output = dict(means=gm_pos["means"] + bias.unsqueeze(-4), logstds=gm_pos["logstds"], logweights=gm_pos["logweights"])

                if output_mode == "mean":
                    denoising_output = gm_to_mean(denoising_output)
                else:
                    denoising_output = gm_to_sample(denoising_output).squeeze(-4)

            else:
                assert output_mode == "mean"
                if use_guidance:
                    mean_neg, mean_pos = denoising_output.chunk(2, dim=0)
                    bias = guidance_jit(mean_pos, mean_neg, guidance_scale, orthogonal_guidance)
                    denoising_output = mean_pos + bias

            x_t = sampler.step(denoising_output, t, x_t, return_dict=False)[0]
            if show_pbar:
                self._progress.update(self._task_id, advance=1)

        if show_pbar:
            self._progress.stop()

        return x_t.to(ori_dtype)

    def forward(self, x_0=None, return_loss=False, **kwargs):
        if return_loss:
            return self.forward_train(x_0, **kwargs)

        return self.forward_test(x_0, **kwargs)
