"""Default RES 2S/2M paths extracted from RES4LYF, without the final correction.

Adapted from ClownsharkBatwing/RES4LYF beta RK samplers (local commit 5e72fe3).
See LICENSE.RES4LYF for the upstream license and commercial-service restriction.
"""

from decimal import Decimal, localcontext

import torch
from tqdm.auto import trange

from comfy.model_sampling import CONST


def _phi(order, h, c=1.0):
    # Match RES4LYF's high-precision analytic coefficients without mpmath.
    with localcontext() as context:
        context.prec = 80
        z = -Decimal.from_float(float(h)) * Decimal.from_float(c)
        if z == 0:
            return 1.0 if order == 1 else 0.5
        remainder = z.exp() - 1
        if order == 2:
            remainder -= z
        return float(remainder / z ** order)


def _reanchor(x_0, x, data, sigma, sub_sigma):
    anchored = (x_0 - data) / sigma
    unmoored = (x - data) / sub_sigma
    return x_0 - sigma * (unmoored + (anchored - unmoored))


def _rebound(x_0, x, sigma, sigma_next):
    eps = (x_0 - x) / (sigma - sigma_next)
    return x_0 - sigma * eps + sigma_next * eps


def _swap_noise(x_0, x, sigma, sigma_next, generator, variance_preserving):
    sigma_up = sigma_next * 0.5
    residual = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    alpha = 1 - sigma_next + residual if variance_preserving else torch.ones_like(sigma_next)
    sigma_down = residual / alpha
    eps = (x_0 - x) / (sigma - sigma_next)
    data = x_0 - sigma * eps
    noise = torch.randn(x.shape, dtype=torch.float64, layout=x.layout, device=x.device, generator=generator)
    noise = (noise - noise.mean()) / noise.std()
    noise.sub_(noise.mean(dim=(-2, -1), keepdim=True)).div_(noise.std(dim=(-2, -1), keepdim=True))
    return alpha * (data + sigma_down * eps) + sigma_up * noise


def _sample_res(model, x, sigmas, extra_args, callback, disable, multistep, sharpness):
    if len(sigmas) <= 1:
        return x

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    variance_preserving = isinstance(model_sampling, CONST)
    sigma_min = model_sampling.sigma_min.to(device=x.device, dtype=torch.float64)
    sigma_max = model_sampling.sigma_max.to(device=x.device, dtype=torch.float64)
    sigmas = sigmas.to(device=x.device, dtype=torch.float64).clone()
    sample_sigmas = sigmas
    sigmas = torch.unique_consecutive(sigmas)
    unsample_from_zero = bool(sigmas[0] == 0 and sigmas[-1] == 0)
    if sigmas[0] == 0:
        sigmas = sigmas[1:]
        if len(sigmas) and sigmas[-1] == 0:
            sigmas = sigmas[:-1]
    if len(sigmas) <= 1:
        return x
    if sigmas[-1] == 0:
        if sigmas[-2] < sigma_min:
            sigmas[-2] = sigma_min
        elif (sigmas[-2] - sigma_min).abs() > 1e-4:
            sigmas = torch.cat((sigmas[:-1], sigma_min.unsqueeze(0), sigmas[-1:]))
    elif unsample_from_zero and not torch.isclose(sigmas[0], sigma_min):
        sigmas = torch.cat((sigma_min.unsqueeze(0), sigmas))
    steps = len(sigmas) - (2 if sigmas[-1] == 0 else 1)

    extra_args = {} if extra_args is None else extra_args.copy()
    options = extra_args["model_options"] = extra_args.get("model_options", {}).copy()
    transformer_options = options["transformer_options"] = options.get("transformer_options", {}).copy()
    transformer_options["sample_sigmas"] = sample_sigmas
    noise = torch.Generator(device=x.device).manual_seed(torch.initial_seed() + 1)
    substep_noise = torch.Generator(device=x.device).manual_seed(9999)
    x = x.to(torch.float32)
    s_in = x.new_ones([x.shape[0]])
    history = None
    stages = torch.zeros((3, *x.shape), dtype=x.dtype, device=x.device)
    eps = torch.zeros((2, *x.shape), dtype=x.dtype, device=x.device)
    data = torch.zeros_like(eps)

    for step in trange(steps, disable=disable):
        sigma, sigma_next = sigmas[step:step + 2]
        h = -(sigma_next / sigma).log()
        use_history = multistep and step >= 2 and bool(h < 1)
        euler = multistep and bool(h >= 1) and bool(sigma < 0.1)
        c2 = float((sigmas[step] / sigmas[step - 1]).log() / h) if use_history else 0.5
        if euler:
            a = torch.zeros(1, device=x.device, dtype=eps.dtype)
            b = torch.ones(1, device=x.device, dtype=eps.dtype)
            stage_sigmas = (-(-sigma.log() + h * sigmas.new_tensor([0, 1]))).exp()
        else:
            a = torch.tensor([c2 * _phi(1, h, c2), 0], device=x.device, dtype=eps.dtype)
            b2 = _phi(2, h) / c2
            b = torch.tensor([_phi(1, h) - b2, b2], device=x.device, dtype=eps.dtype)
            stage_sigmas = (-(-sigma.log() + h * sigmas.new_tensor([0, c2, 1]))).exp()
        zero_coefficients = torch.zeros_like(a)
        rows = 1 if use_history or euler else 2
        stages[0] = x
        x_0 = stages[0].clone()
        if use_history:
            eps[1] = history - x_0

        for row in range(rows):
            sub_sigma = stage_sigmas[row]
            transformer_options.update(row=row, x_tmp=stages[row], sigma_next=sigma_next)
            prediction = model(stages[row], sub_sigma * s_in, **extra_args)
            prediction = _reanchor(x_0, stages[row], prediction, sigma, sub_sigma)
            eps[row] = prediction - x_0
            data[row] = prediction

            final_stage = row == rows - 1
            sub_sigma_next = stage_sigmas[-1] if final_stage else stage_sigmas[1]
            sub_h = -(sub_sigma_next / sigma).log()
            h_new = h * sub_h / sub_h if sub_h != 0 else h
            eps_update = eps
            if not multistep and final_stage and sharpness != 0:
                factor = 1 + sharpness * (step / len(sigmas)) ** 2
                eps_update = eps.clone()
                eps_update[0] += (data[0] * factor - x_0) - (data[0] - x_0)
            coefficients = b if final_stage else a
            stages[row + 1] = x_0 + h_new * torch.einsum('i, i... -> ...', coefficients, eps_update[:len(coefficients)])
            if sigma > sub_sigma_next:
                stages[row + 1] = _rebound(x_0, stages[row + 1], sigma, sub_sigma_next)
            if not final_stage:
                stages[row + 1] = _swap_noise(x_0, stages[row + 1], sigma, sub_sigma_next, substep_noise, variance_preserving)
                # RES4LYF reanchors the noisy intermediate predictor before stage two.
                if stage_sigmas[row] > sigma_min and h < sigma_max / 2 and sigma > 0.03:
                    for _ in range(100):
                        x_0 = stages[1] - h * torch.einsum('i, i... -> ...', a, eps)
                        stages[0] = x_0 + h * torch.einsum('i, i... -> ...', zero_coefficients, eps)
                        eps[0] = _reanchor(x_0, stages[0], data[0], sigma, stage_sigmas[0]) - x_0

        x = _rebound(x_0, stages[rows], sigma, sigma_next)
        x = _swap_noise(x_0, x, sigma, sigma_next, noise, variance_preserving)
        if callback is not None:
            callback({'x': x, 'i': step, 'i_sched': step, 'final': False, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': data[0]})
        history = data[0].clone()
        if multistep and sharpness != 0:
            history *= 1 + sharpness * (step / len(sigmas)) ** 2

    if steps and callback is not None:
        callback({'x': x, 'i': steps, 'i_sched': steps, 'final': True, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': data[0]})
    return x


def sample_res_2s_nc(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, False, 0.0)


def sample_res_2m_nc(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, True, 0.0)


def sample_res_2s_nc_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, sharpness=0.15):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, False, sharpness)


def sample_res_2m_nc_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, sharpness=0.15):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, True, sharpness)
