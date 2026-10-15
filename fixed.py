"""Research 2M/3M paths adapted from research-unipc-dpmpp.

Copyright (c) 2026 Iakov Savelev. MIT license: see LICENSE.research-unipc-dpmpp.
Source: https://github.com/YakovSava/research-unipc-dpmpp
"""

from functools import partial
import math

import numpy as np
from tqdm.auto import trange

from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler, offset_first_sigma_for_snr, sigma_to_half_log_snr


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(12)


def _warmup_3m(x, denoised, old_denoised, h_last_eta, h_eta, x_decay, alpha_t):
    # Preserve the research sampler's two-point quadrature warmup.
    d1 = (old_denoised - denoised) / -h_last_eta
    s = 0.5 * h_eta * (_GL_NODES + 1)
    w = 0.5 * h_eta * _GL_WEIGHTS
    integrand = np.exp(-(h_eta - s))
    i0 = float(np.sum(w * integrand))
    integrand = integrand * (s - 0.0)
    i1 = float(np.sum(w * integrand))
    x_new = x_decay * x
    x_new = x_new + alpha_t * i0 * denoised
    return x_new + alpha_t * i1 * d1


def _sample_research_dpmpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, order=3, sharpness=0.15):
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min = float(sigmas[sigmas > 0].min())
    sigma_max = float(sigmas.max())
    if noise_sampler is None and eta != 0:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=not x.is_cuda)
    s_in = x.new_ones([x.shape[0]])

    model_sampling = None
    if hasattr(model, "inner_model") and hasattr(model.inner_model, "model_patcher"):
        model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    if model_sampling is not None:
        lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
        sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
        s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    else:
        lambda_fn = lambda s: -s.log()

    hist_denoised = []
    hist_h = []
    hist_h_eta = []

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sharpness:
            adjustment_factor = 1 + sharpness * (i / len(sigmas)) ** 2

        if sigmas[i + 1] == 0:
            x = denoised
            h = None
        else:
            lambda_s = lambda_fn(sigmas[i]).item()
            lambda_t = lambda_fn(sigmas[i + 1]).item()
            h = lambda_t - lambda_s
            eta_h = eta * h
            h_eta = h + eta_h
            alpha_t = (sigmas[i + 1] * math.exp(lambda_t)).item() if model_sampling is not None else 1.0
            x_ratio = (sigmas[i + 1] / sigmas[i]).item() if sigmas[i] > 0 else math.exp(-h)
            x_decay = x_ratio * math.exp(-eta_h)

            if len(hist_denoised) == 0:
                x = x_decay * x + alpha_t * (-math.expm1(-h_eta)) * denoised
            elif order == 2:
                r = hist_h[0] / h
                d1 = (denoised - hist_denoised[0]) / r
                x = (
                    x_decay * x
                    + alpha_t * (-math.expm1(-h_eta)) * denoised
                    + 0.5 * alpha_t * (-math.expm1(-h_eta)) * d1
                )
            elif len(hist_denoised) >= 2:
                r0 = hist_h[0] / h
                r1 = hist_h[1] / h
                d1_0 = (denoised - hist_denoised[0]) / r0
                d1_1 = (hist_denoised[0] - hist_denoised[1]) / r1
                d1_val = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2_val = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = math.expm1(-h_eta) / h_eta + 1 if h_eta != 0 else 0.5
                phi_3 = phi_2 / h_eta - 0.5 if h_eta != 0 else 0.0
                x = (
                    x_decay * x
                    + alpha_t * (-math.expm1(-h_eta)) * denoised
                    + alpha_t * phi_2 * d1_val
                    - alpha_t * phi_3 * d2_val
                )
            else:
                x = _warmup_3m(x, denoised, hist_denoised[0], hist_h_eta[0], h_eta, x_decay, alpha_t)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * math.sqrt(-math.expm1(-2 * eta_h)) * s_noise

        if sharpness:
            denoised = denoised * adjustment_factor
        hist_denoised.insert(0, denoised)
        if h is not None:
            hist_h.insert(0, h)
            hist_h_eta.insert(0, h_eta)
        hist_denoised = hist_denoised[:2]
        hist_h = hist_h[:2]
        hist_h_eta = hist_h_eta[:2]

    return x


sample_dpmpp_2m_fixed = partial(_sample_research_dpmpp, order=2, eta=0.0, sharpness=0.0)
sample_dpmpp_2m_sharp_fixed = partial(_sample_research_dpmpp, order=2, eta=0.0)
sample_dpmpp_2m_sde_gpu_sharp_fixed = partial(_sample_research_dpmpp, order=2)
sample_dpmpp_3m_sde_gpu_sharp_fixed = partial(_sample_research_dpmpp, order=3)
