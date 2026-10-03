from functools import partial

import torch

import comfy.k_diffusion.sampling as sampling
import comfy.samplers
from comfy_api.latest import ComfyExtension, io
from tqdm.auto import trange

from .res import sample_res_2s_nc, sample_res_2m_nc, sample_res_2s_nc_sharp, sample_res_2m_nc_sharp


def sample_dpmpp_2m_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, sharpness=0.15):
    """DPM-Solver++(2M) with progressively sharpened denoised history."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + sharpness * sigma_progress * sigma_progress
        old_denoised = denoised * adjustment_factor
    return x


def sample_dpmpp_2m_sde_gpu_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint', sharpness=0.15):
    """DPM-Solver++(2M) SDE with GPU noise and sharpened denoised history."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    lambda_fn = partial(sampling.sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)

    old_denoised = None
    h, h_last = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + sharpness * sigma_progress * sigma_progress
        old_denoised = denoised * adjustment_factor
        h_last = h
    return x


def sample_seeds_2_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=0.5, solver_type="phi_1", sharpness=0.15):
    """SEEDS-2 with progressively scaled first-stage prediction differences."""
    if solver_type not in {"phi_1", "phi_2"}:
        raise ValueError("solver_type must be 'phi_1' or 'phi_2'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = sampling.default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    inject_noise = eta > 0 and s_noise > 0
    sigma_fn = partial(sampling.half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sampling.sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = sampling.offset_first_sigma_for_snr(sigmas, model_sampling)

    fac = 1 / (2 * r)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r)
        sigma_s_1 = sigma_fn(lambda_s_1)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        x_2 = sigma_s_1 / sigmas[i] * (-r * h * eta).exp() * x - alpha_s_1 * sampling.ei_h_phi_1(-r * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        adjustment_factor = 1 + sharpness * (i / len(sigmas)) ** 2
        if solver_type == "phi_1":
            denoised_d = torch.lerp(denoised, denoised_2, fac)
            if sharpness != 0.0:
                denoised_d = denoised_d - fac * (adjustment_factor - 1) * denoised
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * sampling.ei_h_phi_1(-h_eta) * denoised_d
        elif solver_type == "phi_2":
            b2 = sampling.ei_h_phi_2(-h_eta) / r
            b1 = sampling.ei_h_phi_1(-h_eta) - b2
            denoised_d = b1 * denoised + b2 * denoised_2
            if sharpness != 0.0:
                denoised_d = denoised_d - b2 * (adjustment_factor - 1) * denoised
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * denoised_d

        if inject_noise:
            segment_factor = (r - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x


class SamplerDPMPP_2M_Sharp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerDPMPP_2M_Sharp",
            display_name="Sampler DPM++ 2M Sharp",
            category="model/sampling/samplers",
            description="A better sampler for Qwen Image 2.1, with adjustable denoised-history sharpening.",
            inputs=[
                io.Float.Input("sharpness", default=0.15, min=0.0, max=100.0, step=0.01, round=False,
                               tooltip="Progressively scales denoised history, or the first-stage prediction used in RES 2S and SEEDS_2 corrections. 0 disables sharpening. Ignored by samplers without _sharp."),
                io.Combo.Input("sampler_name", options=["dpmpp_2m_sharp", "dpmpp_2m_sde_gpu_sharp", "res_2s_nc", "res_2m_nc", "res_2s_nc_sharp", "res_2m_nc_sharp", "seeds_2_sharp"], default="dpmpp_2m_sharp", optional=True,
                               tooltip="RES _nc variants skip the final clean-latent correction, retaining a small amount of residual noise."),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, sharpness, sampler_name="dpmpp_2m_sharp") -> io.NodeOutput:
        sampler_function = {
            "dpmpp_2m_sharp": sample_dpmpp_2m_sharp,
            "dpmpp_2m_sde_gpu_sharp": sample_dpmpp_2m_sde_gpu_sharp,
            "seeds_2_sharp": sample_seeds_2_sharp,
            "res_2s_nc": sample_res_2s_nc,
            "res_2m_nc": sample_res_2m_nc,
            "res_2s_nc_sharp": sample_res_2s_nc_sharp,
            "res_2m_nc_sharp": sample_res_2m_nc_sharp,
        }[sampler_name]
        extra_options = {"sharpness": sharpness} if sampler_name.endswith("_sharp") else {}
        sampler = comfy.samplers.KSAMPLER(sampler_function, extra_options)
        return io.NodeOutput(sampler)

    get_sampler = execute


class SamplerSEEDS2Sharp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSEEDS2Sharp",
            display_name="Sampler SEEDS_2 Sharp",
            category="model/sampling/samplers",
            description="Experimental SEEDS_2 stage sharpening with adjustable stochastic noise.",
            inputs=[
                io.Combo.Input("solver_type", options=["phi_1", "phi_2"]),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, tooltip="Stochastic strength", advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, tooltip="SDE noise multiplier", advanced=True),
                io.Float.Input("r", default=0.5, min=0.01, max=1.0, step=0.01, round=False, tooltip="Relative step size for the intermediate stage (c2 node)", advanced=True),
                io.Float.Input("sharpness", default=0.15, min=0.0, max=100.0, step=0.01, round=False,
                               tooltip="Progressively scales the first prediction in the stage difference. 0 matches native SEEDS_2."),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, solver_type, eta, s_noise, r, sharpness) -> io.NodeOutput:
        return io.NodeOutput(comfy.samplers.KSAMPLER(sample_seeds_2_sharp, {"solver_type": solver_type, "eta": eta, "s_noise": s_noise, "r": r, "sharpness": sharpness}))

    get_sampler = execute


class DPMPP2MSharpExtension(ComfyExtension):
    async def on_load(self):
        for name, function in (
            ("dpmpp_2m_sde_gpu_sharp", sample_dpmpp_2m_sde_gpu_sharp),
            ("seeds_2_sharp", sample_seeds_2_sharp),
            ("res_2s_nc", sample_res_2s_nc),
            ("res_2m_nc", sample_res_2m_nc),
            ("res_2s_nc_sharp", sample_res_2s_nc_sharp),
            ("res_2m_nc_sharp", sample_res_2m_nc_sharp),
        ):
            setattr(sampling, "sample_" + name, function)
            if name not in comfy.samplers.KSAMPLER_NAMES:
                comfy.samplers.KSAMPLER_NAMES.append(name)
            if name not in comfy.samplers.SAMPLER_NAMES:
                comfy.samplers.SAMPLER_NAMES.append(name)

    async def get_node_list(self):
        return [SamplerDPMPP_2M_Sharp, SamplerSEEDS2Sharp]


async def comfy_entrypoint():
    return DPMPP2MSharpExtension()
