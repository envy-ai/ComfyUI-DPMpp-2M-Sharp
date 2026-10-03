from functools import partial

import comfy.k_diffusion.sampling as sampling
import comfy.samplers
from comfy_api.latest import ComfyExtension, io
from tqdm.auto import trange


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
                               tooltip="Progressively scales the denoised history. 0 disables the adjustment for the selected sampler."),
                io.Combo.Input("sampler_name", options=["dpmpp_2m_sharp", "dpmpp_2m_sde_gpu_sharp"], default="dpmpp_2m_sharp", optional=True,
                               tooltip="Choose DPM++ 2M Sharp or its SDE variant, which adds seeded Brownian noise on the sampling device."),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, sharpness, sampler_name="dpmpp_2m_sharp") -> io.NodeOutput:
        sampler_function = {
            "dpmpp_2m_sharp": sample_dpmpp_2m_sharp,
            "dpmpp_2m_sde_gpu_sharp": sample_dpmpp_2m_sde_gpu_sharp,
        }[sampler_name]
        sampler = comfy.samplers.KSAMPLER(sampler_function, {"sharpness": sharpness})
        return io.NodeOutput(sampler)

    get_sampler = execute


class DPMPP2MSharpExtension(ComfyExtension):
    async def get_node_list(self):
        return [SamplerDPMPP_2M_Sharp]


async def comfy_entrypoint():
    return DPMPP2MSharpExtension()
