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


class SamplerDPMPP_2M_Sharp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerDPMPP_2M_Sharp",
            display_name="Sampler DPM++ 2M Sharp",
            category="model/sampling/samplers",
            description="A better sampler for Qwen Image 2.1, with adjustable denoised-history sharpening.",
            inputs=[
                io.Float.Input("sharpness", default=0.15, min=0.0, max=100.0, step=0.01, round=False),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, sharpness) -> io.NodeOutput:
        sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2m_sharp, {"sharpness": sharpness})
        return io.NodeOutput(sampler)

    get_sampler = execute


class DPMPP2MSharpExtension(ComfyExtension):
    async def get_node_list(self):
        return [SamplerDPMPP_2M_Sharp]


async def comfy_entrypoint():
    return DPMPP2MSharpExtension()
