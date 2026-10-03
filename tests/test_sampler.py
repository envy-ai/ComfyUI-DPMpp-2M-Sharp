import importlib.util
from pathlib import Path
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sampler_pack():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    spec = importlib.util.spec_from_file_location("dpmpp_2m_sharp_nodes", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_for_sampling(model_sampling):
    patcher = types.SimpleNamespace(get_model_object=lambda name: model_sampling)
    return types.SimpleNamespace(model_patcher=patcher)


class Denoiser:
    def __init__(self):
        self.outputs = []

    def __call__(self, x, sigma, **kwargs):
        denoised = x * 0.17 + sigma.reshape((-1,) + (1,) * (x.ndim - 1)) * 0.031 + 0.07
        self.outputs.append(denoised.clone())
        return denoised


def denoiser_for(model_sampling):
    denoiser = Denoiser()
    denoiser.inner_model = types.SimpleNamespace(
        model_patcher=model_for_sampling(model_sampling).model_patcher
    )
    return denoiser


class FixedNoise:
    def __init__(self):
        self.calls = []

    def __call__(self, sigma, sigma_next):
        self.calls.append((sigma.clone(), sigma_next.clone()))
        return torch.full((2, 4, 3, 5), 0.125 * len(self.calls), dtype=sigma.dtype, device=sigma.device)


def flow_sampling(comfy_model_sampling):
    class ConstFlow(comfy_model_sampling.ModelSamplingDiscreteFlow, comfy_model_sampling.CONST):
        pass

    result = ConstFlow()
    result.set_noise_scale(0.65)
    return result


@pytest.mark.parametrize("solver_type", ["midpoint", "heun"])
@pytest.mark.parametrize("eta,s_noise", [(0.0, 1.0), (0.7, 0.0), (0.7, 1.3)])
@pytest.mark.parametrize("model_sampling_kind", ["discrete", "flow"])
def test_zero_sharpness_matches_comfy_sde_and_preserves_callbacks(
    sampler_pack, solver_type, eta, s_noise, model_sampling_kind
):
    import comfy.k_diffusion.sampling as sampling
    import comfy.model_sampling

    model_sampling = (
        comfy.model_sampling.ModelSamplingDiscrete()
        if model_sampling_kind == "discrete"
        else flow_sampling(comfy.model_sampling)
    )
    model = denoiser_for(model_sampling)
    reference_model = denoiser_for(model_sampling)
    initial = torch.linspace(-0.8, 0.9, 120).reshape(2, 4, 3, 5)
    original = initial.clone()
    sigmas = torch.tensor([1.0, 0.62, 0.31, 0.12, 0.0])
    args = {"seed": 534, "marker": "forwarded"}
    new_model = model
    new_noise, reference_noise = FixedNoise(), FixedNoise()
    new_events, reference_events = [], []

    result = sampler_pack.sample_dpmpp_2m_sde_gpu_sharp(
        new_model,
        initial,
        sigmas,
        extra_args=args,
        callback=new_events.append,
        disable=True,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=new_noise,
        solver_type=solver_type,
        sharpness=0.0,
    )
    expected = sampling.sample_dpmpp_2m_sde(
        reference_model,
        initial.clone(),
        sigmas,
        extra_args=args,
        callback=reference_events.append,
        disable=True,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=reference_noise,
        solver_type=solver_type,
    )

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    assert len(new_model.outputs) == len(reference_model.outputs) == len(sigmas) - 1
    assert len(new_events) == len(reference_events) == len(sigmas) - 1
    assert len(new_noise.calls) == len(reference_noise.calls) == (len(sigmas) - 2 if eta > 0 and s_noise > 0 else 0)
    for actual, expected_event in zip(new_events, reference_events):
        assert actual["i"] == expected_event["i"]
        torch.testing.assert_close(actual["sigma"], expected_event["sigma"], rtol=0, atol=0)
        torch.testing.assert_close(actual["denoised"], expected_event["denoised"], rtol=0, atol=0)
    if model_sampling_kind == "flow":
        assert new_events[0]["sigma"] < sigmas[0]
    for actual, expected_output in zip(new_model.outputs, reference_model.outputs):
        torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize("solver_type", ["midpoint", "heun"])
def test_sharpness_changes_history_without_mutating_callback_prediction(sampler_pack, solver_type):
    import comfy.model_sampling

    model = denoiser_for(comfy.model_sampling.ModelSamplingDiscrete())
    events = []
    x = torch.linspace(-1, 1, 120).reshape(2, 4, 3, 5)
    sigmas = torch.tensor([1.4, 0.82, 0.45, 0.2, 0.0])
    result = sampler_pack.sample_dpmpp_2m_sde_gpu_sharp(
        model,
        x.clone(),
        sigmas,
        extra_args={"seed": 93},
        callback=events.append,
        disable=True,
        eta=0,
        noise_sampler=FixedNoise(),
        solver_type=solver_type,
        sharpness=0.8,
    )
    ordinary = sampler_pack.sample_dpmpp_2m_sde_gpu_sharp(
        denoiser_for(comfy.model_sampling.ModelSamplingDiscrete()),
        x.clone(),
        sigmas,
        extra_args={"seed": 93},
        disable=True,
        eta=0,
        noise_sampler=FixedNoise(),
        solver_type=solver_type,
        sharpness=0,
    )
    assert not torch.equal(result, ordinary)
    assert len(events) == len(model.outputs) == len(sigmas) - 1
    for event, output in zip(events, model.outputs):
        torch.testing.assert_close(event["denoised"], output, rtol=0, atol=0)


@pytest.mark.parametrize("sigmas", [torch.tensor([]), torch.tensor([1.0])])
def test_empty_or_single_sigma_returns_input_without_model_or_noise(sampler_pack, sigmas):
    class Unexpected:
        def __call__(self, *args, **kwargs):
            pytest.fail("No model evaluation expected")

    x = torch.ones(2, 4, 3, 5)
    assert sampler_pack.sample_dpmpp_2m_sde_gpu_sharp(Unexpected(), x, sigmas) is x


def test_default_noise_sampler_requests_gpu_brownian_tree_and_forwards_seed(sampler_pack, monkeypatch):
    import comfy.k_diffusion.sampling as sampling
    import comfy.model_sampling

    made = []

    class Brownian:
        def __init__(self, x, sigma_min, sigma_max, seed, cpu):
            made.append((x, sigma_min, sigma_max, seed, cpu))

        def __call__(self, sigma, sigma_next):
            return torch.zeros_like(initial)

    monkeypatch.setattr(sampling, "BrownianTreeNoiseSampler", Brownian)
    initial = torch.ones(2, 4, 3, 5)
    sigmas = torch.tensor([1.4, 0.8, 0.3, 0.0])
    denoiser = denoiser_for(comfy.model_sampling.ModelSamplingDiscrete())
    sampler_pack.sample_dpmpp_2m_sde_gpu_sharp(
        denoiser, initial, sigmas,
        extra_args={"seed": 713}, disable=True, eta=1, sharpness=0.2
    )
    assert len(made) == 1
    assert made[0][0] is initial
    assert made[0][3:] == (713, False)
    torch.testing.assert_close(made[0][1], sigmas[sigmas > 0].min())
    torch.testing.assert_close(made[0][2], sigmas.max())


def test_seeded_brownian_noise_is_repeatable_and_matches_cpu_tree_on_cpu(sampler_pack):
    import comfy.k_diffusion.sampling as sampling
    import comfy.model_sampling

    model_sampling = comfy.model_sampling.ModelSamplingDiscrete()
    model = denoiser_for(model_sampling)
    sigmas = torch.tensor([1.5, 0.75, 0.375, 0.125, 0.0])
    x = torch.linspace(-0.5, 0.5, 32).reshape(2, 4, 2, 2)
    kwargs = {"extra_args": {"seed": 918}, "disable": True, "eta": 0.7, "sharpness": 0.0}
    gpu_variant = sampler_pack.sample_dpmpp_2m_sde_gpu_sharp(model, x.clone(), sigmas, **kwargs)
    repeated = sampler_pack.sample_dpmpp_2m_sde_gpu_sharp(
        denoiser_for(model_sampling), x.clone(), sigmas, **kwargs
    )
    native_cpu_tree = sampling.sample_dpmpp_2m_sde(
        denoiser_for(model_sampling), x.clone(), sigmas,
        extra_args={"seed": 918}, disable=True, eta=0.7
    )
    torch.testing.assert_close(gpu_variant, repeated, rtol=0, atol=0)
    torch.testing.assert_close(gpu_variant, native_cpu_tree, rtol=0, atol=0)


def test_provider_selects_both_samplers_and_keeps_sharpness_option(sampler_pack, monkeypatch):
    import comfy.k_diffusion.sampling as k_sampling

    res_functions = {}
    for name in (
        "sample_res_2s_nc", "sample_res_2m_nc", "sample_res_2s_nc_sharp",
        "sample_res_2m_nc_sharp",
    ):
        function = lambda *args, **kwargs: None
        monkeypatch.setattr(k_sampling, name, function, raising=False)
        res_functions[name.removeprefix("sample_")] = function

    for sampler_name, function in (
        ("dpmpp_2m_sharp", sampler_pack.sample_dpmpp_2m_sharp),
        ("dpmpp_2m_sde_gpu_sharp", sampler_pack.sample_dpmpp_2m_sde_gpu_sharp),
        ("res_2s_nc", res_functions["res_2s_nc"]),
        ("res_2m_nc", res_functions["res_2m_nc"]),
        ("res_2s_nc_sharp", res_functions["res_2s_nc_sharp"]),
        ("res_2m_nc_sharp", res_functions["res_2m_nc_sharp"]),
        ("seeds_2_sharp", sampler_pack.sample_seeds_2_sharp),
    ):
        sampler = sampler_pack.SamplerDPMPP_2M_Sharp.execute(0.375, sampler_name).result[0]
        assert sampler.sampler_function is function
        assert sampler.extra_options == ({"sharpness": 0.375} if sampler_name.endswith("_sharp") else {})

    schema = sampler_pack.SamplerDPMPP_2M_Sharp.define_schema()
    assert schema.node_id == "SamplerDPMPP_2M_Sharp"
    assert [input.id for input in schema.inputs] == ["sharpness", "sampler_name"]
    selector = next(input for input in schema.inputs if input.id == "sampler_name")
    assert selector.options == [
        "dpmpp_2m_sharp", "dpmpp_2m_sde_gpu_sharp", "res_2s_nc", "res_2m_nc",
        "res_2s_nc_sharp", "res_2m_nc_sharp", "seeds_2_sharp",
    ]
    assert selector.default == "dpmpp_2m_sharp"

    default_sampler = sampler_pack.SamplerDPMPP_2M_Sharp.execute(0.375).result[0]
    assert default_sampler.sampler_function is sampler_pack.sample_dpmpp_2m_sharp
    assert default_sampler.extra_options == {"sharpness": 0.375}
