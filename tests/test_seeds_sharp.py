import importlib.util
from pathlib import Path
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def samplers():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.k_diffusion.sampling as sampling
    import comfy.model_sampling

    spec = importlib.util.spec_from_file_location("dpmpp_2m_sharp_nodes_for_seeds", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class EpsSampling(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS):
        pass

    class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    class Model:
        def __init__(self, model_type):
            if model_type == "flow":
                model_sampling = FlowSampling()
                model_sampling.set_parameters(timesteps=100)
            else:
                model_sampling = EpsSampling()
            patcher = types.SimpleNamespace(get_model_object=lambda name: model_sampling)
            self.inner_model = types.SimpleNamespace(model_patcher=patcher)
            self.calls = []

        def __call__(self, x, sigma, **kwargs):
            sigma_view = sigma.reshape((-1,) + (1,) * (x.ndim - 1))
            output = x * 0.19 + sigma_view * 0.027 + 0.06
            self.calls.append((x.clone(), sigma.clone(), output.clone()))
            return output

    return types.SimpleNamespace(sampling=sampling, module=module, Model=Model)


class FixedNoise:
    def __init__(self):
        self.calls = []

    def __call__(self, sigma, sigma_next):
        self.calls.append((sigma.clone(), sigma_next.clone()))
        return torch.full((1, 2, 3, 4), 0.075 * len(self.calls), dtype=sigma.dtype, device=sigma.device)


def run(function, model, x, sigmas, *, callback=None, **kwargs):
    return function(
        model, x, sigmas,
        extra_args={"seed": 824, "marker": "passed"},
        callback=callback,
        disable=True,
        **kwargs,
    )


def record_callbacks(events):
    def record(event):
        events.append({
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in event.items()
        })

    return record


@pytest.mark.parametrize("model_type", ["eps", "flow"])
@pytest.mark.parametrize("solver_type", ["phi_1", "phi_2"])
@pytest.mark.parametrize("eta,s_noise", [(0.0, 1.0), (0.65, 1.25)])
def test_zero_sharpness_matches_native_seeds_2(
    samplers, model_type, solver_type, eta, s_noise
):
    sampling = samplers.sampling
    sharp_fn = samplers.module.sample_seeds_2_sharp
    initial_model = samplers.Model(model_type)
    reference_model = samplers.Model(model_type)
    initial = torch.linspace(-0.9, 0.8, 24).reshape(1, 2, 3, 4)
    sigmas = torch.tensor([1.2, 0.86, 0.59, 0.36, 0.18, 0.0])
    original = initial.clone()
    sharp_noise, native_noise = FixedNoise(), FixedNoise()
    sharp_events, native_events = [], []

    result = sharp_fn(
        initial_model, initial, sigmas, extra_args={"seed": 824, "marker": "passed"},
        callback=record_callbacks(sharp_events), disable=True, eta=eta, s_noise=s_noise,
        noise_sampler=sharp_noise, r=0.5, solver_type=solver_type, sharpness=0.0,
    )
    expected = sampling.sample_seeds_2(
        reference_model, original.clone(), sigmas,
        extra_args={"seed": 824, "marker": "passed"},
        callback=record_callbacks(native_events), disable=True, eta=eta, s_noise=s_noise,
        noise_sampler=native_noise, r=0.5, solver_type=solver_type,
    )

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    assert len(initial_model.calls) == len(reference_model.calls) == 2 * len(sigmas) - 3
    assert len(sharp_events) == len(native_events) == len(sigmas) - 1
    noise_calls = 2 * (len(sigmas) - 2) if eta > 0 and s_noise > 0 else 0
    assert len(sharp_noise.calls) == len(native_noise.calls) == noise_calls
    for left, right in zip(sharp_events, native_events):
        assert left["i"] == right["i"]
        torch.testing.assert_close(left["sigma"], right["sigma"], rtol=0, atol=0)
        torch.testing.assert_close(left["denoised"], right["denoised"], rtol=0, atol=0)
    for left_call, right_call in zip(initial_model.calls, reference_model.calls):
        for left, right in zip(left_call, right_call):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("model_type", ["eps", "flow"])
@pytest.mark.parametrize("solver_type", ["phi_1", "phi_2"])
def test_default_r_sharpness_changes_mix_after_preserving_initial_stages(
    samplers, model_type, solver_type
):
    sharp_fn = samplers.module.sample_seeds_2_sharp
    plain_model = samplers.Model(model_type)
    sharp_model = samplers.Model(model_type)
    initial = torch.linspace(-0.9, 0.8, 24).reshape(1, 2, 3, 4)
    sigmas = torch.tensor([1.2, 0.86, 0.59, 0.36, 0.18, 0.0])
    events = []

    plain = sharp_fn(
        plain_model, initial.clone(), sigmas, extra_args={"seed": 824}, disable=True,
        eta=0.0, noise_sampler=FixedNoise(), solver_type=solver_type, sharpness=0.0,
    )
    sharp = sharp_fn(
        sharp_model, initial.clone(), sigmas, extra_args={"seed": 824},
        callback=record_callbacks(events), disable=True, eta=0.0,
        noise_sampler=FixedNoise(), solver_type=solver_type,
    )

    assert len(events) == len(sigmas) - 1
    assert len(plain_model.calls) == len(sharp_model.calls) == 2 * len(sigmas) - 3
    # Each step's first prediction and its inner evaluation are raw and unchanged.
    for plain_call, sharp_call in zip(plain_model.calls[:4], sharp_model.calls[:4]):
        for plain_value, sharp_value in zip(plain_call, sharp_call):
            torch.testing.assert_close(plain_value, sharp_value, rtol=0, atol=0)
    assert all(torch.isfinite(event["denoised"]).all() for event in events)
    assert not torch.equal(sharp, plain)


def test_default_seeded_noise_sampler_receives_cpu_tensor_and_seed(samplers, monkeypatch):
    import comfy.k_diffusion.sampling as sampling

    seen = []
    fixed_noise = FixedNoise()

    def make_noise(x, seed=None):
        seen.append((x, seed))
        return fixed_noise

    monkeypatch.setattr(sampling, "default_noise_sampler", make_noise)
    model = samplers.Model("flow")
    initial = torch.zeros(1, 2, 3, 4)
    sigmas = torch.tensor([1.2, 0.8, 0.4, 0.0])
    samplers.module.sample_seeds_2_sharp(
        model, initial, sigmas, extra_args={"seed": 824}, disable=True,
        eta=0.5, solver_type="phi_1", sharpness=0.2,
    )
    assert len(seen) == 1
    assert seen[0][0] is initial
    assert seen[0][0].device.type == "cpu"
    assert seen[0][1] == 824
    assert fixed_noise.calls


def test_seeds_sharp_node_forwards_native_options_and_registers_sampler(samplers, monkeypatch):
    import asyncio
    import comfy.samplers

    module = samplers.module
    schema = module.SamplerSEEDS2Sharp.define_schema()
    assert schema.node_id == "SamplerSEEDS2Sharp"
    assert [input.id for input in schema.inputs] == ["solver_type", "eta", "s_noise", "r", "sharpness"]
    assert next(input for input in schema.inputs if input.id == "solver_type").options == ["phi_1", "phi_2"]

    sampler = module.SamplerSEEDS2Sharp.execute(
        solver_type="phi_2", eta=0.4, s_noise=0.3, r=0.7, sharpness=0.375
    ).result[0]
    assert sampler.sampler_function is module.sample_seeds_2_sharp
    assert sampler.extra_options == {
        "solver_type": "phi_2", "eta": 0.4, "s_noise": 0.3, "r": 0.7, "sharpness": 0.375,
    }

    functions = {
        "dpmpp_2m_sde_gpu_sharp": module.sample_dpmpp_2m_sde_gpu_sharp,
        "seeds_2_sharp": module.sample_seeds_2_sharp,
    }
    original_k_names = set(comfy.samplers.KSAMPLER_NAMES)
    original_names = set(comfy.samplers.SAMPLER_NAMES)
    for name in functions:
        monkeypatch.setattr(samplers.sampling, "sample_" + name, None, raising=False)
    try:
        asyncio.run(module.DPMPP2MSharpExtension().on_load())
        asyncio.run(module.DPMPP2MSharpExtension().on_load())
        for name, function in functions.items():
            assert comfy.samplers.KSAMPLER_NAMES.count(name) == 1
            assert comfy.samplers.SAMPLER_NAMES.count(name) == 1
            assert comfy.samplers.KSampler.SAMPLERS.count(name) == 1
            registered = comfy.samplers.sampler_object(name)
            assert registered.sampler_function is function
    finally:
        for name in functions:
            if name not in original_k_names:
                comfy.samplers.KSAMPLER_NAMES.remove(name)
            if name not in original_names:
                comfy.samplers.SAMPLER_NAMES.remove(name)
