from pathlib import Path
import sys
import importlib.util
import types
import subprocess
import textwrap

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_NODES = ROOT.parent


@pytest.fixture(scope="module")
def res_sampler():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.model_sampling

    spec = importlib.util.spec_from_file_location("sharp_res_nodes", ROOT / "__init__.py")
    pack = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pack
    spec.loader.exec_module(pack)

    class DiffusionModel:
        double_stream_blocks = []
        single_stream_blocks = []

        def _get_name(self):
            return "TinyDenoiser"

    class EpsSampling(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS):
        pass

    class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    class Model:
        def __init__(self, model_type="eps"):
            if model_type == "flow":
                self.model_sampling = FlowSampling()
                self.model_sampling.set_parameters(timesteps=100)
            else:
                self.model_sampling = EpsSampling()
            self.diffusion_model = DiffusionModel()
            core = types.SimpleNamespace(
                device=torch.device("cpu"),
                model_sampling=self.model_sampling,
                diffusion_model=self.diffusion_model,
            )
            patcher = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=self.diffusion_model), get_model_object=lambda name: self.model_sampling)
            wrapper = types.SimpleNamespace(inner_model=core, model_patcher=patcher, cfg=1.0)
            self.inner_model = wrapper
            self.outputs = []
            self.calls = []

        def __call__(self, x, sigma, **kwargs):
            sigma = sigma.reshape((-1,) + (1,) * (x.ndim - 1))
            result = 0.21 * x + 0.035 * sigma + 0.08
            self.outputs.append(result.clone())
            self.calls.append((x.clone(), sigma.clone(), result.clone()))
            return result

    yield types.SimpleNamespace(
        pack=pack,
        Model=Model,
    )


@pytest.fixture(scope="module")
def reference():
    if not (CUSTOM_NODES / "RES4LYF").is_dir():
        pytest.skip("Optional RES4LYF comparison requires the original node pack")
    import server

    class Routes:
        def post(self, path):
            return lambda function: function

        def get(self, path):
            return lambda function: function

    missing = object()
    previous_instance = getattr(server.PromptServer, "instance", missing)
    server.PromptServer.instance = types.SimpleNamespace(routes=Routes(), client_id=None, supports=set())
    sys.path.insert(0, str(CUSTOM_NODES))
    try:
        from RES4LYF import beta
        yield beta
    finally:
        sys.path.remove(str(CUSTOM_NODES))
        if previous_instance is missing:
            del server.PromptServer.instance
        else:
            server.PromptServer.instance = previous_instance


def run_sampler(function, model, x, sigmas, *, seed=719, callback=None, **kwargs):
    torch.manual_seed(seed)
    return function(
        model,
        x,
        sigmas,
        extra_args={"model_options": {"transformer_options": {}}},
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


@pytest.mark.parametrize("kind", ["2s", "2m"])
@pytest.mark.parametrize("model_type", ["eps", "flow"])
def test_nc_skips_only_terminal_clean_latent_correction(res_sampler, reference, kind, model_type):
    baseline_fn = getattr(reference, f"sample_res_{kind}")
    nc_fn = getattr(res_sampler.pack, f"sample_res_{kind}_nc")
    sigmas = torch.tensor([1.0, 0.58, 0.29, 0.0])
    initial = torch.linspace(-0.6, 0.8, 24).reshape(1, 2, 3, 4)
    baseline = res_sampler.Model(model_type)
    nc = res_sampler.Model(model_type)
    baseline_calls, nc_calls = [], []
    clean_conversion = baseline.model_sampling.calculate_denoised
    correction_input = []

    def spy_conversion(sigma, eps, x):
        correction_input.append(x.clone())
        return clean_conversion(sigma, eps, x)

    baseline.model_sampling.calculate_denoised = spy_conversion
    result = run_sampler(baseline_fn, baseline, initial.clone(), sigmas, callback=record_callbacks(baseline_calls))
    no_correction = run_sampler(nc_fn, nc, initial.clone(), sigmas, callback=record_callbacks(nc_calls))

    assert len(correction_input) == 1
    torch.testing.assert_close(no_correction, correction_input[0], rtol=0, atol=0)
    assert not torch.equal(no_correction, result)
    assert len(baseline_calls) == len(nc_calls)
    assert len(baseline.outputs) == len(nc.outputs) and len(baseline.outputs) > 2
    assert len(baseline.calls) == len(nc.calls)
    for base_call, nc_call in zip(baseline.calls, nc.calls):
        for base_tensor, nc_tensor in zip(base_call, nc_call):
            torch.testing.assert_close(base_tensor, nc_tensor, rtol=0, atol=0)
    for base_event, nc_event in zip(baseline_calls, nc_calls):
        assert base_event["i"] == nc_event["i"]
        torch.testing.assert_close(base_event["sigma"], nc_event["sigma"], rtol=0, atol=0)
        torch.testing.assert_close(base_event["denoised"], nc_event["denoised"], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["2s", "2m"])
@pytest.mark.parametrize("model_type", ["eps", "flow"])
def test_sharp_nc_zero_matches_nc_and_sharp_preserves_first_prediction(res_sampler, kind, model_type):
    nc_fn = getattr(res_sampler.pack, f"sample_res_{kind}_nc")
    sharp_fn = getattr(res_sampler.pack, f"sample_res_{kind}_nc_sharp")
    sigmas = torch.tensor([1.0, 0.62, 0.36, 0.17, 0.0])
    initial = torch.linspace(-0.7, 0.9, 24).reshape(1, 2, 3, 4)
    plain_model, zero_model, sharp_model = (res_sampler.Model(model_type) for _ in range(3))
    plain_events, zero_events, sharp_events = [], [], []

    plain = run_sampler(nc_fn, plain_model, initial.clone(), sigmas, callback=record_callbacks(plain_events))
    zero = run_sampler(sharp_fn, zero_model, initial.clone(), sigmas, callback=record_callbacks(zero_events), sharpness=0.0)
    sharp = run_sampler(sharp_fn, sharp_model, initial.clone(), sigmas, callback=record_callbacks(sharp_events), sharpness=0.8)

    torch.testing.assert_close(zero, plain, rtol=0, atol=0)
    assert len(plain_events) == len(zero_events)
    for plain_event, zero_event in zip(plain_events, zero_events):
        torch.testing.assert_close(plain_event["denoised"], zero_event["denoised"], rtol=0, atol=0)
    assert len(sharp_events) == len(plain_events) and len(sharp_events) > 2
    unchanged_calls = 4 if kind == "2s" else 2
    for sharp_call, plain_call in zip(sharp_model.calls[:unchanged_calls], plain_model.calls[:unchanged_calls]):
        for sharp_tensor, plain_tensor in zip(sharp_call, plain_call):
            torch.testing.assert_close(sharp_tensor, plain_tensor, rtol=0, atol=0)
    assert all(torch.isfinite(event["denoised"]).all() for event in sharp_events)
    assert not torch.equal(sharp, zero)


@pytest.mark.parametrize("kind", ["2s", "2m"])
@pytest.mark.parametrize("model_type", ["eps", "flow"])
@pytest.mark.parametrize("sharpness", [None, 0.0, 0.8])
@pytest.mark.parametrize("schedule", [
    [1.0, 0.62, 0.36, 0.17, 0.08, 0.0],
    [0.95, 0.78, 0.65, 0.54, 0.44, 0.35, 0.28, 0.22, 0.17, 0.13, 0.095, 0.07, 0.05, 0.03, 0.02, 0.012, 0.0],
    [1.0, 0.5, 0.2],
    [1.0, 0.8, 0.8, 0.45, 0.05, 0.0],
    [1.0, 0.62, 0.29, 0.005, 0.0],
    [1.0, 0.03, 0.005, 0.0],
    [1.0, 0.0],
    [0.0, 0.01, 0.03, 0.2, 0.5, 1.0, 0.0],
])
def test_bundled_res_matches_installed_res4lyf(res_sampler, reference, kind, model_type, sharpness, schedule):
    name = f"sample_res_{kind}_nc" + ("_sharp" if sharpness is not None else "")
    initial = torch.linspace(-0.6, 0.8, 24).reshape(1, 2, 3, 4)
    sigmas = torch.tensor(schedule)
    original_model, bundled_model = (res_sampler.Model(model_type) for _ in range(2))
    old_events, new_events = [], []
    kwargs = {} if sharpness is None else {"sharpness": sharpness}
    expected = run_sampler(getattr(reference, name), original_model, initial.clone(), sigmas, callback=record_callbacks(old_events), **kwargs)
    result = run_sampler(getattr(res_sampler.pack, name), bundled_model, initial.clone(), sigmas, callback=record_callbacks(new_events), **kwargs)
    assert len(original_model.calls) == len(bundled_model.calls)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
    for actual, expected_call in zip(bundled_model.calls, original_model.calls):
        for a, b in zip(actual, expected_call):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    assert len(new_events) == len(old_events)
    for a, b in zip(new_events, old_events):
        assert a["i"] == b["i"]
        torch.testing.assert_close(a["denoised"], b["denoised"], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kind", ["2s", "2m"])
@pytest.mark.parametrize("model_type", ["eps", "flow"])
def test_bundled_res_seed_repeatability_and_input_preservation(res_sampler, kind, model_type):
    function = getattr(res_sampler.pack, f"sample_res_{kind}_nc_sharp")
    initial = torch.linspace(-0.6, 0.8, 48).reshape(1, 2, 2, 3, 4)
    sigmas = torch.tensor([1.0, 0.62, 0.36, 0.17, 0.0])
    original_x, original_sigmas = initial.clone(), sigmas.clone()
    model = res_sampler.Model(model_type)
    args = {"seed": 719, "model_options": {"transformer_options": {"marker": "unchanged"}}}
    def run(seed):
        torch.manual_seed(seed)
        return function(model, initial, sigmas, extra_args=args, disable=True)
    first, second, changed = run(719), run(719), run(720)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, changed)
    assert torch.isfinite(first).all()
    assert all(call[0].dtype == torch.float32 for call in model.calls)
    torch.testing.assert_close(initial, original_x, rtol=0, atol=0)
    torch.testing.assert_close(sigmas, original_sigmas, rtol=0, atol=0)
    assert args == {"seed": 719, "model_options": {"transformer_options": {"marker": "unchanged"}}}


def test_res_loads_registers_and_runs_without_res4lyf():
    script = textwrap.dedent("""
        import asyncio
        import importlib.abc
        import importlib.util
        import sys
        import types
        import torch
        import comfy.cli_args
        comfy.cli_args.args.cpu = True

        class BlockRES4LYF(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if 'RES4LYF' in fullname:
                    raise ImportError('RES4LYF is intentionally unavailable')
        sys.meta_path.insert(0, BlockRES4LYF())
        spec = importlib.util.spec_from_file_location('standalone_sharp', sys.argv[1])
        pack = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pack
        spec.loader.exec_module(pack)
        import comfy.model_sampling
        import comfy.samplers
        class Flow(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass
        class Model:
            def __init__(self):
                self.inner_model = types.SimpleNamespace(model_patcher=types.SimpleNamespace(get_model_object=lambda name: Flow()))
            def __call__(self, x, sigma, **kwargs):
                return 0.2 * x + 0.08
        asyncio.run(pack.DPMPP2MSharpExtension().on_load())
        for name in ('res_2s_nc', 'res_2m_nc', 'res_2s_nc_sharp', 'res_2m_nc_sharp'):
            provider = pack.SamplerDPMPP_2M_Sharp.execute(0.15, name).result[0]
            standard = comfy.samplers.sampler_object(name)
            assert standard.sampler_function is provider.sampler_function
            output = provider.sampler_function(Model(), torch.arange(24).reshape(1, 2, 3, 4).float() / 24, torch.tensor([1., .6, .3, 0.]), disable=True, **provider.extra_options)
            assert output.shape == (1, 2, 3, 4) and torch.isfinite(output).all()
        assert not any('RES4LYF' in name for name in sys.modules)
    """)
    result = subprocess.run([sys.executable, "-c", script, str(ROOT / "__init__.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
