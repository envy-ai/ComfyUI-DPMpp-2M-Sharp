from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_NODES = ROOT.parent


@pytest.fixture(scope="module")
def res_sampler():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    sys.path.insert(0, str(CUSTOM_NODES))
    import server

    class Routes:
        def post(self, path):
            return lambda function: function

        def get(self, path):
            return lambda function: function

    missing = object()
    previous_instance = getattr(server.PromptServer, "instance", missing)
    server.PromptServer.instance = types.SimpleNamespace(routes=Routes(), client_id=None, supports=set())
    import RES4LYF
    import comfy.k_diffusion.sampling as k_sampling
    import comfy.model_sampling

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
            patcher = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=self.diffusion_model))
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
        module=RES4LYF,
        sampling=k_sampling,
        Model=Model,
    )
    if previous_instance is missing:
        del server.PromptServer.instance
    else:
        server.PromptServer.instance = previous_instance
    sys.path.remove(str(CUSTOM_NODES))


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
def test_nc_skips_only_terminal_clean_latent_correction(res_sampler, kind, model_type):
    sampling = res_sampler.sampling
    baseline_fn = getattr(sampling, f"sample_res_{kind}")
    nc_fn = getattr(sampling, f"sample_res_{kind}_nc")
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
    sampling = res_sampler.sampling
    nc_fn = getattr(sampling, f"sample_res_{kind}_nc")
    sharp_fn = getattr(sampling, f"sample_res_{kind}_nc_sharp")
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
@pytest.mark.parametrize("terminal_zero", [False, True])
def test_sharp_nc_partial_run_keeps_original_solver_result(res_sampler, kind, terminal_zero):
    from RES4LYF.beta.rk_sampler_beta import sample_rk_beta

    model_a, model_b = res_sampler.Model(), res_sampler.Model()
    initial = torch.linspace(-0.4, 0.7, 24).reshape(1, 2, 3, 4)
    sigmas = torch.tensor([1.0, 0.5, 0.2, 0.0]) if terminal_zero else torch.tensor([1.0, 0.5, 0.2])
    events_a, events_b = [], []
    original = run_sampler(
        lambda *args, **kwargs: sample_rk_beta(
            *args, rk_type=f"res_{kind}", steps_to_run=1, **kwargs
        ),
        model_a,
        initial.clone(),
        sigmas,
        callback=record_callbacks(events_a),
    )
    sharp = run_sampler(
        lambda *args, **kwargs: sample_rk_beta(
            *args, rk_type=f"res_{kind}", steps_to_run=1,
            extra_options="disable_final_correction\nres_history_sharpness=0.8", **kwargs
        ),
        model_b,
        initial.clone(),
        sigmas,
        callback=record_callbacks(events_b),
    )
    torch.testing.assert_close(sharp, original, rtol=0, atol=0)
    assert len(events_a) == len(events_b)
    assert len(model_a.outputs) == len(model_b.outputs)
    for event_a, event_b in zip(events_a, events_b):
        torch.testing.assert_close(event_a["denoised"], event_b["denoised"], rtol=0, atol=0)
    for call_a, call_b in zip(model_a.calls, model_b.calls):
        for tensor_a, tensor_b in zip(call_a, call_b):
            torch.testing.assert_close(tensor_a, tensor_b, rtol=0, atol=0)
