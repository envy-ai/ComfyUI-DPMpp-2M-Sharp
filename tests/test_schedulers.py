import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CURVES = {
    "qwen_21_turbo_custom": (0.9786, 0.9750, 0.9520, 0.8903, 0.7898, 0.6505, 0.4724, 0.2556, 0.0),
    "qwen_21_turbo": (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0),
}


def original_schedule(steps, anchors):
    # Frozen ComfyUI-Quadratic-Schedulers implementation, independent of that pack.
    slopes = [0.0]
    for start, end in zip(anchors, anchors[1:]):
        slopes.append(2 * (end - start) - slopes[-1])

    sigmas = []
    intervals = len(anchors) - 1
    for step in range(steps):
        position = step * intervals / steps
        interval = int(position)
        fraction = position - interval
        start = anchors[interval]
        slope = slopes[interval]
        curvature = anchors[interval + 1] - start - slope
        sigmas.append(start + fraction * (slope + fraction * curvature))
    sigmas.append(0.0)
    return torch.tensor(sigmas, dtype=torch.float32, device="cpu")


@pytest.fixture
def scheduler_pack():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.k_diffusion.sampling as sampling
    import comfy.samplers

    names_before = list(comfy.samplers.SCHEDULER_NAMES)
    handlers_before = dict(comfy.samplers.SCHEDULER_HANDLERS)
    samplers_before = list(comfy.samplers.SAMPLER_NAMES)
    ksamplers_before = list(comfy.samplers.KSAMPLER_NAMES)
    module_name = "sharp_scheduler_tests"
    original_import = __import__

    def reject_old_pack(name, *args, **kwargs):
        if "Quadratic-Schedulers" in name or "quadratic_schedulers" in name:
            raise AssertionError(f"Imported old scheduler pack: {name}")
        return original_import(name, *args, **kwargs)

    try:
        for name in EXPECTED_CURVES:
            comfy.samplers.SCHEDULER_HANDLERS.pop(name, None)
            comfy.samplers.SCHEDULER_NAMES[:] = [item for item in comfy.samplers.SCHEDULER_NAMES if item != name]
        with patch.dict(sampling.__dict__), patch("builtins.__import__", side_effect=reject_old_pack):
            spec = importlib.util.spec_from_file_location(module_name, ROOT / "__init__.py")
            pack = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = pack
            spec.loader.exec_module(pack)
            assert all(name not in comfy.samplers.SCHEDULER_HANDLERS for name in EXPECTED_CURVES)
            asyncio.run(pack.DPMPP2MSharpExtension().on_load())
            yield pack
    finally:
        comfy.samplers.SCHEDULER_NAMES[:] = names_before
        comfy.samplers.SCHEDULER_HANDLERS.clear()
        comfy.samplers.SCHEDULER_HANDLERS.update(handlers_before)
        comfy.samplers.SAMPLER_NAMES[:] = samplers_before
        comfy.samplers.KSAMPLER_NAMES[:] = ksamplers_before
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                del sys.modules[name]


@pytest.mark.parametrize("name", EXPECTED_CURVES)
def test_eight_steps_match_exact_anchors_and_standard_ksampler(scheduler_pack, name):
    import comfy.samplers

    assert scheduler_pack.CURVES == EXPECTED_CURVES
    model_sampling = SimpleNamespace(sigma_min=0.01, sigma_max=1000.0)
    model = SimpleNamespace(get_model_object=lambda name: model_sampling)
    sampler = comfy.samplers.KSampler(model, 8, "cpu", sampler="euler", scheduler=name)
    torch.testing.assert_close(
        sampler.sigmas, torch.tensor(EXPECTED_CURVES[name], dtype=torch.float32), rtol=0, atol=0
    )
    assert comfy.samplers.KSampler.SCHEDULERS.count(name) == 1


@pytest.mark.parametrize("name", EXPECTED_CURVES)
@pytest.mark.parametrize("steps", [0, 1, 2, 3, 7, 8, 9, 17, 31, 127, 512])
def test_step_counts_match_original_without_old_pack(scheduler_pack, name, steps):
    import comfy.samplers

    anchors = EXPECTED_CURVES[name]
    actual = comfy.samplers.calculate_sigmas(object(), name, steps)
    torch.testing.assert_close(actual, original_schedule(steps, anchors), rtol=0, atol=0)
    assert actual.shape == (steps + 1,)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert (actual[:-1] >= actual[1:]).all()
    assert actual[-1] == 0
    assert actual[0] == torch.tensor(anchors[0] if steps else 0.0, dtype=torch.float32)


def test_on_load_is_idempotent_and_preserves_existing_handlers(scheduler_pack):
    import comfy.samplers

    handlers = dict(comfy.samplers.SCHEDULER_HANDLERS)
    names = list(comfy.samplers.SCHEDULER_NAMES)
    extension = scheduler_pack.DPMPP2MSharpExtension()
    asyncio.run(extension.on_load())
    asyncio.run(extension.on_load())
    assert comfy.samplers.SCHEDULER_NAMES == names
    for name, handler in handlers.items():
        assert comfy.samplers.SCHEDULER_HANDLERS[name] is handler

    # An already loaded scheduler pack owns its handlers regardless of load order.
    for name in EXPECTED_CURVES:
        handler = comfy.samplers.SchedulerHandler(lambda model, steps: torch.zeros(steps + 1))
        comfy.samplers.SCHEDULER_HANDLERS[name] = handler
        asyncio.run(extension.on_load())
        assert comfy.samplers.SCHEDULER_HANDLERS[name] is handler
        assert comfy.samplers.SCHEDULER_NAMES.count(name) == 1
