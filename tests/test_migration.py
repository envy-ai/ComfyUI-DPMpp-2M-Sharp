import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import uuid
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_NODES = ROOT.parent
MIGRATED_NAMES = (
    "dpmpp_2m_fixed",
    "dpmpp_2m_sharp_fixed",
    "dpmpp_2m_sde_gpu_sharp_fixed",
    "dpmpp_3m_sde_gpu_sharp_fixed",
)
RESEARCH_ALIASES = dict(zip((
    "research_dpmpp_2m",
    "research_dpmpp_2m_sharp",
    "research_dpmpp_2m_sde_gpu_sharp",
    "research_dpmpp_3m_sde_gpu_sharp",
), MIGRATED_NAMES))


def load_package(path, prefix):
    name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name, module


def load_reference():
    package_name = f"research_reference_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(CUSTOM_NODES / "research-unipc-dpmpp")]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.sampling_ext",
        CUSTOM_NODES / "research-unipc-dpmpp" / "sampling_ext.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return package_name, module


def unload_package(name):
    for module_name in tuple(sys.modules):
        if module_name == name or module_name.startswith(f"{name}."):
            del sys.modules[module_name]


def sampler_attrs(sampling):
    return {name for name in sampling.__dict__ if name.startswith("sample_")}


def model_sampling(kind, comfy_model_sampling):
    if kind == "eps":
        class EpsSampling(comfy_model_sampling.ModelSamplingDiscrete, comfy_model_sampling.EPS):
            pass

        result = EpsSampling()
    else:
        class FlowSampling(comfy_model_sampling.ModelSamplingDiscreteFlow, comfy_model_sampling.CONST):
            pass

        result = FlowSampling()
        result.set_parameters(timesteps=100)
    return result


class Denoiser:
    def __init__(self, sampling):
        patcher = types.SimpleNamespace(get_model_object=lambda name: sampling)
        self.inner_model = types.SimpleNamespace(model_patcher=patcher)

    def __call__(self, x, sigma, **kwargs):
        sigma = sigma.reshape((-1,) + (1,) * (x.ndim - 1))
        return 0.21 * x + 0.035 * sigma + 0.08


class FixedNoise:
    def __init__(self):
        self.calls = 0

    def __call__(self, sigma, sigma_next):
        self.calls += 1
        return torch.full((1, 2, 3, 4), self.calls * 0.03125, dtype=sigma.dtype, device=sigma.device)


def test_standalone_registers_fixed_names_without_research_pack():
    import comfy.cli_args
    import comfy.k_diffusion.sampling as sampling
    import comfy.samplers

    comfy.cli_args.args.cpu = True
    names_before = list(comfy.samplers.SAMPLER_NAMES)
    ksampler_names_before = list(comfy.samplers.KSAMPLER_NAMES)
    core_names = ("dpmpp_2m", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde_gpu")
    core_functions = {name: getattr(sampling, f"sample_{name}") for name in core_names}
    attrs_before = sampler_attrs(sampling)
    pack_name, pack = load_package(ROOT, "sharp_migration_standalone")
    try:
        assert not any("research_unipc_dpmpp" in name for name in sys.modules)
        original_import = __import__

        def reject_research_import(name, *args, **kwargs):
            if "research-unipc-dpmpp" in name or "research_unipc_dpmpp" in name:
                raise AssertionError(f"Standalone pack imported research dependency: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_research_import):
            asyncio.run(pack.DPMPP2MSharpExtension().on_load())
        for name in MIGRATED_NAMES:
            function = getattr(sampling, f"sample_{name}")
            assert name in comfy.samplers.SAMPLER_NAMES
            assert comfy.samplers.SAMPLER_NAMES.count(name) == 1
            assert comfy.samplers.KSAMPLER_NAMES.count(name) == 1
            assert function is getattr(pack, f"sample_{name}")
            assert comfy.samplers.SAMPLER_NAMES.count(name) == 1
            sampler = pack.SamplerDPMPP_2M_Sharp.execute(0.375, name).result[0]
            assert sampler.sampler_function is function
            expected_options = {"sharpness": 0.375} if "_sharp_fixed" in name else {}
            assert sampler.extra_options == expected_options
        options = next(
            input for input in pack.SamplerDPMPP_2M_Sharp.define_schema().inputs
            if input.id == "sampler_name"
        ).options
        assert set(MIGRATED_NAMES) <= set(options)
        assert all(getattr(sampling, f"sample_{name}") is function for name, function in core_functions.items())
        for name in core_names:
            assert comfy.samplers.SAMPLER_NAMES.count(name) == 1
            assert comfy.samplers.KSAMPLER_NAMES.count(name) == 1
        assert not any(name in comfy.samplers.SAMPLER_NAMES for name in (
            "research_dpmpp_2m",
            "research_dpmpp_2m_sharp",
            "research_dpmpp_2m_sde_gpu_sharp",
            "research_dpmpp_3m_sde_gpu_sharp",
        ))
    finally:
        for attr in sampler_attrs(sampling) - attrs_before:
            if hasattr(sampling, attr):
                delattr(sampling, attr)
        comfy.samplers.SAMPLER_NAMES[:] = names_before
        comfy.samplers.KSAMPLER_NAMES[:] = ksampler_names_before
        unload_package(pack_name)


@pytest.mark.parametrize("model_kind", ["eps", "flow"])
@pytest.mark.parametrize("sharpness", [0.0, 0.15])
@pytest.mark.parametrize(
    "name,order,eta",
    [
        ("dpmpp_2m_fixed", 2, 0.0),
        ("dpmpp_2m_sharp_fixed", 2, 0.0),
        ("dpmpp_2m_sde_gpu_sharp_fixed", 2, 1.0),
        ("dpmpp_3m_sde_gpu_sharp_fixed", 3, 1.0),
    ],
)
def test_migrated_matches_research_sampler(name, order, eta, model_kind, sharpness):
    import comfy.cli_args
    import comfy.model_sampling

    comfy.cli_args.args.cpu = True
    if not (CUSTOM_NODES / "research-unipc-dpmpp" / "sampling_ext.py").exists():
        pytest.skip("Optional comparison requires the research sampler pack")
    pack_name, pack = load_package(ROOT, "sharp_migration_compare")
    reference_name, reference = load_reference()
    try:
        sampling = model_sampling(model_kind, comfy.model_sampling)
        sigmas = torch.tensor([1.5, 0.83, 0.46, 0.24, 0.11, 0.0])
        initial = torch.linspace(-0.7, 0.9, 24).reshape(1, 2, 3, 4)
        noise = FixedNoise()
        reference_noise = FixedNoise()
        kwargs = dict(eta=eta, noise_sampler=noise, disable=True)
        if name.endswith("_sharp_fixed"):
            kwargs["sharpness"] = sharpness
        actual = getattr(pack, f"sample_{name}")(
            Denoiser(sampling), initial.clone(), sigmas.clone(), **kwargs
        )
        expected = reference.sample_dpmpp_Nm_sde(
            Denoiser(sampling), initial.clone(), sigmas.clone(), order=order,
            eta=eta, noise_mode="sde",
            sharpness=sharpness if name.endswith("_sharp_fixed") else 0.0,
            noise_sampler=reference_noise, disable=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        assert noise.calls == reference_noise.calls
    finally:
        unload_package(reference_name)
        unload_package(pack_name)


def test_migrated_and_research_packs_load_in_either_order():
    import comfy.cli_args
    import comfy.k_diffusion.sampling as sampling
    import comfy.samplers

    comfy.cli_args.args.cpu = True
    names_before = list(comfy.samplers.SAMPLER_NAMES)
    ksampler_names_before = list(comfy.samplers.KSAMPLER_NAMES)
    base_names = ("dpmpp_2m", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde_gpu")
    base_functions = {name: getattr(sampling, f"sample_{name}") for name in base_names}
    research_path = CUSTOM_NODES / "research-unipc-dpmpp"
    if not (research_path / "sampling_ext.py").exists():
        pytest.skip("Optional load-order comparison requires the research sampler pack")

    for order in (("standalone", "research"), ("research", "standalone")):
        loaded = []
        attrs_before = sampler_attrs(sampling)
        try:
            for pack_name in order:
                if pack_name == "research":
                    name, module = load_package(research_path, "sharp_migration_research")
                else:
                    name, module = load_package(ROOT, "sharp_migration_both")
                    asyncio.run(module.DPMPP2MSharpExtension().on_load())
                loaded.append(name)

            assert all(getattr(sampling, f"sample_{name}") is function for name, function in base_functions.items())
            for name in MIGRATED_NAMES:
                assert comfy.samplers.SAMPLER_NAMES.count(name) == 1
                assert comfy.samplers.KSAMPLER_NAMES.count(name) == 1
            assert len(set(MIGRATED_NAMES)) == len(MIGRATED_NAMES)
            for old, target in RESEARCH_ALIASES.items():
                assert comfy.samplers.SAMPLER_NAMES.count(old) == 1
                alias_sampler = comfy.samplers.ksampler(old).sampler_function
                target_sampler = comfy.samplers.ksampler(target).sampler_function
                sigmas = torch.tensor([1.5, 0.83, 0.46, 0.24, 0.11, 0.0])
                initial = torch.linspace(-0.7, 0.9, 24).reshape(1, 2, 3, 4)
                actual_callbacks, expected_callbacks = [], []
                alias_noise, target_noise = FixedNoise(), FixedNoise()
                denoiser = lambda x, sigma, **kwargs: x * 0.21 + 0.08
                actual = alias_sampler(
                    denoiser, initial.clone(), sigmas, {"seed": 42},
                    actual_callbacks.append, True, noise_sampler=alias_noise,
                )
                expected = target_sampler(
                    denoiser, initial.clone(), sigmas, {"seed": 42},
                    expected_callbacks.append, True, noise_sampler=target_noise,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert alias_noise.calls == target_noise.calls
                assert len(actual_callbacks) == len(expected_callbacks)
                for actual_step, expected_step in zip(actual_callbacks, expected_callbacks):
                    assert actual_step.keys() == expected_step.keys()
                    for key in actual_step:
                        torch.testing.assert_close(actual_step[key], expected_step[key], rtol=0, atol=0)
        finally:
            for attr in sampler_attrs(sampling) - attrs_before:
                if hasattr(sampling, attr):
                    delattr(sampling, attr)
            comfy.samplers.SAMPLER_NAMES[:] = names_before
            comfy.samplers.KSAMPLER_NAMES[:] = ksampler_names_before
            for name in loaded:
                unload_package(name)


def test_research_aliases_resolve_at_call_time_and_forward_arguments(monkeypatch):
    import comfy.cli_args
    import comfy.k_diffusion.sampling as sampling
    import comfy.samplers

    comfy.cli_args.args.cpu = True
    research_path = CUSTOM_NODES / "research-unipc-dpmpp"
    if not (research_path / "sampling_ext.py").exists():
        pytest.skip("Optional alias comparison requires the research sampler pack")
    names_before = list(comfy.samplers.SAMPLER_NAMES)
    attrs_before = sampler_attrs(sampling)
    research_name, research = load_package(research_path, "sharp_alias_dispatch")
    try:
        for old, target in RESEARCH_ALIASES.items():
            alias = research.SAMPLER_FUNCS[old]
            attr = f"sample_{target}"
            with monkeypatch.context() as registry:
                registry.delattr(sampling, attr, raising=False)
                with pytest.raises(RuntimeError, match="requires ComfyUI-DPMpp-2M-Sharp"):
                    alias(None, None, None)
                for result in (object(), object()):
                    with patch.object(sampling, attr, create=True, return_value=result) as function:
                        args = (object(), object(), object(), {"seed": 42}, object(), True)
                        kwargs = dict(eta=0.5, s_noise=0.75, noise_sampler=object(), sharpness=0.3)
                        assert alias(*args, **kwargs) is result
                        function.assert_called_once_with(*args, **kwargs)
                        function.reset_mock()
                        assert alias(*args[:3]) is result
                        function.assert_called_once_with(*args[:3])
    finally:
        for attr in sampler_attrs(sampling) - attrs_before:
            delattr(sampling, attr)
        comfy.samplers.SAMPLER_NAMES[:] = names_before
        unload_package(research_name)
