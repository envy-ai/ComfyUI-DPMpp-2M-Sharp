# DPM++ 2M Sharp

A better sampler for **Qwen Image 2.1**, with adjustable sharpening of the denoised history used by DPM++ 2M.

The sampler progressively scales the previous denoised prediction before the next multistep update. It preserves the original DPM++ 2M step count and model-call count. Results depend on the prompt, model, and settings; this repository does not include a comparative benchmark.

The same history adjustment is also available as **DPM++ 2M SDE GPU Sharp**, preserving ComfyUI's SDE update and seeded GPU Brownian noise. Its image quality has not been comparatively tested.

**SEEDS_2 Sharp** applies an analogous adjustment to the difference between its two stage predictions. It leaves the intermediate predictor unscaled and keeps native seeded noise, model-call count, and the final zero-sigma denoising step. At `sharpness = 0.0` it matches native SEEDS_2. This is experimental and has not been comparatively benchmarked.

Experimental RES choices are `res_2s_nc`, `res_2m_nc`, `res_2s_nc_sharp`, and `res_2m_nc_sharp`. All four skip RES4LYF's final clean-latent correction, retaining the latent at the minimum positive sigma with a small amount of residual noise. They keep the minimum-sigma insertion and all positive-sigma model evaluations. RES 2M Sharp scales cached denoised history; RES 2S Sharp scales the first prediction only when combining its two stage predictions, preserving the unscaled intermediate predictor. Their image quality has not been comparatively tested.

## Install

Search for **DPM++ 2M Sharp for Qwen Image 2.1** in ComfyUI Manager, or install from the registry:

```sh
comfy node install dpmpp-2m-sharp
```

For a manual install, run this from your ComfyUI directory:

```sh
git clone https://github.com/envy-ai/ComfyUI-DPMpp-2M-Sharp custom_nodes/ComfyUI-DPMpp-2M-Sharp
```

Restart ComfyUI after installation. The package uses ComfyUI's V3 node API and its existing PyTorch and tqdm dependencies. All sampler choices work without additional node packs or core modifications. The default RES 2S/2M implementation is bundled here; RES4LYF is not required.

## Use

1. Add **Sampler DPM++ 2M Sharp** from `model/sampling/samplers`.
2. Select a DPM++, RES, or SEEDS_2 sampler using `sampler_name`.
3. Connect its `SAMPLER` output to **SamplerCustom** or **SamplerCustomAdvanced**, using your existing model, conditioning, noise, latent, and sigma schedule.
4. Start with `sharpness = 0.15`. Set it to `0.0` to disable the history adjustment; the SDE variant then matches ordinary DPM++ 2M SDE GPU. Larger values strengthen the history adjustment and may introduce artifacts.

For RES, `sharpness = 0.0` makes each Sharp variant match its corresponding `_nc` variant. The two RES choices without `_sharp` ignore the sharpness input. Both Sharp RES choices also omit the final correction.

For manual SEEDS_2 controls, add **Sampler SEEDS_2 Sharp** and connect its `SAMPLER` output to **SamplerCustom** or **SamplerCustomAdvanced**. It retains all native controls: `solver_type` (`phi_1` or `phi_2`), `eta` (default `1.0`), `s_noise` (default `1.0`), and `r` (default `0.5`), and adds `sharpness` (default `0.15`). The native advanced controls retain their usual widget settings. Set `eta = 0.0` for deterministic sampling.

Use SamplerCustomAdvanced's `output` for NC comparisons; `denoised_output` comes from the preview prediction rather than the returned NC latent.

The node ID remains `SamplerDPMPP_2M_Sharp`. Existing workflows default to `dpmpp_2m_sharp`. This package exposes sampler provider nodes and adds `dpmpp_2m_sde_gpu_sharp`, `seeds_2_sharp`, and all four RES choices to the standard KSampler dropdown. Sharp variants use `sharpness = 0.15`. RES retains the original default noise strength of `0.5` for both steps and intermediate stages.

## Tests

From the ComfyUI directory, run `python -m pytest --import-mode=importlib custom_nodes/ComfyUI-DPMpp-2M-Sharp/tests`. Tests cover standalone RES loading and execution, sampler registration, seeded noise, and zero-sharpness behavior. Optional comparisons against the locally updated RES4LYF pack check its original sampling paths across EPS/flow models and several schedules; these comparisons skip when that pack is absent.

## License and credits

The DPM++ and SEEDS_2 modifications are GPL-3.0; see [LICENSE](LICENSE). They are adapted from [ComfyUI's DPM++ 2M and SEEDS_2 implementations](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/k_diffusion/sampling.py), based on [k-diffusion](https://github.com/crowsonkb/k-diffusion). SEEDS is described in [Stochastic Exponential Integrators for Diffusion Models](https://arxiv.org/abs/2305.14267). The k-diffusion MIT notice is included in [LICENSE.k-diffusion](LICENSE.k-diffusion).

The bundled RES implementation in [res.py](res.py) is adapted from the default beta sampler paths in [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF), including the locally added NC/Sharp variants. Its upstream license is preserved in [LICENSE.RES4LYF](LICENSE.RES4LYF): AGPL-3.0 with an additional restriction on commercial services. The rest of RES4LYF's samplers, guides, and options are not bundled.
