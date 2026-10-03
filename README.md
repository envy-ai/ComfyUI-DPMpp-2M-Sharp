# DPM++ 2M Sharp

A better sampler for **Qwen Image 2.1**, with adjustable sharpening of the denoised history used by DPM++ 2M.

The sampler progressively scales the previous denoised prediction before the next multistep update. It preserves the original DPM++ 2M step count and model-call count. Results depend on the prompt, model, and settings; this repository does not include a comparative benchmark.

The same history adjustment is also available as **DPM++ 2M SDE GPU Sharp**, preserving ComfyUI's SDE update and seeded GPU Brownian noise. Its image quality has not been comparatively tested.

## Install

Search for **DPM++ 2M Sharp for Qwen Image 2.1** in ComfyUI Manager, or install from the registry:

```sh
comfy node install dpmpp-2m-sharp
```

For a manual install, run this from your ComfyUI directory:

```sh
git clone https://github.com/envy-ai/ComfyUI-DPMpp-2M-Sharp custom_nodes/ComfyUI-DPMpp-2M-Sharp
```

Restart ComfyUI after installation. The package uses ComfyUI's V3 node API and its existing PyTorch and tqdm dependencies; no additional packages or core modifications are required.

## Use

1. Add **Sampler DPM++ 2M Sharp** from `model/sampling/samplers`.
2. Select `dpmpp_2m_sharp` or `dpmpp_2m_sde_gpu_sharp` using `sampler_name`.
3. Connect its `SAMPLER` output to **SamplerCustom** or **SamplerCustomAdvanced**, using your existing model, conditioning, noise, latent, and sigma schedule.
4. Start with `sharpness = 0.15`. Set it to `0.0` to disable the history adjustment; the SDE variant then matches ordinary DPM++ 2M SDE GPU. Larger values strengthen the history adjustment and may introduce artifacts.

The node ID remains `SamplerDPMPP_2M_Sharp`. Existing workflows default to `dpmpp_2m_sharp`. This package exposes a sampler provider node; it does not add entries to the standard KSampler dropdown.

## License and credits

GPL-3.0; see [LICENSE](LICENSE). The sampler is adapted from [ComfyUI's DPM++ 2M implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/k_diffusion/sampling.py), based on [k-diffusion](https://github.com/crowsonkb/k-diffusion). The k-diffusion MIT notice is included in [LICENSE.k-diffusion](LICENSE.k-diffusion).
