# Adapted from ComfyUI-Quadratic-Schedulers; see LICENSE.Quadratic-Schedulers.
import torch


CURVES = {
    "qwen_21_turbo_custom": (0.9786, 0.9750, 0.9520, 0.8903, 0.7898, 0.6505, 0.4724, 0.2556, 0.0),
    "qwen_21_turbo": (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0),
}


def quadratic_scheduler(model_sampling, steps, anchors):
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
