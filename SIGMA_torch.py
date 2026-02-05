from __future__ import annotations
from typing import Callable, Tuple, List, Optional

import torch
Tensor = torch.Tensor


def normalise(image: Tensor, new_min: float, new_max: float) -> Tensor:
    """Normalises image to original bounds [new_min, new_max] using per-tensor min/max."""
    image = image.to(dtype=torch.float32)
    min_val = image.min()
    max_val = image.max()
    scaled = (image - min_val) / (max_val - min_val + 1e-10)
    return scaled * (new_max - new_min) + new_min


def normalise_to_confidence_change(gradient_map: Tensor, confidence_change: Tensor) -> Tensor:
    """Scales gradient map so that sum of pixels equals confidence_change."""
    gradient_sum = gradient_map.sum()
    scaling_factor = confidence_change / (gradient_sum + 1e-10)
    return scaling_factor * gradient_map


@torch.no_grad()
def compute_confidence(model, image, target_class, preprocess_fn):
    model.eval()
    image = image.to(dtype=torch.float32)
    image_scaled = normalise(image, 0.0, 255.0)
    image_processed = preprocess_fn(image_scaled).unsqueeze(0)

    out = model(image_processed)
    logits = out.logits if hasattr(out, "logits") else out
    probs = torch.softmax(logits, dim=1) 
    return probs[0, target_class]


def compute_gradients(model, image, target_class, preprocess_fn):
    model.eval()

    image = image.to(dtype=torch.float32)

    image_scaled = normalise(image, 0.0, 255.0).requires_grad_(True)  

    image_processed = preprocess_fn(image_scaled)                     
    image_expanded = image_processed.unsqueeze(0)               

    preds = model(image_expanded)
    logits = preds.logits if hasattr(preds, "logits") else preds
    confidence = logits[:, target_class]                            

    grad_hwc = torch.autograd.grad(
        outputs=confidence,
        inputs=image_scaled,
        grad_outputs=torch.ones_like(confidence),
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]

    return grad_hwc



@torch.no_grad()
def batched_confidences(model, images, target_class, preprocess_fn):
    model.eval()
    images = images.to(dtype=torch.float32)
    processed = torch.stack([preprocess_fn(normalise(img, 0.0, 255.0)) for img in images], dim=0)

    out = model(processed)
    logits = out.logits if hasattr(out, "logits") else out
    probs = torch.softmax(logits, dim=1)
    return probs[:, target_class]


def single_SIGMA_path(
    image: Tensor,
    model: torch.nn.Module,
    target_class: int,
    beta: float,
    alpha: float,
    epsilon: float,
    preprocess_fn: Callable[[Tensor], Tensor],
    maximum_iterations: int = 1000,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes a SIGMA path:
      returns (attribution_map, confidence_curve, final_estimate_SIGMA)
    """
    device = image.device
    estimate_SIGMA = image.clone()
    attribution = torch.zeros_like(image, dtype=torch.float32, device=device)

    original_confidence = compute_confidence(model, image, target_class, preprocess_fn)
    prev_confidence = original_confidence
    confidence_path: List[Tensor] = [original_confidence]

    step = 0
    while (prev_confidence > epsilon) and (step < maximum_iterations):
        shape = estimate_SIGMA.shape
        pattern = torch.randint(0, 2, shape, device=device, dtype=torch.int32).to(torch.float32) * 255.0

        increase = estimate_SIGMA + beta * pattern
        decrease = estimate_SIGMA - beta * pattern
        batched = torch.stack([increase, decrease], dim=0)

        confidences = batched_confidences(model, batched, target_class, preprocess_fn)
        perturbation_grad = (confidences[0] - confidences[1]) / (2.0 * beta)  # scalar

        estimate_SIGMA = estimate_SIGMA - alpha * perturbation_grad * pattern
        estimate_SIGMA_norm = normalise(estimate_SIGMA, 0.0, 255.0)

        new_confidence = compute_confidence(model, estimate_SIGMA_norm, target_class, preprocess_fn)
        model_gradient = compute_gradients(model, estimate_SIGMA_norm, target_class, preprocess_fn)

        conf_diff = prev_confidence - new_confidence
        attribution = attribution + normalise_to_confidence_change(model_gradient, conf_diff)

        confidence_path.append(new_confidence)
        prev_confidence = new_confidence
        step += 1

    attribution_map = torch.abs(attribution).sum(dim=-1)
    confidence_curve = torch.stack(confidence_path, dim=0)
    return attribution_map, confidence_curve, estimate_SIGMA


def single_SIGMA_path_adaptive(
    image: Tensor,
    model: torch.nn.Module,
    target_class: int,
    alpha_range: Tuple[float, float],
    beta: float,
    epsilon: float,
    preprocess_fn: Callable[[Tensor], Tensor],
    maximum_iterations: int = 1000,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes a SIGMA path with adaptive alpha
      returns (attribution_map, confidence_curve, final_estimate_SIGMA)
    """
    device = image.device
    estimate_SIGMA = image.clone()
    attribution = torch.zeros_like(image, dtype=torch.float32, device=device)

    min_bound = torch.tensor(alpha_range[0], dtype=torch.float32, device=device)
    max_bound = torch.tensor(alpha_range[1], dtype=torch.float32, device=device)

    prev_estimate_SIGMA = image.clone()
    prev_perturbation_grad = torch.zeros_like(image, dtype=torch.float32, device=device)

    original_confidence = compute_confidence(model, image, target_class, preprocess_fn)
    prev_confidence = original_confidence
    confidence_path: List[Tensor] = [original_confidence]

    step = 0
    while (prev_confidence > epsilon) and (step < maximum_iterations):
        shape = estimate_SIGMA.shape
        pattern = torch.randint(0, 2, shape, device=device, dtype=torch.int32).to(torch.float32) * 255.0

        increase = estimate_SIGMA + beta * pattern
        decrease = estimate_SIGMA - beta * pattern
        batched = torch.stack([increase, decrease], dim=0)

        confidences = batched_confidences(model, batched, target_class, preprocess_fn)

        perturbation_grad = ((confidences[0] - confidences[1]) / (2.0 * beta)) * pattern

        delta_x = estimate_SIGMA - prev_estimate_SIGMA
        delta_g = perturbation_grad - prev_perturbation_grad

        numerator = (delta_x * delta_x).sum()
        denominator = (delta_x * delta_g).sum() + 1e-8

        adaptive_alpha = numerator / denominator
        adaptive_alpha_prime = torch.clamp(adaptive_alpha, min=min_bound, max=max_bound)

        updated_estimate_SIGMA = estimate_SIGMA - adaptive_alpha_prime * perturbation_grad
        estimate_SIGMA_norm = normalise(updated_estimate_SIGMA, 0.0, 255.0)

        new_confidence = compute_confidence(model, estimate_SIGMA_norm, target_class, preprocess_fn)
        model_gradient = compute_gradients(model, estimate_SIGMA_norm, target_class, preprocess_fn)

        conf_diff = prev_confidence - new_confidence
        attribution = attribution + normalise_to_confidence_change(model_gradient, conf_diff)

        confidence_path.append(new_confidence)

        prev_estimate_SIGMA = estimate_SIGMA
        prev_perturbation_grad = perturbation_grad
        estimate_SIGMA = updated_estimate_SIGMA
        prev_confidence = new_confidence
        step += 1

    attribution_map = torch.abs(attribution).sum(dim=-1)
    confidence_curve = torch.stack(confidence_path, dim=0)
    return attribution_map, confidence_curve, estimate_SIGMA


def SIGMA_attribution(
    model: torch.nn.Module,
    image: Tensor,
    target_class: int,
    n: int,
    beta: float,
    alpha: float,
    epsilon: float,
    preprocess_fn: Callable[[Tensor], Tensor],
    maximum_iterations: int = 1000,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes SIGMA attribution averaged over n paths (matches TF Code 1 returns):
      returns (avg_attribution_map , error_map, final_SIGMA_img)
    """
    maps: List[Tensor] = []
    final_imgs: List[Tensor] = []

    for _ in range(int(n)):
        attribution_map, _, estimate_SIGMA = single_SIGMA_path(
            image=image,
            model=model,
            target_class=target_class,
            beta=beta,
            alpha=alpha,
            epsilon=epsilon,
            preprocess_fn=preprocess_fn,
            maximum_iterations=maximum_iterations,
        )
        maps.append(attribution_map)
        final_imgs.append(estimate_SIGMA)

    all_maps = torch.stack(maps, dim=0)        
    final_SIGMA_img = torch.stack(final_imgs, 0)

    avg_attribution_map = all_maps.mean(dim=0)

    error_map = all_maps.std(dim=0, correction=0)

    return avg_attribution_map, error_map, final_SIGMA_img


def SIGMA_attribution_adaptive(
    model: torch.nn.Module,
    image: Tensor,
    target_class: int,
    n: int,
    alpha_range: Tuple[float, float],
    beta: float,
    epsilon: float,
    preprocess_fn: Callable[[Tensor], Tensor],
    maximum_iterations: int = 1000,
) -> Tuple[Tensor, List[Tensor], Tensor]:
    """
    Computes SIGMA attribution averaged over n paths with adaptive alpha 
    """
    maps: List[Tensor] = []
    confidences: List[Tensor] = []
    final_imgs: List[Tensor] = []

    for _ in range(int(n)):
        attribution_map, confidence_curve, estimate_SIGMA = single_SIGMA_path_adaptive(
            image=image,
            model=model,
            target_class=target_class,
            alpha_range=alpha_range,
            beta=beta,
            epsilon=epsilon,
            preprocess_fn=preprocess_fn,
            maximum_iterations=maximum_iterations,
        )
        maps.append(attribution_map)
        confidences.append(confidence_curve)
        final_imgs.append(estimate_SIGMA)

    all_maps = torch.stack(maps, dim=0)          
    final_SIGMA_img = torch.stack(final_imgs, 0)
    avg_attribution_map = all_maps.mean(dim=0)
    return avg_attribution_map, confidences, final_SIGMA_img
