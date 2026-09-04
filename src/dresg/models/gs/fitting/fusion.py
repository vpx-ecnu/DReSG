"""Multi-view appearance-gradient fusion rules."""

from __future__ import annotations

import torch

APPEARANCE_UPDATE_RULES = frozenset({"standard", "pcgrad"})


def _project_conflicting_appearance_gradients(
    grad_stack: torch.Tensor,
) -> torch.Tensor:
    if grad_stack.shape[0] == 1:
        return grad_stack[0]
    original_shape = grad_stack.shape[1:]
    references = grad_stack.detach().reshape(grad_stack.shape[0], -1).float()
    projected = references.clone()
    eps = (references.norm(dim=1).mean() * 1.0e-8).clamp_min(1.0e-12)
    denominators = references.square().sum(dim=1).clamp_min(eps)
    gradient_indices = torch.arange(
        references.shape[0],
        device=references.device,
    )
    for reference_index in range(references.shape[0]):
        reference = references[reference_index]
        dots = torch.mv(projected, reference)
        conflicts = (dots < 0.0) & (gradient_indices != reference_index)
        coefficients = torch.where(
            conflicts,
            dots / denominators[reference_index],
            torch.zeros_like(dots),
        )
        projected.addcmul_(
            coefficients.unsqueeze(1),
            reference.unsqueeze(0),
            value=-1.0,
        )
    return projected.mean(dim=0).reshape(original_shape).to(grad_stack)


def fuse_appearance_gradients(
    grad_stack: torch.Tensor,
    *,
    rule: str,
) -> torch.Tensor:
    """Fuse one non-empty stack of per-view appearance gradients."""
    if rule == "standard":
        return grad_stack.mean(dim=0)
    if rule == "pcgrad":
        return _project_conflicting_appearance_gradients(grad_stack)
    raise ValueError(
        f"Unsupported appearance_update.rule: {rule}. "
        f"Supported: {', '.join(sorted(APPEARANCE_UPDATE_RULES))}"
    )
