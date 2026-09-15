from __future__ import annotations


def fedprox_proximal_penalty(model, global_parameters, mu: float):
    """Return (mu/2)||w-w_global||^2 for a PyTorch model without altering gradients."""
    if mu < 0:
        raise ValueError("mu must be non-negative")
    import torch
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for local, global_p in zip(model.parameters(), global_parameters):
        penalty = penalty + torch.sum((local - global_p.detach().to(local.device)) ** 2)
    return 0.5 * float(mu) * penalty
