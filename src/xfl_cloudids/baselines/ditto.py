from __future__ import annotations


def ditto_personalization_penalty(personal_model, global_parameters, lam: float):
    """Return Ditto's proximal personalized-model regularizer (lambda/2)||v-w||^2."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    import torch
    penalty = torch.zeros((), device=next(personal_model.parameters()).device)
    for personal, global_p in zip(personal_model.parameters(), global_parameters):
        penalty = penalty + torch.sum((personal - global_p.detach().to(personal.device)) ** 2)
    return 0.5 * float(lam) * penalty
