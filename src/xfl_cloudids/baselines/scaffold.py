from __future__ import annotations
import numpy as np


def scaffold_corrected_gradient(gradient, server_control, client_control):
    """SCAFFOLD correction g - c_i + c for one parameter tensor."""
    g = np.asarray(gradient, dtype=float)
    c = np.asarray(server_control, dtype=float)
    ci = np.asarray(client_control, dtype=float)
    if g.shape != c.shape or g.shape != ci.shape:
        raise ValueError("gradient/control-variate shapes must match")
    return g - ci + c
