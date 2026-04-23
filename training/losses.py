# training/losses.py
# Loss functions for RA regression model

import torch
import torch.nn as nn


def huber_loss(pred: torch.Tensor, target: torch.Tensor,
               delta: float = 25.0) -> torch.Tensor:
    """
    Huber loss — robust to outlier RA measurements.

    L = 0.5·(pred-target)²       if |pred-target| < δ
      = δ·(|pred-target| - 0.5δ) otherwise

    More robust than MSE for RA measurements which can have
    large outliers due to surface contamination or sensor errors.

    Args:
        pred:   Predicted RA values
        target: Ground truth RA values
        delta:  Transition point from quadratic to linear loss

    Returns:
        Scalar loss value
    """
    diff = torch.abs(pred - target)
    return torch.where(
        diff < delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    ).mean()


def confidence_loss(pred: torch.Tensor, target: torch.Tensor,
                    conf: torch.Tensor, sigma: float = 25.0) -> torch.Tensor:
    """
    Confidence calibration loss.

    c_target = exp(-|RA_pred - RA_true| / σ)
    L_conf = BCE(conf_pred, c_target)

    Forces model to output high confidence when prediction error is small,
    low confidence when prediction error is large.

    Args:
        pred:   Predicted RA values (detached for gradient flow)
        target: Ground truth RA values
        conf:   Predicted confidence scores (0-1)
        sigma:  Scaling parameter for target confidence

    Returns:
        Scalar BCE loss
    """
    c_target = torch.exp(-torch.abs(pred.detach() - target) / sigma)
    return nn.functional.binary_cross_entropy(conf, c_target)


def total_loss(ra_pred: torch.Tensor, ra_true: torch.Tensor,
               conf_pred: torch.Tensor,
               lambda1: float = 0.3, lambda2: float = 0.1) -> torch.Tensor:
    """
    Combined loss function.

    L_total = L_ra + λ₁·L_conf + λ₂·L_consistency

    Where:
        L_ra:          Huber loss on RA predictions
        L_conf:        Confidence calibration loss
        L_consistency: Variance penalty for smooth predictions

    Args:
        ra_pred:    Predicted RA values
        ra_true:    Ground truth RA values
        conf_pred:  Predicted confidence scores
        lambda1:    Weight for confidence loss
        lambda2:    Weight for consistency loss

    Returns:
        Scalar total loss
    """
    l_ra = huber_loss(ra_pred, ra_true)
    l_conf = confidence_loss(ra_pred, ra_true, conf_pred)
    l_cons = ra_pred.var()  # penalize high variance predictions
    return l_ra + lambda1 * l_conf + lambda2 * l_cons
