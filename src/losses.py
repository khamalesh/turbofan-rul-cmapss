import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.5, delta=1.0, penalize_under=False, under_penalty_weight=1.5):
        """
        Composite loss combining MSE and Huber (SmoothL1) loss.

        Args:
            alpha (float): Weight for MSE component. (1 - alpha) is weight for Huber.
            delta (float): Huber loss threshold (PyTorch ≥1.10 uses 'beta').
            penalize_under (bool): If True, adds extra penalty for under-predictions.
            under_penalty_weight (float): Scaling factor for under-prediction penalty.
        """
        super().__init__()
        self.alpha = alpha
        self.penalize_under = penalize_under
        self.under_penalty_weight = under_penalty_weight
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss(beta=delta)

    def forward(self, preds, targets):
        # Ensure float32 precision for safety with AMP/mixed precision training
        preds = preds.float()
        targets = targets.float()

        loss_mse = self.mse(preds, targets)
        loss_huber = self.huber(preds, targets)
        total_loss = self.alpha * loss_mse + (1 - self.alpha) * loss_huber

        # Optional: penalize under-predictions (predicted RUL < actual)
        if self.penalize_under:
            under_mask = preds < targets
            if under_mask.any():
                under_penalty = torch.mean((targets[under_mask] - preds[under_mask]) ** 2)
                total_loss += self.under_penalty_weight * under_penalty

        return total_loss
