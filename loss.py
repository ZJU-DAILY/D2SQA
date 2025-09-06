import torch
import torch.nn as nn
import numpy as np


class EnhancedVACCLoss(nn.Module):
    def __init__(self, epsilon=0.1, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Regression loss - Use MAE to focus more on absolute errors
        self.regression_loss = nn.L1Loss()
        # 1. Define the pos_weight vector
        pos_weight = torch.tensor(
            [1.862, 5.826, 1.989, 1.726, 4.840, 1.626, 5.691, 1.661, 1.611],
            device="cuda:0"  # Ensure it matches the device (CPU/GPU) of the model and data to avoid computation errors
        )
        # Binary classification loss - Use BCE to directly optimize classification accuracy
        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, preds, labels, opt_labels):
        """
        preds: Model predictions [batch, num_labels]
        labels: Original multi-label classification labels (not used for binary classification)
        opt_labels: Original impact values
        """
        batch_size = preds.size(0)

        # 1. Dynamically generate binary classification targets (valid root causes = opt_labels > epsilon)
        binary_targets = (opt_labels > self.epsilon).float()

        # 2. Binary classification loss - Directly optimize valid root cause judgment
        binary_loss = self.binary_loss(preds, binary_targets)
        regression_loss = 0
        threshold_loss = 0
        # Combined loss
        total_loss = (
                self.alpha * binary_loss +
                self.beta * regression_loss +
                self.gamma * threshold_loss
        )

        return total_loss, binary_loss, regression_loss, threshold_loss