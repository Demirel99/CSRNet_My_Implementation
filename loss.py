# loss.py

import torch.nn as nn

# Section 3.2.3 of the paper describes the loss function as the Euclidean distance
# between the estimated density map and the ground truth density map.
# L(Θ) = (1 / 2N) * Σ ||Z(X_i; Θ) - Z_i^GT||^2
# This is equivalent to the Mean Squared Error (MSE) Loss.
# The 1/2 factor is a constant that doesn't affect optimization, so we can
# use the standard MSELoss implementation from PyTorch.

# We use reduction='sum' to sum the squared errors across all pixels and then
# divide by the batch size ourselves to closely match the paper's formula.
# Using reduction='mean' is also a common and valid alternative.
loss_fn = nn.MSELoss(reduction='sum')