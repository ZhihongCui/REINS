# modules/mlp_structure_fn.py

import torch
import torch.nn as nn
from typing import List, Dict


class FunctionMLP(nn.Module):
    """
    MLP-based structure function learner.
    Given a set of cause variables, predicts the effect variable.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super(FunctionMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape [batch_size, input_dim]
        :return: Tensor of shape [batch_size, output_dim]
        """
        return self.model(x)


def prepare_input(cause_vars: List[str], sample: Dict[str, float]) -> torch.Tensor:
    """
    Converts a dict of variable values into a tensor input for the model.

    :param cause_vars: List of variable names serving as input
    :param sample: Dictionary containing all variable values
    :return: 1D Tensor of input values
    """
    values = [sample[var] for var in cause_vars]
    return torch.tensor(values, dtype=torch.float32)


def train_structure_function(model: FunctionMLP,
                             data: List[Dict[str, float]],
                             cause_vars: List[str],
                             effect_var: str,
                             epochs: int = 50,
                             lr: float = 1e-3) -> FunctionMLP:
    """
    Trains the MLP to predict effect_var from cause_vars.

    :param model: The MLP model
    :param data: List of samples (each is a dict of variable values)
    :param cause_vars: List of input variable names
    :param effect_var: The target output variable name
    :param epochs: Training epochs
    :param lr: Learning rate
    :return: Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for sample in data:
            x = prepare_input(cause_vars, sample).unsqueeze(0)  # shape: [1, input_dim]
            y = torch.tensor([[sample[effect_var]]], dtype=torch.float32)  # shape: [1, 1]

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    return model
