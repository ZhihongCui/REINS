# modules/gnn_structure_fn.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from typing import List, Dict


class StructureFunctionGNN(nn.Module):
    """
    GNN-based structure function learner.
    Given a causal graph, this module learns a function that maps cause variables to effect variables.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, num_heads: int = 4):
        super(StructureFunctionGNN, self).__init__()

        self.gnn1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gnn2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=True)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the GNN model.

        :param data: PyG Data object with `x`, `edge_index`, and optionally `batch`
        :return: Prediction tensor of shape [num_target_nodes, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gnn1(x, edge_index)
        x = self.gnn2(x, edge_index)

        pooled = global_mean_pool(x, batch)
        out = self.regressor(pooled)
        return out


def build_graph_from_variables(variables: Dict[str, float], edge_list: List[List[int]], node_order: List[str]) -> Data:
    """
    Construct a PyG graph from variable dictionary and causal edge list.

    :param variables: Dict mapping variable names to float values (e.g., from a sampled state)
    :param edge_list: List of [source_idx, target_idx] edges
    :param node_order: List of variable names corresponding to node index order
    :return: torch_geometric.data.Data object
    """
    x = torch.tensor([[variables[var]] for var in node_order], dtype=torch.float32)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def train_structure_function(model: StructureFunctionGNN,
                             dataset: List[Data],
                             epochs: int = 50,
                             lr: float = 1e-3,
                             target_fn=lambda x: x.mean()) -> StructureFunctionGNN:
    """
    Train the GNN model to approximate structure functions on a dataset.

    :param model: GNN model
    :param dataset: List of PyG Data samples
    :param epochs: Training epochs
    :param lr: Learning rate
    :param target_fn: Function to extract target from data.x
    :return: Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataset:
            optimizer.zero_grad()

            out = model(data)
            target = target_fn(data.x).unsqueeze(0)  # shape: [1]
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    return model
