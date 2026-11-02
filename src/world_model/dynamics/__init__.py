# src/world_model/dynamics/__init__.py
"""Dynamics function models"""
from .mlp import FunctionMLP
from .transformer import FunctionTransformer

# GNN requires torch_geometric - import only when needed
try:
    from .gnn import StructureFunctionGNN
except ImportError:
    StructureFunctionGNN = None
