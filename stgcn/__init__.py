"""A lightweight, faithful implementation of the original ST-GCN model."""

from .graph import Graph
from .model import STGCN

__all__ = ["Graph", "STGCN"]
