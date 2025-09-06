import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import re
import ast

from torch_geometric.data import Data


class PlanFeatureExtractor:
    def __init__(self):
        # Automatically build category mappings
        self.node_type_map = defaultdict(lambda: len(self.node_type_map))
        self.relation_map = defaultdict(lambda: len(self.relation_map))
        self.join_type_map = defaultdict(lambda: len(self.join_type_map))

        # Initialize unknown categories
        self._init_mappings()

    def _init_mappings(self):
        # Predefine common types
        self.node_type_map["Unknown"]  # ID 0
        self.relation_map["Unknown"]  # ID 0
        self.join_type_map["Unknown"]  # ID 0

    def parse_plan_string(self, plan_str):
        """Parse the execution plan string into a Python object."""
        # Method 1: Use ast.literal_eval for safe parsing of Python literals
        try:
            # Clean string: Remove indices at the beginning (e.g., "0    ")
            clean_str = re.sub(r'^\d+\s+', '', plan_str.strip())
            clean_str = re.sub(r'Name:.*dtype: object$', '', clean_str.strip())  # Remove trailing descriptions
            # Use abstract syntax tree for parsing
            return ast.literal_eval(clean_str)
        except (ValueError, SyntaxError) as e:
            print(f"AST parsing failed: {e}")

        # Method 2: Convert to JSON format for parsing
        try:
            # Fix common issues
            json_str = plan_str.replace("'", '"')  # Convert single quotes to double quotes
            json_str = json_str.replace("True", "true").replace("False", "false")  # Boolean values
            json_str = json_str.replace("None", "null")  # None values

            # Try parsing as JSON
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return []

    def extract_features(self, node):
        """Extract feature vector from a single node."""
        # Ensure the node is a dictionary
        if not isinstance(node, dict):
            print(f"Warning: Encountered non-dictionary node: {node}")
            return self.get_default_features()

        # 1. Node type features
        node_type = node.get("Node Type", "Unknown")
        node_type_idx = self.node_type_map[node_type]

        # 2. Relation features
        relation_name = node.get("Relation Name", "Unknown")
        # Handle cases where relation name is a list
        if isinstance(relation_name, list):
            relation_name = relation_name[0] if relation_name else "Unknown"
        relation_idx = self.relation_map[relation_name]

        # 3. Join type features
        join_type = node.get("Join Type", "Unknown")
        join_type_idx = self.join_type_map[join_type]

        # 4. Numerical features (handle missing values)
        startup_cost = float(node.get("Startup Cost", 0.0))
        total_cost = float(node.get("Total Cost", 0.0))
        plan_rows = float(node.get("Plan Rows", 0.0))
        plan_width = float(node.get("Plan Width", 0.0))

        # 5. Boolean features
        parallel_aware = 1 if node.get("Parallel Aware", False) else 0
        has_filter = 1 if "Filter" in node else 0
        has_hash_cond = 1 if "Hash Cond" in node else 0

        # 6. Derived features
        cost_ratio = startup_cost / max(total_cost, 1e-5)  # Avoid division by zero

        return torch.tensor([
            float(node_type_idx),
            float(relation_idx),
            float(join_type_idx),
            startup_cost,
            total_cost,
            plan_rows,
            plan_width,
            float(parallel_aware),
            float(has_filter),
            float(has_hash_cond),
            float(cost_ratio)
        ], dtype=torch.float)

    def get_default_features(self):
        """Return default feature vector."""
        return torch.tensor([
            float(self.node_type_map["Unknown"]),
            float(self.relation_map["Unknown"]),
            float(self.join_type_map["Unknown"]),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=torch.float)

    def build_graph(self, plan_root):
        """Build graph structure from the plan root node."""
        nodes = []  # Node feature list
        edges = []  # Edge list [(parent index, child index)]
        index_counter = 0
        node_queue = [(plan_root, -1)]  # (current node, parent node index)

        while node_queue:
            current_node, parent_idx = node_queue.pop(0)
            current_idx = index_counter
            index_counter += 1

            # Extract features for the current node
            features = self.extract_features(current_node)
            nodes.append(features)

            # Add parent-child edges
            if parent_idx >= 0:
                edges.append((parent_idx, current_idx))

            # Process child nodes - Handle major issues
            if "Plans" in current_node:
                children = current_node["Plans"]

                # Handle different types of child node representations
                if isinstance(children, dict):
                    # If dictionary, treat as a single child node
                    node_queue.append((children, current_idx))
                elif isinstance(children, list):
                    # If list, iterate through all child nodes
                    for child in children:
                        node_queue.append((child, current_idx))
                else:
                    print(f"Warning: Unknown type of child node: {type(children)}")

        # Convert to PyG format
        edge_index = torch.tensor([], dtype=torch.long)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return {
            "x": torch.stack(nodes) if nodes else torch.tensor([]),
            "edge_index": edge_index
        }


class PlanGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        self.conv1 = GATConv(64, hidden_channels, heads=4)
        self.conv2 = GATConv(4 * hidden_channels, hidden_channels)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # Feature encoding
        x = self.encoder(x)

        # GAT processing
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        # Global pooling (graph embedding)
        x = x.unsqueeze(0).permute(0, 2, 1)  # [1, features, nodes]
        x = self.pool(x).squeeze()  # [features]

        return self.fc(x)


if __name__ == '__main__':
    data = pd.read_csv('./data/tpc_c/tpc_c_val.csv')
    # Example usage

    extractor = PlanFeatureExtractor()
    plan_data = data["plan_json"][0]
    print(plan_data)
    parsed_data = extractor.parse_plan_string(plan_data)
    print(f"Parsed data type: {type(parsed_data)}")

    if not isinstance(parsed_data, list) or len(parsed_data) == 0:
        print("Parsing failed or result is not a list")

    # Get plan root node
    if 'Plan' not in parsed_data[0]:
        print("Missing 'Plan' key in data")

    plan_root = parsed_data[0]['Plan']

    # Build graph data
    graph_data = extractor.build_graph(plan_root)

    # Print results
    print("\nGraph structure information:")
    print(f"Number of nodes: {graph_data['x'].shape[0]}")
    print(f"Feature dimensions: {graph_data['x'].shape[1]}")
    if graph_data['edge_index'].numel() > 0:
        print(f"Number of edges: {graph_data['edge_index'].shape[1]}")
    else:
        print("Number of edges: 0")

    # Print node types (for debugging)
    print("\nFirst 5 node types:")
    for i in range(min(5, graph_data['x'].shape[0])):
        # Reverse lookup for node type
        type_id = int(graph_data['x'][i, 0].item())
        for k, v in extractor.node_type_map.items():
            if v == type_id:
                print(f"Node {i}: {k}")
                break
        else:
            print(f"Node {i}: Unknown")

    input_dim = graph_data['x'].shape[1]
    model = PlanGAT(in_channels=input_dim, out_channels=64)

    x = graph_data['x'].float()  # Convert to float32
    edge_index = graph_data['edge_index'].long()  # Ensure long type

    input_data = Data(x=x, edge_index=edge_index)

    model.eval()
    with torch.no_grad():
        embedding = model(input_data)


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, q, output_dim, num_layers=1):
        """
        :param input_size: Number of features per time step (features dimension of each k_i)
        :param hidden_size: LSTM hidden layer dimension
        :param q: Sequence count (the q in k₁, k₂, ..., k_q)
        :param output_dim: Output dimension after linear transformation for each sequence
        :param num_layers: Number of LSTM layers (default is 1)
        """
        super().__init__()
        self.q = q  # Number of sequences (k₁~k_q)
        self.num_layers = num_layers

        # LSTM layer (process each k_i)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Input format: [batch, time_steps, features]
        )

        # Linear transformation layer
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        :param x: Input tensor with shape [batch_size, q, T, input_size]
                  - batch_size: Batch size
                  - q: Number of sequences (number of k)
                  - T: Time steps for each k_i
                  - input_size: Number of features per time step
        :return: Concatenated output h_K, shape [batch_size, q * output_dim]
        """
        batch_size = x.shape[0]
        h_list = []  # Store processed h'_ki for each k_i

        for i in range(self.q):
            # 1. Extract the i-th sequence k_i, shape: [batch_size, T, input_size]
            ki = x[:, i, :, :]

            # 2. Process k_i with LSTM, get final hidden state h_T and cell state c_T
            _, (h_T, _) = self.lstm(ki)

            # 3. Extract the final hidden state from the last layer
            h_ki = h_T[-1, :, :]  # Shape: [batch_size, hidden_size]

            # 4. Linear transformation
            h_prime_ki = self.linear(h_ki)  # Shape: [batch_size, output_dim]

            # 5. Save the result for the current k_i
            h_list.append(h_prime_ki)

        # 6. Concatenate all h'_ki
        h_K = torch.cat(h_list, dim=1)  # Shape: [batch_size, q * output_dim]
        return h_K

