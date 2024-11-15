import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import torch


def create_graph_from_data(csv_file):
    df = pd.read_csv(csv_file)
    G = nx.Graph()

    # Add nodes and edges
    for _, row in df.iterrows():
        G.add_node(row['lane_id'], speed=row['speed'], position=row['position'])

    # Add edges between lanes (customize based on your road network layout)
    # G.add_edge('lane_1', 'lane_2') - Example

    # Convert to PyTorch Geometric Data
    x = torch.tensor([G.nodes[node]['speed'] for node in G.nodes], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    return Data(x=x, edge_index=edge_index)


if __name__ == "__main__":
    graph_data = create_graph_from_data("data/raw_traffic_data.csv")
    torch.save(graph_data, "data/processed_graph.pt")
