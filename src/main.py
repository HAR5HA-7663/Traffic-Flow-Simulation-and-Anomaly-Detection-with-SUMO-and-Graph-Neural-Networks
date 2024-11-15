import torch
from gnn_model import TrafficAnomalyGNN, train
from data_preprocessing import create_graph_from_data
from data_collection import run_sumo_simulation


def main():
    # Step 1: Run SUMO simulation and collect data
    sumo_cmd = ["sumo", "-c", "data/sumo_config/your_simulation.sumocfg"]
    traffic_data = run_sumo_simulation(sumo_cmd)
    traffic_data.to_csv("data/raw_traffic_data.csv", index=False)

    # Step 2: Preprocess data into graph format
    graph_data = create_graph_from_data("data/raw_traffic_data.csv")

    # Step 3: Initialize and train the GNN model
    model = TrafficAnomalyGNN(in_channels=1, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  # Anomaly detection as a regression task

    # Train the model (for example, on normal data)
    epochs = 10
    for epoch in range(epochs):
        loss = train(model, graph_data, optimizer, criterion)
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    main()
