# Traffic Flow Simulation and Anomaly Detection with SUMO and Graph Neural Networks (GNNs)

## Project Overview
This project simulates traffic flow using **SUMO** (Simulation of Urban MObility) and detects traffic anomalies using a **Graph Neural Network (GNN)**. By modeling traffic patterns in a simulated environment, this system learns normal traffic behaviors and identifies unusual conditions, such as congestion or accidents.



## Requirements
- **SUMO**: Traffic simulation software ([Eclipse SUMO](https://www.eclipse.org/sumo/))
- **Python 3.7+**
- **Python Libraries**: Listed in `requirements.txt`

To install required libraries, run:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Project Workflow

### 1. Traffic Flow Simulation with SUMO
   - **Setup**: Create a road network and define vehicle routes in the `sumo_config` folder.
   - **Run Simulation**: `data_collection.py` uses SUMO’s TraCI interface to run the simulation and collect traffic data such as speed, density, and vehicle counts.

### 2. Data Collection
   - **Generate Traffic Data**: Run `data_collection.py` to start SUMO, simulate traffic scenarios, and export traffic metrics to `data/raw_traffic_data.csv`.

### 3. Data Preprocessing for GNN
   - **Graph Representation**: `data_preprocessing.py` converts traffic data into a graph where intersections or road segments are nodes, and roads are edges.
   - **Graph Format for GNN**: The processed graph data is saved for use in the GNN model.

### 4. Anomaly Detection with Graph Neural Network (GNN)
   - **GNN Model**: `gnn_model.py` defines a GNN to analyze traffic patterns and detect anomalies.
   - **Training**: Run `main.py` to load the traffic data, train the GNN model on normal traffic patterns, and identify unusual behaviors.

## Running the Project
1. **Simulate and Collect Traffic Data**:
   \`\`\`bash
   python src/data_collection.py
   \`\`\`
   This generates traffic data and saves it in `data/raw_traffic_data.csv`.

2. **Preprocess Data for GNN**:
   \`\`\`bash
   python src/data_preprocessing.py
   \`\`\`
   This script converts traffic data into a graph format and saves it as `data/processed_graph.pt`.

3. **Train the Anomaly Detection Model**:
   \`\`\`bash
   python src/main.py
   \`\`\`
   The `main.py` script integrates data collection, preprocessing, and model training for anomaly detection, outputting training loss per epoch.

## Expected Results
The GNN model should identify anomalies such as:
- Unusual vehicle speeds
- High traffic density
- Simulated incidents like congestion or accidents

## Future Extensions
1. **Real-World Data Integration**: Integrate real-time traffic data to extend the model to real-world applications.
2. **Complex Scenarios**: Simulate advanced traffic scenarios, such as multi-lane intersections and roundabouts.
3. **Domain Adaptation**: Adapt the model for real-world environments with transfer learning techniques.

## Troubleshooting
- **SUMO Errors**: Ensure SUMO is installed correctly, and paths in `sumo_cmd` are accurate.
- **Dependency Issues**: Check `requirements.txt` for any missing libraries.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
