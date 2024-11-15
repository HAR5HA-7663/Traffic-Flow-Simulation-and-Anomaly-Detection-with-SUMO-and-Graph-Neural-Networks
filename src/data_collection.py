import traci
import pandas as pd


def run_sumo_simulation(sumo_cmd, steps=1000):
    traci.start(sumo_cmd)
    traffic_data = []

    for step in range(steps):
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:
            speed = traci.vehicle.getSpeed(vehicle_id)
            pos = traci.vehicle.getPosition(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)

            traffic_data.append({
                "vehicle_id": vehicle_id,
                "speed": speed,
                "position": pos,
                "lane_id": lane_id
            })

    traci.close()
    return pd.DataFrame(traffic_data)


if __name__ == "__main__":
    sumo_cmd = ["sumo", "-c", "data/sumo_config/your_simulation.sumocfg"]
    traffic_data = run_sumo_simulation(sumo_cmd)
    traffic_data.to_csv("data/raw_traffic_data.csv", index=False)
