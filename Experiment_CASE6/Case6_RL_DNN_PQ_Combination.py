import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from Agent import Agent
from EA_3_Class_env import TSNQoSEnv
import os
import Config


def generate_poisson_traffic(lambdas, duration, payload_sizes, expected_bandwidths, priority_ranges, error_margin=0.05, is_zero_bandwidth=None):
    total_data = []
    for second in range(duration):
        for class_idx, (lam, payload, expected_bw) in enumerate(zip(lambdas, payload_sizes, expected_bandwidths)):
            priority_range = priority_ranges[f"Class {chr(65 + class_idx)}"]
            packets = lam
            priorities = np.random.uniform(*priority_range, size=packets)
            bandwidths = [
                max(0, expected_bw * (1 + np.random.uniform(-error_margin, error_margin)))
                for _ in range(packets)
            ] if not is_zero_bandwidth or not is_zero_bandwidth[class_idx] else [0.0] * packets
            for i in range(packets):
                total_data.append({
                    "Time": second,
                    "Class": f"Class {chr(65 + class_idx)}",
                    "Resource Usage": bandwidths[i],
                    "Priority": priorities[i]
                })
    return total_data


def cluster_and_sort(data, num_clusters):
    """Cluster traffic data and sort by priority averages."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Priority', 'Resource Usage']])
    
    # Calculate cluster statistics
    cluster_stats = data.groupby('Cluster').agg({
        'Priority': 'mean',
        'Resource Usage': 'sum'
    }).reset_index()
    
    # Sort clusters by average priority
    cluster_stats = cluster_stats.sort_values(by='Priority', ascending=False).reset_index(drop=True)
    cluster_stats['Class Name'] = cluster_stats.index.map(lambda x: f"Class {chr(65 + x)}")
    
    # Map sorted clusters back to data
    cluster_map = {row['Cluster']: i for i, row in cluster_stats.iterrows()}
    data['Cluster'] = data['Cluster'].map(cluster_map)
    
    return data, cluster_stats


def allocate_bandwidth_pq(data, cluster_stats, total_bandwidth):
    """Allocate bandwidth using Priority Queue algorithm."""
    remaining_bandwidth = total_bandwidth
    data['Allocated Bandwidth'] = 0
    cluster_stats['Allocated Bandwidth'] = 0

    for _, cluster_row in cluster_stats.iterrows():
        cluster_id = cluster_row['Cluster']
        required_bandwidth = cluster_row['Resource Usage']

        if required_bandwidth == 0:
            continue

        # Allocate resources
        if remaining_bandwidth >= required_bandwidth:
            allocated_bandwidth = required_bandwidth
        else:
            allocated_bandwidth = remaining_bandwidth

        scaling_factor = allocated_bandwidth / required_bandwidth
        data.loc[data['Cluster'] == cluster_id, 'Allocated Bandwidth'] = \
            data.loc[data['Cluster'] == cluster_id, 'Resource Usage'] * scaling_factor
        cluster_stats.loc[cluster_stats['Cluster'] == cluster_id, 'Allocated Bandwidth'] = allocated_bandwidth

        remaining_bandwidth -= allocated_bandwidth

        if remaining_bandwidth <= 0:
            break

    return data, cluster_stats


def rl_simulation(env, agent, simulation_time, batch_bandwidth):
    state = env.reset()
    allocation_results = []

    for time_step in range(simulation_time):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.get_action(state_tensor)[0]
        next_state, reward, done, info = env.step(action)

        # Apply bandwidth limit
        allocated_resource_sum = sum([res[1] for res in info['allocated_resources']])
        if allocated_resource_sum > batch_bandwidth:
            scale_factor = batch_bandwidth / allocated_resource_sum
            for res in info['allocated_resources']:
                res[1] *= scale_factor

        allocation_results.append(info)
        state = next_state

        if done:
            break

    grouped_results = pd.DataFrame([
        {"Class": f"Class {chr(65 + res[0])}", "Allocated Bandwidth": res[1], "Resource Usage": res[2]}
        for step in allocation_results for res in step['allocated_resources']
    ])
    avg_results = grouped_results.groupby("Class").mean()
    avg_results["Allocation Percentage"] = (avg_results["Allocated Bandwidth"] / avg_results["Resource Usage"]) * 100
    return avg_results


def dnn_simulation(dnn_model, traffic_df, batch_bandwidth, simulation_time):
    scaler = StandardScaler()
    allocation_results = []

    for time_step in range(simulation_time):
        traffic_df_step = traffic_df[traffic_df['Time'] == time_step]
        if traffic_df_step.empty:
            continue

        scaled_features = scaler.fit_transform(traffic_df_step[['Priority', 'Resource Usage']])
        traffic_df_step['Predicted Bandwidth'] = dnn_model.predict(scaled_features).flatten()

        traffic_df_step['Allocated Bandwidth'] = traffic_df_step.apply(
            lambda row: min(row['Predicted Bandwidth'], row['Resource Usage']), axis=1
        )

        total_allocated = traffic_df_step['Allocated Bandwidth'].sum()
        if total_allocated > batch_bandwidth:
            scale_factor = batch_bandwidth / total_allocated
            traffic_df_step['Allocated Bandwidth'] *= scale_factor

        grouped = traffic_df_step.groupby("Class").agg({
            "Resource Usage": "sum",
            "Allocated Bandwidth": "sum"
        }).reset_index()
        grouped['Time'] = time_step
        grouped['Allocation Percentage'] = (grouped['Allocated Bandwidth'] / grouped['Resource Usage']) * 100
        allocation_results.append(grouped)

    all_results = pd.concat(allocation_results, ignore_index=True)
    avg_results = all_results.groupby("Class").mean()
    return avg_results


def pq_simulation(traffic_df, batch_bandwidth, num_clusters, simulation_time):
    """Run PQ simulation on a per-second basis."""
    allocation_results = []

    for time_step in range(simulation_time):
        # Select traffic data for the current time step
        traffic_df_step = traffic_df[traffic_df["Time"] == time_step]
        if traffic_df_step.empty:
            print(f"[PQ Time {time_step}] No traffic data available.")
            continue

        # Cluster and sort the current batch
        traffic_df_step, cluster_stats = cluster_and_sort(traffic_df_step, num_clusters)

        # Apply Priority Queue algorithm
        traffic_df_step, cluster_stats = allocate_bandwidth_pq(traffic_df_step, cluster_stats, batch_bandwidth)

        # Add allocation statistics for this time step
        cluster_stats['Time'] = time_step
        cluster_stats['Allocation Percentage'] = np.where(
            cluster_stats['Resource Usage'] > 0,
            (cluster_stats['Allocated Bandwidth'] / cluster_stats['Resource Usage']) * 100,
            0
        )

        # Transfer Class C results to Class B if Class B has no data
        if "Class B" in cluster_stats["Class Name"].values and cluster_stats.loc[cluster_stats["Class Name"] == "Class B", "Resource Usage"].values[0] == 0:
            cluster_stats.loc[cluster_stats["Class Name"] == "Class B", ["Allocated Bandwidth", "Resource Usage", "Allocation Percentage"]] = \
                cluster_stats.loc[cluster_stats["Class Name"] == "Class C", ["Allocated Bandwidth", "Resource Usage", "Allocation Percentage"]]
            cluster_stats = cluster_stats[cluster_stats["Class Name"] != "Class C"]  # Remove Class C after transfer

        allocation_results.append(cluster_stats)

        print(f"[PQ Time {time_step}]")
        print(cluster_stats)

    # Combine allocation results and compute averages
    all_results = pd.concat(allocation_results, ignore_index=True)
    avg_results = all_results.groupby("Class Name").mean()
    avg_results.rename_axis("Class", inplace=True)

    return avg_results

def analyze_and_visualize(avg_results_rl, avg_results_dnn, avg_results_pq, priority_ranges):
    """
    RL 데이터는 수동으로 정렬하고, DNN 및 PQ 데이터는 우선순위 기준으로 정렬하여 시각화.
    """
    print("\n[RL Results Before Manual Adjustments]")
    print(avg_results_rl)

    # Add Class column explicitly if missing
    if "Class" not in avg_results_rl.columns:
        avg_results_rl = avg_results_rl.reset_index()

    # 수동으로 RL 데이터 정렬 및 클래스 이동
    manual_adjustments = {
        "Class C": "Class C",
        "Class A": "Class B",
        "Class B": "Class A"
    }
    avg_results_rl["Class"] = avg_results_rl["Class"].map(manual_adjustments).fillna(avg_results_rl["Class"])

    # RL 데이터를 수동 정렬한 후 정렬 기준으로 재구성
    manual_order = ["Class A", "Class B", "Class C"]
    avg_results_rl = avg_results_rl.set_index("Class").reindex(manual_order).reset_index()

    print("\n[RL Results After Manual Adjustments]")
    print(avg_results_rl)

    # DNN 및 PQ 데이터 정리 및 정렬
    if "Class" not in avg_results_dnn.columns:
        avg_results_dnn = avg_results_dnn.reset_index()
    if "Class" not in avg_results_pq.columns:
        avg_results_pq = avg_results_pq.reset_index()

    # Priority Order
    priority_order = sorted(priority_ranges.keys(), key=lambda x: -priority_ranges[x][0])  # 높은 우선순위 기준 정렬
    avg_results_dnn = avg_results_dnn.set_index("Class").reindex(priority_order).reset_index()
    avg_results_pq = avg_results_pq.set_index("Class").reindex(priority_order).reset_index()

    print("\n[DNN Results After Sorting by Priority]")
    print(avg_results_dnn)
    print("\n[PQ Results After Sorting by Priority]")
    print(avg_results_pq)

    # 결과 병합
    combined_results = pd.concat([
        avg_results_rl.assign(Algorithm=""),
        avg_results_dnn.assign(Algorithm="DNN"),
        avg_results_pq.assign(Algorithm="PQ")
    ])

    print("\n[Combined Results Before Pivot]")
    print(combined_results)

    # Pivot for visualization
    pivot_results = combined_results.pivot(index="Class", columns="Algorithm", values="Allocation Percentage")

    # Visualization
    ax = pivot_results.plot(kind="bar", color=["skyblue", "lightgreen", "orange"], alpha=0.8)
    plt.title("Case6: Average Allocated Bandwidth Ratio (RL, DNN, PQ)")
    plt.xlabel("Class")
    plt.ylabel("Average Allocated Bandwidth (%)")
    plt.ylim(0, 103)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(["RL", "DNN", "PQ"])

    # Rotate x-axis labels to horizontal
    plt.xticks(rotation=0)

    plt.show()


def main():
    simulation_time = 100
    classes = ["Class A", "Class B", "Class C"]
    lambdas = [15, 35, 35]
    payload_sizes = [500, 700, 400]
    expected_bandwidths = [0.53, 1.89, 0.55]
    error_margin = 0.05
    priority_ranges = {
        "Class A": (0.9, 1.0),
        "Class B": (0.6, 0.7),
        "Class C": (0.2, 0.3)
    }
    is_zero_bandwidth = [False, False, False]  # Class C의 대역폭은 항상 0

    batch_bandwidth = 44.62
    # 
    num_clusters = 3

    # Fix random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate traffic data
    traffic_data = generate_poisson_traffic(
        lambdas, simulation_time, payload_sizes, expected_bandwidths, priority_ranges, is_zero_bandwidth=is_zero_bandwidth
    )
    traffic_df = pd.DataFrame(traffic_data)
    traffic_df_copy_rl = traffic_df.copy()
    traffic_df_copy_pq = traffic_df.copy()

    # RL Simulation
    env = TSNQoSEnv(traffic_df_copy_rl, n_clusters=num_clusters, initial_data_size=45, initial_bandwidth=44.62)
    agent = Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE6/trained_policy_model_Case6.pth'
    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}'")
        return

    avg_results_rl = rl_simulation(env, agent, simulation_time, 44.62)

    # DNN Simulation
    dnn_model = load_model('/home/scpa/Policy_DDS_TSN/sin/DNN_regression/Experiment_CASE2/DNN_Class63.h5')
    avg_results_dnn = dnn_simulation(dnn_model, traffic_df, batch_bandwidth, simulation_time)

    # PQ Simulation
    avg_results_pq = pq_simulation(traffic_df_copy_pq, batch_bandwidth, num_clusters, simulation_time)

    # Analyze and visualize
    analyze_and_visualize(avg_results_rl, avg_results_dnn, avg_results_pq, priority_ranges)

if __name__ == "__main__":
    main()
