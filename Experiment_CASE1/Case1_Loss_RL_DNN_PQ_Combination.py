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

        # Add time information to allocation results
        for res in info['allocated_resources']:
            allocation_results.append({
                "Class": f"Class {chr(65 + res[0])}",
                "Time": time_step,
                "Allocated Bandwidth": res[1],
                "Resource Usage": res[2]
            })

        state = next_state

        if done:
            break

    details = pd.DataFrame(allocation_results)
    # Calculate Allocation Percentage
    details["Allocation Percentage"] = (details["Allocated Bandwidth"] / details["Resource Usage"]) * 100
    avg_results = details.groupby("Class").mean()

    return avg_results, details


def dnn_simulation(dnn_model, traffic_df, batch_bandwidth, simulation_time):
    """
    DNN 시뮬레이션 수행 및 디버깅용 출력 추가
    """
    scaler = StandardScaler()
    allocation_results = []

    for time_step in range(simulation_time):
        traffic_df_step = traffic_df[traffic_df['Time'] == time_step]
        if traffic_df_step.empty:
            print(f"Warning: No traffic data for time step {time_step}.")
            continue

        scaled_features = scaler.fit_transform(traffic_df_step[['Priority', 'Resource Usage']])
        predictions = dnn_model.predict(scaled_features).flatten()
        
        # Debugging predicted bandwidth
        print(f"Time Step {time_step}: Predicted Bandwidths (First 5): {predictions[:5]}")

        traffic_df_step['Predicted Bandwidth'] = predictions

        traffic_df_step['Allocated Bandwidth'] = traffic_df_step.apply(
            lambda row: min(row['Predicted Bandwidth'], row['Resource Usage']), axis=1
        )

        total_allocated = traffic_df_step['Allocated Bandwidth'].sum()
        if total_allocated > batch_bandwidth:
            scale_factor = batch_bandwidth / total_allocated
            traffic_df_step['Allocated Bandwidth'] *= scale_factor

        for _, row in traffic_df_step.iterrows():
            allocation_results.append({
                "Class": row["Class"],
                "Time": time_step,
                "Allocated Bandwidth": row["Allocated Bandwidth"],
                "Resource Usage": row["Resource Usage"]
            })

    details = pd.DataFrame(allocation_results)

    # Debugging Allocation Details
    print("\nDNN Simulation Allocation Details (First 5 Rows):")
    print(details.head())

    # Calculate Allocation Percentage
    details["Allocation Percentage"] = details.apply(
        lambda row: (row["Allocated Bandwidth"] / row["Resource Usage"]) * 100 if row["Resource Usage"] > 0 else 0,
        axis=1
    )

    avg_results = details.groupby("Class").mean()

    return avg_results, details
def pq_simulation(traffic_df, batch_bandwidth, num_clusters, simulation_time):
    allocation_results = []

    for time_step in range(simulation_time):
        # Select traffic data for the current time step
        traffic_df_step = traffic_df[traffic_df["Time"] == time_step]
        if traffic_df_step.empty:
            continue

        # Cluster and sort the current batch
        traffic_df_step, cluster_stats = cluster_and_sort(traffic_df_step, num_clusters)

        # Apply Priority Queue algorithm
        traffic_df_step, cluster_stats = allocate_bandwidth_pq(traffic_df_step, cluster_stats, batch_bandwidth)

        for _, row in cluster_stats.iterrows():
            allocation_results.append({
                "Class": row["Class Name"],
                "Time": time_step,
                "Allocated Bandwidth": row["Allocated Bandwidth"],
                "Resource Usage": row["Resource Usage"]
            })

    details = pd.DataFrame(allocation_results)
    # Calculate Allocation Percentage
    details["Allocation Percentage"] = (details["Allocated Bandwidth"] / details["Resource Usage"]) * 100
    avg_results = details.groupby("Class").mean()

    return avg_results, details


def analyze_loss_and_visualize(avg_results_rl, avg_results_dnn, avg_results_pq, rl_details, dnn_details, pq_details):
    """
    각 알고리즘의 전체 Loss Rate을 Box Plot으로 시각화하고 최대값, 최소값, 평균값 및 평균의 최대값과 최소값을 계산해 표시.
    """
    # Allocation Percentage 확인 및 Loss Rate 계산
    if "Allocation Percentage" in rl_details.columns:
        rl_details["Loss Rate"] = 100 - rl_details["Allocation Percentage"]
    else:
        print("RL details are missing 'Allocation Percentage'.")
        print(rl_details.head())
        return
    
    if "Allocation Percentage" in dnn_details.columns:
        dnn_details["Loss Rate"] = 100 - dnn_details["Allocation Percentage"]
    else:
        print("DNN details are missing 'Allocation Percentage'.")
        print(dnn_details.head())
        return
    
    if "Allocation Percentage" in pq_details.columns:
        pq_details["Loss Rate"] = 100 - pq_details["Allocation Percentage"]
    else:
        print("PQ details are missing 'Allocation Percentage'.")
        print(pq_details.head())
        return

    # Box Plot 데이터 준비
    loss_rate_data = [
        rl_details["Loss Rate"].dropna().values,
        dnn_details["Loss Rate"].dropna().values,
        pq_details["Loss Rate"].dropna().values
    ]
    algorithm_labels = ["RL", "DNN", "PQ"]

    # 최대값, 최소값, 평균값 계산 및 출력
    print("\nLoss Rate Statistics:")
    mean_values = []
    for label, loss_data in zip(algorithm_labels, loss_rate_data):
        if len(loss_data) > 0:
            max_loss = np.max(loss_data)
            min_loss = np.min(loss_data)
            mean_loss = np.mean(loss_data)
            mean_values.append(mean_loss)
            print(f"{label} - Max: {max_loss:.2f}, Min: {min_loss:.2f}, Mean: {mean_loss:.2f}")
        else:
            print(f"{label} - No data available.")

    # 평균의 최대값과 최소값
    if mean_values:
        max_mean = np.max(mean_values)
        min_mean = np.min(mean_values)
        print(f"\nMean Loss Rate - Max: {max_mean:.2f}, Min: {min_mean:.2f}")
    else:
        print("\nNo mean values available.")

    # Box Plot 생성
    plt.figure(figsize=(10, 6))
    plt.boxplot(loss_rate_data, labels=algorithm_labels, patch_artist=True,
                boxprops=dict(facecolor="white"),
                medianprops=dict(color="yellow", linewidth=2),
                whiskerprops=dict(color="blue", linewidth=1.5))

    # 평균값 표시 (세모)
    for i, loss_data in enumerate(loss_rate_data):
        if len(loss_data) > 0:
            mean_loss = np.mean(loss_data)
            plt.plot(i + 1, mean_loss, marker='^', color='green', markersize=10)

    # 그래프 설정
    plt.title("Loss Rate Distribution by Algorithm", fontsize=16)
    plt.xlabel("Algorithm", fontsize=14)
    plt.ylabel("Loss Rate (%)", fontsize=14)
    plt.ylim(0, 103)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()



def main():
    simulation_time = 100
    lambdas = [50, 30, 20]
    payload_sizes = [500, 400, 1500]
    expected_bandwidths = [0.53, 1.89, 0.05]
    priority_ranges = {"Class A": (0.9, 1.0), "Class B": (0.6, 0.7), "Class C": (0.2, 0.3)}
    is_zero_bandwidth = [False, False, False]
    batch_bandwidth = 80
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
    env = TSNQoSEnv(traffic_df_copy_rl, n_clusters=num_clusters, initial_data_size=100, initial_bandwidth=batch_bandwidth)
    agent = Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/trained_policy_model_After_Case1.pth'
    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}'")
        return

    avg_results_rl, rl_details = rl_simulation(env, agent, simulation_time, batch_bandwidth)

    # DNN Simulation
    dnn_model = load_model('/home/scpa/Policy_DDS_TSN/sin/DNN_regression/Experiment_CASE1/DNN_Class3.h5')
    avg_results_dnn, dnn_details = dnn_simulation(dnn_model, traffic_df, batch_bandwidth, simulation_time)

    # PQ Simulation
    avg_results_pq, pq_details = pq_simulation(traffic_df_copy_pq, batch_bandwidth, num_clusters, simulation_time)

    # Analyze and visualize loss rates using details
    analyze_loss_and_visualize(avg_results_rl, avg_results_dnn, avg_results_pq, rl_details, dnn_details, pq_details)


if __name__ == "__main__":
    main()
