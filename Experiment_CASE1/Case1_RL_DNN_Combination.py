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

        allocation_results.append({"time": time_step, "allocated_resources": info['allocated_resources']})
        state = next_state

        if done:
            break

    # Convert allocation results to detailed DataFrame
    detailed_results = pd.DataFrame([
        {
            "Class": f"Class {chr(65 + res[0])}",
            "Time": step["time"],
            "Allocation Percentage": (res[1] / res[2]) * 100 if res[2] > 0 else 0
        }
        for step in allocation_results for res in step["allocated_resources"]
    ])

    # Compute average results
    avg_results = detailed_results.groupby("Class").mean().reset_index()
    return avg_results, detailed_results



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

        traffic_df_step['Allocation Percentage'] = (traffic_df_step['Allocated Bandwidth'] / traffic_df_step['Resource Usage']) * 100
        allocation_results.append(traffic_df_step)

    detailed_results = pd.concat(allocation_results, ignore_index=True)

    # Calculate average results
    avg_results = detailed_results.groupby("Class")["Allocation Percentage"].mean().reset_index()

    return avg_results, detailed_results

def pq_simulation(traffic_df, batch_bandwidth, num_clusters, simulation_time):
    allocation_results = []

    for time_step in range(simulation_time):
        traffic_df_step = traffic_df[traffic_df["Time"] == time_step]
        if traffic_df_step.empty:
            print(f"[PQ Time {time_step}] No traffic data available.")
            continue

        # Cluster and sort the current batch
        traffic_df_step, cluster_stats = cluster_and_sort(traffic_df_step, num_clusters)

        # Apply Priority Queue algorithm
        traffic_df_step, cluster_stats = allocate_bandwidth_pq(traffic_df_step, cluster_stats, batch_bandwidth)

        traffic_df_step["Allocation Percentage"] = (traffic_df_step["Allocated Bandwidth"] / traffic_df_step["Resource Usage"]) * 100
        allocation_results.append(traffic_df_step)

        print(f"[PQ Time {time_step}]")
        print(cluster_stats)

    detailed_results = pd.concat(allocation_results, ignore_index=True)

    # Calculate average results
    avg_results = detailed_results.groupby("Class")["Allocation Percentage"].mean().reset_index()

    return avg_results, detailed_results


def analyze_and_visualize(avg_results_rl, avg_results_dnn, avg_results_pq, rl_details, dnn_details, pq_details, priority_ranges):
    print("\n[RL Results Before Processing]")
    print(avg_results_rl)

    # 중복 처리: DNN 및 PQ에서 Class 값 확인
    if avg_results_dnn["Class"].duplicated().any():
        avg_results_dnn = avg_results_dnn.groupby("Class").mean().reset_index()
    if avg_results_pq["Class"].duplicated().any():
        avg_results_pq = avg_results_pq.groupby("Class").mean().reset_index()

    # RL 클래스 수동 변경
    manual_adjustments = {
        "Class A": "Class A",
        "Class B": "Class B",
        "Class C": "Class C",
    }
    avg_results_rl["Class"] = avg_results_rl["Class"].map(manual_adjustments).fillna(avg_results_rl["Class"])

    # RL 데이터 수동 정렬
    manual_order = ["Class A", "Class B", "Class C"]
    avg_results_rl = avg_results_rl.set_index("Class").reindex(manual_order).reset_index()

    print("\n[RL Results After Manual Adjustments]")
    print(avg_results_rl)

    # 정렬 기준 설정
    priority_order = sorted(priority_ranges.keys(), key=lambda x: -priority_ranges[x][0])

    # DNN 및 PQ 데이터를 정렬
    avg_results_dnn = avg_results_dnn.set_index("Class").reindex(priority_order).reset_index()
    avg_results_pq = avg_results_pq.set_index("Class").reindex(priority_order).reset_index()

    # 클래스별 최소, 최대, 중간값 계산
    rl_stats = rl_details.groupby("Class")["Allocation Percentage"].agg(["min", "median", "max"]).reindex(manual_order).reset_index()
    dnn_stats = dnn_details.groupby("Class")["Allocation Percentage"].agg(["min", "median", "max"]).reindex(priority_order).reset_index()
    pq_stats = pq_details.groupby("Class")["Allocation Percentage"].agg(["min", "median", "max"]).reindex(priority_order).reset_index()

    print("\n[RL Stats]")
    print(rl_stats)
    print("\n[DNN Stats]")
    print(dnn_stats)
    print("\n[PQ Stats]")
    print(pq_stats)

    # 시각화
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    x = np.arange(len(manual_order))

    # RL 데이터 시각화
    plt.bar(x - bar_width, rl_stats["median"], width=bar_width, label="RL Median", color="skyblue", alpha=0.8)
    plt.vlines(x - bar_width, rl_stats["min"], rl_stats["max"], color="black", alpha=0.6, lw=2, label="RL Min-Max")
    plt.scatter(x - bar_width, rl_stats["median"], color="red", zorder=5)

    # DNN 데이터 시각화
    plt.bar(x, dnn_stats["median"], width=bar_width, label="DNN Median", color="lightgreen", alpha=0.8)
    plt.vlines(x, dnn_stats["min"], dnn_stats["max"], color="black", alpha=0.6, lw=2, label="DNN Min-Max")
    plt.scatter(x, dnn_stats["median"], color="red", zorder=5)

    # PQ 데이터 시각화
    plt.bar(x + bar_width, pq_stats["median"], width=bar_width, label="PQ Median", color="orange", alpha=0.8)
    plt.vlines(x + bar_width, pq_stats["min"], pq_stats["max"], color="black", alpha=0.6, lw=2, label="PQ Min-Max")
    plt.scatter(x + bar_width, pq_stats["median"], color="red", zorder=5)

    # X축, Y축 레이블 및 제목 설정
    plt.xticks(x, manual_order)
    plt.xlabel("Class")
    plt.ylabel("Allocation Percentage (%)")
    plt.title("Allocation Percentage with Median, Min, and Max")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()



def main():
    simulation_time = 100
    lambdas = [50, 30, 20]
    payload_sizes = [500, 400, 1500]
    expected_bandwidths = [0.53, 1.89, 0.05]
    priority_ranges = {"Class A": (0.9, 1.0), "Class B": (0.6, 0.7), "Class C": (0.2, 0.3)}
    is_zero_bandwidth = [False, False, True]
    batch_bandwidth = 59.24
    num_clusters = 3

    # Fix random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate traffic data
    traffic_data = generate_poisson_traffic(
        lambdas, simulation_time, payload_sizes, expected_bandwidths, priority_ranges, is_zero_bandwidth=is_zero_bandwidth
    )
    traffic_df = pd.DataFrame(traffic_data)

    # RL Simulation
    traffic_df_copy_rl = traffic_df.copy()
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
    dnn_model_path = '/home/scpa/Policy_DDS_TSN/sin/DNN_regression/Experiment_CASE1/DNN_Class3.h5'
    dnn_model = load_model(dnn_model_path)
    avg_results_dnn, dnn_details = dnn_simulation(dnn_model, traffic_df.copy(), batch_bandwidth, simulation_time)

    # PQ Simulation
    traffic_df_copy_pq = traffic_df.copy()
    avg_results_pq, pq_details = pq_simulation(traffic_df_copy_pq, batch_bandwidth, num_clusters, simulation_time)

    # Analyze and visualize
    analyze_and_visualize(
        avg_results_rl, rl_details,
        avg_results_dnn, dnn_details,
        avg_results_pq, pq_details,
        priority_ranges
    )

if __name__ == "__main__":
    main()
