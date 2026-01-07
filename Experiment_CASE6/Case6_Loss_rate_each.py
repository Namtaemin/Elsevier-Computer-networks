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

    for time_step in range(20):
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
    RL 데이터만 수동으로 클래스 데이터를 이동 및 재구성하고, Loss Rate를 시각화.
    또한 RL에서 최대 중 최소값과 최소 중 최소값을 계산하고 출력.
    박스플롯에 평균값을 세모로 표시 (라벨 없음), 이상치 비활성화.
    """
    # 수동으로 변경할 클래스 맵핑 (하드코딩)
    manual_adjustments = {
        "Class C": "Class A",
        "Class A": "Class B",
        "Class B": "Class C"
    }

    # 수동 정렬 순서 (하드코딩)
    manual_order = ["Class A", "Class B", "Class C"]

    # RL 데이터만 클래스 이동
    if "Class" in rl_details.columns:
        rl_details["Class"] = rl_details["Class"].map(manual_adjustments).fillna(rl_details["Class"])
        print("RL: Classes adjusted according to manual adjustments.")
    
    # RL 데이터만 클래스 순서 재정렬
    if "Class" in rl_details.columns:
        rl_details["Class"] = pd.Categorical(rl_details["Class"], categories=manual_order, ordered=True)
        rl_details.sort_values("Class", inplace=True)
        print("RL: Classes sorted in manual order.")

    # Allocation Percentage 확인 및 Loss Rate 계산
    for details, algorithm in zip([rl_details, dnn_details, pq_details], ["RL", "DNN", "PQ"]):
        if "Loss Rate" not in details.columns:
            details["Loss Rate"] = 100 - details["Allocation Percentage"]

    # RL의 Time별 Loss Rate 최대값과 최소값 계산
    rl_max_by_time = rl_details.groupby("Time")["Loss Rate"].max()
    rl_min_by_time = rl_details.groupby("Time")["Loss Rate"].min()

    # 최대 중 최소값 및 최소 중 최소값 계산
    rl_max_min = rl_max_by_time.min()
    rl_min_min = rl_min_by_time.min()

    # 결과 출력
    print("\nRL Loss Rate Analysis:")
    print(f"  Max Loss (Min of Max): {rl_max_min:.2f}%")
    print(f"  Min Loss (Min of Min): {rl_min_min:.2f}%")

    # 클래스별 평균 Loss Rate 계산 및 출력
    print("\nClass-wise Loss Rate Statistics:")
    for details, algorithm in zip([rl_details, dnn_details, pq_details], ["RL", "DNN", "PQ"]):
        print(f"\n{algorithm} Class-wise Loss Rate:")
        class_loss_means = details.groupby("Class")["Loss Rate"].mean()
        for class_name, loss_mean in class_loss_means.items():
            print(f"  {class_name}: {loss_mean:.2f}%")

    # 전체 알고리즘의 Loss Rate 박스 플롯 생성
    plt.figure(figsize=(10, 6))
    loss_rate_data = [
        rl_details["Loss Rate"].dropna().values,
        dnn_details["Loss Rate"].dropna().values,
        pq_details["Loss Rate"].dropna().values
    ]
    algorithm_labels = ["RL", "DNN", "PQ"]

    # 평균값 계산
    mean_values = [np.mean(data) for data in loss_rate_data]

    # 박스플롯 생성 (이상치 비활성화)
    box_plot = plt.boxplot(loss_rate_data, labels=algorithm_labels, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor="white"),
                           medianprops=dict(color="yellow", linewidth=2),
                           whiskerprops=dict(color="blue", linewidth=1.5))

    # 평균값을 세모로 표시 (라벨 없음)
    for i, mean in enumerate(mean_values):
        plt.plot(i + 1, mean, marker='^', color='green', markersize=10)

    # 그래프 설정
    plt.ylabel("Loss Rate (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # RL 클래스별 Loss Rate 박스 플롯 생성
    unique_classes = rl_details["Class"].unique()
    for class_name in unique_classes:
        class_data = {
            "RL": rl_details[rl_details["Class"] == class_name]["Loss Rate"].dropna().values,
            "DNN": dnn_details[dnn_details["Class"] == class_name]["Loss Rate"].dropna().values,
            "PQ": pq_details[pq_details["Class"] == class_name]["Loss Rate"].dropna().values,
        }

        # 평균값 계산
        mean_values = [np.mean(data) for data in class_data.values()]

        plt.figure(figsize=(10, 10))
        box_plot = plt.boxplot(class_data.values(), labels=class_data.keys(), patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor="white"),
                               medianprops=dict(color="yellow", linewidth=2),
                               whiskerprops=dict(color="blue", linewidth=1.5))

        # 평균값을 세모로 표시 (라벨 없음)
        for i, mean in enumerate(mean_values):
            plt.plot(i + 1, mean, marker='^', color='green', markersize=10)

        # 그래프 설정
        plt.ylabel("Loss Rate (%)", fontsize=14)
        plt.ylim(-5, 105)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
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
    env = TSNQoSEnv(traffic_df_copy_rl, n_clusters=num_clusters, initial_data_size=85, initial_bandwidth=60.62)
    agent = Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE5/trained_policy_model_Case5.pth'
    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}'")
        return

    avg_results_rl, rl_details = rl_simulation(env, agent, simulation_time, 60.62)

    # DNN Simulation
    dnn_model = load_model('/home/scpa/Policy_DDS_TSN/sin/DNN_regression/Experiment_CASE2/DNN_Class63.h5')
    avg_results_dnn, dnn_details = dnn_simulation(dnn_model, traffic_df, batch_bandwidth, simulation_time)

    # PQ Simulation
    avg_results_pq, pq_details = pq_simulation(traffic_df_copy_pq, batch_bandwidth, num_clusters, simulation_time)

    # Analyze and visualize loss rates using details
    analyze_loss_and_visualize(avg_results_rl, avg_results_dnn, avg_results_pq, rl_details, dnn_details, pq_details)


if __name__ == "__main__":
    main()
