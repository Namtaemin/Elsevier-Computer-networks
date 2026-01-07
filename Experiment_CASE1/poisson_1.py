import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Config
import torch
import os
from Agent import Agent  # Agent 클래스 정의 필요
from EA_3_Class_env import TSNQoSEnv  # 환경 클래스 정의 필요

def generate_poisson_traffic(lam, duration, payload_size, expected_bandwidth, error_margin=0.05):
    """
    Poisson 분포 기반 트래픽 생성
    :param lam: 패킷 발생률 λ (패킷/초)
    :param duration: 시뮬레이션 시간 (초)
    :param payload_size: 각 패킷의 크기 (bytes)
    :param expected_bandwidth: 예측 대역폭 (Mbps)
    :param error_margin: 예측 대역폭에 대한 오차 범위 (±%)
    :return: 초당 대역폭 리스트 (Mbps)
    """
    packets_per_second = np.random.poisson(lam=lam, size=duration)
    bandwidth_per_second = [(packets * (payload_size * 8)) / 1e6 for packets in packets_per_second]
    adjusted_bandwidth = [bw + np.random.uniform(-error_margin, error_margin) * expected_bandwidth for bw in bandwidth_per_second]
    return adjusted_bandwidth

def main():
    # 트래픽 시나리오 설정
    simulation_time = 100
    classes = ["Class A", "Class B", "Class C"]
    lambdas = [50, 30, 30]
    payload_sizes = [500, 400, 1500]
    expected_bandwidths = [0.53, 1.89, 0.0]
    error_margin = 0.05
    priority_ranges = {
        "Class A": (0.9, 1.0),
        "Class B": (0.5, 0.9),
        "Class C": (0.3, 0.5)
    }

    # 트래픽 데이터 생성
    traffic_data = []
    for i, (class_name, lam, payload, expected_bw) in enumerate(zip(classes, lambdas, payload_sizes, expected_bandwidths)):
        resource_usage = generate_poisson_traffic(lam, simulation_time, payload, expected_bw, error_margin)
        priority = np.random.uniform(*priority_ranges[class_name], size=simulation_time)
        for t in range(simulation_time):
            traffic_data.append({"Priority": priority[t], "Resource Usage": resource_usage[t]})

    # 데이터프레임 생성 및 저장
    traffic_df = pd.DataFrame(traffic_data)
    csv_file_path = "/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/traffic_scenario_case1.csv"
    traffic_df.to_csv(csv_file_path, index=False)
    print(f"Generated Traffic Scenario CSV with Priority and Resource Usage:\n{traffic_df.head()}")

    # 환경 초기화
    env = TSNQoSEnv(traffic_df, n_clusters=3, initial_data_size=300, initial_bandwidth=200)

    # 모델 로드
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/trained_policy_model_After_Case1.pth'
    agent = Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)

    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
        print(f"Loaded trained model from '{pre_trained_model_path}'")
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}'")
        exit()

    # 강화학습 실행
    state = env.reset()
    done = False
    episode_reward = 0
    allocation_results = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.get_action(state_tensor)[0]
        next_state, reward, done, info = env.step(action)

        allocation_results.append(info)
        state = next_state
        episode_reward += reward

    # 클래스 이름을 CLASS A, CLASS B, CLASS C로 매핑
    class_mapping = {f'class_{chr(65 + idx)}': f'Class {chr(65 + idx)}' for idx in range(len(classes))}

    # 클래스별 대역폭 할당 비율 계산
    allocated_bandwidth_ratios = {
        class_mapping[f'class_{chr(65 + idx)}']: np.mean([
            allocated_resource / total_resource_needed * 100
            if total_resource_needed > 0 else 0
            for step in allocation_results
            for class_index, allocated_resource, total_resource_needed in step['allocated_resources']
            if class_index == idx
        ])
        for idx in range(len(allocation_results[0]['allocated_resources']))
    }

    # 클래스별 손실률 계산
    loss_rates = {
        class_mapping[f'class_{chr(65 + idx)}']: np.mean([
            100 * max(0, (total_resource_needed - allocated_resource) / total_resource_needed)
            if total_resource_needed > 0 else 0
            for step in allocation_results
            for class_index, allocated_resource, total_resource_needed in step['allocated_resources']
            if class_index == idx
        ])
        for idx in range(len(allocation_results[0]['allocated_resources']))
    }

    # 최종 결과 저장
    result_df = pd.DataFrame({
        "Class": list(allocated_bandwidth_ratios.keys()),
        "Average Allocated Bandwidth (%)": list(allocated_bandwidth_ratios.values()),
        "Average Loss Rate (%)": list(loss_rates.values())
    })
    result_df.to_csv("/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/case1_results.csv", index=False)
    print("Final Results:\n", result_df)

    # 결과 시각화
    # 그래프 1: 클래스별 대역폭 할당 비율
    plt.figure(figsize=(8, 5))
    plt.bar(allocated_bandwidth_ratios.keys(), allocated_bandwidth_ratios.values(), color='skyblue', alpha=0.8)
    plt.title("Allocated Bandwidth Ratio by Class (Case 1)")
    plt.xlabel("Class")
    plt.ylabel("Allocated Bandwidth (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(allocated_bandwidth_ratios.values()):
        plt.text(i, val + 2, f"{val:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.show()

    # 그래프 2: 클래스별 손실률
    plt.figure(figsize=(8, 5))
    plt.bar(loss_rates.keys(), loss_rates.values(), color='orange', alpha=0.8)
    plt.title("Loss Rate by Class (Case 1)")
    plt.xlabel("Class")
    plt.ylabel("Loss Rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(loss_rates.values()):
        plt.text(i, val + 2, f"{val:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.show()


    result_df = pd.DataFrame({
        "Class": classes,
        "Average Allocated Bandwidth (%)": [allocated_bandwidth_ratios[class_name] for class_name in classes],
        "Average Loss Rate (%)": [loss_rates[class_name] for class_name in classes]
    })
    result_df.to_csv("/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/case1_results.csv", index=False)
    print("Final Results:\n", result_df)


if __name__ == "__main__":
    main()
