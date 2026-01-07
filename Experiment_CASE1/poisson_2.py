import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Config
import torch
import os
from Agent import Agent  # Agent 클래스 정의 필요
from EA_3_Class_env import TSNQoSEnv  # 환경 클래스 정의 필요

def generate_poisson_traffic(lam, duration, payload_size, expected_bandwidth, error_margin=0.05, is_zero_bandwidth=False):
    """
    Poisson 분포 기반 트래픽 생성
    :param lam: 패킷 발생률 λ (패킷/초)
    :param duration: 시뮬레이션 시간 (초)
    :param payload_size: 각 패킷의 크기 (bytes)
    :param expected_bandwidth: 예측 대역폭 (Mbps)
    :param error_margin: 예측 대역폭에 대한 오차 범위 (±%)
    :param is_zero_bandwidth: True이면 대역폭 값을 항상 0으로 설정
    :return: 초당 대역폭 리스트 (Mbps)
    """
    if is_zero_bandwidth:
        return [0.0] * duration  # Class C의 경우 대역폭은 항상 0

    packets_per_second = np.random.poisson(lam=lam, size=duration)
    bandwidth_per_second = [(packets * (payload_size * 8)) / 1e6 for packets in packets_per_second]
    adjusted_bandwidth = [bw + np.random.uniform(-error_margin, error_margin) * expected_bandwidth for bw in bandwidth_per_second]
    return adjusted_bandwidth

# 시뮬레이션 결과 분석 함수
def analyze_results(allocation_results, classes):
    # 클래스 이름 매핑
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

    return allocated_bandwidth_ratios, loss_rates

def visualize_results(allocated_bandwidth_ratios, loss_rates, allocation_results, classes):
    # 클래스별 평균 우선순위 계산
    average_priorities = {
        class_name: np.mean([
            step['allocated_resources'][idx][2] if idx < len(step['allocated_resources']) else 0
            for step in allocation_results
        ])
        for idx, class_name in enumerate(classes)
    }

    # 할당률 기준으로 정렬
    sorted_classes = sorted(allocated_bandwidth_ratios.keys(), key=lambda x: allocated_bandwidth_ratios[x], reverse=True)
    sorted_ratios = [allocated_bandwidth_ratios[class_name] for class_name in sorted_classes]

    # 시간별 손실률 계산
    time_steps = list(range(len(allocation_results)))
    loss_rates_per_time = {class_name: [
        100 * max(0, (step['allocated_resources'][idx][2] - step['allocated_resources'][idx][1]) / step['allocated_resources'][idx][2])
        if idx < len(step['allocated_resources']) and step['allocated_resources'][idx][2] > 0 else 0
        for step in allocation_results
    ] for idx, class_name in enumerate(classes)}

    # 그래프 1: 클래스별 대역폭 할당 비율
    plt.figure(figsize=(8, 5))
    plt.bar(sorted_classes, sorted_ratios, color='skyblue', alpha=0.8)
    plt.title("Allocated Bandwidth Ratio by Class")
    plt.xlabel("Class (Sorted by Allocation)")
    plt.ylabel("Allocated Bandwidth (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(sorted_ratios):
        plt.text(i, val + 2, f"{val:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.show()

    # 그래프 2: 시간에 따른 클래스별 손실률
    plt.figure(figsize=(10, 6))
    for class_name in classes:
        plt.plot(time_steps, loss_rates_per_time[class_name], label=class_name, marker='o', linestyle='-')
    plt.title("Loss Rate Over Time (Per Second)")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Loss Rate (%)")
    plt.xticks(range(0, len(time_steps) + 1, 1))  # X축 눈금을 10초 단위로 설정
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()




# 메인 실행 함수
def main():
    # 트래픽 시나리오 설정
    simulation_time = 100
    classes = ["Class A", "Class B", "Class C"]
    lambdas = [50, 30, 20]
    payload_sizes = [500, 400, 1500]
    expected_bandwidths = [0.53, 1.89, 0.0]
    error_margin = 0.05
    priority_ranges = {
        "Class A": (0.9, 1.0),
        "Class B": (0.6, 0.7),
        "Class C": (0.2, 0.3)
    }

    # 트래픽 데이터 생성
    traffic_data = []
    for i, (class_name, lam, payload, expected_bw) in enumerate(zip(classes, lambdas, payload_sizes, expected_bandwidths)):
        is_zero_bandwidth = (class_name == "Class C")  # Class C는 항상 0 대역폭
        resource_usage = generate_poisson_traffic(lam, simulation_time, payload, expected_bw, error_margin, is_zero_bandwidth)
        priority = np.linspace(*priority_ranges[class_name], num=simulation_time).tolist()  # 각 클래스 우선순위 생성
        print(f"Class {class_name}: Priority Range = {priority_ranges[class_name]}")
        print(f"Generated Priorities: {priority[:5]}... (Total: {len(priority)})")  # 확인용 출력
        for t in range(simulation_time):
            traffic_data.append({"Class": class_name, "Resource Usage": resource_usage[t], "Priority": priority[t]})


    # 데이터프레임 생성 및 저장
    traffic_df = pd.DataFrame(traffic_data)
    csv_file_path = "/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/traffic_scenario_case1.csv"
    traffic_df.to_csv(csv_file_path, index=False)
    print(f"Generated Traffic Scenario CSV with Priority and Resource Usage:\n{traffic_df.head()}")

    # 환경 초기화 및 강화학습 실행 (이후는 동일)
    env = TSNQoSEnv(traffic_df, n_clusters=3, initial_data_size=100, initial_bandwidth=23)

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
    allocation_results = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.get_action(state_tensor)[0]
        next_state, reward, done, info = env.step(action)

        allocation_results.append(info)
        state = next_state

    # 결과 분석 및 시각화
    allocated_bandwidth_ratios, loss_rates = analyze_results(allocation_results, classes)
    visualize_results(allocated_bandwidth_ratios, loss_rates, allocation_results, classes)

if __name__ == "__main__":
    main()