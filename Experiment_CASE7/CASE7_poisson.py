import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import Config
from Agent import Agent  # Agent 클래스 정의 필요
from EA_4_Class_env import TSNQoSEnv  # 환경 클래스 정의 필요

def generate_poisson_traffic(lambdas, duration, payload_sizes, expected_bandwidths, priority_ranges, error_margin=0.05, is_zero_bandwidth=None):
    """
    매 초마다 각 클래스별로 고정된 패킷 수를 생성하는 트래픽 생성 함수
    """
    total_data = []

    for second in range(duration):
        for class_idx, (lam, payload, expected_bw) in enumerate(zip(lambdas, payload_sizes, expected_bandwidths)):
            # priority_ranges 딕셔너리에서 현재 클래스의 우선순위 범위 가져오기
            priority_range = priority_ranges[f"Class {chr(65 + class_idx)}"]
            packets = lam  # 고정된 패킷 수
            priorities = np.random.uniform(*priority_range, size=packets)  # 우선순위 생성

            # 대역폭 계산 (is_zero_bandwidth를 사용하여 특정 클래스 처리)
            if is_zero_bandwidth and is_zero_bandwidth[class_idx]:  # is_zero_bandwidth가 참이라면 대역폭을 0으로 설정
                bandwidths = [0.0] * packets
            else:
                bandwidths = [
                    max(0, expected_bw * (1 + np.random.uniform(-error_margin, error_margin)))  # expected_bw ± 5% 범위
                    for _ in range(packets)
                ]

            # 각 패킷 데이터를 저장
            for i in range(packets):
                total_data.append({
                    "Time": second,
                    "Class": f"Class {chr(65 + class_idx)}",  # Class A, B, C, ...로 이름 지정
                    "Resource Usage": bandwidths[i],
                    "Priority": priorities[i]
                })

    return total_data


def analyze_results(allocation_results, classes):
    """
    결과 분석
    """
    allocated_bandwidth_ratios = {
        f"Class {chr(65 + idx)}": np.mean([
            allocated_resource / total_resource_needed * 100
            if total_resource_needed > 0 else 0
            for step in allocation_results
            for class_index, allocated_resource, total_resource_needed in step['allocated_resources']
            if class_index == idx
        ])
        for idx in range(len(allocation_results[0]['allocated_resources']))
    }

    loss_rates_per_time = {f"Class {chr(65 + idx)}": [
        100 * max(0, (step['allocated_resources'][idx][2] - step['allocated_resources'][idx][1]) / step['allocated_resources'][idx][2])
        if idx < len(step['allocated_resources']) and step['allocated_resources'][idx][2] > 0 else 0
        for step in allocation_results
    ] for idx in range(len(allocation_results[0]['allocated_resources']))}

    return allocated_bandwidth_ratios, loss_rates_per_time

def visualize_results(allocated_bandwidth_ratios, loss_rates_per_time):
    """
    결과 시각화 - 겹치는 로스 그래프를 클래스별로 잘 구분하여 표현
    """
    # 평균 대역폭 할당률 (값 기준으로 정렬)
    sorted_ratios = dict(sorted(allocated_bandwidth_ratios.items(), key=lambda item: item[1], reverse=True))
    sorted_values = list(sorted_ratios.values())
    original_order_classes = allocated_bandwidth_ratios.keys()

    # 평균 대역폭 할당률 그래프
    plt.figure(figsize=(8, 5))
    plt.bar(original_order_classes, sorted_values, color='skyblue', alpha=0.8)
    plt.title("Average Allocated Bandwidth Ratio Over 100 Seconds")
    plt.xlabel("Class")
    plt.ylabel("Average Allocated Bandwidth (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(sorted_values):
        plt.text(i, val + 2, f"{val:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.show()

    # 시간별 손실률 - 클래스별로 분리된 그래프
    time_steps = list(range(1, 101))  # 1초부터 100초
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'brown']  # 클래스별 색상
    markers = ['o', 'o', 'o', 'o', 'o', 'o']  # 클래스별 마커 스타일

    plt.figure(figsize=(12, 8))
    for idx, (class_name, loss_rates) in enumerate(loss_rates_per_time.items()):
        plt.plot(
            time_steps,
            loss_rates[:100],
            label=class_name,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            markersize=6,
            linewidth=2
        )

    plt.title("Case7: Loss Rate", fontsize=16)
    plt.xlabel("Time (Seconds)", fontsize=14)
    plt.ylabel("Loss Rate (%)", fontsize=14)
    plt.xticks(range(0, 101, 10))
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    """
    메인 실행 함수
    """
    simulation_time = 100
    classes = ["Class A", "Class B", "Class C", "Class D"]
    lambdas = [50, 50, 30, 20]
    payload_sizes = [500, 500, 300, 200]
    expected_bandwidths = [0.53, 1.89, 0.83, 0.41]
    error_margin = 0.05
    priority_ranges = {
        "Class A": (0.9, 1.0),
        "Class B": (0.6, 0.8),
        "Class C": (0.7, 0.5),
        "Class D": (0.2, 0.4),
    }
    is_zero_bandwidth = [False, False, False, False]  # Class F의 대역폭은 항상 0

    # 트래픽 데이터 생성
    traffic_data = generate_poisson_traffic(
        lambdas, simulation_time, payload_sizes, expected_bandwidths, priority_ranges, error_margin, is_zero_bandwidth
    )

    # 데이터프레임 생성
    traffic_df = pd.DataFrame(traffic_data)
    csv_file_path = "/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE4/traffic_scenario_case4.csv"
    traffic_df.to_csv(csv_file_path, index=False)

    # 환경 초기화
    env = TSNQoSEnv(traffic_df, n_clusters=4, initial_data_size=150, initial_bandwidth=110)

    # 모델 로드
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE7/trained_policy_model_Case7.pth'
    agent = Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)

    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
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
    allocated_bandwidth_ratios, loss_rates_per_time = analyze_results(allocation_results, classes)
    visualize_results(allocated_bandwidth_ratios, loss_rates_per_time)

if __name__ == "__main__":
    main()
