import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import Agent  # Agent 클래스는 Agent.py에 정의되어 있다고 가정
from EA_4_Class_env import TSNQoSEnv  # 환경 파일이 TSNQoSEnv.py에 있다고 가정

# 설정 (Config은 Config.py에 설정값들이 정의되어 있다고 가정)
import Config
print("Current Working Directory:", os.getcwd())

# 지수 표기법 대신 소수점 표기법 사용
np.set_printoptions(suppress=True, precision=8, floatmode='fixed')
pd.set_option('display.float_format', lambda x: '%.8f' % x)

def run_case1_experiment(pre_trained_model_path):
    # 33개의 데이터를 포함한 CSV 파일 로드
    csv_file = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE3/Case3_experiment.csv'
    dataset = pd.read_csv(csv_file)
    
    # 환경 초기화 (33개의 데이터를 사용하는 Case 1 환경 설정)
    env = TSNQoSEnv(dataset, n_clusters=4, initial_data_size=28, initial_bandwidth=110)  # 초기 데이터 수를 33으로 설정
    agent = Agent.Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)

    # 사전 학습된 모델 로드
    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
        print(f"Loaded trained model from '{pre_trained_model_path}' for Case 3")
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}', starting from scratch for Case 3")
        return None  # 모델이 없는 경우, 실험을 중단

    # 학습 없이 에피소드 1번 실행
    state = env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    # 첫 번째 스텝의 자원 할당 정보를 저장할 변수
    first_step_info = None

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.get_action(state_tensor)[0]
        
        # 환경에서 한 스텝 실행하고, 할당된 자원을 추적
        next_state, reward, done, info = env.step(action)

        # 첫 번째 스텝 정보만 저장
        if step_count == 0:
            first_step_info = {
                'step': step_count,
                'allocated_resources': info['allocated_resources'],
                'remaining_resource': info['remaining_resource']
            }

        episode_reward += reward
        step_count += 1
        state = next_state

        # 첫 번째 스텝 이후에는 반복문 탈출
        break

    # 자원 할당 비율과 우선순위 계산
    allocation_ratios = {}
    class_priorities = {}
    for class_index, allocated_resource, total_resource in first_step_info['allocated_resources']:
        # 요청 대비 할당 비율 계산
        if total_resource > 0:
            allocation_ratio = (allocated_resource / total_resource) * 100
        else:
            allocation_ratio = 0
        allocation_ratios[f'class_{class_index}'] = allocation_ratio
        class_priorities[f'class_{class_index}'] = state[class_index * 2]  # 우선순위는 state 배열의 짝수 인덱스에서 추출

    return allocation_ratios, class_priorities

def main():
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE3/trained_policy_model_Case7.pth'
    
    num_experiments = 11
    all_runs_ratios = {'class_0': [], 'class_1': [], 'class_2': [], 'class_3': []}  # 클래스별 할당 비율 저장
    all_class_priorities = {'class_0': [], 'class_1': [], 'class_2': [], 'class_3': []}  # 클래스별 우선순위 저장

    # 100번 실험 반복
    for i in range(num_experiments):
        print(f"Starting experiment run {i+1}")
        allocation_ratios, class_priorities = run_case1_experiment(pre_trained_model_path)
        
        if allocation_ratios is None:
            print("No pre-trained model found or experiment failed")
            return

        # 클래스별 할당 비율 및 우선순위 저장
        for class_name, ratio in allocation_ratios.items():
            all_runs_ratios[class_name].append(ratio)
        for class_name, priority in class_priorities.items():
            all_class_priorities[class_name].append(priority)
    
    # 평균 우선순위에 따라 정렬
    avg_priorities = {class_name: np.mean(all_class_priorities[class_name]) for class_name in all_class_priorities}
    sorted_classes = sorted(avg_priorities.keys(), key=lambda x: avg_priorities[x], reverse=True)
    print(avg_priorities)
    # 클래스별 평균, 최소, 최대, 중간값 계산
    avg_ratios = {class_name: np.mean(all_runs_ratios[class_name]) for class_name in sorted_classes}
    min_ratios = {class_name: np.min(all_runs_ratios[class_name]) for class_name in sorted_classes}
    max_ratios = {class_name: np.max(all_runs_ratios[class_name]) for class_name in sorted_classes}
    med_ratios = {class_name: np.median(all_runs_ratios[class_name]) for class_name in sorted_classes}

    for class_name in sorted_classes:
        print(f'{class_name}: Min={min_ratios[class_name]:.2f}%, Max={max_ratios[class_name]:.2f}%')

    # 클래스 이름을 우선순위 순서대로 가져옴
    class_names = [f'Class {chr(65 + i)}' for i in range(len(sorted_classes))]

    # 오차 범위 계산 (위쪽: max - avg, 아래쪽: avg - min)
    yerr_lower = [max(0, avg_ratios[class_name] - min_ratios[class_name]) for class_name in sorted_classes]
    yerr_upper = [max(0, max_ratios[class_name] - avg_ratios[class_name]) for class_name in sorted_classes]
    yerr = [yerr_lower, yerr_upper]

    # 막대 그래프 그리기
    plt.figure(figsize=(5, 5))
    bars = plt.bar(class_names, avg_ratios.values(), color='green', yerr=yerr, capsize=5, alpha=0.6)

    # 최소-최대 범위 및 중간값을 나타내는 선 추가
    for idx, class_name in enumerate(class_names):
        avg_val = avg_ratios[sorted_classes[idx]]
        min_val = min_ratios[sorted_classes[idx]]
        max_val = max_ratios[sorted_classes[idx]]
        med_val = med_ratios[sorted_classes[idx]]

        # 막대 위에 최소-최대 선 표시
        plt.plot([idx, idx], [min_val, max_val], color='black', lw=2)
        # 중간값 표시
        plt.scatter([idx], [med_val], color='red', zorder=5)

        # 막대 위에 평균값 텍스트 표시
        #plt.text(idx, avg_val, f'{avg_val:.2f}%', ha='center', va='bottom')

    plt.title(f'Case1-3 Resource Allocation Ratios with Min/Max/Median')
    plt.xlabel('Class (Sorted by Priority)')
    plt.ylabel('Allocated Resource Ratio (%)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
