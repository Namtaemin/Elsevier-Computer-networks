import torch
import numpy as np
from torch.optim import Adam
import Agent  # Agent 클래스는 Agent.py에 정의되어 있다고 가정
from EA_6_Class_env import TSNQoSEnv  # 환경 파일이 TSNQoSEnv.py에 있다고 가정
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import os

# 설정 (Config은 Config.py에 설정값들이 정의되어 있다고 가정)
import Config
print("Current Working Directory:", os.getcwd())

# 지수 표기법 대신 소수점 표기법 사용
np.set_printoptions(suppress=True, precision=8, floatmode='fixed')
pd.set_option('display.float_format', lambda x: '%.8f' % x)

def run_case2_experiment(pre_trained_model_path):
    # 33개의 데이터를 포함한 CSV 파일 로드
    csv_file = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE2/Case2_experiment.csv'  # Case 2에 해당하는 데이터 파일
    dataset = pd.read_csv(csv_file)
    
    # 환경 초기화 (33개의 데이터를 사용하는 Case 2 환경 설정)
    env = TSNQoSEnv(dataset, n_clusters=6, initial_data_size=100, initial_bandwidth=70)  # 초기 데이터 수를 33으로 설정
    agent = Agent.Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)

    # 사전 학습된 모델 로드
    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
        print(f"Loaded trained model from '{pre_trained_model_path}' for Case 2")
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}', starting from scratch for Case 2")
        return  # 모델이 없는 경우, 실험을 중단

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

    # 원하는 스텝의 자원 할당량 출력 (첫 번째 스텝 출력)
    print(f"\nFirst Step: {first_step_info['step']}")

    allocation_ratios = {}
    for class_index, allocated_resource, total_resource in first_step_info['allocated_resources']:
        # 요청 대비 할당 비율 계산
        if total_resource > 0:
            allocation_ratio = (allocated_resource / total_resource) * 100
        else:
            allocation_ratio = 0
        allocation_ratios[f'class_{class_index}'] = allocation_ratio
        print(f"Allocated {allocated_resource:.2f} resources to class_{class_index} (Needed: {total_resource:.2f}). Allocation Ratio: {allocation_ratio:.2f}%")
    
    print(f"Remaining Resources: {first_step_info['remaining_resource']:.2f}")

    # 에피소드 종료 후 각 클래스별 데이터와 우선순위 평균 출력
    class_priority_means = {}
    for cls, class_data in env.class_data.items():
        priority_mean = class_data['Priority'].mean() if not class_data.empty else 0
        class_priority_means[cls] = priority_mean
        print(f"\nClass {cls} ({len(class_data)} data items) - Priority Mean: {priority_mean:.2f}")
        print(class_data[['Priority', 'Resource Usage']])  # 주요 데이터 속성 출력

    # 우선순위 평균에 따라 정렬된 클래스 리스트
    sorted_classes = sorted(class_priority_means.keys(), key=lambda x: class_priority_means[x], reverse=True)
    
    # 정렬된 클래스 순서대로 할당 비율 리스트 생성
    sorted_allocation_ratios = [allocation_ratios.get(f'class_{cls}', 0) for cls in sorted_classes]
    
    # 우선순위 순서에 따라 클래스 이름 지정 (Class A, Class B, Class C ...)
    class_names = [f'Class {chr(65 + i)}' for i in range(len(sorted_classes))]

    # 정렬된 클래스별 자원 할당 비율 시각화 (첫 번째 스텝 할당량 비율)
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, sorted_allocation_ratios, color='blue')
    plt.title('Initial Resource Allocation Ratios per Class for Case 1-2')
    plt.xlabel('Class')
    plt.ylabel('Allocated Resource Ratio (%)')  # 비율로 표시
    plt.grid()
    plt.show()

    return sorted_allocation_ratios, episode_reward  # 최종 클래스별 자원 할당 비율과 보상 반환

def main():
    # 모델 파일 경로 (Case 2 전용)
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE2/trained_policy_model_Case2.pth'
    
    # Case 2 실험 실행
    print(f"\nStarting experiment for Case 2")
    class_resource_allocation, total_reward = run_case2_experiment(pre_trained_model_path)

if __name__ == "__main__":
    main()
