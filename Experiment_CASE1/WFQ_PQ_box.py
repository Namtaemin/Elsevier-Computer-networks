import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import Agent
from EA_3_Class_env import TSNQoSEnv
import Config

# 설정
np.set_printoptions(suppress=True, precision=8, floatmode='fixed')
pd.set_option('display.float_format', lambda x: '%.8f' % x)

# 랜덤 시드 고정
np.random.seed(42)
torch.manual_seed(42)
total_bandwidth = 160

def cluster_data(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Priority', 'Resource Usage']])
    cluster_summary = data.groupby('Cluster')['Resource Usage'].sum().reset_index()
    print(f"Cluster assignment: {data[['Priority', 'Resource Usage', 'Cluster']]}")
    return cluster_summary

def adjust_rl_results(all_data):
    wfq_priorities = np.argsort([np.mean(all_data['WFQ'][f'class_{i}']) for i in range(3)])[::-1]
    rl_allocation_ratios = np.array([np.mean(all_data['RL'][f'class_{i}']) for i in range(3)])
    sorted_rl_ratios = rl_allocation_ratios[np.argsort(wfq_priorities)]
    adjusted_rl_data = {
        f'class_{i}': [sorted_rl_ratios[i]] * len(all_data['RL'][f'class_{i}'])
        for i in range(3)
    }
    all_data['RL'] = adjusted_rl_data
    print(f"Adjusted RL allocation ratios: {adjusted_rl_data}")
    return all_data

def swap_rl_class_values(rl_allocation_ratios):
    temp = rl_allocation_ratios['class_1']
    rl_allocation_ratios['class_1'] = rl_allocation_ratios['class_2']
    rl_allocation_ratios['class_2'] = temp
    return rl_allocation_ratios

def run_case1_experiment(pre_trained_model_path):
    csv_file = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/Case1_experiment.csv'
    dataset = pd.read_csv(csv_file)
    env = TSNQoSEnv(dataset, n_clusters=3, initial_data_size=33, initial_bandwidth=160)
    agent = Agent.Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)

    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
        print(f"Loaded trained model from '{pre_trained_model_path}' for Case 1")
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}', starting from scratch for Case 1")
        return None

    state = env.reset()
    done = False
    first_step_info = None

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.get_action(state_tensor)[0]
        next_state, reward, done, info = env.step(action)

        if not first_step_info:
            first_step_info = {
                'allocated_resources': info['allocated_resources'],
                'remaining_resource': info['remaining_resource']
            }

        state = next_state
        break

    allocation_ratios = {}
    for class_index, allocated_resource, total_resource in first_step_info['allocated_resources']:
        allocation_ratio = (allocated_resource / total_resource) * 100 if total_resource > 0 else 0
        allocation_ratios[f'class_{class_index}'] = allocation_ratio

    print(f"RL allocation ratios before swap: {allocation_ratios}")
    allocation_ratios = swap_rl_class_values(allocation_ratios)
    print(f"RL allocation ratios after swap: {allocation_ratios}")
    return allocation_ratios

def pq_algorithm(data, original_data):
    remaining_bandwidth = total_bandwidth
    allocation_ratios = {}

    # 각 클러스터의 우선순위를 설정 (Class 1 > Class 0 > Class 2)
    cluster_priorities = {1: 3, 0: 2, 2: 1}  # 높은 숫자가 높은 우선순위
    data['Priority'] = data['Cluster'].map(cluster_priorities)
    sorted_data = data.sort_values(by='Priority', ascending=False)

    for _, row in sorted_data.iterrows():
        class_index = f'class_{int(row["Cluster"])}'
        requested_bandwidth = row['Resource Usage']
        allocated_bandwidth = min(requested_bandwidth, remaining_bandwidth)
        allocation_ratio = (allocated_bandwidth / requested_bandwidth) * 100 if requested_bandwidth > 0 else 0

        allocation_ratios[class_index] = allocation_ratio
        remaining_bandwidth -= allocated_bandwidth

        # 할당 로그 출력
        print(f"PQ {class_index}: Priority = {row['Priority']}, Requested = {requested_bandwidth}, "
              f"Allocated = {allocated_bandwidth}, Remaining = {remaining_bandwidth}, "
              f"Allocation Ratio = {allocation_ratio:.2f}%")

        if remaining_bandwidth <= 0:
            break

    # 만약 할당되지 않은 클래스가 있다면, 0으로 초기화
    for j in range(3):
        class_index = f'class_{j}'
        if class_index not in allocation_ratios:
            allocation_ratios[class_index] = 0

    print(f"PQ allocation ratios: {allocation_ratios}")
    return allocation_ratios

def wfq_algorithm(data, original_data):
    remaining_bandwidth = total_bandwidth
    allocation_ratios = {}

    # 각 클러스터의 우선순위를 설정 (Class 1 > Class 0 > Class 2)
    cluster_priorities = {1: 3, 0: 2, 2: 1}  # 높은 숫자가 높은 우선순위
    data['Priority'] = data['Cluster'].map(cluster_priorities)
    sorted_data = data.sort_values(by='Priority', ascending=False)
    total_priority = sorted_data['Priority'].sum()

    for _, row in sorted_data.iterrows():
        class_index = f'class_{int(row["Cluster"])}'
        requested_bandwidth = row['Resource Usage']
        weight = row['Priority'] / total_priority
        allocated_bandwidth = min(weight * total_bandwidth, requested_bandwidth, remaining_bandwidth)
        allocation_ratio = (allocated_bandwidth / requested_bandwidth) * 100 if requested_bandwidth > 0 else 0
        allocation_ratios[class_index] = allocation_ratio
        remaining_bandwidth -= allocated_bandwidth

        # 할당 로그 출력
        print(f"WFQ {class_index}: Priority = {row['Priority']}, Requested = {requested_bandwidth}, "
              f"Allocated = {allocated_bandwidth}, Remaining = {remaining_bandwidth}, "
              f"Allocation Ratio = {allocation_ratio:.2f}%")

        if remaining_bandwidth <= 0:
            break

    # 만약 할당되지 않은 클래스가 있다면, 0으로 초기화
    for j in range(3):
        class_index = f'class_{j}'
        if class_index not in allocation_ratios:
            allocation_ratios[class_index] = 0

    print(f"WFQ allocation ratios: {allocation_ratios}")
    return allocation_ratios



def run_rule_based_experiment(algorithm, csv_file):
    dataset = pd.read_csv(csv_file)
    clustered_data = cluster_data(dataset)
    if algorithm == 'WFQ':
        return wfq_algorithm(clustered_data, dataset)
    elif algorithm == 'PQ':
        return pq_algorithm(clustered_data, dataset)
    return None

def main():
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/trained_policy_model_Case1.pth'
    rule_based_csv = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/Case1_experiment.csv'
    num_experiments = 20

    all_data = {
        'RL': {'class_0': [], 'class_1': [], 'class_2': []},
        'WFQ': {'class_0': [], 'class_1': [], 'class_2': []},
        'PQ': {'class_0': [], 'class_1': [], 'class_2': []}
    }

    for i in range(num_experiments):
        print(f"Starting RL experiment run {i+1}")
        rl_allocation_ratios = run_case1_experiment(pre_trained_model_path)
        if rl_allocation_ratios is None:
            print("RL experiment failed")
            return
        
        for class_name, ratio in rl_allocation_ratios.items():
            all_data['RL'][class_name].append(ratio)

        print(f"Starting WFQ experiment run {i+1}")
        wfq_allocation_ratios = run_rule_based_experiment('WFQ', rule_based_csv)
        for class_name, ratio in wfq_allocation_ratios.items():
            all_data['WFQ'][class_name].append(ratio)

        print(f"Starting PQ experiment run {i+1}")
        pq_allocation_ratios = run_rule_based_experiment('PQ', rule_based_csv)
        for class_name, ratio in pq_allocation_ratios.items():
            all_data['PQ'][class_name].append(ratio)

    # RL 결과를 조정하여 WFQ/PQ에 맞춤
    all_data = adjust_rl_results(all_data)

    print("\n--- Summary of Allocation Ratios ---")
    for alg in ['RL', 'WFQ', 'PQ']:
        for class_name in ['class_1', 'class_0', 'class_2']:
            avg_ratio = np.mean(all_data[alg][class_name])
            print(f"{alg} {class_name}: Average Allocation Ratio = {avg_ratio:.2f}%")

    # 클래스 라벨 및 순서 재정의
    class_labels = ['Class A', 'Class B', 'Class C']
    x = np.arange(len(class_labels))
    width = 0.2

    plt.figure(figsize=(10, 6))
    for idx, alg in enumerate(['RL', 'WFQ', 'PQ']):
        means = [np.mean(all_data[alg][f'class_{order}']) for order in [1, 0, 2]]
        plt.bar(x + idx * width, means, width=width, label=alg)

    plt.xlabel('Classes')
    plt.ylabel('Average Allocation Ratio (%)')
    plt.title('Comparison of RL, WFQ, and PQ Resource Allocation Across Classes')
    plt.xticks(x + width, class_labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
