import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import Agent  # Agent 클래스는 Agent.py에 정의되어 있다고 가정
from EA_3_Class_env import TSNQoSEnv  # 환경 파일이 3개의 클래스로 설정되어 있다고 가정

import Config

# 지수 표기법 대신 소수점 표기법 사용
np.set_printoptions(suppress=True, precision=8, floatmode='fixed')
pd.set_option('display.float_format', lambda x: '%.8f' % x)
total_bandwidth =350

def cluster_data(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data[['Priority', 'Resource Usage']])
    
    cluster_summary = data.groupby('Cluster')['Priority'].mean().reset_index()
    sorted_clusters = cluster_summary.sort_values(by='Priority', ascending=False)['Cluster'].tolist()

    class_mapping = {sorted_clusters[i]: f'class_{i}' for i in range(len(sorted_clusters))}
    data['Class'] = data['Cluster'].map(class_mapping)
    
    return data

def run_case1_experiment(pre_trained_model_path):
    csv_file = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE6/Case6_experiment.csv'
    dataset = pd.read_csv(csv_file)
    
    env = TSNQoSEnv(dataset, n_clusters=3, initial_data_size=114, initial_bandwidth=350)
    agent = Agent.Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)

    if os.path.exists(pre_trained_model_path):
        agent.agent_control.policy_nn.load_state_dict(torch.load(pre_trained_model_path))
        print(f"Loaded trained model from '{pre_trained_model_path}' for Case 2-2")
    else:
        print(f"No pre-trained model found at '{pre_trained_model_path}', starting from scratch for Case 2-2")
        return None

    state = env.reset()
    done = False
    step_count = 0
    first_step_info = None

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = agent.get_action(state_tensor)[0]
        next_state, reward, done, info = env.step(action)

        if step_count == 0:
            first_step_info = {
                'step': step_count,
                'allocated_resources': info['allocated_resources'],
                'remaining_resource': info['remaining_resource']
            }

        step_count += 1
        state = next_state
        break

    allocation_ratios = {}
    total_allocated = sum(allocated_resource for _, allocated_resource, _ in first_step_info['allocated_resources'])
    for class_index, allocated_resource, _ in first_step_info['allocated_resources']:
        allocation_ratio = (allocated_resource / total_allocated) * 100 if total_allocated > 0 else 0
        allocation_ratios[f'class_{class_index}'] = allocation_ratio

    return allocation_ratios

def wfq_algorithm(data, original_data):
    remaining_bandwidth = total_bandwidth
    allocation_ratios = {}
    allocated_resources = {f'class_{i}': 0 for i in range(3)}

    sorted_data = data.sort_values(by='Priority', ascending=False)
    total_priority = sorted_data['Priority'].sum()

    for _, row in sorted_data.iterrows():
        class_index = row['Class']
        requested_bandwidth = row['Resource Usage']
        weight = row['Priority'] / total_priority if total_priority > 0 else 1 / 3
        max_allocatable = weight * total_bandwidth
        allocated_bandwidth = min(max_allocatable, requested_bandwidth, remaining_bandwidth)
        
        allocated_resources[class_index] += allocated_bandwidth
        remaining_bandwidth -= allocated_bandwidth

    total_allocated = sum(allocated_resources.values())
    allocation_ratios = {class_index: (allocated_resources[class_index] / total_allocated) * 100 if total_allocated > 0 else 0 
                         for class_index in allocated_resources}

    return allocation_ratios

def pq_algorithm(data, original_data):
    remaining_bandwidth = total_bandwidth
    allocation_ratios = {}
    allocated_resources = {f'class_{i}': 0 for i in range(3)}

    sorted_data = data.sort_values(by='Priority', ascending=False)

    for _, row in sorted_data.iterrows():
        class_index = row['Class']
        requested_bandwidth = row['Resource Usage']
        allocated_bandwidth = min(requested_bandwidth, remaining_bandwidth)
        
        allocated_resources[class_index] += allocated_bandwidth
        remaining_bandwidth -= allocated_bandwidth

    total_allocated = sum(allocated_resources.values())
    allocation_ratios = {class_index: (allocated_resources[class_index] / total_allocated) * 100 if total_allocated > 0 else 0 
                         for class_index in allocated_resources}

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
    pre_trained_model_path = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE6/trained_policy_model_Case6.pth'
    rule_based_csv = '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE6/Case6_experiment.csv'
    
    num_experiments = 20
    all_data = {
        'RL': {'class_0': [], 'class_1': [], 'class_2': []},
        'WFQ': {'class_0': [], 'class_1': [], 'class_2': []},
        'PQ': {'class_0': [], 'class_1': [], 'class_2': []}
    }

    for i in range(num_experiments):
        rl_allocation_ratios = run_case1_experiment(pre_trained_model_path)
        if rl_allocation_ratios is None:
            return
        
        for class_name, ratio in rl_allocation_ratios.items():
            all_data['RL'][class_name].append(ratio)

        wfq_allocation_ratios = run_rule_based_experiment('WFQ', rule_based_csv)
        for class_name, ratio in wfq_allocation_ratios.items():
            all_data['WFQ'][class_name].append(ratio)

        pq_allocation_ratios = run_rule_based_experiment('PQ', rule_based_csv)
        for class_name, ratio in pq_allocation_ratios.items():
            all_data['PQ'][class_name].append(ratio)

    avg_priorities = {
        'RL': {class_name: np.mean(all_data['RL'][class_name]) for class_name in all_data['RL']},
        'WFQ': {class_name: np.mean(all_data['WFQ'][class_name]) for class_name in all_data['WFQ']},
        'PQ': {class_name: np.mean(all_data['PQ'][class_name]) for class_name in all_data['PQ']}
    }

    # 각 알고리즘에 대해 우선순위에 따라 정렬된 클래스 순서 얻기
    sorted_rl_classes = sorted(avg_priorities['RL'], key=avg_priorities['RL'].get, reverse=True)
    sorted_wfq_classes = sorted(avg_priorities['WFQ'], key=avg_priorities['WFQ'].get, reverse=True)
    sorted_pq_classes = sorted(avg_priorities['PQ'], key=avg_priorities['PQ'].get, reverse=True)

    plt.figure(figsize=(10, 6))
    class_labels = ['Class A', 'Class B', 'Class C']
    x = np.arange(len(class_labels))
    width = 0.2

    # RL, WFQ, PQ 결과를 각각의 우선순위에 따라 정렬된 순서대로 그래프에 표시
    rl_means = [avg_priorities['RL'][class_name] for class_name in sorted_rl_classes]
    bars_rl = plt.bar(x, rl_means, width=width, label='RL')

    wfq_means = [avg_priorities['WFQ'][class_name] for class_name in sorted_wfq_classes]
    bars_wfq = plt.bar(x + width, wfq_means, width=width, label='WFQ')

    pq_means = [avg_priorities['PQ'][class_name] for class_name in sorted_pq_classes]
    bars_pq = plt.bar(x + 2 * width, pq_means, width=width, label='PQ')

    # 그래프에 각 막대 위에 할당 비율 표시
    for bars in [bars_rl, bars_wfq, bars_pq]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f'{height:.2f}%',
                ha='center',
                va='bottom'
            )

    plt.xlabel('Classes (Sorted by Priority)')
    plt.ylabel('Normalized Allocation Ratio (%)')
    plt.title('Comparison of RL, WFQ, and PQ Resource Allocation Across Classes')
    plt.xticks(x + width, class_labels)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
