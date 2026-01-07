import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random

class TSNQoSEnv:
    def __init__(self, dataset, n_clusters=6, initial_data_size=100, initial_bandwidth=19000):
        self.dataset = dataset.copy()
        self.n_clusters = n_clusters
        self.initial_data_size = initial_data_size
        self.initial_bandwidth = initial_bandwidth  # 제한된 자원 양 설정
        self.previous_total_reward = None  # 적응형 리워드를 위한 이전 리워드 저장
        self.max_steps = 1500
        self.current_step = 0

        # 초기화
        self.usable_resource = self.initial_bandwidth

        # 초기 데이터로 클러스터링 수행
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        self.dataset['cluster'] = self.initial_clustering()

        # 각 클래스에 대한 데이터를 사전 형태로 저장
        self.class_data = {cls: self.dataset[self.dataset['cluster'] == cls] for cls in range(self.n_clusters)}
        self.verify_class_data()

    def initial_clustering(self):
        initial_data = self.dataset.iloc[:self.initial_data_size][['Priority']]
        self.kmeans.fit(initial_data)
        clusters = self.kmeans.predict(initial_data)

        full_clusters = np.full(self.dataset.shape[0], -1)
        full_clusters[:self.initial_data_size] = clusters

        self.classes = np.arange(self.n_clusters)
        self.class_names = [f'class_{chr(65 + i)}' for i in range(len(self.classes))]
        
        return full_clusters

    def verify_class_data(self):
        total_items = 0
        for cls in self.classes:
            class_data = self.class_data.get(cls, pd.DataFrame())
            num_items = len(class_data)
            priorities = list(class_data['Priority']) if 'Priority' in class_data.columns else ['No Priority data']
            index_range = (class_data.index.min(), class_data.index.max()) if not class_data.empty else ('N/A', 'N/A')
            class_name = self.class_names[cls]
            print(f"{class_name} contains {num_items} data items from index {index_range[0]} to {index_range[1]}")
            print(f"Priorities in {class_name}: {priorities}")
            assert num_items == len(priorities), f"Mismatch in item count for {class_name}: Expected {num_items}, found {len(priorities)}"
            total_items += num_items
        print(f"Total data items: {total_items}")

    def get_state(self):
        state = []

        for cls in range(self.n_clusters):
            if cls in self.class_data:
                class_data = self.class_data[cls]
                if not class_data.empty:
                    avg_priority = class_data['Priority'].mean()
                    total_resource_usage = class_data['Resource Usage'].sum()
                else:
                    avg_priority = 0
                    total_resource_usage = 0
            else:
                avg_priority = 0
                total_resource_usage = 0

            state.extend([avg_priority, total_resource_usage])

        while len(state) < 14:  
            state.extend([0, 0])

        state.append(self.initial_bandwidth)  # 초기 대역폭 추가

        print(f"Current State: {state}")

        return np.array(state, dtype=float)

    def step(self, action):
        # 매 스텝마다 usable_resource 초기화
        self.usable_resource = self.initial_bandwidth

        print(f"Step: {self.current_step}")
        
        # 우선순위에 따라 클래스 정렬
        sorted_class_states = sorted(self.class_data.items(), key=lambda x: x[1]['Priority'].mean(), reverse=True)

        # 새로 이름을 매긴 클래스 이름을 생성
        new_class_names = [f'class_{chr(65 + i)}' for i in range(len(sorted_class_states))]

        allocated_resources = []
        class_states = []
        total_allocated_resource = 0  # 총 할당된 자원량 추적 변수

        for i, (class_index, class_data) in enumerate(sorted_class_states):
            avg_priority = class_data['Priority'].mean()
            total_resource_usage = class_data['Resource Usage'].sum()

            # 할당 가능한 자원을 initial_bandwidth 내로 제한
            if i == 0:  # 우선순위가 가장 높은 클래스에 우선적으로 자원 할당
                allocated_resource = min(total_resource_usage, self.usable_resource) * 0.997
            else:
                satisfaction_rate = max(0.0001, action[i - 1])  # 최소 만족도를 0.0001로 설정
                allocated_resource = satisfaction_rate * min(self.usable_resource, total_resource_usage)

            max_allocatable_resource = min(allocated_resource, total_resource_usage)

            # 총 할당 자원이 initial_bandwidth를 초과하지 않도록 제한
            if total_allocated_resource + max_allocatable_resource > self.initial_bandwidth:
                max_allocatable_resource = self.initial_bandwidth - total_allocated_resource

            allocated_resources.append((class_index, max_allocatable_resource, total_resource_usage))
            class_states.append((class_index, avg_priority, total_resource_usage))

            total_allocated_resource += max_allocatable_resource
            self.usable_resource -= max_allocatable_resource

            print(f"Allocated {max_allocatable_resource} resources to {new_class_names[i]}. (Needed: {total_resource_usage})")
            print(f"Remaining Resources: {self.usable_resource}")

            # 총 할당 자원이 initial_bandwidth를 초과하면 더 이상 자원 할당 중지
            if total_allocated_resource >= self.initial_bandwidth:
                print("Total allocated resources exceed or equal initial bandwidth, stopping allocation.")
                done = True

        print("\nAction applied:")
        for i, (class_index, allocated_resource, total_resource_usage) in enumerate(allocated_resources):
            avg_priority = class_states[i][1]
            print(f"{new_class_names[i]}: Action={action[i-1] if i > 0 else '100% (highest priority)'}, Priority = {avg_priority}, Allocated Resource={allocated_resource}, Total Resource Needed={total_resource_usage}")

        # 에피소드 종료 조건 체크 및 이유 출력
        if self.current_step >= self.max_steps:
            done = True
            print(f"Episode ended because max steps {self.max_steps} reached.")
        elif self.usable_resource <= 0:
            done = True
            print(f"Episode ended because usable resources are depleted: {self.usable_resource}.")
        else:
            done = False

        reward = self.calculate_reward(allocated_resources, class_states, action)

        # 데이터 플로우 관리
        self.manage_data_flow()

        # 다음 상태 및 정보 업데이트
        next_state = self.get_state()
        self.current_step += 1

        # info 딕셔너리 확장: 클래스별 할당 자원량 추가
        info = {
            'allocated_resources': allocated_resources,  # 각 클래스의 (index, 할당된 자원, 총 필요 자원) 튜플
            'total_allocated_resource': total_allocated_resource,  # 총 할당 자원량
            'remaining_resource': self.usable_resource  # 남은 자원량
        }

        return next_state, reward, done, info

    def calculate_reward(self, allocated_resources, class_states, action):
        total_reward = 0
        threshold = 0.6
        total_allocated_resource = 0
        sorted_class_states = sorted(class_states, key=lambda x: x[1], reverse=True)
        total_required_resources = sum([x[2] for x in sorted_class_states])

        if self.initial_bandwidth >= total_required_resources:
            for i, (class_name, avg_priority, total_resource_usage) in enumerate(sorted_class_states):
                allocated_resource = allocated_resources[i][1]
                
                # 우선순위가 가장 높은 클래스는 리워드 계산에서 제외
                if i == 0:
                    continue

                action_value = action[i - 1] if i > 0 else 1.0
                if total_resource_usage > 0:
                    allocation_ratio = allocated_resource / total_resource_usage
                    reward = avg_priority * allocation_ratio * 1000
                else:
                    reward = 0

                # 이전 클래스의 액션 값과 비교하여 패널티 부여
                prev_action_value = action[i - 2] if i > 1 else 1.0
                prev_prev_action_value = action[i - 3] if i > 1 else 1.0
                prev_prev_prev_action_value = action[i - 4] if i > 1 else 1.0
                next_class_index = (i) % len(self.class_names)
                if action_value > prev_action_value and self.usable_resource > 0:
                    penalty = -1500 * abs(prev_action_value - action_value)
                    reward += penalty
                     
                    print(f"Penalty Applied for {self.class_names[next_class_index]}: Action value lower than previous class, Penalty: {penalty}")
                else:
                    if action_value <= prev_action_value and self.usable_resource > 0:
                        if 0.9 < action_value <= 1.0:
                            reward += (action_value * 2) * 3000
                        elif 0.8 < action_value <= 0.9:
                            reward += (action_value * 2) * 2000
                        elif 0.7 < action_value <= 0.8:
                            reward += (action_value * 2) * 100
                        elif 0.6 < action_value <= 0.7:
                            reward += (action_value * 2) * 50
                        elif 0.5 < action_value <= 0.6:
                            reward += (action_value * 2) * 10
                        else:
                            reward += action_value * 2

                # 낮은 우선순위 클래스가 더 높은 액션을 받았을 경우 페널티 부여
                if i == len(sorted_class_states) - 1:
                    # 마지막 클래스에서는 next_action_value가 없으므로, prev_action_value만 고려
                    if action_value > prev_action_value or action_value > prev_prev_action_value or action_value > prev_prev_prev_action_value:
                        penalty = -2000 * abs(prev_action_value - action_value)
                        reward += penalty
                        print(f"Penalty Applied for {self.class_names[next_class_index]}: Last class received more resources than previous class, Penalty: {penalty}")

                elif i < len(sorted_class_states) - 1:
                    next_action_value = action[i]
                    next_class_index = (i) % len(self.class_names)
                    if action_value < 0.8 and next_action_value > 0.7 and self.usable_resource > 0:
                        penalty = -1500 * abs(next_action_value - 0.2)  # 낮은 우선순위 클래스가 0.2 이상 할당받은 경우 페널티
                        reward += penalty
                        print(f"Penalty Applied for {self.class_names[next_class_index]}: Lower priority class received more resources than allowed, Penalty: {penalty}")
                        if action_value - prev_action_value > threshold:
                            reward += penalty
                total_reward += reward
                total_allocated_resource += allocated_resource

        else:
            
            # 자원이 부족할 경우
            for i, (class_name, avg_priority, total_resource_usage) in enumerate(sorted_class_states):
                allocated_resource = allocated_resources[i][1]

                # 우선순위가 가장 높은 클래스는 리워드 계산에서 제외
                if i == 0:
                    continue
                action_value = action[i - 1] if i > 0 else 1.0

                if total_resource_usage > 0:
                    allocation_ratio = allocated_resource / total_resource_usage
                    reward = avg_priority * allocation_ratio * 1000
                else:
                    reward = 0

                prev_action_value = action[i - 2] if i > 1 else 1.0
                prev_prev_action_value = action[i - 3] if i > 1 else 1.0
                prev_prev_prev_action_value = action[i - 4] if i > 1 else 1.0
                next_class_index = (i) % len(self.class_names)
                if action_value > prev_action_value and self.usable_resource > 0:
                    penalty = -1500 * abs(prev_action_value - action_value)
                    reward += penalty
                    print(f"Penalty Applied for {self.class_names[next_class_index]}: Action value lower than previous class, Penalty: {penalty}")

                else:
                    if action_value <= prev_action_value and self.usable_resource > 0:
                        if 0.9 < action_value <= 1.0:
                            reward += (action_value * 2) * 3000
                        elif 0.8 < action_value <= 0.9:
                            reward += (action_value * 2) * 2000
                        elif 0.7 < action_value <= 0.8:
                            reward += (action_value * 2) * 100
                        elif 0.6 < action_value <= 0.7:
                            reward += (action_value * 2) * 50
                        elif 0.5 < action_value <= 0.6:
                            reward += (action_value * 2) * 10
                        else:
                            reward += action_value * 2

                if i == len(sorted_class_states) - 1:
                    # 마지막 클래스에서는 next_action_value가 없으므로, prev_action_value만 고려
                    if action_value > prev_action_value or action_value > prev_prev_action_value or action_value > prev_prev_prev_action_value:
                        penalty = -2000 * abs(prev_action_value - action_value)
                        reward += penalty
                        print(f"Penalty Applied for {self.class_names[next_class_index]}: Last class received more resources than previous class, Penalty: {penalty}")
                # 낮은 우선순위 클래스가 더 높은 액션을 받았을 경우 페널티 부여
                elif i < len(sorted_class_states) - 1:
                    next_action_value = action[i]
                    next_class_index = (i) % len(self.class_names)
                    if action_value < 0.8 and next_action_value > 0.7 and self.usable_resource > 0:
                        penalty = -1500 * abs(next_action_value - 0.2)  # 낮은 우선순위 클래스가 0.2 이상 할당받은 경우 페널티
                        reward += penalty
                        print(f"Penalty Applied for {self.class_names[next_class_index]}: Lower priority class received more resources than allowed, Penalty: {penalty}")
                        if action_value - prev_action_value > threshold:
                            reward += penalty
                total_reward += reward
                total_allocated_resource += allocated_resource

        if self.previous_total_reward is not None:
            improvement = total_reward - self.previous_total_reward
            adaptive_bonus = improvement * 0.05 if improvement > 0 else 0  # 음수 보너스 제거
            total_reward += adaptive_bonus
            print(f"Adaptive Bonus Applied: {adaptive_bonus}")

        self.previous_total_reward = total_reward

        return total_reward

    def manage_data_flow(self):
        all_data = pd.concat(self.class_data.values())

        num_remove = min(random.randint(1, 5), len(all_data))
        if num_remove > 0:
            all_data = all_data.drop(all_data.sample(num_remove).index)

        if not all_data.empty:
            last_index = int(all_data.index.max()) + 1
        else:
            last_index = 0

        num_add = random.randint(1, 5)
        new_data_indices = range(last_index, min(last_index + num_add, self.dataset.shape[0]))

        if len(new_data_indices) > 0 and last_index < self.dataset.shape[0]:
            new_data = self.dataset.iloc[new_data_indices]
            all_data = pd.concat([all_data, new_data])
        else:
            print("No more data to add. Ending episode.")
            self.current_step = self.max_steps
            return

        all_data['cluster'] = self.kmeans.predict(all_data[['Priority']])
        self.classes = np.unique(all_data['cluster'])
        self.class_data = {cls: all_data[all_data['cluster'] == cls] for cls in self.classes}

        print(f"Data re-clustered. Classes updated with new stream data.")
        self.verify_class_data()

    def reset(self):
        self.current_step = 0
        self.usable_resource = self.initial_bandwidth  # 매 에피소드 시작 시 초기화
        self.class_data = {cls: self.dataset[self.dataset['cluster'] == cls] for cls in self.classes}
        return self.get_state()
