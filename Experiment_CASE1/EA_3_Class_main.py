import torch
import numpy as np
from torch.optim import Adam
import torch.distributions as dist

import Agent  # Agent 클래스는 Agent.py에 정의되어 있다고 가정
from EA_3_Class_env import TSNQoSEnv  # 환경 파일이 TSNQoSEnv.py에 있다고 가정
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt  # 추가된 부분: matplotlib 임포트
import time  # 추가된 부분: time 모듈 임포트
from datetime import datetime  # 추가된 부분: datetime 모듈 임포트

# 설정 (Config은 Config.py에 설정값들이 정의되어 있다고 가정)
import Config

if Config.WRITER_FLAG:
    writer = SummaryWriter(log_dir='/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/tensorboard/' + Config.WRITER_NAME)
    print(Config.WRITER_NAME)
# 지수 표기법 대신 소수점 표기법 사용
np.set_printoptions(suppress=True, precision=8, floatmode='fixed')
pd.set_option('display.float_format', lambda x: '%.8f' % x)

def format_array(arr):
    return " ".join(f"{x:.8f}" for x in arr)

def main():
    # CSV 파일에서 데이터 로드
    csv_file = 'domain_edit.csv'
    dataset = pd.read_csv(csv_file)
    
    # 환경 초기화
    env = TSNQoSEnv(dataset)
    agent = Agent.Agent(15, action_size=Config.ACTION_SIZE, batch_size=Config.BATCH_SIZE)
    
    # 손실 값을 저장할 리스트 초기화
    all_losses = []
    start_time = datetime.now()  # 학습 시작 시간
    
    # 학습 루프
    for episode in range(Config.NUM_EPISODES):
        
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0  # 스텝 카운트를 초기화
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, action_log_prob, mean, std = agent.get_action(state_tensor)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            step_count += 1  # 스텝 카운트를 증가
            
            # 경험을 메모리에 추가
            agent.add_to_memory(state, action, action_log_prob, next_state, reward, done, episode)
            state = next_state
            
            # 스텝별 결과 출력
            formatted_state = format_array(state)
            print(f"Episode {episode + 1}, Step {step_count}: Reward = {reward:.8f}, State = {formatted_state}")

            if done:
                # 에이전트가 저장된 경험을 사용하여 학습
                writer.add_scalar('reward', episode_reward, episode)
                elapsed_time = (datetime.now() - start_time).total_seconds()  # 경과 시간 계산
                
                for batch in agent.memory.batch_iter(Config.BATCH_SIZE):
                    loss = agent.update(batch)
                    if loss is not None:
                        print(f"Episode {episode}, Loss: {loss.item()}, Elapsed Time: {elapsed_time}")
                        all_losses.append(loss.item())
                        writer.add_scalar('loss_over_time', loss.item(), elapsed_time)  # 로스를 시간에 따라 기록

                        # 클래스별 로스 기록
                        for cls, class_data in env.class_data.items():
                            if not class_data.empty:
                                print(f"Episode {episode}, Class {cls}, Loss: {loss.item()}")
                                writer.add_scalar(f'loss_per_class/class_{cls}', loss.item(), episode)
                        writer.flush()  # 강제적으로 로그 파일에 기록
                    else:
                        print(f"Warning: Received None for loss at episode {episode} step {step_count}")

                
                print(f"Episode {episode + 1} ended with total Reward = {episode_reward:.8f}")
                
                agent.calculate_old_value_state()
                agent.calculate_advantage()
    
    torch.save(agent.agent_control.policy_nn.state_dict(), '/home/scpa/Policy_DDS_TSN/sin/PPO-RL-base/Experiment_CASE1/trained_policy_model_After_Case1.pth')
    print("Trained model saved as 'trained_agent_model_Case1.pth'")
    print("Training completed.")
    
if __name__ == "__main__":
    main()
