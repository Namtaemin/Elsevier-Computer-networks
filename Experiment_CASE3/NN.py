#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import Config

torch.manual_seed(Config.SEED)
class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape, Config):
        super(PolicyNN, self).__init__()
        self.actions_mean = nn.Sequential(
            nn.Linear(input_shape, Config.hidden_size_1),
            nn.Tanh(),
            nn.Linear(Config.hidden_size_1, Config.hidden_size_2),
            nn.Tanh(),
            nn.Linear(Config.hidden_size_2, Config.hidden_size_3),
            nn.Tanh(),
            nn.Linear(Config.hidden_size_3, output_shape)
            )
        self.actions_logstd = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x, actions=None):
        actions_mean = self.actions_mean(x)
        actions_logstd = self.actions_logstd
        actions_std = torch.exp(actions_logstd)

        # 정규 분포 생성
        normal_dist = Normal(actions_mean, actions_std)

        if actions is None:
            actions = normal_dist.rsample()
        
        # sigmoid를 적용하여 액션을 [0, 1] 사이로 제한

        # 로그 확률 계산
        actions_logprob = normal_dist.log_prob(actions)
        actions = torch.sigmoid(actions)
        return actions, actions_logprob, torch.sum(normal_dist.entropy(), dim=-1), actions_mean, actions_std

class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)