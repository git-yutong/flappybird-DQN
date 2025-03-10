import numpy as np
import torch
import torch.nn as nn
import pygame
import random

# 设备检测
device = torch.device("cpu")

# 画面设置
WIDTH, HEIGHT = 400, 500
WHITE = (255, 255, 255)
BIRD_COLOR = (255, 255, 0)
PIPE_COLOR = (0, 255, 0)

# Flappy Bird 环境（可视化版）
class FlappyBirdEnv:
    def __init__(self):
        self.gravity = 1
        self.lift = -10
        self.bird_y = 250
        self.bird_vel = 0
        self.pipe_x = 400
        self.pipe_gap = 150
        self.pipe_height = random.randint(100, 300)
        self.done = False
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.bird_y = 250
        self.bird_vel = 0
        self.pipe_x = 400
        self.pipe_height = random.randint(100, 300)
        self.done = False
        return np.array([self.bird_y, self.pipe_x, self.pipe_height])

    def step(self, action):
        if action == 1:
            self.bird_vel = self.lift
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel
        self.pipe_x -= 5

        if self.pipe_x < -50:
            self.pipe_x = 400
            self.pipe_height = random.randint(100, 300)

        reward = 1
        if self.bird_y > HEIGHT or self.bird_y < 0 or (
                self.pipe_x < 50 and (self.bird_y < self.pipe_height or self.bird_y > self.pipe_height + self.pipe_gap)):
            self.done = True
            reward = -100

        return np.array([self.bird_y, self.pipe_x, self.pipe_height]), reward, self.done
    
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, PIPE_COLOR, (self.pipe_x, 0, 50, self.pipe_height))
        pygame.draw.rect(self.screen, PIPE_COLOR, (self.pipe_x, self.pipe_height + self.pipe_gap, 50, HEIGHT))
        pygame.draw.circle(self.screen, BIRD_COLOR, (50, int(self.bird_y)), 10)
        pygame.display.flip()
        pygame.time.wait(10)
        self.clock.tick(30)

# DQN 神经网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# 载入模型进行测试
def test():
    env = FlappyBirdEnv()
    state_dim = 3
    action_dim = 2

    dqn = DQN(state_dim, action_dim).to(device)
    dqn.load_state_dict(torch.load("flappy_bird_best.pth", map_location=device))
    dqn.eval()  # 设为评估模式

    state = torch.FloatTensor(env.reset()).to(device)

    while True:
        env.render()
        with torch.no_grad():
            action = torch.argmax(dqn(state)).item()

        next_state, reward, done = env.step(action)
        state = torch.FloatTensor(next_state).to(device)

        if done:
            state = torch.FloatTensor(env.reset()).to(device)

if __name__ == "__main__":
    test()
