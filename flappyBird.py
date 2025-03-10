import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
from collections import deque

# 检测是否有 CUDA 设备
device=torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化 pygame
pygame.init()

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

        reward  = 1 - abs(self.bird_y - (self.pipe_height + self.pipe_gap / 2)) * 0.01
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

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=5000):  # 限制大小，避免内存占用过多
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 训练 DQN
env = FlappyBirdEnv()
state_dim = 3
action_dim = 2

dqn = DQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
replay_buffer = ReplayBuffer(5000)  # 限制经验回放大小

epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
gamma = 0.99
batch_size = 32
episodes = 5000
best_reward = float('-inf')

def train():
    global epsilon, best_reward
    for episode in range(episodes):
        state = torch.FloatTensor(env.reset()).to(device)
        total_reward = 0
        while True:

            # if episode % 500 == 0 and best_reward>0:
            #     env.render()

            action = np.random.choice(action_dim) if random.random() < epsilon else torch.argmax(dqn(state)).item()
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
            state = next_state
            total_reward += reward

            if done:
                break
            
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = dqn(states).gather(1, actions).squeeze(1)
                next_q_values = dqn(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                loss = loss_fn(q_values, target_q_values.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
        
        # 仅在达到更高奖励时保存模型并渲染
        if total_reward > best_reward and total_reward > 0:
            best_reward = total_reward
            torch.save(dqn.state_dict(), "flappy_bird_best.pth")
            print(f"New best model saved with reward: {best_reward}")
            

if __name__ == "__main__":
    train()
