import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
from PIL import Image
from torchvision import transforms
import imageio
import tkinter as tk
from tkvideo import tkvideo


class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        # kernel size diz respeito a matrix de convolução e stride a quantos
        # pixels a matriz vai andar em cada iteração
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(10 * 10 * 128, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, action_size)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Montando o ambiente Pac-Man usando a biblioteca Gymnastics
env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

# Hiperparâmetros
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99


# Pré-Processamento de frames
def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    return preprocess(frame).unsqueeze(0)


# Implementando DCQN
class Agent():

    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > minibatch_size:
            experiences = random.sample(self.memory, k=minibatch_size)
            self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Inicialização

agent = Agent(number_actions)

# Treinamento

number_episodes = 2000
maximum_number_timesteps_per_episode = 10000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen=100)

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end="")
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    if np.mean(scores_on_100_episodes) >= 500:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                     np.mean(scores_on_100_episodes)))
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break


# Cria um vídeo com uma execução do cenário com o modelo treinado - Testado no Windows 11


def resize_frame(frame, block_size=16):
    height, width, _ = frame.shape
    new_width = (width + block_size - 1) // block_size * block_size
    new_height = (height + block_size - 1) // block_size * block_size
    resized_frame = np.zeros((new_height, new_width, 3), dtype=frame.dtype)
    resized_frame[:height, :width, :] = frame
    return resized_frame


def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        resized_frame = resize_frame(frame)  # Resize frame
        frames.append(resized_frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    if frames:
        imageio.mimsave('video.mp4', frames, fps=30)
    else:
        print("No frames captured.")


show_video_of_model(agent, 'MsPacmanDeterministic-v0')


def play_video(video_path):
    window = tk.Tk()
    window.title("Windows 11 Video Player")
    window.geometry("800x600")  # Set initial window size
    window.configure(bg="#f3f3f3")  # Set the background color to a light Windows 11 style

    video_label = tk.Label(window)
    video_label.pack(expand=True, fill="both", padx=20, pady=20)

    player = tkvideo(video_path, video_label, loop=1, size=(800, 450))
    player.play()

    window.eval('tk::PlaceWindow . center')

    style_frame = tk.Frame(window, bg="#f3f3f3", bd=0)
    style_frame.pack(fill="x")
    tk.Label(style_frame, text="Windows 11 Video Player", font=("Segoe UI", 14), bg="#f3f3f3").pack(pady=5)

    window.mainloop()


play_video(os.path.abspath('video.mp4'))