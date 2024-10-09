
# Instalar as seguintes bibliotecas - Talvez outras sejam necessárias - Visual Studio C++  necessário para o box2d
##pip install gymnasium
##pip install "gymnasium[atari, accept-rom-license]"
##apt-get install -y swig
##pip install gymnasium[box2d]

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym

# Criando o "Cérebro" - Neural Networks

class Cerebro(nn.Module):
    def __init__(self, state_size, action_size, seed = 42):
        super(Cerebro, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcl = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self,state):
        x = self.fcl(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
    
    
#Carregando o cenário Lunar Lander
    
cenario = gym.make("LunarLander-v2")
state_size = cenario.observation_space.shape[0]
state_shape = cenario.observation_space.shape
acoes = cenario.action_space.n
print('State_Shape: ', state_shape)
print('State_Size: ', state_size)
print('Ações: ', acoes)
    
# Inicializando Hiperparâmetros
    
learning_rate = 5e-4
mini_batch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3
    
# Implementando o "Experience Replay"
class ReplayMemory (object):
    def __init__(self, capacidade):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacidade = capacidade
        self.memoria = []
        
    def push(self, evento):
        self.memoria.append(evento)
        if len(self.memoria) > self.capacidade: 
            del self.memoria[0]
    
    def sample(self, batch_size):
        experiencia = random.sample(self.memoria, k = batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiencia if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiencia if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiencia if e is not None])).float().to(self.device)
        next_state = torch.from_numpy(np.vstack([e[3] for e in experiencia if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiencia if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_state, actions, rewards, dones

#Implementando DQN

class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Cerebro(state_size, action_size).to(self.device)
        self.target_qnetwork = Cerebro(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memoria) > mini_batch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)    
                
                
    def act(self, state, epsilon =  0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())    
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)
        
    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data )
            

agent = Agent(state_size, acoes)

number_episodes = 2000
maximum_steps = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores = deque(maxlen=100)

for episodes in range(1, number_episodes + 1):
    state, _ = cenario.reset()
    score = 0
    for t in range(maximum_steps):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = cenario.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value *epsilon)
    
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episodes, np.mean(scores)), end = "")
    if episodes % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episodes, np.mean(scores)))
    if np.mean(scores)>=250.0:
        print('\nCenário Solucionado em {:d} episódios\tAverage Score: {:.2f}'.format(episodes - 100, np.mean(scores)))
        try:
            save_path = os.path.abspath('checkpoint.pth')
            torch.save(agent.local_qnetwork.state_dict(), save_path)
            print(f"Model saved successfully at {save_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
        break

# Cria um vídeo com uma execução do cenário com o modelo treinado - Testado no Windows 11   
     

import imageio

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

show_video_of_model(agent, 'LunarLander-v2')




import tkinter as tk
from tkvideo import tkvideo

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