import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import tqdm
import imageio
import tkinter as tk
from tkvideo import tkvideo
import os

class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 1024)
        self.fc2a = torch.nn.Linear(1024, out_features=action_size)
        self.fc2s = torch.nn.Linear(1024, 1)

    def forward(self, state):
        x = self.conv1(state)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.conv3(x)
        x = f.relu(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = f.relu(x)
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)
        return action_values, state_value

class PreprocessAtari(ObservationWrapper):

  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()
state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("Observation shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.get_action_meanings())

learning_rate = 1e-4
discount_factor = 0.99
number_enviroments = 10

class Agent():
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def act(self, state):
        if state.ndim == 3:
            state = [state]
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        action_values, _ = self.network.forward(state)
        policy = f.softmax(action_values, dim=-1)
        return np.array([np.random.choice(len(p), p=p.cpu().numpy()) for p in policy.detach()])

    def step(self, state, action, reward, next_state, done):
        batch_size = state.shape[0]
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).to(dtype=torch.float32)
        action_values, state_value = self.network.forward(state)
        _, next_state_value = self.network.forward(next_state)
        target_state_value = reward + discount_factor * next_state_value.squeeze(-1) * (1-done)
        advantage = target_state_value - state_value.squeeze(-1)
        probs = f.softmax(action_values, dim=-1)
        logprobs = f.log_softmax(action_values, dim=-1)
        entropy = -torch.sum(probs * logprobs, axis=-1)
        batch_idx = torch.arange(batch_size, device=self.device)
        logp_actions = logprobs[batch_idx, action]
        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = f.mse_loss(target_state_value.detach(), state_value.squeeze(-1))
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


agent = Agent(number_actions)


def evaluate(agent, env, n_episodes = 1):
    episodes_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_rewards = 0
        while True:
            action = agent.act(state)
            state, rewards, done, info, _ = env.step(action[0])
            total_rewards += rewards
            if done:
                break
        episodes_rewards.append(total_rewards)
    return episodes_rewards

class EnvBatch():
    def __init__(self,  n_envs=10):
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        _states = []
        for env in self.envs:
            _states.append(env.reset()[0])
        return np.array(_states)

    def step(self, actions):
        next_state, rewards, dones, infos, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
        for i in range(len(self.envs)):
            if dones[i]:
                next_state[i] = self.envs[i].reset()[0]
        return next_state, rewards, dones, infos, _


env_batch = EnvBatch(n_envs=number_enviroments)
batch_states = env_batch.reset()

with tqdm.trange(0, 100001) as progress_bar:
    for i in progress_bar:
        batch_actions = agent.act(batch_states)
        batch_next_states, batch_rewards, batch_dones, batc_infos, _ = env_batch.step(batch_actions)
        batch_rewards *= 0.01
        agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
        batch_states = batch_next_states
        if i % 10000 == 0:
            print(f"Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))



def resize_frame(frame, block_size=16):
    height, width, _ = frame.shape
    new_width = (width + block_size - 1) // block_size * block_size
    new_height = (height + block_size - 1) // block_size * block_size
    resized_frame = np.zeros((new_height, new_width, 3), dtype=frame.dtype)
    resized_frame[:height, :width, :] = frame
    return resized_frame

def show_video_of_model(agent):
    env = make_env()
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action[0])
    env.close()
    if frames:
        imageio.mimsave('video.mp4', frames, fps=23)
    else:
        print("No frames captured.")

show_video_of_model(agent)




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
