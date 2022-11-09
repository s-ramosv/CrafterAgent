#I tried to use the teachers rainbow solution for our problem

import argparse
import pickle
from pathlib import Path

import torch

from src.crafter_wrapper import Env

import itertools
import random
from argparse import Namespace
from collections import deque, defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as O
from torchvision import transforms as T
from PIL import Image

import gymnasium as gym #pip install gymnasium, install all other stuff with conda first and then use pip for the rest


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

sns.set()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"OpenAI Gym: {gym.__version__}. \t\tShould be: ~0.26.x")
print(f"PyTorch   : {torch.__version__}.  \tShould be: >=1.2.x+cu100")
print(f"DEVICE    : {DEVICE}. \t\tShould be: cuda")

def reset_rng(seed=42):
    print(f"Setting all rngs to seed={seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

reset_rng()

# Define some helpers: Gym Wrappers and visualization functions

class TorchWrapper(gym.ObservationWrapper):
    """ Applies a couple of transformations depending on the mode.
        Receives numpy arrays and returns torch tensors.
    """

    def __init__(self, env):
        super().__init__(env)
        self._transform = T.Compose([
            lambda obs: (obs * int(255 / 9)).swapaxes(1, 0),
            lambda obs: torch.from_numpy(obs).permute(2, 1, 0)
        ])
    
    def observation(self, obs):
        return self._transform(obs).unsqueeze(0).to(DEVICE)


class FrameStack(gym.Wrapper):
    """Stack k last frames. """

    def __init__(self, env, k, verbose=False):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        observation, info = self.env.reset()
        for _ in range(self.k):
            self.frames.append(observation)
        return self._get_ob(), info

    def step(self, action):
        observation, reward, termination, truncation, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_ob(), reward, termination, truncation, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        if self.k == 1:
            return self.frames.pop()
        return np.concatenate(list(self.frames), axis=2)


def plot_stats(stats, y="ep_rewards", hue=None, window=10):
    df = pd.DataFrame(stats)

    if window:
        new_col = f"avg_{y}"
        if hue is not None:
            df[new_col] = df.groupby(hue)[y].rolling(window=window).mean().reset_index(0,drop=True)
        else:
            df[new_col] = df[y].rolling(window=window).mean()
    
    y = f"avg_{y}" if window else y
    with matplotlib.rc_context({'figure.figsize':(10, 6)}):
        sns.lineplot(x="step_idx", y=y, hue=hue, data=df)

#define trainings routine
def train(agent, env, step_num=100_000):
    
    stats, N = {"step_idx": [0], "ep_rewards": [0.0], "ep_steps": [0.0]}, 0

    (state, info), done = env.reset(), False
    for step in range(step_num):

        action = agent.step(state)
        
        # separate episode termination and episode truncation signals
        # is a very recent change in the Gym API. In Crafter, these two signals
        # are subsumed by `done`.
        state_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.learn(state, action, reward, state_, done)

        # some envs just update the state and are not returning a new one
        state = state_.clone()

        # stats
        stats["ep_rewards"][N] += reward
        stats["ep_steps"][N] += 1

        if done:
            # episode done, reset env!
            (state, info), done = env.reset(), False
        
            # some more stats
            if N % 10 == 0:
                print("[{0:3d}][{1:6d}], R/ep={2:6.2f}, steps/ep={3:2.0f}.".format(
                    N, step,
                    torch.tensor(stats["ep_rewards"][-10:]).mean().item(),
                    torch.tensor(stats["ep_steps"][-10:]).mean().item(),
                ))

            stats["ep_rewards"].append(0.0)  # reward accumulator for a new episode
            stats["ep_steps"].append(0.0)    # reward accumulator for a new episode
            stats["step_idx"].append(step)
            N += 1

    print("[{0:3d}][{1:6d}], R/ep={2:6.2f}, steps/ep={3:2.0f}.".format(
        N, step, torch.tensor(stats["ep_rewards"][-10:]).mean().item(),
        torch.tensor(stats["ep_steps"][-10:]).mean().item(),
    ))
    stats["agent"] = [agent.__class__.__name__ for _ in range(N+1)]
    return stats

#implmenet sample method
class ReplayMemory:
    def __init__(self, size=1000, batch_size=32):
        self._buffer = deque(maxlen=size)
        self._batch_size = batch_size
    
    def push(self, transition):
        self._buffer.append(transition)
    
    def sample(self):
        """ Sample from self._buffer

            Should return a tuple of tensors of size: 
            (
                states:     N * (C*K) * H * W,  (torch.uint8)
                actions:    N * 1, (torch.int64)
                rewards:    N * 1, (torch.float32)
                states_:    N * (C*K) * H * W,  (torch.uint8)
                done:       N * 1, (torch.uint8)
            )

            where N is the batch_size, C is the number of channels = 3 and
            K is the number of stacked states.
        """
        # sample
        s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))

        # reshape, convert if needed, put on device (use torch.to(DEVICE))
        return (
            torch.cat(s, 0).to(DEVICE),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.cat(s_, 0).to(DEVICE),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(DEVICE)
        )
    
    def __len__(self):
        return len(self._buffer)

#ϵ-greedy schedule
def get_epsilon_schedule(start=1.0, end=0.1, steps=500):
    """ Returns either:
        - a generator of epsilon values
        - a function that receives the current step and returns an epsilon

        The epsilon values returned by the generator or function need
        to be degraded from the `start` value to the `end` within the number 
        of `steps` and then continue returning the `end` value indefinetly.

        You can pick any schedule (exp, poly, etc.). I tested with linear decay.
    """
    eps_step = (start - end) / steps
    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step
    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))


# test it, it needs to look nice
epsilon = get_epsilon_schedule(1.0, 0.1, 100)
plt.plot([next(epsilon) for _ in range(500)])

# or if you prefer a function
# epsilon_fn = get_epsilon_schedule(1.0, 0.1, 100)
# plt.plot([epsilon(step_idx) for step_idx in range(500)])

#Define a Neural Network Approximator for your Agents
class ByteToFloat(nn.Module):
    """ Converts ByteTensor to FloatTensor and rescales.
    """
    def forward(self, x):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor."
        return x.float().div_(255)


class View(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_estimator(action_num, input_ch=3, lin_size=32):
    return nn.Sequential(
        ByteToFloat(), #change that if we use environment hack
        nn.Conv2d(input_ch, 16, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, kernel_size=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, kernel_size=2),
        nn.ReLU(inplace=True),
        View(),
        nn.Linear(9 * 16, lin_size),
        nn.ReLU(inplace=True),
        nn.Linear(lin_size, action_num),
    ).to(DEVICE)

#Define a DQN Agent
class DQN:
    def __init__(
        self,
        estimator,
        buffer,
        optimizer,
        epsilon_schedule,
        action_num,
        gamma=0.92,
        update_steps=4,
        update_target_steps=10,
        warmup_steps=100,
    ):
        self._estimator = estimator
        self._target_estimator = deepcopy(estimator)
        self._buffer = buffer
        self._optimizer = optimizer
        self._epsilon = epsilon_schedule
        self._action_num = action_num
        self._gamma = gamma
        self._update_steps=update_steps
        self._update_target_steps=update_target_steps
        self._warmup_steps = warmup_steps
        self._step_cnt = 0
        assert warmup_steps > self._buffer._batch_size, (
            "You should have at least a batch in the ER.")
    
    def step(self, state):
        # implement an epsilon greedy policy using the
        # estimator and epsilon schedule attributes.

        # warning, you should make sure you are not including
        # this step into torch computation graph
        
        if self._step_cnt < self._warmup_steps:
            return torch.randint(self._action_num, (1,)).item()

        if next(self._epsilon) < torch.rand(1).item():
            with torch.no_grad():
                qvals = self._estimator(state)
                return qvals.argmax()
        else:
            return torch.randint(self._action_num, (1,)).item()


    def learn(self, state, action, reward, state_, done):

        # add transition to the experience replay
        self._buffer.push((state, action, reward, state_, done))

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            # sample from experience replay and do an update
            batch = self._buffer.sample() 
            self._update(*batch)
        
        # update the target estimator
        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(self._estimator.state_dict())

        self._step_cnt += 1

    def _update(self, states, actions, rewards, states_, done):
        # compute the DeepQNetwork update. Carefull not to include the
        # target network in the computational graph.

        # Compute Q(s, * | θ) and Q(s', . | θ^)
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)
        
        # compute Q(s, a) and max_a' Q(s', a')
        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.max(1, keepdim=True)[0]

        # compute target Q(s', a')
        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        # at this step you should check the target values
        # are looking about right :). You can use this code.
        # if rewards.squeeze().sum().item() > 0.0:
        #     print("R: ", rewards.squeeze())
        #     print("T: ", target_qsa.squeeze())
        #     print("D: ", done.squeeze())

        # compute the loss and average it over the entire batch
        loss = (qsa - target_qsa).pow(2).mean()

        # backprop and optimize
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
env = Env() #correct? #gym.make(envs.easy) #change that, it gives back an environment from gym
env = TorchWrapper(ImgObsWrapper(env))
net = get_estimator(env.action_space.n) #I think our env has no .n but an action space


stats = train(
    DQN(
        net,
        ReplayMemory(size=1000, batch_size=32),
        O.Adam(net.parameters(), lr=1e-3, eps=1e-4),
        get_epsilon_schedule(start=1.0, end=0.1, steps=4000),
        env.action_space.n, #I think our env has no .n but an action space
        warmup_steps=100,
        update_steps=2,
    ),
    env,
    step_num=7_000  # change the experiment length if it's learning but not reaching about .95
)

#there is also something with a partial observable maze - maybe this would fit better?

class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},64,64),"
        + "with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    #opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = RandomAgent(env.action_space.n)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        step_cnt += 1

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
