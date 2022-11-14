#I tried to use the teachers rainbow solution for our problem and used the random agent as a template

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

import gymnasium as gym #pip install gymnasium, install all other stuff with conda first and then use pip for the rest, not used


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #this is not safe!!!

#define trainings routine
def train(agent, env, step_num, opt):
    
    stats, N = {"step_idx": [0], "ep_rewards": [0.0], "ep_steps": [0.0]}, 0

    state, done = env.reset(), False #before: (state, info), done = env.reset(), False
    for step in range(step_num):

        action = agent.step(state)
        
        # separate episode termination and episode truncation signals
        # is a very recent change in the Gym API. In Crafter, these two signals
        # are subsumed by `done`.
        state_, reward, done, info = env.step(action) #before: state_, reward, terminated, truncated, info = env.step(action)
        #done = terminated or truncated
        
        agent.learn(state, action, reward, state_, done, opt)

        # some envs just update the state and are not returning a new one
        state = state_.clone()

        # stats
        stats["ep_rewards"][N] += reward
        stats["ep_steps"][N] += 1

        # evaluate once in a while
        if step % opt.eval_interval == 0:
            eval(agent, env, step, opt)

        if done:
            # episode done, reset env!
            state, done = env.reset(), False #before: (state, info), done = env.reset(), False
        
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

class ReplayMemory:
    def __init__(self, size=1000, batch_size=32):
        self._buffer = deque(maxlen=size)
        self._batch_size = batch_size
    
    def push(self, transition):
        self._buffer.append(transition)
    
    def sample(self, opt):
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

        #only for testing
        #t = torch.tensor(d)
        #t.unsqueeze(1)
        #t.to(opt.device)

        # reshape, convert if needed, put on device (use torch.to(DEVICE))
        return (
            torch.cat(s, 0).to(opt.device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(opt.device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(opt.device),
            torch.cat(s_, 0).to(opt.device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(opt.device)
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
        return x.float() #at crafter we have gray scales which is already between 0 and 1
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor."
        return x.float().div_(255)


class View(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_estimator(action_num, opt, input_ch=1, lin_size=32):
    #the input of the network is [1, 128, 64, 64] and the output is [lin_size, action_num]
    return nn.Sequential(
        ByteToFloat(),
        nn.Conv2d(input_ch, 64, kernel_size=4, stride=3),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        View(),
        nn.Linear(4096, lin_size),
        nn.ReLU(),
        nn.Linear(lin_size, action_num),
    ).to(opt.device)

""" def get_estimator(action_num, opt, input_ch=1, lin_size=32):
    #the input of the network is [1, 128, 64, 64] and the output is [lin_size, action_num]
    return nn.Sequential(
        ByteToFloat(),
        nn.Conv2d(input_ch, 32, kernel_size=8, stride=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        View(),
        nn.Linear(1024, lin_size),
        nn.ReLU(inplace=True),
        nn.Linear(lin_size, action_num),
    ).to(opt.device) """

""" def get_estimator(action_num, opt, input_ch=3, lin_size=32):
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
    ).to(opt.device)
 """

class DQLAgent:
    """Deep Q Learning Agent"""

    def __init__(self, estimator,
        buffer,
        optimizer,
        epsilon_schedule,
        action_num,
        gamma=0.92,
        update_steps=4,
        update_target_steps=10,
        warmup_steps=100,
    ) -> None:
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

        #update self policy here? maybe like this:
        # self._policy = EpsilonGreedyPolicy(estimator, epsilon_schedule)
        #self._policy = GreedyPolicy(estimator)

        # self.action_num = action_num
        # # a uniformly random policy
        # self.policy = torch.distributions.Categorical(
        #     torch.ones(action_num) / action_num
        #)

    def step(self, state):
        # implement an epsilon greedy policy using the
        # estimator and epsilon schedule attributes.

        # warning, you should make sure you are not including
        # this step into torch computation graph
        
        if self._step_cnt < self._warmup_steps:
            return torch.randint(self._action_num, (1,)).item()

        if next(self._epsilon) < torch.rand(1).item():
            with torch.no_grad():
                qvals = self._estimator(torch.unsqueeze(state, 0).transpose(0, 1)) #before: qvals = self._estimator(state)
                return qvals.argmax()
        else:
            return torch.randint(self._action_num, (1,)).item()


    def learn(self, state, action, reward, state_, done, opt):

        # add transition to the experience replay
        self._buffer.push((state, action, reward, state_, done))

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            # sample from experience replay and do an update
            batch = self._buffer.sample(opt) #before: batch = self._buffer.sample()
            self._update(*batch)
        
        # update the target estimator
        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(self._estimator.state_dict())

        self._step_cnt += 1

    def _update(self, states, actions, rewards, states_, done):
        # compute the DeepQNetwork update. Carefull not to include the
        # target network in the computational graph.

        # Compute Q(s, * | θ) and Q(s', . | θ^)
        #print(states.shape)
        #states = torch.unsqueeze(states, 0)
        #print(states.shape)
        q_values = self._estimator(torch.unsqueeze(states, 0).transpose(0, 1)) #before: q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(torch.unsqueeze(states_, 0).transpose(0, 1))
        
        # compute Q(s, a) and max_a' Q(s', a')
        #print("actions", actions.shape)
        #print("q_values.shape: ", q_values.shape)
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
    
    #update this method
    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        #return self.policy.sample().item()


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
            #action = agent.act(obs) #agent.step gives also an action -> use agent.step
            action = agent.step(state=obs)

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

    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)

    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"OpenAI Gym: {gym.__version__}. \t\tShould be: ~0.26.x")
    print(f"PyTorch   : {torch.__version__}.  \tShould be: >=1.2.x+cu100")
    print(f"DEVICE    : {opt.device}. \t\tShould be: cuda")

    print(env.action_space.n)
    
    net = get_estimator(env.action_space.n, opt) 

    agent = DQLAgent(
        net,
        ReplayMemory(size=1000, batch_size=64),
        O.Adam(net.parameters(), lr=1e-3, eps=1e-4),
        get_epsilon_schedule(start=1.0, end=0.1, steps=4000),
        env.action_space.n,
        warmup_steps=100,
        update_steps=2,
    )
                    
    train(agent=agent, env=env, step_num=opt.steps, opt=opt) #like this and then no loop?

    """ # main loop
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
            eval(agent, eval_env, step_cnt, opt) """


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
        default=1, #before =4
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