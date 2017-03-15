# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


class Bandit:
    def __init__(self, arms = 10, epsilon = 0, step_size = 0.1, initial = 0, stationary = True,
                 sample_average = True, UCB_param = None, gradient_param = None, aver_reward = 0):
        self.arms = arms
        self.epsilon = epsilon
        self.step_size = step_size
        self.indices = np.arange(self.arms)
        self.true_q = np.zeros(self.arms)
        self.est_q = np.zeros(self.arms)
        self.time = 0
        self.act_times = np.zeros(self.arms)
        self.aver_reward = 0
        self.stationary = stationary
        self.sample_average = sample_average
        self.UCB_param = UCB_param
        self.gradient_param = gradient_param
        self.act_pref = np.zeros(self.arms)
        self.aver_reward = aver_reward
        self.act_pi = np.zeros(self.arms)

        for i in range(self.arms):
            if self.stationary:
                self.true_q[i] = np.random.randn()
            else:
                self.true_q[i] = 0
            self.est_q[i] = initial

        if self.gradient_param is not None:
            for i in range(self.arms):
                self.true_q[i] = np.random.normal(self.aver_reward, 1)


        self.best_act = np.argmax(self.true_q)

    def get_action(self):
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon):  # explore
                np.random.shuffle(self.indices)
                return self.indices[0]

        if self.UCB_param is not None:
            UCB_est = self.est_q + self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.act_times + 1))
            return np.argmax(UCB_est)

        if self.gradient_param is not None:
            pref_exp = np.exp(self.act_pref)
            self.act_pi = pref_exp / sum(pref_exp)
            return np.random.choice(self.indices, p = self.act_pi)

        return np.argmax(self.est_q)

    def take_action(self, action):
        self.time += 1
        reward = self.true_q[action] + np.random.randn()
        if self.gradient_param is not None:
            for i in range(self.arms):
                if i != action:
                    self.act_pref[i] -= self.gradient_param * (reward - self.aver_reward) * self.act_pi[i]
                else:
                    self.act_pref[i] += self.gradient_param * (reward - self.aver_reward) * (1 - self.act_pi[i])
            return reward
        if not self.stationary:
            self.best_act = np.argmax(self.true_q)
            for i in range(self.arms):
                self.true_q[i] += np.random.randn()
        self.act_times[action] += 1
        if self.sample_average:
            self.est_q[action] += (reward - self.est_q[action])  / self.act_times[action]
        else:
            self.est_q[action] += (reward - self.est_q[action]) * self.step_size
        return reward


def figure2_1():
    sns.violinplot(data = np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    plt.show()


def bandit_simulation(bandits, times, n_bandits):
    aver_rewards = [np.zeros(times, dtype='float') for _ in range(len(bandits))]
    best_acts = [np.zeros(times) for _ in range(len(bandits))]
    for bandit_idx, bandit in enumerate(bandits):
        for i in range(n_bandits):
            for t in range(times):
                action = bandit[i].get_action()
                reward = bandit[i].take_action(action)
                aver_rewards[bandit_idx][t] += reward
                if action == bandit[i].best_act:
                    best_acts[bandit_idx][t] += 1
        aver_rewards[bandit_idx] /= n_bandits
        best_acts[bandit_idx] /= n_bandits
    return aver_rewards, best_acts


def epsilon_greedy(n_bandits, times):
    bandits = []
    epsilons = [0, 0.01, 0.1]
    for eps in epsilons:
        bandits.append([Bandit(epsilon= eps, stationary = True) for _ in range(n_bandits)])
    aver_reward, best_act = bandit_simulation(bandits, times, n_bandits)
    plt.subplot(211)
    for eps, reward in zip(epsilons, aver_reward):
        plt.plot(reward, label = 'epsilon = ' + str(eps))
    plt.xlabel('steps')
    plt.ylabel('average rewards')
    plt.legend()
    plt.subplot(212)
    for eps, act in zip(epsilons, best_act):
        plt.plot(act, label = 'epsilon = ' + str(eps))
    plt.xlabel('steps')
    plt.ylabel('%optimal action')
    plt.legend()

def optimistic_init(n_bandits, times):
    bandits = []
    epsilons = [0, 0.1]
    initials = [5, 0]
    for eps, init in zip(epsilons, initials):
        bandits.append([Bandit(epsilon=eps, initial=init, sample_average= True) for _ in range(n_bandits)])
    aver_reward, best_act = bandit_simulation(bandits, times, n_bandits)
    plt.subplot(211)
    for eps, init, reward in zip(epsilons, initials, aver_reward):
        plt.plot(reward, label='epsilon = ' + str(eps) + ',' + 'initial value = ' + str(init))
    plt.xlabel('steps')
    plt.ylabel('average rewards')
    plt.legend()
    plt.subplot(212)
    for eps, init, act in zip(epsilons, initials, best_act):
        plt.plot(act, label='epsilon = ' + str(eps) + ',' + 'initial value = ' + str(init))
    plt.xlabel('steps')
    plt.ylabel('%optimal action')
    plt.legend()


def UCB(n_bandits, times):
    epsilons = [0, 0.1]
    UCB_params = [2, None]
    bandits = []
    for eps, UCB_param in zip(epsilons, UCB_params):
        bandits.append([Bandit(epsilon= eps, UCB_param= UCB_param, sample_average= False) for _ in range(n_bandits)])
    aver_reward, _ = bandit_simulation(bandits, times, n_bandits)
    plt.plot(aver_reward[0], label = 'UCB c=2')
    plt.plot(aver_reward[1], label = 'e-greedy e = 0.1')
    plt.xlabel('steps')
    plt.ylabel('Average reward')
    plt.legend()


def gradient_bandit(n_bandits, times):
    gradient_params = [0.1, 0.4]
    baselines = [0, 4]
    bandits = []
    for grad, baseline in itertools.product(gradient_params, baselines):
        bandits.append([Bandit(gradient_param= grad, aver_reward= baseline) for _ in range(n_bandits)])
    _, best_act = bandit_simulation(bandits, times, n_bandits)
    for i, item in enumerate(itertools.product(gradient_params, baselines)):
        plt.plot(best_act[i], label = 'alpha = ' + str(item[0]) + ', baseline = ' + str(item[1]))
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.legend()

def figure2_6(n_bandits, times):
    parameters = [np.arange(-7, 0, dtype= 'float'),
                  np.arange(-5, 3, dtype= 'float'),
                  np.arange(-5, 3, dtype= 'float'),
                  np.arange(-2, 3, dtype= 'float')]
    labels = ['epsilon-greedy',
              'gradient bandit',
              'UCB',
              'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon= epsilon),
                 lambda step_size: Bandit(gradient_param= step_size, sample_average= False, aver_reward= 0),
                 lambda coef: Bandit(sample_average= False, UCB_param= coef),
                 lambda initial: Bandit(sample_average= False, initial= initial)]
    bandits = [[generator(pow(2, param)) for _ in range(n_bandits)] for parameter, generator in zip(parameters, generators) for param in parameter]
    aver_rewards, _ = bandit_simulation(bandits, times, n_bandits)
    aver_rewards = np.sum(aver_rewards, axis= 1) / times
    i = 0
    for parameter, label in zip(parameters, labels):
        l = len(parameter)
        plt.plot(parameter, aver_rewards[i:i+l], label = label)
        i += l
    plt.xlabel('parameters(2^x)')
    plt.ylabel('Average reward over first 1000 steps')
    plt.legend()

#optimistic_init(2000, 1000)
#UCB(2000, 1000)
#gradient_bandit(2000, 1000)
figure2_6(2000, 1000)
plt.show()
