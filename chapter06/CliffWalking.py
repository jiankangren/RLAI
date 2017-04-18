import numpy as np
import matplotlib.pyplot as plt

class Cliff:
    def __init__(self):
        self.ROWS = 4
        self.COLS = 12
        self.ACTIONS = list(range(4))
        self.UP, self.DOWN, self.LEFT, self.RIGHT = self.ACTIONS
        self.state_action_values = np.zeros((self.ROWS, self.COLS,len(self.ACTIONS)))
        self.start = [self.ROWS - 1, 0]
        self.goal = [self.ROWS - 1, self.COLS - 1]
        self.n_episodes = 500


    def move(self, direction):
        reward = -1
        if direction == self.UP:
            self.cur_pos[0] = max(self.cur_pos[0] - 1, 0)
        elif direction == self.DOWN:
            if self.cur_pos[0] == 2 and 1 <= self.cur_pos[1] <= 10:
                reward = -100
                self.cur_pos = self.start[:]
                return reward
            elif self.cur_pos[0] == 2 and self.cur_pos[1] == 11:
                reward = 0
            self.cur_pos[0] = min(self.cur_pos[0] + 1, self.ROWS - 1)
        elif direction == self.LEFT:
            self.cur_pos[1] = max(self.cur_pos[1] - 1, 0)
        else:
            if self.cur_pos == [self.ROWS - 1, 0]:
                self.cur_pos = self.start[:]
                reward = -100
                return reward
            self.cur_pos[1] = min(self.cur_pos[1] + 1, self.COLS - 1)
        return reward

    def ctrl(self, method, alpha = 0.5, epsilon = 0.1):
        rewards = []
        self.state_action_values = np.zeros((self.ROWS, self.COLS, len(self.ACTIONS)))
        for ep in range(self.n_episodes):
            self.cur_pos = self.start[:]
            total_reward = 0
            while self.cur_pos != self.goal:
                row, col = self.cur_pos
                prob = 1 - epsilon
                if np.random.binomial(1, prob):
                    action = np.argmax(self.state_action_values[row, col, :])
                else:
                    action = np.random.choice(self.ACTIONS)
                reward = self.move(action)
                total_reward += reward
                next_row, next_col = self.cur_pos
                if method == 'sarsa':
                    if np.random.binomial(1, prob):
                        next_action = np.argmax(self.state_action_values[next_row, next_col, :])
                    else:
                        next_action = np.random.choice(self.ACTIONS)
                elif method == 'q':
                    next_action = np.argmax(self.state_action_values[next_row, next_col, :])
                cur_value = self.state_action_values[row, col, action]
                next_value = self.state_action_values[next_row, next_col, next_action]
                self.state_action_values[row, col, action] += alpha * (reward + next_value - cur_value)
            rewards.append(total_reward)
        return np.asarray(rewards)

def figure6_5():
    cliff = Cliff()
    run = 20
    sarsa_reward = cliff.ctrl(method = 'sarsa')
    q_reward = cliff.ctrl(method='q')
    for i in range(run - 1):
        print('run:', i)
        sarsa_reward += cliff.ctrl(method = 'sarsa')
        q_reward += cliff.ctrl(method='q')
    sarsa_reward = sarsa_reward / run
    q_reward = q_reward / run
    for i in range(len(sarsa_reward)):
        sarsa_reward[i] = max(np.mean(sarsa_reward[i: i + 10]), -100)
        q_reward[i] = max(np.mean(q_reward[i : i + 10]), -100)
    plt.figure()
    plt.plot([x for x in range(cliff.n_episodes)], sarsa_reward, label = 'Sarsa')
    plt.plot([x for x in range(cliff.n_episodes)], q_reward, label = 'Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()
    plt.show()

figure6_5()