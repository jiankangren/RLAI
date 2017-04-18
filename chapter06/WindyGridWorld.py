import numpy as np
import matplotlib.pyplot as plt

ROWS = 7
COLS = 10


class WindyWorld:
    def __init__(self):
        self.left = 0
        self.right = 1
        self.up = 2
        self.down = 3
        self.actions = [self.left, self.right, self.up, self.down]
        self.state_action_values = np.zeros((ROWS, COLS, len(self.actions)))
        self.states = []
        for i in range(ROWS):
            for j in range(COLS):
                self.states.append([i, j])
        self.start = [3, 0]
        self.goal = [3, 7]
        self.cur_pos = self.start
        self.policy = np.zeros((ROWS, COLS))


    def move(self, direction):
        if direction == self.left:
            row, col = self.cur_pos
            if 3 <= col <= 5 or col == 8:
                self.cur_pos[0] = max(row - 1, 0)
                self.cur_pos[1] = col - 1
            elif 6 <= col <= 7:
                self.cur_pos[0] = max(row - 2, 0)
                self.cur_pos[1] = col - 1
            else:
                self.cur_pos[1] = max(col - 1, 0)
        elif direction == self.right:
            row, col = self.cur_pos
            if 3 <= col <= 5 or col == 8:
                self.cur_pos[0] = max(row - 1, 0)
                self.cur_pos[1] = col + 1
            elif 6 <= col <= 7:
                self.cur_pos[0] = max(row - 2, 0)
                self.cur_pos[1] = col + 1
            else:
                self.cur_pos[1] = min(col + 1, COLS - 1)
        elif direction == self.up:
            row, col = self.cur_pos
            if 3 <= col <= 5 or col == 8:
                self.cur_pos[0] = max(row - 2, 0)
            elif 6 <= col <= 7:
                self.cur_pos[0] = max(row - 3, 0)
            else:
                self.cur_pos[0] = max(row - 1, 0)
        else:
            row, col = self.cur_pos
            if 3 <= col <= 5 or col == 8:
                pass
            elif 6 <= col <= 7:
                self.cur_pos[0] = max(row - 1, 0)
            else:
                self.cur_pos[0] = min(row + 1, ROWS - 1)
        if self.cur_pos == self.goal:
            return 0
        else:
            return -1

    def online_td_ctrl(self, epsilon = 0.1, alpha = 0.5):
        episodes = 0
        episode_list = []
        time_step = 0
        while True:
            self.cur_pos = self.start[:]
            while self.cur_pos != self.goal:
                row, col = self.cur_pos
                prob = 1 - epsilon
                if np.random.binomial(1, prob):
                    action = np.argmax(self.state_action_values[row, col, :])
                else:
                    action = np.random.randint(len(self.actions))
                reward = self.move(action)
                episode_list.append(episodes)
                time_step += 1
                print('episodes:', episodes, 'time step:', time_step)
                next_row, next_col = self.cur_pos
                if np.random.binomial(1, prob):
                    next_action = np.argmax(self.state_action_values[next_row, next_col, :])
                else:
                    next_action = np.random.randint(len(self.actions))
                cur_value = self.state_action_values[row, col, action]
                next_value = self.state_action_values[next_row, next_col, next_action]
                self.state_action_values[row, col, action] += alpha * (reward + next_value - cur_value)
            episodes += 1
            if time_step >= 8000:
                break
        plt.figure()
        plt.plot([x for x in range(8000)], episode_list[:8000])
        plt.xlabel('time steps')
        plt.ylabel('episodes')
        plt.show()

windy_grid_world = WindyWorld()
windy_grid_world.online_td_ctrl()


