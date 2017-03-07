# -*- coding: utf-8 -*-

import numpy as np
import pickle
BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_value = None
        self.end = None

    # 计算当前状态的hash值,每种状态都有一个唯一的hash值
    def get_hash(self):
        if self.hash_value is not None:
            return self.hash_value
        self.hash_value = 0
        for i in self.data.reshape(BOARD_SIZE):
            if i == -1:
                i = 2
            self.hash_value = self.hash_value * 3 + i
        return self.hash_value

    def is_end(self):
        results = []
        # 检查行
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # 检查列
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        # 检查对角线
        results.append(0)
        for i in range(BOARD_ROWS):
            results[-1] += self.data[i, i]
        results.append(0)
        for i in range(BOARD_ROWS):
            results[-1] += self.data[i, BOARD_ROWS - i - 1]

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end
        # 检查是否平局
        sum = np.sum(np.abs(self.data))
        if sum == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end
        self.end = False
        return self.end

    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def show(self):
        print('-------------')
        for i in range(BOARD_ROWS):
            out = '|'
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + '|'
            print(out)
        print('-------------')


def get_all_states_impl(current_symbol, current_state, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i, j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                #print(new_state.data)
                hash = new_state.get_hash()
                if hash not in all_states.keys():
                    is_end = new_state.is_end()
                    all_states[hash] = (new_state, is_end)
                    if not is_end:
                        get_all_states_impl(-current_symbol, new_state, all_states)


def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.get_hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_symbol, current_state, all_states)
    return all_states
# 获得所有可能的局势
all_states = get_all_states()


class Judger():
    def __init__(self, player1, player2, feedback = True):
        self.p1 = player1
        self.p2 = player2
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.feedback = feedback
        self.current_player = None
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.current_state = State()
        self.current_player = None

    def feed_current_state(self):
        """更新p1与p2的状态"""
        self.p1.feed_state(self.current_state)
        self.p2.feed_state(self.current_state)

    def give_reward(self):
        """给予获胜者的reward为1"""
        if self.current_state.winner == self.p1_symbol:
            self.p1.feed_reward(1)
            self.p2.feed_reward(0)
        elif self.current_state.winner == self.p2_symbol:
            self.p1.feed_reward(0)
            self.p2.feed_reward(1)
        else:
            self.p1.feed_reward(0)
            self.p2.feed_reward(1)

    def play(self, show = False):
        self.reset()
        self.feed_current_state()
        while True:
            if self.current_player == self.p1:
                self.current_player = self.p2
            else:
                self.current_player = self.p1
            if show:
                self.current_state.show()
            i, j, symbol = self.current_player.take_action()
            self.current_state = self.current_state.next_state(i, j, symbol)
            hash = self.current_state.get_hash()
            self.current_state, is_end = all_states[hash]
            self.feed_current_state()
            if is_end:
                if self.feedback:
                    # 如果是训练的话，feedback为True，此时给予获胜者reward，人机对战时feedback为False
                    self.give_reward()
                return self.current_state.winner


class Player:
    def __init__(self, step_size = 0.1, explore_rate = 0.1):
        self.states = []
        self.estimations = dict()
        self.step_size = step_size
        self.explore_rate = explore_rate

    def reset(self):
        self.states = []

    def feed_state(self, state):
        self.states.append(state)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash in all_states.keys():
            # 给每个状态的value赋初始值，若该状态为获胜则赋值为1，失败则赋值为0，其他情况赋值0.5
            state, is_end = all_states[hash]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash] = 1.
                else:
                    self.estimations[hash] = 0
            else:
                self.estimations[hash] = 0.5

    def feed_reward(self, reward):
        if len(self.states) == 0:
            return
        self.states = [state.get_hash() for state in self.states]
        target = reward
        for latest_state in reversed(self.states):
            # 更新公式v(s) = v(s) + step_size * (v(s') - v(s))
            value = self.estimations[latest_state] + self.step_size * (target - self.estimations[latest_state])
            self.estimations[latest_state] = value
            target = value
        self.states = []

    def take_action(self):
        state = self.states[-1]
        next_states = []
        next_positions = []

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_states.append(state.next_state(i, j, self.symbol).get_hash())
                    next_positions.append([i, j])

        if np.random.binomial(1, self.explore_rate):
            # epsilon-greedy策略，以explore_rate的概率选择随机策略
            np.random.shuffle(next_positions)
            action = next_positions[0]
            action.append(self.symbol)
            return action
        values = []
        for hash, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash], pos))
        # 选择贪心策略
        values.sort(key = lambda x:x[0], reverse = True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        fw = open('optimal_policy_' + str(self.symbol), 'wb')
        pickle.dump(self.estimations, fw)
        fw.close()

    def load_policy(self):
        fr = open('optimal_policy_' + str(self.symbol), 'rb')
        self.estimations = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self):
        self.symbol = None
        self.current_state = None
        return

    def take_action(self):
        data = int(input('Input your position:'))
        data -= 1
        i = data // BOARD_ROWS
        j = data % BOARD_COLS
        if self.current_state.data[i, j] != 0:
            return self.take_action()
        return (i, j, self.symbol)

    def set_symbol(self, symbol):
        self.symbol = symbol

    def feed_reward(self, reward):
        pass

    def feed_state(self, state):
        self.current_state = state

    def reset(self):
        pass


def play():
    while True:
        player1 = Player(explore_rate=0)
        player2 = HumanPlayer()
        judger = Judger(player2, player1, False)
        player1.load_policy()
        winner = judger.play(True)
        if winner == player2.symbol:
            print('Win!')
        elif winner == player1.symbol:
            print('Lose!')
        else:
            print('Tie!')

def train(epochs = 20000):
    player1 = Player()
    player2 = Player()
    judger = Judger(player1, player2, True)  #第三个参数为True表示没进行完一局比赛便更新valuefunction
    player1_win = 0
    player2_win = 0
    for i in range(epochs):
        print('Epoch', i)
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print(player1_win / epochs)
    print(player2_win / epochs)
    player1.save_policy()
    player2.save_policy()


#train()
play()
