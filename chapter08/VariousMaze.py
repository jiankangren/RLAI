import numpy as np
import random
import matplotlib.pyplot as plt
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ROWS, COLS = 6, 9

state_action_values = np.zeros((ROWS, COLS, len(ACTIONS)), dtype=np.float32)
START = [2, 0]
GOAL = [0, 8]
OBSTACLES = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

EPSILON = 0.1
STEP_SIZE = 0.1
gamma = 0.9

before = []
after = []
steps = np.zeros(51)
results = []


def walk(pos, direction):
    if direction == UP:
        new_pos = [max(0, pos[0] - 1), pos[1]]

    if direction == DOWN:
        new_pos = [min(ROWS - 1, pos[0] + 1), pos[1]]

    if direction == LEFT:
        new_pos = [pos[0], max(0, pos[1] - 1)]

    if direction == RIGHT:
        new_pos = [pos[0], min(COLS - 1, pos[1] + 1)]

    if new_pos not in OBSTACLES:
        return new_pos
    else:
        return pos


for n in [0, 5, 50]:
    for repetition in range(30):
        episodes = [0]

        state_action_values = np.zeros((ROWS, COLS, len(ACTIONS)), dtype=np.float32)
        while len(episodes) <= 50:
            cur_pos = START

            step = 0

            while cur_pos != GOAL:
                #print('episode %d, step %d' % (len(episodes), step))

                step += 1
                if np.random.binomial(1, EPSILON):
                    action = np.random.choice(ACTIONS)
                    #print(action)
                else:
                    action = np.argmax(state_action_values[cur_pos[0], cur_pos[1]])
                    rando = []
                    for tmp in ACTIONS:
                        if state_action_values[cur_pos[0], cur_pos[1], action] == state_action_values[cur_pos[0], cur_pos[1], tmp]:
                            rando.append(tmp)
                    action = random.choice(rando)
                new_pos = walk(cur_pos, action)
                if new_pos == GOAL:
                    reward = 1
                else:
                    reward = 0

                state_action_values[cur_pos[0], cur_pos[1], action] += STEP_SIZE * (reward
                        + gamma * max(state_action_values[new_pos[0], new_pos[1]]) - state_action_values[cur_pos[0], cur_pos[1], action])

                if [cur_pos[0], cur_pos[1], action] not in before:
                    before.append([cur_pos[0], cur_pos[1], action])
                    after.append([new_pos[0], new_pos[1], reward])
                else:
                    idx = before.index([cur_pos[0], cur_pos[1], action])
                    after[idx] = [new_pos[0], new_pos[1], reward]

                assert len(before) == len(after)
                for i in range(n):
                    select = np.random.randint(len(before))
                    before_state = [before[select][0], before[select][1]]
                    action = before[select][2]
                    after_state = [after[select][0], after[select][1]]
                    reward = after[select][2]
                    state_action_values[before_state[0], before_state[1], action] += STEP_SIZE * (reward + gamma * max(
                        state_action_values[after_state[0], after_state[1]]) - state_action_values[before_state[0], before_state[1], action])

                cur_pos = new_pos

            episodes.append(step)
        #print(episodes)
        steps += (np.asarray(episodes) - steps) / (repetition + 1)
    print(steps)
    results.append(np.copy(steps))

print(results)
plt.plot(results[0][1:], label='0 step')
plt.plot(results[1][1:], label='5 step')
plt.plot(results[2][1:], label='50 step')
plt.legend()
plt.show()


