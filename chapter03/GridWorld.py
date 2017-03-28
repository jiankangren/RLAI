import numpy as np

SIZE = 5
actions = ['U', 'D', 'L', 'R']
world = np.zeros((SIZE, SIZE))
next_state = []
action_reward = []
A_pos = [0, 1]
A_prime_pos = [4, 1]
B_pos = [0, 3]
B_prime_pos = [2, 3]
discount = 0.9

for i in range(SIZE):
    next_state.append([])
    action_reward.append([])
    for j in range(SIZE):
        next = dict()
        reward = dict()
        if i == 0:
            next['U'] = [i, j]
            reward['U'] = -1
        else:
            next['U'] = [i - 1, j]
            reward['U'] = 0

        if i == SIZE - 1:
            next['D'] = [i, j]
            reward['D'] = -1
        else:
            next['D'] = [i + 1, j]
            reward['D'] = 0

        if j == SIZE - 1:
            next['R'] = [i, j]
            reward['R'] = -1
        else:
            next['R'] = [i, j + 1]
            reward['R'] = 0

        if j == 0:
            next['L'] = [i, j]
            reward['L'] = -1
        else:
            next['L'] = [i, j - 1]
            reward['L'] = 0

        if [i, j] == A_pos:
            next['L']=next['R']=next['U']=next['D'] = A_prime_pos
            reward['L']=reward['R']=reward['U']=reward['D'] = 10
        if [i, j] == B_pos:
            next['L'] = next['R'] = next['U'] = next['D'] = B_prime_pos
            reward['L'] = reward['R'] = reward['U'] = reward['D'] = 5

        next_state[i].append(next)
        action_reward[i].append(reward)

#figure 3.5
while True:
    new_world = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            for action in actions:
                new_pos = next_state[i][j][action]
                new_world[i, j] += 0.25 * (action_reward[i][j][action] + discount * world[new_pos[0], new_pos[1]])
    if np.sum(np.abs(world - new_world)) < 1e-4:
        print('random policy')
        print(new_world)
        break
    world = new_world

#figure 3.8
while True:
    new_world = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            values = []
            for action in actions:
                new_pos = next_state[i][j][action]
                values.append(action_reward[i][j][action] + discount * world[new_pos[0], new_pos[1]])
            new_world[i, j] = np.max(values)
    if np.sum(np.abs(world - new_world)) < 1e-4:
        print('optimal policy')
        print(new_world)
        break
    world = new_world