import numpy as np

SIZE = 4
actions = ['U', 'D', 'L', 'R']
world = np.zeros((SIZE, SIZE))
next_state = []
action_reward = []
discount = 1

for i in range(SIZE):
    next_state.append([])
    action_reward.append([])
    for j in range(SIZE):
        next = dict()
        reward = dict()
        if i == 0:
            next['U'] = [i, j]
        else:
            next['U'] = [i - 1, j]
        reward['U'] = -1

        if i == SIZE - 1:
            next['D'] = [i, j]
        else:
            next['D'] = [i + 1, j]
        reward['D'] = -1

        if j == SIZE - 1:
            next['R'] = [i, j]
        else:
            next['R'] = [i, j + 1]
        reward['R'] = -1

        if j == 0:
            next['L'] = [i, j]
        else:
            next['L'] = [i, j - 1]
        reward['L'] = -1

        if i == j and (i == 0 or i == SIZE - 1):
            next['L'] = next['R'] = next['U'] = next['D'] = [i, j]
            reward['L'] = reward['R'] = reward['U'] = reward['D'] = 0
        next_state[i].append(next)
        action_reward[i].append(reward)

#figure 4.1
k = 0
while True:
    new_world = np.zeros((SIZE, SIZE))
    if k == 0 or k == 1 or k == 2 or k == 3 or k == 10:
        print('k = ', k)
        print(world)
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
    k += 1

