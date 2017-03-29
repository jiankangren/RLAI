import numpy as np
import matplotlib.pyplot as plt

TARGET = 100
P_H = 0.4
REWARD = 1000
states = np.arange(1, TARGET, dtype = 'int')
state_values = np.zeros(TARGET + 1)
policy = dict()
best_policy = np.zeros(TARGET + 1)
for state in states:
    policy[state] = [x for x in range(1, min(state, TARGET - state) + 1)]
    best_policy[state] = 1
    state_values[state] = 0
state_values[TARGET] = 1

sweep = 0
while True:
    sweep += 1
    new_state_values = np.zeros(TARGET + 1)
    for state in states:
        action_value = []
        for action in policy[state]:
            suc_return = P_H * (REWARD if action + state >= TARGET else 0 + state_values[action + state])
            fail_return = (1 - P_H) * (state_values[state - action])
            action_value.append(suc_return + fail_return)
        new_state_values[state] = max(action_value)
        best_policy[state] = policy[state][np.argmax(action_value)]
    if sweep in [1, 2, 3, 32]:
        plt.plot([x for x in range(1, TARGET)], new_state_values[1 : TARGET], label = 'sweep ' + str(sweep))
    if np.sum(np.abs(state_values - new_state_values)) < 1e-9:
        break
    state_values[:] = new_state_values
plt.legend()
plt.figure()
plt.scatter([x for x in range(1, TARGET)], best_policy[1 : TARGET])
plt.show()



