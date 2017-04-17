import numpy as np
import matplotlib.pyplot as plt

LEFT = -1
RIGHT = 1
ACTIONS = [LEFT, RIGHT]
STATES = list(range(7))
state_values = np.ones(7) / 2
state_values[0] = state_values[-1] = 0
sum_of_error = np.zeros(6)
trajectorys = []
mc_rewards = []

def init_values():
    global state_values
    state_values = np.ones(7) / 2
    state_values[0] = state_values[-1] = 0

def init_errors():
    global sum_of_error
    sum_of_error = np.zeros(6)


def td(alpha = 0.1, batch = False):
    global state_values, sum_of_error, trajectorys
    cur = 3
    cur_trajectory = [cur]
    while cur != 0 and cur != 6:
        action = np.random.choice(ACTIONS)
        next = cur + action
        cur_trajectory.append(next)
        if cur == 5 and next == 6:
            reward = 1
        else:
            reward = 0
        if not batch:
            state_values[cur] += alpha * (reward + state_values[next] - state_values[cur])
        cur = next
    if batch:
        trajectorys.append(cur_trajectory)
        while True:
            init_errors()
            for trajectory in trajectorys:
                for i in range(len(trajectory) - 1):
                    if trajectory[i] == 5 and trajectory[i + 1] == 6:
                        reward = 1
                    else:
                        reward = 0
                    sum_of_error[trajectory[i]] += reward + state_values[trajectory[i + 1]] - state_values[trajectory[i]]
            if alpha * np.sum(np.abs(sum_of_error)) < 1e-3:
                break
            for state in range(1, 6):
                state_values[state] += alpha * sum_of_error[state]

def mc(alpha = 0.1, batch = False):
    global state_values, sum_of_error, trajectorys, mc_rewards
    cur = 3
    states = [cur]
    cur_trajectory = [cur]
    while cur != 0 and cur != 6:
        action = np.random.choice(ACTIONS)
        next = cur + action
        cur_trajectory.append(next)
        if next not in states:
            states.append(next)
        cur = next
    if cur == 6:
        reward = 1
    else:
        reward = 0
    mc_rewards.append(reward)
    for state in states:
        if not batch:
            state_values[state] += alpha * (reward - state_values[state])
    if batch:
        trajectorys.append(cur_trajectory)
        while True:
            init_errors()
            for trajectory, reward in zip(trajectorys, mc_rewards):
                for i in range(len(trajectory) - 1):
                    sum_of_error[trajectory[i]] += reward  - state_values[trajectory[i]]
            if alpha * np.sum(np.abs(sum_of_error)) < 1e-3:
                break
            for state in range(1, 6):
                state_values[state] += alpha * sum_of_error[state]


def figure6_2_left():
    plt.figure()
    n_episodes = 100
    for i in range(n_episodes + 1):
        if i == 0 or i == 1 or i == 10 or i == 100:
            plt.plot([x for x in range(1, 6)], state_values[1:6], label=str(i) + ' episodes')
        td()
    plt.plot([x for x in range(1, 6)], [y * 1./6 for y in range(1, 6)], label= 'true value')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()
    plt.show()

def figure6_2_right():
    plt.figure()
    n_episodes = 100
    alphas = [0.05, 0.1, 0.15]
    true_values = np.asarray([x * 1. / 6 for x in range(1, 6)])
    for alpha in alphas:
        init_values()
        rms = np.zeros(n_episodes + 1)
        for i in range(n_episodes + 1):
            rms[i] = np.sqrt(np.sum(np.power(true_values - state_values[1:6], 2)) / 5.0)
            td(alpha)
        plt.plot([x for x in range(n_episodes + 1)], rms, label = 'td alpha = ' + str(alpha))
    alphas = [0.01, 0.02, 0.03, 0.04]
    for alpha in alphas:
        init_values()
        rms = np.zeros(n_episodes + 1)
        for i in range(n_episodes + 1):
            rms[i] = np.sqrt(np.sum(np.power(true_values - state_values[1:6], 2)) / 5.0)
            mc(alpha)
        plt.plot([x for x in range(n_episodes + 1)], rms, label = 'mc alpha = ' + str(alpha))
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()
    plt.show()

def figure6_3():
    global state_values, trajectorys, mc_rewards
    n_episodes = 100
    runs = 100
    true_values = np.asarray([x * 1. / 6 for x in range(1, 6)])

    rms = np.zeros(n_episodes + 1)
    for i in range(runs):
        init_values()
        trajectorys = []
        print('td run:' + str(i))
        for episode in range(1, n_episodes + 1):
                init_errors()
                td(alpha=1e-3, batch=True)
                rms[episode] += np.sqrt(np.sum(np.power(state_values[1:6] - true_values, 2)) / 5.0)
    rms /= runs
    plt.plot(range(1, n_episodes + 1), rms[1:], label='TD')




    rms = np.zeros(n_episodes + 1)
    for i in range(runs):
        init_values()
        trajectorys = []
        mc_rewards = []
        print('mc run:' + str(i))
        for episode in range(1, n_episodes + 1):
            init_errors()
            mc(alpha=1e-3, batch=True)
            rms[episode] += np.sqrt(np.sum(np.power(state_values[1:6] - true_values, 2)) / 5.0)
    rms /= runs
    plt.plot(range(1, n_episodes + 1), rms[1:], label = 'MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()
    plt.show()


#figure6_2_right()
figure6_3()



