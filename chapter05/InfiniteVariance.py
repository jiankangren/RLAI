import numpy as np
import matplotlib.pyplot as plt

BACK = 0
END = 1
ACTIONS = [BACK, END]

def get_return():
    trajectory = []
    while np.random.binomial(1, 0.5):
        trajectory.append(BACK)
        if np.random.binomial(1, 0.1):
            return 1, trajectory
    trajectory.append(END)
    return 0, trajectory

def figure5_5():
    runs = 10
    n_episodes = 10000000
    for run in range(runs):
        value = 0
        sum_of_above = [0]
        for episode in range(n_episodes):
            reward, trajectory = get_return()
            rho = 1.
            for action in trajectory:
                if action == END:
                    rho = 0
                    break
                else:
                    rho *= 2.
            sum_of_above.append(sum_of_above[-1] + rho * reward)
        del sum_of_above[0]
        sum_of_above = np.asarray(sum_of_above) / np.arange(1, n_episodes + 1)
        plt.plot([np.log10(x) for x in range(1, n_episodes + 1)], sum_of_above)
        plt.xlabel('episodes(10^x)')
        plt.ylabel('ordinary importance sampling')

figure5_5()
plt.show()

