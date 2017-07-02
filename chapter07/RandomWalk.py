import numpy as np
import matplotlib.pyplot as plt
import time

ACTIONS = [-1, 1]
alphas = np.arange(0, 1.1, 0.1)
TRUE_VALUE = [x / 10 for x in range(-9, 10)]
state_value = [0 for _ in range(-9, 10)]
START = 9
cur_pos = START
n_episode = 10
n_steps = [2**x for x in range(10)]
state_his = []
reward_his = []
total_reward = 0
repetition = 100
rms = 0
rmses = []
plt.figure()


for n_step in n_steps:
    rmses.clear()
    for alpha in list(alphas):
        rms = 0
        for n in range(repetition):
            state_value = [0 for _ in range(-9, 10)]
            for episode in range(n_episode):
                state_his = [START]
                reward_his.clear()
                cur_pos = START
                #state_value = [0 for _ in range(-9, 10)]
                while cur_pos != -1 and cur_pos != 19:
                    action = np.random.choice(ACTIONS)
                    cur_pos += action
                    reward = 0
                    if cur_pos ==-1 and action == -1:
                        reward = -1
                    if cur_pos == 19 and action == 1:
                        reward = 1
                    reward_his.append(reward)
                    state_his.append(cur_pos)
                state_his.pop()
                for i, state in enumerate(state_his):
                    total_reward = sum(reward_his[i:i+n_step])
                    g = total_reward
                    if i + n_step < len(state_his):
                        #print('i+n_step', i + n_step, len(state_his))
                        #print('state_his[i+nstep]', state_his[i + n_step], len(state_value))
                        g += state_value[state_his[i+n_step]]

                    state_value[state] += alpha * (g - state_value[state])
                rms += np.sqrt(np.sum(np.square((np.asarray(TRUE_VALUE) - np.asarray(state_value)))) / len(TRUE_VALUE))
        rmses.append(rms / repetition / n_episode)
    #print(rmses)
    plt.plot(alphas, rmses, label='n='+str(n_step))
    plt.xlabel('alpha')
    plt.ylabel('Average RMS error over 19 states and first 10 episodes')
    plt.legend()
print('end')
plt.show()







