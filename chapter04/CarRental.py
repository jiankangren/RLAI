import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


MAX_CAR_NUM = 20
MAX_MOV_OF_CARS = 5
EXPT_REQUEST_FIR_LOC = 3
EXPT_REQUEST_SEC_LOC = 4
EXPT_RETURN_FIR_LOC = 3
EXPT_RETURN_SEC_LOC = 2
MOV_COST = 2
DISCOUNT = 0.9
CREDIT = 10
POSSION_UP_BOUND = 11

state_value = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1))
policy = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1))
states = []
actions = np.arange(-MAX_MOV_OF_CARS, MAX_MOV_OF_CARS + 1)

figure_num = 0
def display(data, labels):
    global figure_num
    fig = plt.figure(figure_num)
    figure_num += 1
    ax = fig.add_subplot(111, projection = '3d')
    zs = []
    for i, j in states:
        zs.append(data[i, j])
    ax.scatter([x for x, y in states], [y for x, y in states], zs)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

for i in range(MAX_CAR_NUM + 1):
    for j in range(MAX_CAR_NUM + 1):
        states.append([i, j])

possion_backup = dict()
def possion(n ,lambd):
    key = 10 * n + lambd
    if key not in possion_backup.keys():
        possion_backup[key] = pow(lambd, n) * np.exp(-lambd) / np.math.factorial(n)
    return possion_backup[key]

def expect_return(state, action):
    global state_value
    returns = 0.0
    returns -= MOV_COST * abs(action)
    for request_fir_loc in range(POSSION_UP_BOUND):
        for request_sec_loc in range(POSSION_UP_BOUND):
            cars_fir_loc = min(state[0] - action, MAX_CAR_NUM)
            cars_sec_loc = min(state[1] + action, MAX_CAR_NUM)
            real_rental_fir_loc = min(request_fir_loc, cars_fir_loc)
            real_rental_sec_loc = min(request_sec_loc, cars_sec_loc)
            cars_fir_loc -= real_rental_fir_loc
            cars_sec_loc -= real_rental_sec_loc
            reward = (real_rental_fir_loc + real_rental_sec_loc) * CREDIT
            prob = possion(request_fir_loc, EXPT_REQUEST_FIR_LOC) * possion(request_sec_loc, EXPT_REQUEST_SEC_LOC)
            const_ret = False
            if const_ret:
                for ret_fir_loc in range(POSSION_UP_BOUND):
                    for ret_sec_loc in range(POSSION_UP_BOUND):
                        prob_ = possion(ret_fir_loc, EXPT_RETURN_FIR_LOC) * possion(ret_sec_loc, EXPT_RETURN_SEC_LOC) * prob
                        cars_fir_loc_ = int(min(cars_fir_loc + ret_fir_loc, MAX_CAR_NUM))
                        cars_sec_loc_ = int(min(cars_sec_loc + ret_sec_loc, MAX_CAR_NUM))
                        returns += prob_ * (reward + DISCOUNT * state_value[cars_fir_loc_, cars_sec_loc_])
            else:
                cars_fir_loc_ = int(min(cars_fir_loc + EXPT_RETURN_FIR_LOC, MAX_CAR_NUM))
                cars_sec_loc_ = int(min(cars_sec_loc + EXPT_RETURN_SEC_LOC, MAX_CAR_NUM))
                returns += prob * (reward + DISCOUNT * state_value[cars_fir_loc_, cars_sec_loc_])

    return returns

new_state_value = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1))
improve_policy = False
policy_improve_idx = 1
while True:
    if improve_policy:
        new_policy = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1))
        print('improvement', policy_improve_idx)
        policy_improve_idx += 1
        for i, j in states:
            action_value = []
            for action in actions:
                if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                    action_value.append(expect_return([i, j], action))
                else:
                    action_value.append(-float('inf'))
            best_action = actions[np.argmax(action_value)]
            new_policy[i, j] = best_action
        policy_changed_num = np.sum(new_policy != policy)
        policy[:] = new_policy
        print(policy_changed_num,'policies changed')
        improve_policy = False
        if policy_changed_num == 0:
            break

    #policy evaluation
    for i, j in states:
        new_state_value[i, j] = expect_return([i,j], policy[i, j])
    if np.sum(np.abs(new_state_value - state_value)) < 1e-4:
        state_value[:] = new_state_value
        improve_policy = True
        continue
    state_value[:] = new_state_value

display(policy, ['cars in first location', 'cars in second location', 'cars to move during night'])
display(state_value, ['cars in first location', 'cars in second location', 'expected returns'])
plt.show()

