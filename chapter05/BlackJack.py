import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import pickle
ACTION_HIT = 0
ACTION_STICK = 1
USABLE_ACE = 1
NO_USABLE_ACE = 0
PLAYER_WIN = 0
DEALER_WIN = 1
DRAW = 2
ACTIONS = [ACTION_HIT, ACTION_STICK]

policy_player = np.zeros((22, 11, 2))
policy_dealer = np.zeros(22)
for i in range(12, 22):
    for j in range(1, 11):
        for k in range(2):
            if i < 21:
                policy_player[i][j][k] = ACTION_HIT
            else:
                policy_player[i][j][k] = ACTION_STICK

for i in range(4, 22):
    if i < 17:
        policy_dealer[i] = ACTION_HIT
    else:
        policy_dealer[i] = ACTION_STICK

state_values_usable_ace = np.zeros((22, 11))
state_values_not_usable_ace = np.zeros((22, 11))
state_times_usable_ace = np.zeros((22, 11), dtype='int')
state_times_not_usable_ace = np.zeros((22, 11),dtype='int')
state_action_values = np.zeros((22, 11, 2, 2))
state_action_times = np.zeros((22, 11, 2, 2), dtype='int')



def get_card():
    card = np.random.randint(1, 14)
    return min(card, 10)

def play(state = None, action = None):
    state_action_pairs = []
    if state is None and action is None:
        player_usable_ace = USABLE_ACE
        player_sum = 0
        player_card1 = get_card()
        while player_sum < 12:
            player_card2 = get_card()
            player_usable_ace = USABLE_ACE
            if player_card1 == 1 and player_card2 == 1:
                player_sum = 12
            elif player_card1 == 1 and player_card2 != 1:
                player_sum = 11 + player_card2
            elif player_card1 != 1 and player_card2 == 1:
                player_sum = player_card1 + 11
            else:
                player_sum = player_card1 + player_card2
                player_usable_ace = NO_USABLE_ACE
            dealer_card1 = get_card()
    else:
        player_sum, dealer_card1, player_usable_ace = state
        while action == ACTION_HIT:
            state_action_pairs.append([(player_sum, dealer_card1, player_usable_ace), ACTION_HIT])
            card = get_card()
            if card == 1 and player_usable_ace == NO_USABLE_ACE and player_sum + 11 < 22:
                player_sum += 11
                player_usable_ace = USABLE_ACE
            else:
                player_sum += card
            if player_sum > 21:
                return -1, state_action_pairs
            action = np.argmax(state_action_values[player_sum, dealer_card1, player_usable_ace, :])
        else:
            state_action_pairs.append([(player_sum, dealer_card1, player_usable_ace), ACTION_STICK])
    player_init_sum = player_sum
    player_init_ace = player_usable_ace

    dealer_card2 = get_card()
    dealer_usable_ace = USABLE_ACE
    if dealer_card1 == 1 and dealer_card2 == 1:
        dealer_sum = 12
    elif dealer_card1 == 1 and dealer_card2 != 1:
        dealer_sum = 11 + dealer_card2
    elif dealer_card1 != 1 and dealer_card2 == 1:
        dealer_sum = dealer_card1 + 11
    else:
        dealer_sum = dealer_card1 + dealer_card2
        dealer_usable_ace = NO_USABLE_ACE

    while policy_dealer[dealer_sum] == ACTION_HIT:
        card = get_card()
        if card == 1 and dealer_usable_ace == NO_USABLE_ACE:
            dealer_sum += 11
            dealer_usable_ace = USABLE_ACE
        else:
            dealer_sum += card
        if dealer_sum > 21:
            if state is not None and action is not None:
                return 1, state_action_pairs
            else:
                return PLAYER_WIN, player_init_ace, player_init_sum, dealer_card1
    if state is not None and action is not None:
        if player_sum == dealer_sum:
            return 0, state_action_pairs
        elif player_sum > dealer_sum:
            return 1, state_action_pairs
        else:
            return -1, state_action_pairs
    else:
        if player_sum == dealer_sum:
            return DRAW, player_init_ace, player_init_sum, dealer_card1
        elif player_sum > dealer_sum:
            return PLAYER_WIN, player_init_ace, player_init_sum, dealer_card1
        else:
            return DEALER_WIN, player_init_ace, player_init_sum, dealer_card1

def mc_eval(ES = False):
    global state_values_not_usable_ace, state_values_usable_ace
    global state_times_usable_ace, state_times_not_usable_ace
    iter = 0

    while True:
        iter += 1
        old_state_values_usable_ace = state_values_usable_ace.copy()
        old_state_values_not_usable_ace = state_values_not_usable_ace.copy()
        winner, usable_ace, player_sum, dealer_show = play()
        if usable_ace == True:
            state_times_usable_ace[player_sum][dealer_show] += 1
            if winner == PLAYER_WIN:
                state_values_usable_ace[player_sum][dealer_show] += (1 -
                    state_values_usable_ace[player_sum][dealer_show]) / state_times_usable_ace[player_sum][dealer_show]
            elif winner == DRAW:
                state_values_usable_ace[player_sum][dealer_show] += (0 -
                    state_values_usable_ace[player_sum][dealer_show]) / state_times_usable_ace[player_sum][dealer_show]
            else:
                state_values_usable_ace[player_sum][dealer_show] += (-1 -
                    state_values_usable_ace[player_sum][dealer_show]) / state_times_usable_ace[player_sum][dealer_show]
        else:
            state_times_not_usable_ace[player_sum][dealer_show] += 1
            if winner == PLAYER_WIN:
                state_times_not_usable_ace[player_sum][dealer_show] += (1 -
                    state_times_not_usable_ace[player_sum][dealer_show]) / state_times_not_usable_ace[player_sum][dealer_show]
            elif winner == DRAW:
                state_values_not_usable_ace[player_sum][dealer_show] += (0 -
                    state_values_not_usable_ace[player_sum][dealer_show]) / state_times_not_usable_ace[player_sum][dealer_show]
            else:
                state_values_not_usable_ace[player_sum][dealer_show] += (-1 -
                    state_values_not_usable_ace[player_sum][dealer_show]) / state_times_not_usable_ace[player_sum][dealer_show]
        if iter >= 500000:
            break


def mc_ctrl(n_episodes):
    global state_action_times, state_action_values
    state_action_times = np.ones((22, 11, 2, 2), dtype='int')
    for i in range(n_episodes):
        player_sum = np.random.choice(range(12, 22))
        dealer_show = np.random.choice(range(1, 11))
        ace = np.random.choice(range(2))
        action = np.random.choice(ACTIONS)
        reward, pairs = play(state = (player_sum, dealer_show, ace), action = action)
        for pair in pairs:
            player_sum1, dealer_show1, usable_ace1 = pair[0]
            action1 = int(pair[1])
            state_action_times[player_sum1][dealer_show1][usable_ace1][action1] += 1
            state_action_values[player_sum1][dealer_show1][usable_ace1][action1] += reward
    state_action_values /= state_action_times
    print(state_action_times)




figure_num = 0
def display(data, labels, title):
    global  figure_num
    fig = plt.figure(figure_num)
    figure_num += 1
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter([x for x,y in product(range(12, 22), range(1,11))], [y for x, y in product(range(12, 22), range(1,11))], data)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.title(title)

def display5_3():
    action_usable_ace = np.zeros((10, 10), dtype='int')
    action_no_usable_ace = np.zeros((10, 10), dtype='int')
    value_usable_ace = np.zeros((10, 10))
    value_no_usable_ace = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            action_usable_ace[i, j] = np.argmax(state_action_values[i + 12, j + 1, 1, :])
            value_usable_ace[i, j] = np.max(state_action_values[i + 12, j + 1, 1, :])
            action_no_usable_ace[i, j] = np.argmax(state_action_values[i + 12, j + 1, 0, :])
            value_no_usable_ace[i, j] = np.max(state_action_values[i + 12, j + 1, 0, :])
    display(action_usable_ace, ['player sum', 'dealer show', 'best policy'], 'usable ace')
    display(action_no_usable_ace, ['player sum', 'dealer show', 'best policy'], 'no usable ace')
    display(value_usable_ace, ['player sum', 'dealer show', 'state-value value'], 'usable ace')
    display(value_no_usable_ace, ['player sum', 'dealer show', 'state-value value'], 'no usable ace')

mc_ctrl(1000000)
display5_3()
plt.show()



