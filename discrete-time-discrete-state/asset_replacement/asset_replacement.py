import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'D:\Projects\Computational Economics\discrete-time-discrete-state\Solution Algorithms')

from function_iteration import fxn_iter

max_age = 5
# replacement cost
c = 75

delta = 0.9

state_space = np.arange(1, max_age + 1)

action_space = np.array([0, 1]) # 0: keep, 1: replace

# state_transition_matrix = np.concatenate((np.arange(2, max_age + 2).T, np.array([1]*max_age).T), axis = 1)
state_transition_matrix = np.zeros((len(state_space), len(action_space)))
for i in range(len(state_space)):
    state_transition_matrix[i, 0] = np.where(min(state_space[i] + 1, max_age) == state_space)[0][0]
    state_transition_matrix[i, 1] = np.where(1 == state_space)[0][0]

state_transition_matrix = state_transition_matrix.astype(int)

reward_matrix = np.zeros((len(state_space), len(action_space)))
for i in range(len(state_space)):
    reward_matrix[i, 0] = 50 - 2.5*state_space[i] - 2.5*state_space[i]**2
    reward_matrix[i, 1] = 50 - c
reward_matrix[len(state_space) - 1, 0] = -np.inf
# print(reward_matrix)
v = np.zeros(len(state_space))
# print(state_transition_matrix)

opt_val, opt_policy = fxn_iter(v, delta, reward_matrix, state_transition_matrix)

plt.plot(state_space, opt_val)
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([160, 170, 180, 190, 200, 210, 220])
plt.xlabel("Machine's age")
plt.ylabel('Value')
# plt.title('The plot shows the expected value to the manager for the possible values of the stock of ore')
plt.savefig('d:/Projects/Computational Economics/discrete-time-discrete-state/asset_replacement/Optimal Value Function.png')
plt.show()