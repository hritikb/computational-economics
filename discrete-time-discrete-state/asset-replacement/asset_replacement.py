import numpy as np
import matplotlib.pyplot as plt
import sys
# add the path of the folder containing the function_iteration.py file to the system path
sys.path.insert(0, 'D:\Projects\Computational Economics\discrete-time-discrete-state\Solution Algorithms')
from function_iteration import fxn_iter

# maximum age of the machine
max_age = 5
# replacement cost
c = 75
# discount factor
delta = 0.9
# state space is the possible age of the machine. Since once the machine is replaced we use it for an year and get some reward and
# end up with a machine of age 1, the maximum age of the machine is 5
state_space = np.arange(1, max_age + 1)
action_space = np.array([0, 1]) # 0: keep, 1: replace

# initialize state transition matrix. It gives the index of the state to which the system is transitioned when an action is taken in a
state_transition_matrix = np.zeros((len(state_space), len(action_space)))
for i in range(len(state_space)):
    state_transition_matrix[i, 0] = np.where(min(state_space[i] + 1, max_age) == state_space)[0][0]
    state_transition_matrix[i, 1] = np.where(1 == state_space)[0][0]

state_transition_matrix = state_transition_matrix.astype(int)

# initialize reward matrix for each state-action pair
reward_matrix = np.zeros((len(state_space), len(action_space)))
for i in range(len(state_space)):
    reward_matrix[i, 0] = 50 - 2.5*state_space[i] - 2.5*state_space[i]**2
    reward_matrix[i, 1] = 50 - c
# since the machine cannot be kept after 5 years we put reward to be -inf to ensure that
reward_matrix[len(state_space) - 1, 0] = -np.inf 

# initialize value function
v = np.zeros(len(state_space))

# run the function iteration algorithm
opt_val, opt_policy = fxn_iter(v, delta, reward_matrix, state_transition_matrix)

# plot the optimal value function
plt.plot(state_space, opt_val)
plt.xticks([1, 2, 3, 4, 5])
plt.yticks([160, 170, 180, 190, 200, 210, 220])
plt.xlabel("Machine's age")
plt.ylabel('Value')
# plt.title('The plot shows the expected value to the manager for the possible values of the stock of ore')
plt.savefig('d:/Projects/Computational Economics/discrete-time-discrete-state/asset-replacement/optimal_value_function.png')
plt.show()

# find the optimal state path and plot it
s = 1
age = [1]
year = 0
rewards = []
# print(state_transition_matrix)
while year <= 12:
    action = opt_policy[np.where(state_space == s)][0]
    rewards.append(reward_matrix[np.where(state_space == s), action][0][0])

    s = state_space[state_transition_matrix[np.where(state_space == s)[0][0], action]]
    age.append(s)
    year += 1

plt.plot([i for i in range(year + 1)], age)
plt.xticks([0, 2, 4, 6, 8, 10, 12])
plt.yticks([1, 2, 3, 4])
plt.xlabel('Year')
plt.ylabel('Age if machine')
plt.savefig('D:/Projects/Computational Economics/discrete-time-discrete-state/asset-replacement/optimal_state_path.png')
plt.show()