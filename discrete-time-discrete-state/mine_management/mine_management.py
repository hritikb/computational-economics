import numpy as np
import matplotlib.pyplot as plt
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, 'D:\Projects\Computational Economics\discrete-time-discrete-state\Solution Algorithms')
from function_iteration import fxn_iter


# let initial stock of ore be 100 ton and price of 1 dollar per ton
s0 = 100
price = 1

state_space = np.arange(101)  # state space is the possible amount of ore that can be available at each step
action_space = np.arange(101) # action space is the amount of ore (in integer quantities) that can be mined

# setting the discount factor to 0.9 and convergence tolerance to 10^-5
delta = 0.9
tol = 1e-5

    
# defining cost function by the assumption: cost = action^2/(1 + state)
def cost(state, action):
    return action**2/(1 + state)

# reward is (price*quantity of ore mined - cost of mining the ore). To put the constraint that mining more than available ore is not
# allowed we put reward to be negative infinity in the reward matrix when such is the case.
def reward(state, action):
    if action > state:
        return -np.inf
    
    else:
        return price*action - cost(state, action)

# this will be a deterministic state transition rule. Negative inf reward will prevent mining more than available ore however in
# state transition function we put 0 as the remaining stock in such cases
def state_transition(state, action):
    return max(state - action, 0)

# initialize reward matrix for each state-action pair
reward_matrix = np.zeros((len(state_space), len(action_space)))
for i in range(len(state_space)):
    for j in range(len(action_space)):
        reward_matrix[i, j] = reward(state_space[i], action_space[j])

# initialize state transition function

state_transition_matrix = np.zeros((len(state_space), len(action_space)))

for i in range(len(state_space)):
    for j in range(len(action_space)):
        transitioned_value = state_transition(state_space[i], action_space[j])
        state_transition_matrix[i, j] = np.where(state_space == transitioned_value)[0][0]
state_transition_matrix = state_transition_matrix.astype(int)
v = np.zeros(len(state_space))

opt_val, opt_policy = fxn_iter(v, delta, reward_matrix, state_transition_matrix)

plt.plot(state_space, opt_val)
plt.xticks([0, 20, 40, 60, 80, 100])
plt.yticks([0, 10, 20, 30, 40, 50, 60])
plt.xlabel('Stock')
plt.ylabel('Value')
# plt.title('The plot shows the expected value to the manager for the possible values of the stock of ore')
plt.savefig('D:/Projects/Computational Economics/discrete-time-discrete-state/mine_management/Optimal Value Function')
plt.show()

plt.plot(state_space, action_space[opt_policy])
plt.xticks([0, 20, 40, 60, 80, 100])
plt.yticks([0, 5, 10, 15, 20, 25])
plt.xlabel('Stock')
plt.ylabel('Extraction')
# plt.title('The plot shows the optimal extraction strategy for the possible values of the stock of ore')
plt.savefig('D:/Projects/Computational Economics/discrete-time-discrete-state/mine_management/Optimal Extraction Policy')
plt.show()

s = s0
content = [s0]
year = 0
rewards = []
while s > 0:
    action = action_space[opt_policy[np.where(state_space == s)]]
    rewards.append(reward(s, action)[0])
    s = state_transition(s, action)[0]
    content.append(s)
    year += 1

plt.plot([i for i in range(year + 1)], content)
plt.xticks([0, 5, 10, 15])
plt.yticks([0, 20, 40, 60, 80, 100])
plt.xlabel('Year')
plt.ylabel('Stock')
# plt.title('The plot shows the optimal extraction strategy for the possible values of the stock of ore')
plt.savefig('D:/Projects/Computational Economics/discrete-time-discrete-state/mine_management/Optimal State Path')
plt.show()

plt.plot([i + 1 for i in range(year)], rewards)
plt.xticks([0, 5, 10, 15])
plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
plt.xlabel('Year')
plt.ylabel('Reward')
# plt.title('The plot shows the optimal extraction strategy for the possible values of the stock of ore')
plt.savefig('D:/Projects/Computational Economics/discrete-time-discrete-state/mine_management/Yearwise reward')
plt.show()