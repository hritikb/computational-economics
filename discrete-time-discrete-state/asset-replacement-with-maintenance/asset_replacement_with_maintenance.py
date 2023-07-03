import numpy as np
import matplotlib.pyplot as plt
import sys
 
# adding folder with solution algorithms to the system path
sys.path.insert(0, 'D:\Projects\Computational Economics\discrete-time-discrete-state\Solution Algorithms')
from function_iteration import fxn_iter

max_age = 5
rep_cost = 75
maint_cost = 10
delta = 0.9

age_space = np.arange(1, max_age + 1)
service_space = np.arange(0, max_age)

# state space is a max_age x max_age matrix with age as rows and number of services as columns
state_space = []
for i in range(len(age_space)):
    for j in range(len(service_space)):
        state_space.append((age_space[i], service_space[j]))

# state_space = np.array(state_space)
action_space = np.array([0, 1, 2]) # 0: keep, 1: service, 2: replace

def transition_fxn(age, service, action):
    if action == 0:
        return min(age + 1, max_age), service
    elif action == 1:
        return min(age + 1, max_age), min(service + 1, max_age - 1)
    else:
        return 1, 0
    
def reward_fxn(age, service):
    return (1 - (age - service)/5)*(50 - 2.5*age - 2.5*age**2)


# state_transition_function = np.zeros((3, max_age, max_age))
# for i in range(len(age_space)):
#     for j in range(len(service_space)):
#         new_age, new_service = transition_fxn(age_space[i], service_space[j], 0)
#         state_transition_function[0, i, j] = [int(np.where(age_space == new_age)[0][0]), int(np.where(service_space == new_service)[0][0])]

#         new_age, new_service = transition_fxn(age_space[i], service_space[j], 1)
#         state_transition_function[1, i, j] = [int(np.where(age_space == new_age)[0][0]), int(np.where(service_space == new_service)[0][0])]

#         new_age, new_service = transition_fxn(age_space[i], service_space[j], 2)
#         state_transition_function[2, i, j] = [0, 0]

state_transition_matrix = np.zeros((len(state_space), len(action_space)), dtype=int)
for i in range(len(state_space)):
    new_age, new_service = transition_fxn(state_space[i][0], state_space[i][1], 0)
    state_transition_matrix[i, 0] = int(np.where(age_space == new_age)[0][0]*len(service_space) + np.where(service_space == new_service)[0][0])

    
    new_age, new_service = transition_fxn(state_space[i][0], state_space[i][1], 1)
    state_transition_matrix[i, 1] = int(np.where(age_space == new_age)[0][0]*len(service_space) + np.where(service_space == new_service)[0][0])

    
    new_age, new_service = transition_fxn(state_space[i][0], state_space[i][1], 2)
    state_transition_matrix[i, 2] = int(np.where(age_space == new_age)[0][0]*len(service_space) + np.where(service_space == new_service)[0][0])

# reward_matrix = np.zeros((3, max_age, max_age))
# for i in range(len(age_space)):
#     for j in range(len(service_space)):
#         reward_matrix[0, i, j] = reward_fxn(age_space[i], service_space[j])
#         reward_matrix[1, i, j] = reward_fxn(age_space[i], service_space[j] + 1) - maint_cost
#         reward_matrix[2, i, j] = reward_fxn(0, 0) - rep_cost

reward_matrix = np.zeros((len(state_space), len(action_space)))
for i in range(len(state_space)):
    reward_matrix[i, 0] = reward_fxn(state_space[i][0], state_space[i][1])
    reward_matrix[i, 1] = reward_fxn(state_space[i][0], state_space[i][1] + 1) - maint_cost
    reward_matrix[i, 2] = reward_fxn(0, 0) - rep_cost

reward_matrix[-len(service_space):, 0:2] = -np.inf

v = np.zeros((len(state_space)))

opt_val, opt_policy = fxn_iter(v, delta, reward_matrix, state_transition_matrix)

a0 = 1
s0 = 0
age = [a0]
service = [s0]
year = 0
while year < 12:
    # print(age_space == a0)
    # print(np.where(age_space == a0)[0][0]*len(service_space) + np.where(service_space == s0)[0][0])
    act = opt_policy[np.where(age_space == a0)[0][0]*len(service_space) + np.where(service_space == s0)[0][0]]

    a0, s0 = transition_fxn(a0, s0, act)
    age.append(a0)
    service.append(s0)
    year += 1

plt.plot(age, label='Age')
plt.plot(service, label='Service')
plt.xlabel('Year')
plt.ylabel('Age/Service')
plt.legend()
plt.savefig('D:/Projects/Computational Economics/discrete-time-discrete-state/asset-replacement-with-maintenance/optimal_state_path.png')
plt.show()


