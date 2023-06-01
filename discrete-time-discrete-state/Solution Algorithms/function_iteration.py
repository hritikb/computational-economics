import numpy as np
# the below function fxn_iter executes the function iteration algorithm for solving the optimization problem
# embedded in the bellman equation

def fxn_iter(val, delta, reward_matrix, state_transition_matrix):
    # the loop continues till the values converge
    while True:

        # to get the best possible action for each state, we use amax with axis = 1 on the bellman equation modifies such that for
        # each state-action pair we add the reward matrix to the value of the state we will get to after taking the step.
        # state_transition_matrix contains the index of the state we will reach and passing these indices to val will give the value
        # for the state at that index.

        val_new = np.amax(reward_matrix + delta*val[state_transition_matrix], axis = 1)
        del_val = abs(val_new - val)

        # check convergence by euclidean norm
        if np.sqrt(sum(del_val**2)) < delta:
            break
        
        val = val_new

    # getting the best action for each state by argmax 
    policy = np.argmax(reward_matrix + delta*val[state_transition_matrix], axis = 1)
    return val_new, policy