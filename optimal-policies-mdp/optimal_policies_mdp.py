# ---------------(b)---------------

import random as rd
import numpy as np
import matplotlib.pyplot as plt

max_state = 10    # 1 = new, 10 = failed
deter_P = 0.1     # probability of deterioration per component per time step
correct_cost = 25 # corrective cost per failed component 
up_reward = 1     # reward when the system is up and operates

# 1.------------------------------
def step(state):
    """
    One time step under the 'no preventive repair' policy.
    state: (i, j) with i, j in {1, ..., 10}
    returns: (next_state, reward)
    """
    i, j = state
    # Policy: if both components are working -> operate, else corrective repair
    if i < max_state and j < max_state:
        # operate
        reward = up_reward
        # independent deterioration of each component
        if rd.random() < deter_P:
            i = min(i+1, max_state)
        if rd.random() < deter_P:
            j = min(j+1, max_state)
    else:
        # corrective repair of all failed components
        failed = (1 if i == max_state else 0) + (1 if j == max_state else 0)
        reward = - correct_cost*failed
        # repair all failed components, others stay as they are
        if i == max_state:
            i = 1
        if j == max_state:
            j = 1

    return (i, j), reward  


def simulate(T, seed):
    """
    Simulate T time steps and return the estimated long-run average reward
    """
    rd.seed(seed)       # Fix seed to ensure reproducible simulation results
    state = (1, 1)      # start with two new components
    total_reward = 0.0
    for _ in range(T):
        state, r = step(state)
        total_reward += r

    return total_reward / T


avg_reward = simulate(T=1_000_000, seed=543)
print(f"Estimated long-run average reward (no preventive repair): {avg_reward:.4f}")


# 2.------------------------------

n_levels = max_state
n_states = n_levels ** 2

def idx(i, j):             # indexing helper 1
    """Map (i,j) with i,j in {1,...,10} to index in {0,...,99}."""
    return (i - 1) * n_levels + (j - 1)

def inv_idx(k):            # indexing helper 2
    """Inverse mapping: index -> (i,j)."""
    i = k // n_levels + 1
    j = k % n_levels + 1
    return i, j

# Note:
# The state space has only 100 states (10x10), so building and storing the
# full 100x100 transition matrix is extremely cheap. This makes the
# stationary-distribution and Poisson-equation methods run very fast.

def transit_P():            # transition matrix P
    P = np.zeros((n_states, n_states))

    for i in range(1, max_state + 1):
        for j in range(1, max_state + 1):
            s = idx(i, j)

            if i < max_state and j < max_state:
                # both components are working
                i_same = i
                i_up = min(i + 1, max_state)
                j_same = j
                j_up = min(j + 1, max_state)

                # possible next states
                P[s, idx(i_same, j_same)] += (1 - deter_P) * (1 - deter_P)
                P[s, idx(i_up,   j_same)] += deter_P * (1 - deter_P)
                P[s, idx(i_same, j_up)] += (1 - deter_P) * deter_P
                P[s, idx(i_up,   j_up)] += deter_P * deter_P

            else:
                # corrective repair of all failed components
                i_next = 1 if i == max_state else i
                j_next = 1 if j == max_state else j
                P[s, idx(i_next, j_next)] = 1.0

    return P


def reward_v():               # Reward vector
    r = np.zeros((n_states))

    for k in range(n_states):
        i, j = inv_idx(k)

        if i < max_state and j < max_state:
            # operate
            r[k] = up_reward
        else:
            # corrective repair of all failed components
            failed = (1 if i == max_state else 0) + (1 if j == max_state else 0)
            r[k] = - correct_cost * failed

    return r

# forward recursion to approximate stationary distribution
def stat_distr(P, tol=1e-12, max_iter=1_000_000): 
    # start from uniform distribution
    pi = np.ones((n_states)) / n_states

    for _ in range(max_iter):
        pi_next = pi@P
        if np.max(np.abs(pi_next - pi)) < tol:
            break
        pi = pi_next

    pi /= pi.sum()         # ensure normalization 

    return pi


def long_run_average_reward():
    P  = transit_P()
    r  = reward_v()
    pi = stat_distr(P)

    g = pi @ r

    return g, pi


g, pi = long_run_average_reward()
print(f"Long-run average reward from stationary distribution: {g:.4f}")


# 3.------------------------------

def poisson_value_iteration(P, r, tol=1e-12, max_iter=1_000_000):
    """
    Solving the average-reward Poisson equation
        g + V(s) = r(s) + sum_{s'} P(s,s') V(s')
    via relative value iteration.

    Returns:
        g_est  : estimated long-run average reward
        V     : differential value function (normalized with V(s0) = 0)
    """
    N  = len(r)
    V  = np.zeros(N, dtype=float)
    s0 = 0                     # # reference state index

    for _ in range(max_iter):
        V_tilde = r + P@V
        g_est = V_tilde[s0] - V[s0]
        V_new = V_tilde - V_tilde[s0]

        diff = np.max(np.abs(V_new - V))
        if diff < tol:
            return g_est, V_new
        
        V = V_new

    # if not converged within max_iter, still return last estimate
    return g_est, V

P = transit_P()
r = reward_v()
g_est, V = poisson_value_iteration(P, r, tol=1e-12)
print(f"Average reward from Poisson equation / value iteration: {g_est:.4f}")


#--------------(c)---------------
# average reward value iteration with preventive repair of the healthy component
# when the other has failed

prevent_cost = 5                # cost per preventive repair
# actions:
# 0 = operate
# 1 = correct_only
# 2 = correct_and_prev_other
# 3 = correct_both

def step_Q(i, j, action, V):
    """
    Computing Q(i,j,action) = r + sum p * V(next_state)
    without building the full transition matrix.
    """
    # operate (only if both working)
    if action == 0:
        r = up_reward
        q = 0.0
        for di, pi in [(0, 1 - deter_P), (1, deter_P)]:
            ni = min(i + di, max_state)
            for dj, pj in [(0, 1 - deter_P), (1, deter_P)]:
                nj = min(j + dj, max_state)
                prob = pi*pj
                q += prob * V[idx(ni, nj)]
        return r + q
    
    # corrective only (repair all failed, no preventive)
    if action == 1:
        failed = (1 if i == max_state else 0) + (1 if j == max_state else 0)
        r = - correct_cost * failed
        ni = 1 if i == max_state else i
        nj = 1 if j == max_state else j
        return r + V[idx(ni, nj)]
    
    # corrective + preventive of the healthy other
    if action == 2:
        # exactly one failed, one healthy
        r = - correct_cost - prevent_cost
        return r + V[idx(1, 1)]
    
    # both failed: corrective of both
    if action == 3:
        r = - 2 * correct_cost
        return r + V[idx(1, 1)]

    raise ValueError("unknown action")


def allowed_actions(i, j):
    # both working
    if i < 10 and j < 10:
        return [0]          # operate only
    # both failed
    if i == 10 and j == 10:
        return [3]          # correct_both
    # one failed, one working
    if (i == 10) ^ (j == 10):
        return [1, 2]       # correct_only or correct_and_prev_other
    # should not happen
    return []


def average_reward_value_iteration(tol=1e-8, max_iter=10000):
    # iteration stops when diff = max(abs(V_new[k] - V[k])<1e-8
    # which is more than sufficient given the scale of the rewards.
    N = max_state * max_state
    V = [0.0] * N
    policy = [0] * N
    s0 = 0   # reference state (1,1)

    for it in range(max_iter):
        V_new = [0.0] * N
        g_est = 0.0

        for k in range(N):
            i, j = inv_idx(k)
            best_q = -1e18
            best_a = 0
            for a in allowed_actions(i, j):
                q = step_Q(i, j, a, V)
                if q > best_q:
                    best_q = q
                    best_a = a
            V_new[k] = best_q
            policy[k] = best_a
            if k == s0:
                g_est = best_q - V[k]  # estimate of average reward

        # normalize so that V_new(s0) = 0
        ref = V_new[s0]
        for k in range(N):
            V_new[k] -= ref

        # check convergence
        diff = max(abs(V_new[k] - V[k]) for k in range(N))
        # print(it, diff, g_est)  # debug if you want

        V = V_new
        if diff < tol:
            break

    return g_est, V, policy


g_star, V, policy = average_reward_value_iteration()
print(f"Optimal long-run average reward (part c): {g_star:.4f}")

# build a 10x10 table: 1 if we do corrective+preventive, else 0
table = [[0 for _ in range(max_state)] for _ in range(max_state)]
for k in range(max_state * max_state):
    i, j = inv_idx(k)
    if policy[k] == 2:       # correct_and_prev_other
        table[i-1][j-1] = 1

print("Policy table (1 = corrective + preventive of healthy component):")
for row in table:
    print(row)


# Plot of Optimal Policy

policy_array = np.zeros((max_state, max_state), dtype=int)
for k in range(max_state * max_state):
    i, j = inv_idx(k)
    if policy[k] == 2:   # correct_and_prev_other
        policy_array[i-1, j-1] = 1

plt.figure(figsize=(6, 6))
plt.imshow(policy_array, cmap="Greys", origin="lower")
plt.colorbar(label="1 = corrective + preventive")
plt.xlabel("State of component 2")
plt.ylabel("State of component 1")
plt.title("Optimal policy for part (c)")
plt.xticks(range(0, 10), range(1, 11))
plt.yticks(range(0, 10), range(1, 11))
plt.savefig("policy_part_c.png", dpi=300, bbox_inches='tight')
plt.show()


#--------------(d)---------------
# average reward value iteration with unrestricted preventive repair
# preventive actions allowed in all states

# actions:
# 0 = operate
# 1 = correct_only
# 2 = correct_and_prev_other
# 3 = correct_both
# 4 = prev1          (preventive repair of component 1)
# 5 = prev2          (preventive repair of component 2)
# 6 = prev_both      (preventive repair of both components)

def step_Q(i, j, action, V):
    """
    Compute Q(i,j,action) = r + sum p * V(next_state)
    without building the full transition matrix.
    """

    # operate (only if both working)
    if action == 0:
        r = up_reward
        q = 0.0
        for di, pi in [(0, 1 - deter_P), (1, deter_P)]:
            ni = min(i + di, max_state)
            for dj, pj in [(0, 1 - deter_P), (1, deter_P)]:
                nj = min(j + dj, max_state)
                prob = pi * pj
                q += prob * V[idx(ni, nj)]
        return r + q

    # pure preventive repair, no failure 
    if action == 4:          # prev1
        r = - prevent_cost
        ni, nj = 1, j        # comp 1 repaired, comp 2 frozen
        return r + V[idx(ni, nj)]

    if action == 5:          # prev2
        r = - prevent_cost
        ni, nj = i, 1        # comp 2 repaired, comp 1 frozen
        return r + V[idx(ni, nj)]

    if action == 6:          # prev_both
        r = - 2 * prevent_cost
        ni, nj = 1, 1
        return r + V[idx(ni, nj)]

    # corrective only (repair all failed, no preventive)
    if action == 1:
        failed = (1 if i == max_state else 0) + (1 if j == max_state else 0)
        r = - correct_cost * failed
        ni = 1 if i == max_state else i
        nj = 1 if j == max_state else j
        return r + V[idx(ni, nj)]

    # corrective + preventive of the healthy other 
    if action == 2:
        # exactly one failed, one healthy
        r = - correct_cost - prevent_cost
        ni, nj = 1, 1
        return r + V[idx(ni, nj)]

    # both failed: corrective of both
    if action == 3:
        r = - 2 * correct_cost
        ni, nj = 1, 1
        return r + V[idx(ni, nj)]

    raise ValueError("unknown action")


def allowed_actions(i, j):
    """
    Part (d): preventive repair of 1 or 2 components in ANY state.
    """
    # both working
    if i < 10 and j < 10:
        return [0, 4, 5, 6]      # operate, prev1, prev2, prev_both

    # both failed
    if i == 10 and j == 10:
        return [3]               # correct_both

    # exactly one failed
    if (i == 10) ^ (j == 10):
        return [1, 2]            # correct_only or correct_and_prev_other

    # should not happen
    return []


def average_reward_value_iteration(tol=1e-8, max_iter=10000):
    # iteration stops when diff = max(abs(V_new[k] - V[k])<1e-8
    # which is more than sufficient given the scale of the rewards.
    N = max_state * max_state
    V = [0.0] * N
    policy = [0] * N
    s0 = 0   # reference state (1,1)

    for it in range(max_iter):
        V_new = [0.0] * N
        g_est = 0.0

        for k in range(N):
            i, j = inv_idx(k)
            best_q = -1e18
            best_a = 0
            for a in allowed_actions(i, j):
                q = step_Q(i, j, a, V)
                if q > best_q:
                    best_q = q
                    best_a = a
            V_new[k] = best_q
            policy[k] = best_a
            if k == s0:
                g_est = best_q - V[k]  # estimate of average reward

        # normalize so that V_new(s0) = 0
        ref = V_new[s0]
        for k in range(N):
            V_new[k] -= ref

        # check convergence
        diff = max(abs(V_new[k] - V[k]) for k in range(N))
        # print(it, diff, g_est)  # debug if needed

        V = V_new
        if diff < tol:
            break

    return g_est, V, policy


g_star, V, policy = average_reward_value_iteration()
print(f"Optimal long-run average reward (part d): {g_star:.4f}")

# build a 10x10 table: 1 if the chosen action contains preventive repair, else 0
table = [[0 for _ in range(max_state)] for _ in range(max_state)]
for k in range(max_state * max_state):
    i, j = inv_idx(k)
    if policy[k] in [2, 4, 5, 6]:    # actions that use preventive repair
        table[i-1][j-1] = 1

print("Policy table (1 = action includes preventive repair):")
for row in table:
    print(row)

# Plot of Optimal Policy (where preventive is used)

policy_array = np.zeros((max_state, max_state), dtype=int)
for k in range(max_state * max_state):
    i, j = inv_idx(k)
    if policy[k] in [2, 4, 5, 6]:
        policy_array[i-1, j-1] = 1

plt.figure(figsize=(6, 6))
plt.imshow(policy_array, cmap="Greys", origin="lower")
plt.colorbar(label="1 = action includes preventive repair")
plt.xlabel("State of component 2")
plt.ylabel("State of component 1")
plt.title("Optimal policy for part (d)")
plt.xticks(range(0, 10), range(1, 11))
plt.yticks(range(0, 10), range(1, 11))
plt.savefig("policy_part_d.png", dpi=300, bbox_inches='tight')
plt.show()

