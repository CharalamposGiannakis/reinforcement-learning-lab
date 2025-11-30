

# --------Dynamic Programming Algorithm--------


import numpy as np

T = 150      # time periods
x_max = 80   # practical cap

V = np.zeros((T+2, x_max+1))              # V[t, x]
pi = np.zeros((T+1, x_max+1), dtype=int)   # optimal action

for t in range(T, 0, -1):
    p_t = t / T    # demand prob
    for x in range(x_max+1):

        # ---- a = 0 (no order) ----
        if x > 0:
            # D=1: sell 1, next state x-1
            ev_sell = 1 - 0.1*(x-1) + V[t+1, x-1]
            # D=0: sell 0, next state x
            ev_nosell = 0 - 0.1*x + V[t+1, x]
            val_no = p_t*ev_sell + (1-p_t)*ev_nosell
        else:
            # no stock => no sale regardless of D
            val_no = 0 - 0.0 + V[t+1, 0]

        # ---- a = 1 (order; arrival before demand) ----
        EV_yes = 0.0
        for y, py in [(0, 0.5), (1, 0.5)]:
            xb = min(x + y, x_max)  # stock before demand
            if xb > 0:
                ev_sell = 1 - 0.1*(xb-1) + V[t+1, xb-1]
                ev_nosell = 0 - 0.1*xb + V[t+1, xb]
                EV_yes += py * (p_t*ev_sell + (1-p_t)*ev_nosell)
            else:
                EV_yes += py * (0 - 0.0 + V[t+1, 0])
        val_yes = EV_yes

        # ---- choose action; tie-break to a=0 ----
        if val_yes > val_no:
            V[t, x] = val_yes
            pi[t, x] = 1
        else:
            V[t, x] = val_no
            pi[t, x] = 0

print("Optimal expected reward V1(5):", V[1, 5])


#----------Policy Plot----------


import matplotlib.pyplot as plt

# Zoom to x in [0, 8]
xmax_show = 9
mask = pi[1:T+1, :xmax_show]

plt.figure(figsize=(7,4))
plt.imshow(mask, aspect='auto', origin='lower', cmap='binary')
plt.xlabel("Inventory x (0–8)"); plt.ylabel("Time t (1–150)")
plt.title("Optimal policy (1=order, 0=not)")

# the boundary where action changes
boundary = []
for t in range(1, T+1):
    row = mask[t-1]
    # first x where we do NOT order 
    xs = np.where(row==0)[0]
    boundary.append(xs[0] if xs.size>0 else xmax_show-0.5)
plt.plot(boundary, np.arange(1, T+1), lw=1)

plt.tight_layout()
plt.savefig("policy_heatmap_zoom.png", dpi=200)
plt.show()


# --------Simulation (1000 runs) + CI + histogram--------


rng = np.random.default_rng(42)  # for reproducibility

def simulate_one():
    x = 5
    total = 0.0
    for t in range(1, T+1):
        a = pi[t, x]
        if a == 1 and rng.random() < 0.5:
            x = min(x+1, x_max)
        # demand
        if x > 0 and rng.random() < (t/T):
            x -= 1
            total += 1
        total -= 0.1 * x
    return total

rewards = np.array([simulate_one() for _ in range(10000)])
mean = rewards.mean()
std  = rewards.std(ddof=1)
ci95 = (mean - 1.96*std/np.sqrt(len(rewards)), mean + 1.96*std/np.sqrt(len(rewards)))

print(f"Sim mean = {mean:.3f}, 95% CI = [{ci95[0]:.3f}, {ci95[1]:.3f}]")

plt.figure(figsize=(8,5))
plt.hist(rewards, bins=30, edgecolor='black')
plt.xlabel("Total reward")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reward_histogram.png", dpi=200)
plt.close()


