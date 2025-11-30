# Two-Component Maintenance MDP (Average Reward)

This project analyzes a maintenance problem involving a system with **two independent components** that deteriorate over time. The task is to formulate and solve an **average-reward Markov Decision Process (MDP)** and compare several methods for computing the long-run performance under a chosen maintenance policy.

---

## Problem Summary

Each component can be in one of three conditions:
- **0** – good  
- **1** – degraded  
- **2** – failed  

At every decision epoch, the maintenance team may choose between:
- **No maintenance**
- **Corrective maintenance** (on failed components)
- **Preventive maintenance** (on components in state 1 or 2)
- **Combination actions** depending on the state of both components

Every action has:
- Specific **transition probabilities**
- **Costs** associated with downtime and repair
- Impact on the long-run system performance

The problem is to:
1. Compute the **stationary distribution** of the Markov chain under a fixed policy.  
2. Use it to estimate the **long-run average reward**.  
3. Solve the **Poisson equation** to obtain relative values.  
4. Run **simulation** to verify theoretical results.  
5. Compare these methods and comment on consistency.  

---

## Method

The solution uses several approaches commonly applied to average-reward MDPs:

### 1. Stationary Distribution  
The transition matrix under the chosen maintenance policy is constructed, and the stationary distribution is computed. This allows direct calculation of the long-run average reward.

### 2. Poisson Equation (Relative Value Function)  
A linear-algebra formulation is used to obtain relative values for each system state, confirming the consistency of the policy’s long-run performance.

### 3. Simulation  
A long-run simulation (with a large number of time steps) is performed to empirically estimate:
- the average reward,
- state visit frequencies,
- and the stability of the chain.

Simulation results are compared with the stationary distribution and Poisson-equation results to confirm correctness.

---

## Results

Key findings include:

- The stationary distribution, Poisson equation, and simulation all produced **consistent values** for the long-run performance under the chosen policy.  
- Preventive maintenance on moderately degraded components significantly reduces long-run costs.  
- Allowing components to degrade to failure leads to high corrective maintenance and downtime costs.  
- The optimal policy balances preventive actions with the cost of unnecessary repairs.

Details, tables, and visualizations are provided in the accompanying report.

---

## Files

- **optimal_policies_mdp.py**  
  Python implementation of the transition matrix construction, stationary distribution computation, Poisson equation solution, and long-run simulation.

- **Optimal_Policies_under_MDP_report.pdf**  
  Full report including:
  - model description  
  - transition structure  
  - stationary distribution results  
  - Poisson equation solution  
  - simulation analysis  
  - policy interpretation  

---

## How to Run

Run the script using Python 3:

```bash
python optimal_policies_mdp.py
```
