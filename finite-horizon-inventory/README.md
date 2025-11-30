# Finite-Horizon Inventory Optimization

This project implements a dynamic programming solution for a seasonal inventory problem with:
- 150 time periods
- initial stock of 5 items
- time-dependent demand probability
- unreliable delivery (50% chance of arrival)
- profit per sale = 1
- holding cost = 0.1 per item per period
- no backorders and no end-of-season salvage value

The goal is to compute the optimal policy (order vs. not order) for every time and inventory level,
and to validate the results using simulation.

---

## Problem Summary

Each period:
- Demand is either 0 or 1, and the probability of demand increases from 1/150 up to 1.
- You may choose to order 1 item.
- If you order, delivery succeeds with probability 0.5 before demand occurs.
- If demand occurs and you have stock, you sell 1 item.
- You pay a small cost for every item you keep at the end of the period.

The challenge is deciding when it is worthwhile to order, considering:
- uncertainty in demand
- unreliable deliveries
- holding costs
- the limited selling horizon

---

## Method

A backward dynamic programming algorithm is used to compute the optimal decision for each 
period and each inventory level. The state space is truncated safely to a maximum inventory of 80 units,
which is far above any inventory level reached in realistic simulations.

After computing the optimal policy, the script simulates many sample paths to verify that the observed
average reward matches the theoretical optimal reward.

---

## Results

- Optimal expected reward from the start: **33.62**
- Simulation confirms the result (10,000 runs): average ≈ **33.64**
- The optimal policy:
  - avoids ordering early (when demand is low)
  - orders only when demand is high and inventory is low
  - keeps inventory in a narrow range (about 1–5 units)

This matches the economic intuition: holding costs prevent stock buildup, and ordering late makes sense
because demand becomes almost certain.

---

## Files

- **finite_horizon_inventory.py**  
  Contains the full dynamic programming algorithm and the simulation code.

- **Finite_Horizon_Inventory_report.pdf**  
  Full report with:
  - clear explanation of the problem  
  - methodology and implementation details  
  - optimal policy plot  
  - simulation results and histogram  
  - interpretation of findings  

---

## How to Run

Run the script with Python 3:

```bash
python finite_horizon_inventory.py
```
