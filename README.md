# Reinforcement Learning Lab

A collection of reinforcement learning and dynamic programming mini-projects.
Each folder contains a standalone project with its own Python implementation and report.
The projects explore different core RL concepts: finite-horizon dynamic programming,
average-reward MDPs, and Monte Carlo Tree Search.

---

## 📁 Projects Included

### 1. Finite-Horizon Inventory Optimization
A dynamic programming solution to a seasonal inventory control problem with:
- time-dependent demand
- unreliable replenishment
- holding costs and no salvage value
- a 150-period selling horizon

The project computes the optimal policy, verifies it via simulation, and analyzes policy structure.

**Folder:** `finite-horizon-inventory`  
**Contains:** Python implementation + full report (PDF)

---

### 2. Two-Component Maintenance MDP (Average Reward)
A long-run average-reward Markov Decision Process for a system made of two deteriorating components.
Features:
- preventive and corrective maintenance
- deterioration probabilities
- downtime costs and repair costs
- solving via simulation, stationary distribution, and Poisson equation/value iteration

The project determines optimal repair policies and compares multiple computation methods.

**Folder:** `optimal-policies-mdp`   
**Contains:** Python implementation + report

---

### 3. Tic-Tac-Toe Monte Carlo Tree Search (MCTS)
An MCTS agent for Tic-Tac-Toe that must reliably beat a random opponent.
The project includes:
- encoding states/actions
- terminal condition logic
- MCTS search structure (selection, expansion, simulation, backpropagation)
- visualization of games and convergence analysis

**Folder:** `mcts-tictactoe` *(to be added)*  
**Contains:** Python implementation + report

---

## 🧰 Technologies Used
- Python 3.x  
- NumPy / Matplotlib (when required)  

Each project runs as an independent script with no external frameworks.

---

## 🎯 Purpose

This repository serves as a compact portfolio of reinforcement learning coursework and
independent study, showcasing:
- clean implementations of RL algorithms,
- reproducible experiments,
- interpretation of policies and results,
- and solid understanding of dynamic programming and simulation methods.

More projects will be added as the Reinforcement Learning course progresses.

---

