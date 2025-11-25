"""
Value Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class ValueIteration:
    """
    Value Iteration solver for gridworld MDP.

    Computes optimal value function V* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Value Iteration solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.grid_size ** 2
        self.n_actions = env.action_space

    def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int]:
        """
        Run value iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            values: Converged value function V(s)
            n_iterations: Number of iterations until convergence
        """
        # TODO: Initialize value function to zeros
        # TODO: Iterate until convergence:
        #       - For each state:
        #           - Compute Q(s,a) for all actions using Bellman backup
        #           - Set V(s) = max_a Q(s,a)
        #       - Check convergence: max|V_new - V_old| < epsilon
        #       - Update value function
        # TODO: Return final values and iteration count

        V_old = np.zeros(self.n_states)

        for iteration in range(max_iterations):
            V_new = np.zeros_like(V_old)

            for state in range(self.n_states):
                V_new[state] = self.bellman_backup(state, V_old)

            if np.max(np.abs(V_new - V_old)) < self.epsilon:
                return V_new, iteration + 1

            V_old = V_new

        return V_new, max_iterations

    def compute_q_values(self, state: int, values: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions in a state.

        Args:
            state: State index
            values: Current value function

        Returns:
            q_values: Array of Q(s,a) for each action
        """
        # TODO: For each action:
        #       - Get transition probabilities P(s'|s,a)
        #       - Compute expected value:
        #           Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
        # TODO: Return Q-values array

        q_values = np.zeros(self.n_actions)

        for action in range(self.n_actions):
            probs = self.env.get_transition_prob(state, action)
            q_val = 0.0
            for next_state, prob in probs.items():
                reward = self.env.get_reward(state, action, next_state)
                q_val += prob * (reward + self.gamma * values[next_state])
            q_values[action] = q_val

        return q_values

    def extract_policy(self, values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from value function.

        Args:
            values: Optimal value function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        #       - Compute Q-values for all actions
        #       - Select action with maximum Q-value
        # TODO: Return policy array

        policy = np.zeros(self.n_states, dtype=int)

        for state in range(self.n_states):
            q_values = self.compute_q_values(state, values)
            policy[state] = np.argmax(q_values)

        return policy

    def bellman_backup(self, state: int, values: np.ndarray) -> float:
        """
        Perform Bellman backup for a single state.

        Args:
            state: State index
            values: Current value function

        Returns:
            Updated value for state
        """
        # TODO: If terminal state, return 0
        # TODO: Compute Q-values for all actions
        # TODO: Return maximum Q-value

        if self.env.is_terminal(state):
            return 0.0

        return np.max(self.compute_q_values(state, values))

    def compute_bellman_error(self, values: np.ndarray) -> float:
        """
        Compute Bellman error for current value function.

        Bellman error = max_s |V(s) - max_a Q(s,a)|

        Args:
            values: Current value function

        Returns:
            Maximum Bellman error across all states
        """
        # TODO: For each state:
        #       - Compute optimal value using Bellman backup
        #       - Calculate absolute difference from current value
        # TODO: Return maximum error

        max_err = 0.0

        for state in range(self.n_states):
            max_err = max(max_err, abs(values[state] - self.bellman_backup(state, values)))

        return max_err