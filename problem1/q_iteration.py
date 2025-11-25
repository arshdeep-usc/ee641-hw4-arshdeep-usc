"""
Q-Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class QIteration:
    """
    Q-Iteration solver for gridworld MDP.

    Computes optimal action-value function Q* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Q-Iteration solver.

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
        Run Q-iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            q_values: Converged Q-function Q(s,a)
            n_iterations: Number of iterations until convergence
        """
        # TODO: Initialize Q-function to zeros (shape: [n_states, n_actions])
        # TODO: Iterate until convergence:
        #       - For each state-action pair:
        #           - Compute updated Q-value using Bellman equation:
        #             Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q(s',a')]
        #       - Check convergence: max|Q_new - Q_old| < epsilon
        #       - Update Q-function
        # TODO: Return final Q-values and iteration count

        q_values = np.zeros((self.n_states, self.n_actions))

        for i in range(max_iterations):
            q_new = np.zeros_like(q_values)

            for state in range(self.n_states):
                for act in range(self.n_actions):
                    q_new[state, act] = self.bellman_update(state, act, q_values)

            if np.max(np.abs(q_new - q_values)) < self.epsilon:
                return q_new, i + 1

            q_values = q_new

        return q_values, max_iterations
        

    def bellman_update(self, state: int, action: int, q_values: np.ndarray) -> float:
        """
        Compute updated Q-value for a state-action pair.

        Args:
            state: State index
            action: Action index
            q_values: Current Q-function

        Returns:
            Updated Q-value for (s,a)
        """
        # TODO: Get transition probabilities P(s'|s,a)
        # TODO: For each possible next state:
        #       - Get reward R(s,a,s')
        #       - Get max Q-value for next state: max_a' Q(s',a')
        #       - Accumulate: prob * [reward + gamma * max_q_next]
        # TODO: Return updated Q-value

        t_prob = self.env.get_transition_prob(state, action)

        new_q = 0.0

        for next_state, prob in t_prob.items():
            r = self.env.get_reward(state, action, next_state)
            
            if self.env.is_terminal(next_state):
                max_q_next = 0.0 
            else:
                max_q_next = np.max(q_values[next_state])
                
            new_q += prob * (r + self.gamma * max_q_next)

        return new_q

    def extract_policy(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from Q-function.

        Args:
            q_values: Optimal Q-function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        #       - Select action with maximum Q-value: argmax_a Q(s,a)
        # TODO: Return policy array

        return np.argmax(q_values, axis=1)

    def extract_values(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract value function from Q-function.

        Args:
            q_values: Q-function

        Returns:
            values: State value function V(s) = max_a Q(s,a)
        """
        # TODO: For each state:
        #       - Compute V(s) = max_a Q(s,a)
        # TODO: Return value function

        return np.max(q_values, axis=1)

    def compute_bellman_error(self, q_values: np.ndarray) -> float:
        """
        Compute Bellman error for current Q-function.

        Args:
            q_values: Current Q-function

        Returns:
            Maximum Bellman error across all state-action pairs
        """
        # TODO: For each state-action pair:
        #       - Compute updated Q-value using Bellman update
        #       - Calculate absolute difference from current Q-value
        # TODO: Return maximum error

        max_err = 0.0

        for state in range(self.n_states):
            for act in range(self.n_actions):
                err = abs(self.bellman_update(state, act, q_values) - q_values[state, act])
                max_err = max(max_err, err)

        return max_err