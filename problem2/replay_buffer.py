"""
Experience replay buffer for multi-agent DQN training.
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores joint experiences from both agents for coordinated learning.
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store a transition in the buffer.

        Args:
            state_A: Agent A's observation
            state_B: Agent B's observation
            action_A: Agent A's action
            action_B: Agent B's action
            comm_A: Communication from A to B
            comm_B: Communication from B to A
            reward: Shared reward
            next_state_A: Agent A's next observation
            next_state_B: Agent B's next observation
            done: Whether episode terminated
        """
        # TODO: Create transition tuple
        # TODO: Add to buffer (automatic removal of oldest if at capacity)

        transition = (
            state_A, state_B,
            action_A, action_B,
            comm_A, comm_B,
            reward,
            next_state_A, next_state_B,
            done
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions as separate arrays for each component
        """
        # TODO: Sample batch_size transitions randomly
        # TODO: Separate components into individual arrays
        # TODO: Convert to appropriate numpy arrays
        # TODO: Return tuple of arrays

        batch = random.sample(self.buffer, batch_size)

        (state_A, state_B,
         action_A, action_B,
         comm_A, comm_B,
         reward,
         next_A, next_B,
         done) = zip(*batch)

        return (
            np.array(state_A, dtype=float),
            np.array(state_B, dtype=float),
            np.array(action_A, dtype=int),
            np.array(action_B, dtype=int),
            np.array(comm_A, dtype=float),
            np.array(comm_B, dtype=float),
            np.array(reward, dtype=float),
            np.array(next_A, dtype=float),
            np.array(next_B, dtype=float),
            np.array(done, dtype=bool)
        )

    def __len__(self) -> int:
        """
        Get current size of buffer.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for importance sampling.

    Samples transitions based on TD-error magnitude.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_steps: Steps to anneal beta to 1.0
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.frame = 1

        # TODO: Initialize data storage
        # TODO: Initialize priority tree (sum-tree or similar)
        # TODO: Set random seed if provided

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.tree = SumTree(capacity)

        self.max_priority = 1.0

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store transition with maximum priority.

        New transitions get maximum priority to ensure they're sampled at least once.
        """
        # TODO: Store transition
        # TODO: Assign maximum priority to new transition
        
        transition = (
            state_A, state_B,
            action_A, action_B,
            comm_A, comm_B,
            reward,
            next_state_A, next_state_B,
            done
        )
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch with prioritization.

        Returns:
            transitions: Batch of transitions
            weights: Importance sampling weights
            indices: Indices for updating priorities
        """
        # TODO: Update beta based on schedule
        # TODO: Sample transitions based on priorities
        # TODO: Calculate importance sampling weights
        # TODO: Return transitions, weights, and indices

        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_steps)
        self.frame += 1

        segment = self.tree.total_priority / batch_size

        batch = []
        indices = []
        priorities = []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.sample(s)

            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        (state_A, state_B,
         action_A, action_B,
         comm_A, comm_B,
         reward,
         next_A, next_B,
         done) = zip(*batch)

        probs = np.array(priorities) / self.tree.total_priority

        weights = (len(self.tree) * probs) ** (-self.beta)
        weights = weights / weights.max()

        return (
            np.array(state_A, dtype=float),
            np.array(state_B, dtype=float),
            np.array(action_A, dtype=int),
            np.array(action_B, dtype=int),
            np.array(comm_A, dtype=float),
            np.array(comm_B, dtype=float),
            np.array(reward, dtype=float),
            np.array(next_A, dtype=float),
            np.array(next_B, dtype=float),
            np.array(done, dtype=bool)
        ), weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD-errors)
        """
        # TODO: Update priorities for given indices
        # TODO: Apply alpha exponent for prioritization

        priorities = np.abs(priorities) + 1e-6

        self.max_priority = max(self.max_priority, priorities.max())

        for idx, prio in zip(indices, priorities):
            self.tree.update(idx, prio ** self.alpha)

    def __len__(self):
        """
        Number of transitions stored.
        """
        
        return len(self.tree)

class SumTree:
    """
    Sum-tree data structure for prioritized experience replay.

    Provides O(log N) sampling and priority updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0  # number of valid transitions

    def __len__(self):
        """Return number of stored transitions."""
        return self.size

    def add(self, priority: float, data: object) -> None:
        """
        Add new transition with given priority.
        """
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx: int, priority: float) -> None:
        """
        Update priority at index.
        """
        delta = priority - self.tree[idx]
        self.tree[idx] = priority

        # Propagate update
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def sample(self, s: float) -> Tuple[int, float, object]:
        """
        Sample leaf given cumulative priority s.
        """
        idx = 0

        while idx < self.capacity - 1:  # while not a leaf
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        leaf_idx = idx
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]
