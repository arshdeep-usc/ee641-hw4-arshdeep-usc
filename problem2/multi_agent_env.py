"""
Multi-agent gridworld environment with partial observations and communication.
"""

import numpy as np
from typing import Tuple, Optional, List


class MultiAgentEnv:
    """
    Two-agent cooperative gridworld with partial observations.

    Agents must coordinate to simultaneously reach a target cell.
    Each agent observes a 3x3 local patch and exchanges communication signals.
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
                 max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize multi-agent environment.

        Args:
            grid_size: Tuple defining grid dimensions (default 10x10)
            obs_window: Size of local observation window (must be odd, default 3)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid components
        self._initialize_grid()

        # Agent state
        self.agent_positions = [None, None]
        self.comm_signals = [0.0, 0.0]
        self.step_count = 0

    def _initialize_grid(self) -> None:
        """
        Create grid with obstacles and target.

        Grid values:
        - 0: Free cell
        - 1: Obstacle
        - 2: Target
        """
        # TODO: Create empty grid of size grid_size
        # TODO: Randomly place up to 6 obstacles (avoiding corners)
        # TODO: Randomly place exactly 1 target cell
        # TODO: Store grid as self.grid

        rows, cols = self.grid_size
        grid = np.zeros((rows, cols), dtype=int)

        max_obstacles = 6
        free_positions = [(r, c) for r in range(rows) for c in range(cols)]
        corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
        free_positions = [p for p in free_positions if p not in corner_positions]

        num_obstacles = np.random.randint(3, max_obstacles + 1)
        obstacles = list(np.random.choice(len(free_positions), num_obstacles, replace=False))
        for idx in obstacles:
            r, c = free_positions[idx]
            grid[r, c] = 1

        free_non_obstacle = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 0]
        tr, tc = free_non_obstacle[np.random.randint(len(free_non_obstacle))]
        grid[tr, tc] = 2

        self.grid = grid

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state.

        Returns:
            obs_A: Observation for Agent A (11-dimensional vector)
            obs_B: Observation for Agent B (11-dimensional vector)

        Observation format:
        - Elements 0-8: Flattened 3x3 grid patch (row-major order)
        - Element 9: Communication signal from other agent
        - Element 10: Normalized L2 distance between agents
        """
        # TODO: Reset step counter
        # TODO: Randomly place both agents on free cells (not obstacles or target)
        # TODO: Initialize communication signals to 0.0
        # TODO: Generate observations for both agents
        # TODO: Return (obs_A, obs_B)

        self.step_count = 0
        self.comm_signals = [0.0, 0.0]

        free_cells = self._find_free_cells()
        start_positions = np.random.choice(len(free_cells), 2, replace=False)
        self.agent_positions[0] = free_cells[start_positions[0]]
        self.agent_positions[1] = free_cells[start_positions[1]]

        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)
        
        return obs_A, obs_B

    def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
        """
        Execute one environment step.

        Args:
            action_A: Agent A's movement action (0:Up, 1:Down, 2:Left, 3:Right, 4:Stay)
            action_B: Agent B's movement action
            comm_A: Communication signal from Agent A to B
            comm_B: Communication signal from Agent B to A

        Returns:
            observations: Tuple of (obs_A, obs_B), each 11-dimensional
            reward: +10 if both agents at target, +2 if one agent at target, -0.1 per step
            done: True if both agents at target or max steps reached
        """
        # TODO: Update agent positions based on actions
        #       - Check boundaries and obstacles
        #       - Invalid moves result in no position change
        # TODO: Store new communication signals for next observation
        # TODO: Check reward condition (both agents at target)
        # TODO: Update step count and check termination
        # TODO: Generate new observations with updated comm signals
        # TODO: Return ((obs_A, obs_B), reward, done)

        self.agent_positions[0] = self._apply_action(self.agent_positions[0], action_A)
        self.agent_positions[1] = self._apply_action(self.agent_positions[1], action_B)

        self.comm_signals = [comm_A, comm_B]

        rA = self.grid[self.agent_positions[0]]
        rB = self.grid[self.agent_positions[1]]

        if rA == 2 and rB == 2:
            reward = 10.0
        elif rA == 2 or rB == 2:
            reward = 2.0
        else:
            reward = -0.1

        self.step_count += 1
        done = (rA == 2 and rB == 2) or (self.step_count >= self.max_steps)

        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)

        return (obs_A, obs_B), reward, done

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Extract local observation for an agent.

        Args:
            agent_idx: Agent index (0 for A, 1 for B)

        Returns:
            observation: 10-dimensional vector
        """
        # TODO: Get agent position
        # TODO: Extract 3x3 patch centered on agent
        #       - Cells outside grid should be -1
        #       - Use grid values (0: free, 1: obstacle, 2: target)
        # TODO: Flatten patch to 9 elements
        # TODO: Append communication signal from other agent
        # TODO: Return 10-dimensional observation

        r, c = self.agent_positions[agent_idx]
        half = self.obs_window // 2

        patch = []
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr, cc = r + dr, c + dc
                if self._is_valid_position((rr, cc)):
                    patch.append(self.grid[rr, cc])
                else:
                    patch.append(-1)  # Outside grid

        comm_in = self.comm_signals[1 - agent_idx]

        (rA, cA) = self.agent_positions[0]
        (rB, cB) = self.agent_positions[1]
        dist = np.sqrt((rA - rB) ** 2 + (cA - cB) ** 2)
        max_dist = np.sqrt(self.grid_size[0] ** 2 + self.grid_size[1] ** 2)
        dist_norm = dist / max_dist

        obs = np.array(patch + [comm_in, dist_norm], dtype=np.float32)
        
        return obs

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        # TODO: Check if position is not an obstacle (grid value != 1)

        if not (0 < pos[0] < self.grid_size[0] and 0 < pos[1] < self.grid_size[1]):
            return False
            
        return True

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Apply movement action to position.

        Args:
            pos: Current position (row, col)
            action: Movement action (0-4)

        Returns:
            new_pos: Updated position (stays same if invalid)
        """
        # TODO: Map action to position delta
        #       0: Up (-1, 0)
        #       1: Down (+1, 0)
        #       2: Left (0, -1)
        #       3: Right (0, +1)
        #       4: Stay (0, 0)
        # TODO: Calculate new position
        # TODO: Return new position if valid, else return original position

        deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
            4: (0, 0)    # Stay
        }
        dr, dc = deltas[action]
        new_pos = (pos[0] + dr, pos[1] + dc)

        return new_pos if self._is_valid_position(new_pos) else pos

    def _find_free_cells(self) -> List[Tuple[int, int]]:
        """
        Find all free cells in the grid.

        Returns:
            List of (row, col) positions that are free
        """
        # TODO: Iterate through grid
        # TODO: Collect positions where grid value is 0 (free)
        # TODO: Return list of free positions

        free = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if self.grid[r, c] == 0:  # free
                    free.append((r, c))
        return free

    def render(self) -> None:
        """
        Render current environment state.
        """
        # TODO: Create visual representation of grid
        # TODO: Show agent positions (A, B)
        # TODO: Show target (T)
        # TODO: Show obstacles (X)
        # TODO: Display current communication values

        grid_vis = np.full(self.grid.shape, '.', dtype='<U1')

        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if self.grid[r, c] == 1:
                    grid_vis[r, c] = 'X'
                elif self.grid[r, c] == 2:
                    grid_vis[r, c] = 'T'

        # Place agents
        rA, cA = self.agent_positions[0]
        rB, cB = self.agent_positions[1]
        grid_vis[rA, cA] = 'A'
        grid_vis[rB, cB] = 'B'

        print("\n".join(" ".join(row) for row in grid_vis))
        print(f"Comm A→B: {self.comm_signals[0]:.2f} | Comm B→A: {self.comm_signals[1]:.2f}")
        print(f"Step: {self.step_count}\n")