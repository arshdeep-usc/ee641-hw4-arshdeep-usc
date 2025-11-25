"""
Visualization utilities for gridworld and policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple
import os


class GridWorldVisualizer:
    """
    Visualizer for gridworld environment, value functions, and policies.
    """

    def __init__(self, grid_size: int = 5):
        """
        Initialize visualizer.

        Args:
            grid_size: Size of grid
        """
        self.grid_size = grid_size

        # Define special positions
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

    def plot_value_function(self, values: np.ndarray, title: str = "Value Function") -> None:
        """
        Plot value function as heatmap.

        Args:
            values: Value function V(s) for each state
            title: Plot title
        """
        # TODO: Reshape values to 2D grid
        # TODO: Create heatmap with appropriate colormap
        # TODO: Mark special cells (start, goal, obstacles, penalties)
        # TODO: Add colorbar and labels
        # TODO: Save figure to results/visualizations/

        grid = values.reshape(self.grid_size, self.grid_size)

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap="viridis", origin="upper")   # CONSISTENT ORIENTATION
        plt.title(title)
        plt.colorbar(label="Value")

        ax = plt.gca()

        # Mark special cells
        for r, c in self.obstacles:
            ax.text(c, r, "X", ha="center", va="center", fontsize=18)

        for r, c in self.penalties:
            ax.text(c, r, "P", ha="center", va="center", fontsize=18, color="darkred")

        sr, sc = self.start_pos
        ax.text(sc, sr, "S", ha="center", va="center", fontsize=18, color="blue")

        gr, gc = self.goal_pos
        ax.text(gc, gr, "G", ha="center", va="center", fontsize=18, color="green")

        ax.set_xticks(np.arange(-.5, self.grid_size, 1))
        ax.set_yticks(np.arange(-.5, self.grid_size, 1))
        ax.grid(color="black")

        plt.savefig(f"results/visualizations/{title.replace(' ', '_').lower()}.png")
        plt.close()


    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
        """
        # TODO: Create grid plot
        # TODO: For each state:
        #       - Draw arrow indicating action direction
        #       - Handle special cells appropriately
        # TODO: Mark start, goal, obstacles, penalties
        # TODO: Save figure to results/visualizations/

        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # Movement arrows: (dx, dy)
        arrow_map = {
            0: (0, -0.4),    # UP
            1: (0.4, 0),     # RIGHT
            2: (0, 0.4),     # DOWN
            3: (-0.4, 0)     # LEFT
        }

        for state in range(self.grid_size * self.grid_size):
            r = state // self.grid_size
            c = state % self.grid_size

            if (r, c) in self.obstacles or (r, c) in self.penalties or (r, c) == self.goal_pos:
                continue

            dx, dy = arrow_map[policy[state]]
            ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.15,
                     fc='black', ec='black')

        # Special cells
        for r, c in self.obstacles:
            ax.text(c, r, "X", ha="center", va="center", fontsize=18)

        for r, c in self.penalties:
            ax.text(c, r, "P", ha="center", va="center", fontsize=18, color="darkred")

        sr, sc = self.start_pos
        ax.text(sc, sr, "S", ha="center", va="center", fontsize=18, color="blue")

        gr, gc = self.goal_pos
        ax.text(gc, gr, "G", ha="center", va="center", fontsize=18, color="green")

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)

        ax.set_xticks(np.arange(-.5, self.grid_size, 1))
        ax.set_yticks(np.arange(-.5, self.grid_size, 1))
        ax.grid(color="black")

        plt.title(title)
        plt.savefig(f"results/visualizations/{title.replace(' ', '_').lower()}.png")
        plt.close()

    def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function") -> None:
        """
        Plot Q-function with multiple subplots for each action.

        Args:
            q_values: Q-function Q(s,a)
            title: Plot title
        """
        # TODO: Create subplot for each action
        # TODO: For each action:
        #       - Show Q-values as heatmap
        #       - Mark special cells
        # TODO: Add overall title and save

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        actions = ["UP", "RIGHT", "DOWN", "LEFT"]

        for a in range(4):
            grid = q_values[:, a].reshape(self.grid_size, self.grid_size)

            im = axes[a].imshow(grid, cmap="viridis", origin="upper")
            axes[a].set_title(actions[a])

            # Mark special cells
            for r, c in self.obstacles:
                axes[a].text(c, r, "X", ha="center", va="center", fontsize=16)

            for r, c in self.penalties:
                axes[a].text(c, r, "P", ha="center", va="center", fontsize=16, color="darkred")

            gr, gc = self.goal_pos
            axes[a].text(gc, gr, "G", ha="center", va="center", fontsize=16, color="green")

            axes[a].set_xticks(np.arange(-.5, self.grid_size, 1))
            axes[a].set_yticks(np.arange(-.5, self.grid_size, 1))
            axes[a].grid(color="black")

        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(title)

        plt.savefig(f"results/visualizations/{title.replace(' ', '_').lower()}.png")
        plt.close()


    def plot_convergence(self, vi_history: list, qi_history: list) -> None:
        """
        Plot convergence curves for both algorithms.

        Args:
            vi_history: Value iteration convergence history
            qi_history: Q-iteration convergence history
        """
        # TODO: Plot Bellman error vs iteration for both algorithms
        # TODO: Use log scale for y-axis
        # TODO: Add legend and labels
        # TODO: Save figure

        plt.figure(figsize=(8, 6))
        plt.plot(vi_history, label="Value Iteration")
        plt.plot(qi_history, label="Q-Iteration")

        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Bellman Error")
        plt.title("Convergence Curves")
        plt.legend()

        plt.savefig("results/visualizations/convergence_curves.png")
        plt.close()


    def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
                                vi_policy: np.ndarray, qi_policy: np.ndarray) -> None:
        """
        Create comparison figure showing both algorithms' results.

        Args:
            vi_values: Value function from Value Iteration
            qi_values: Value function from Q-Iteration
            vi_policy: Policy from Value Iteration
            qi_policy: Policy from Q-Iteration
        """
        # TODO: Create 2x2 subplot
        #       - Top left: VI value function
        #       - Top right: QI value function
        #       - Bottom left: VI policy
        #       - Bottom right: QI policy
        # TODO: Highlight any differences
        # TODO: Save comprehensive comparison figure

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Value heatmaps
        axes[0, 0].imshow(vi_values.reshape(self.grid_size, self.grid_size),
                          cmap="viridis", origin="upper")
        axes[0, 0].set_title("VI Values")

        axes[0, 1].imshow(qi_values.reshape(self.grid_size, self.grid_size),
                          cmap="viridis", origin="upper")
        axes[0, 1].set_title("QI Values")

        arrow_map = {0:(0,-0.4),1:(0.4,0),2:(0,0.4),3:(-0.4,0)}

        # VI policy
        ax = axes[1, 0]
        for s in range(self.grid_size * self.grid_size):
            r, c = divmod(s, self.grid_size)
            if (r, c) not in self.obstacles and (r, c) != self.goal_pos:
                dx, dy = arrow_map[vi_policy[s]]
                ax.arrow(c, r, dx, dy, head_width=0.15)
        ax.set_title("VI Policy")
        ax.set_ylim(self.grid_size - 0.5, -0.5)

        # QI policy
        ax = axes[1, 1]
        for s in range(self.grid_size * self.grid_size):
            r, c = divmod(s, self.grid_size)
            if (r, c) not in self.obstacles and (r, c) != self.goal_pos:
                dx, dy = arrow_map[qi_policy[s]]
                ax.arrow(c, r, dx, dy, head_width=0.15)
        ax.set_title("QI Policy")
        ax.set_ylim(self.grid_size - 0.5, -0.5)

        plt.tight_layout()
        plt.savefig("results/visualizations/comparison.png")
        plt.close()


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    # TODO: Load saved value functions and policies
    # TODO: Create visualizer instance
    # TODO: Generate all visualization plots
    # TODO: Print summary statistics

    vis = GridWorldVisualizer()

    vi_values = np.load("results/value_iteration_values.npy")
    vi_policy = np.load("results/value_iteration_policy.npy")

    qi_values = np.load("results/q_iteration_values.npy")
    qi_policy = np.load("results/q_iteration_policy.npy")
    q_values = np.load("results/q_iteration_qvalues.npy")

    # Generate all figures
    vis.plot_value_function(vi_values, "Value_Iteration_Values")
    vis.plot_value_function(qi_values, "Q_Iteration_Values")

    vis.plot_policy(vi_policy, "Value_Iteration_Policy")
    vis.plot_policy(qi_policy, "Q_Iteration_Policy")

    vis.plot_q_function(q_values, "Q_Iteration_Q_Functions")

    vis.create_comparison_figure(vi_values, qi_values, vi_policy, qi_policy)

    print("All visualization images saved to results/visualizations/")


if __name__ == '__main__':
    visualize_results()