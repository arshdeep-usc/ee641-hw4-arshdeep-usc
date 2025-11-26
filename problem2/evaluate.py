"""
Evaluation script for trained multi-agent models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from multi_agent_env import MultiAgentEnv
from models import AgentDQN


class MultiAgentEvaluator:
    """
    Evaluator for analyzing trained multi-agent policies.
    """

    def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module):
        """
        Initialize evaluator.

        Args:
            env: Multi-agent environment
            model_A: Trained model for Agent A
            model_B: Trained model for Agent B
        """
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        # Use CPU for small networks
        self.device = torch.device("cpu")

        # Move models to device and set to evaluation mode
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        self.model_A.eval()
        self.model_B.eval()

    def run_episode(self, render: bool = False) -> Tuple[float, bool, Dict]:
        """
        Run single evaluation episode.

        Args:
            render: Whether to render environment

        Returns:
            reward: Episode reward
            success: Whether target was reached
            info: Episode statistics
        """
        # TODO: Reset environment
        # TODO: Initialize episode tracking
        # TODO: Run episode with greedy policy
        # TODO: Track communication patterns
        # TODO: Return results and statistics

        sA, sB = self.env.reset()
        total_reward = 0.0
        success = False
        comm_signals_A, comm_signals_B = [], []
        positions_A, positions_B = [], []

        for step in range(self.env.max_steps):
            state_A = torch.tensor(sA, dtype=torch.float32).unsqueeze(0).to(self.device)
            state_B = torch.tensor(sB, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                qA, commA = self.model_A(state_A)
                qB, commB = self.model_B(state_B)
                actionA = qA.argmax(dim=1).item()
                actionB = qB.argmax(dim=1).item()
                commA, commB = commA.item(), commB.item()

            (nA, nB), r, done = self.env.step(actionA, actionB, commA, commB)
            total_reward += r

            comm_signals_A.append(commA)
            comm_signals_B.append(commB)
            positions_A.append(self.env.agent_positions[0])
            positions_B.append(self.env.agent_positions[1])

            if render:
                self.env.render()

            if r == 10.0:
                success = True
                break
            if done:
                success = (r >= 10.0)
                break

            sA, sB = nA, nB

        info = {
            "steps": step + 1,
            "comm_A": comm_signals_A,
            "comm_B": comm_signals_B,
            "positions_A": positions_A,
            "positions_B": positions_B
        }

        return total_reward, success, info


    def evaluate_performance(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate overall performance statistics.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Statistics dictionary
        """
        # TODO: Run multiple episodes
        # TODO: Compute success rate
        # TODO: Analyze path lengths
        # TODO: Measure coordination efficiency
        # TODO: Return comprehensive statistics

        rewards, successes, lengths = [], [], []

        for _ in range(num_episodes):
            reward, success, info = self.run_episode()
            rewards.append(reward)
            successes.append(success)
            lengths.append(info["steps"])

        stats = {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(np.mean(successes)),
            "total_successes": float(np.count_nonzero(successes)),
            "avg_episode_length": float(np.mean(lengths)),
            "std_reward": float(np.std(rewards))
        }
        
        return stats

    def analyze_communication(self, num_episodes: int = 20) -> Dict:
        """
        Analyze emergent communication protocols.

        Returns:
            Communication analysis results
        """
        # TODO: Track communication signals over episodes
        # TODO: Analyze signal patterns (magnitude, variance, correlation)
        # TODO: Identify communication strategies
        # TODO: Return analysis results

        all_comm_A, all_comm_B = [], []

        for _ in range(num_episodes):
            _, _, info = self.run_episode()
            all_comm_A.extend(info["comm_A"])
            all_comm_B.extend(info["comm_B"])

        commA = np.array(all_comm_A)
        commB = np.array(all_comm_B)

        analysis = {
            "mean_comm_A": float(np.mean(commA)),
            "mean_comm_B": float(np.mean(commB)),
            "var_comm_A": float(np.var(commA)),
            "var_comm_B": float(np.var(commB)),
            "correlation": float(np.corrcoef(commA, commB)[0, 1]) if len(commA) > 5 else 0.0
        }
        
        return analysis

    def visualize_trajectory(self, save_path: str = 'results/trajectory.png') -> None:
        """
        Visualize agent trajectories in an episode.

        Args:
            save_path: Path to save visualization
        """
        # TODO: Run episode while tracking positions
        # TODO: Create grid visualization
        # TODO: Plot agent paths
        # TODO: Mark key events (near target, coordination points)
        # TODO: Save figure

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Run one episode to collect positions
        _, _, info = self.run_episode(render=False)
        posA = np.array(info["positions_A"])
        posB = np.array(info["positions_B"])

        rows, cols = self.env.grid_size

        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # Show grid cells as an image:
        #   0: free  -> light gray
        #   1: obstacle -> black
        #   2: target -> green
        grid_img = np.zeros((rows, cols, 3), dtype=float)

        for r in range(rows):
            for c in range(cols):
                val = self.env.grid[r, c]
                if val == 0:      # free
                    grid_img[r, c] = [0.9, 0.9, 0.9]
                elif val == 1:    # obstacle
                    grid_img[r, c] = [0.0, 0.0, 0.0]
                elif val == 2:    # target
                    grid_img[r, c] = [0.3, 0.8, 0.3]

        ax.imshow(grid_img, origin="upper")

        # Overlay agent paths: note x = col, y = row
        if len(posA) > 0:
            ax.plot(posA[:, 1], posA[:, 0], 'r-o', label='Agent A path', linewidth=2, markersize=4)
            ax.scatter(posA[0, 1], posA[0, 0], c='darkred', marker='x', s=80, label='A start')
            ax.scatter(posA[-1, 1], posA[-1, 0], c='yellow', edgecolors='k',
                       marker='o', s=70, label='A end')

        if len(posB) > 0:
            ax.plot(posB[:, 1], posB[:, 0], 'b-s', label='Agent B path', linewidth=2, markersize=4)
            ax.scatter(posB[0, 1], posB[0, 0], c='darkblue', marker='x', s=80, label='B start')
            ax.scatter(posB[-1, 1], posB[-1, 0], c='cyan', edgecolors='k',
                       marker='o', s=70, label='B end')

        # Grid lines to show cell boundaries
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=False)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=False)
        ax.grid(color='k', linestyle='-', linewidth=0.5)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Full Environment and Agent Trajectories")
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_communication_heatmap(self, save_path: str = 'results/comm_heatmap.png') -> None:
        """
        Create heatmap of communication signals across grid positions.

        Args:
            save_path: Path to save figure
        """
        # TODO: Sample communication signals at each grid position
        # TODO: Create heatmaps for both agents
        # TODO: Show correlation with distance to target
        # TODO: Save visualization

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        rows, cols = self.env.grid_size
        heat_A = np.zeros((rows, cols), dtype=np.float32)
        heat_B = np.zeros((rows, cols), dtype=np.float32)
        count_A = np.zeros((rows, cols), dtype=np.float32)
        count_B = np.zeros((rows, cols), dtype=np.float32)

        for _ in range(100):
            _, _, info = self.run_episode(render=False)
            posA = info["positions_A"]
            posB = info["positions_B"]
            commA = info["comm_A"]
            commB = info["comm_B"]

            for (r, c), s in zip(posA, commA):
                heat_A[r, c] += abs(s)
                count_A[r, c] += 1.0
            for (r, c), s in zip(posB, commB):
                heat_B[r, c] += abs(s)
                count_B[r, c] += 1.0

        # Avoid division by zero
        maskA = count_A > 0
        maskB = count_B > 0
        avg_A = np.zeros_like(heat_A)
        avg_B = np.zeros_like(heat_B)
        avg_A[maskA] = heat_A[maskA] / count_A[maskA]
        avg_B[maskB] = heat_B[maskB] / count_B[maskB]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        im0 = axs[0].imshow(avg_A, origin='upper', cmap='viridis')
        axs[0].set_title("Agent A |comm|")
        axs[0].set_xlabel("Column")
        axs[0].set_ylabel("Row")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(avg_B, origin='upper', cmap='viridis')
        axs[1].set_title("Agent B |comm|")
        axs[1].set_xlabel("Column")
        axs[1].set_ylabel("Row")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def test_generalization(self, num_configs: int = 10) -> Dict:
        """
        Test generalization to new environment configurations.

        Args:
            num_configs: Number of test configurations

        Returns:
            Generalization performance statistics
        """
        # TODO: Generate new obstacle configurations
        # TODO: Test performance on each configuration
        # TODO: Compare to training performance
        # TODO: Return generalization metrics

        rewards, successes = [], []

        for i in range(num_configs):
            self.env._initialize_grid()
            self.env.reset()
            reward, success, _ = self.run_episode()
            rewards.append(reward)
            successes.append(success)

        return {
            "mean_reward_generalization": float(np.mean(rewards)),
            "success_rate_generalization": float(np.mean(successes))
        }


def load_trained_models(checkpoint_dir: str) -> Tuple[nn.Module, nn.Module]:
    """
    Load trained agent models from checkpoint.

    Args:
        checkpoint_dir: Directory containing saved models

    Returns:
        model_A: Agent A's trained model
        model_B: Agent B's trained model
    """
    # TODO: Load model architectures
    # TODO: Load trained weights
    # TODO: Return initialized models

    model_A = AgentDQN(11, 64, 5)
    model_B = AgentDQN(11, 64, 5)

    model_A.load_state_dict(torch.load(os.path.join(checkpoint_dir, "agent_A.pt"), map_location="cpu"))
    model_B.load_state_dict(torch.load(os.path.join(checkpoint_dir, "agent_B.pt"), map_location="cpu"))

    return model_A, model_B


def create_evaluation_report(results: Dict, save_path: str = 'results/evaluation_report.json') -> None:
    """
    Create comprehensive evaluation report.

    Args:
        results: Evaluation results
        save_path: Path to save report
    """
    # TODO: Format results
    # TODO: Add summary statistics
    # TODO: Save as JSON report

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation report saved to {save_path}")


def main():
    """
    Run full evaluation suite on trained models.
    """
    # TODO: Load trained models
    # TODO: Create environment
    # TODO: Initialize evaluator
    # TODO: Run performance evaluation
    # TODO: Analyze communication
    # TODO: Test generalization
    # TODO: Create visualizations
    # TODO: Generate report

    checkpoint_dir = "results/agent_models"
    model_A, model_B = load_trained_models(checkpoint_dir)

    env = MultiAgentEnv(grid_size=(10, 10), max_steps=50, seed=61)
    evaluator = MultiAgentEvaluator(env, model_A, model_B)

    performance = evaluator.evaluate_performance(num_episodes=200)
    comm_analysis = evaluator.analyze_communication(num_episodes=200)
    generalization = evaluator.test_generalization(num_configs=10)

    evaluator.visualize_trajectory('results/trajectory.png')
    evaluator.plot_communication_heatmap('results/comm_heatmap.png')

    results = {
        "performance": performance,
        "communication": comm_analysis,
        "generalization": generalization
    }

    create_evaluation_report(results)
    print("Full evaluation complete.")


if __name__ == '__main__':
    main()