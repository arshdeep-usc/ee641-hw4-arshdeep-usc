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

        info = {
            "positions_A": [],
            "positions_B": [],
            "comm_A": [],
            "comm_B": [],
            "steps": 0
        }

        episode_reward = 0.0
        success = False

        for t in range(self.env.max_steps):

            aA, comA = self._greedy(self.model_A, sA)
            aB, comB = self._greedy(self.model_B, sB)

            (nA, nB), r, done = self.env.step(aA, aB, comA, comB)

            info["positions_A"].append(self.env.agent_positions[0])
            info["positions_B"].append(self.env.agent_positions[1])
            info["comm_A"].append(comA)
            info["comm_B"].append(comB)

            episode_reward += r
            sA, sB = nA, nB

            if render:
                self.env.render()

            if r == 10.0:
                success = True
                break

            if done:
                success = (r == 10)
                break

        info["steps"] = t + 1

        return episode_reward, success, info

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

        rewards, successes, steps = [], [], []

        for _ in range(num_episodes):
            r, s, info = self.run_episode(render=False)
            rewards.append(r)
            successes.append(int(s))
            steps.append(info["steps"])

        stats = {
            "mean_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "success_rate": float(np.mean(successes)),
            "avg_steps": float(np.mean(steps)),
            "episodes": num_episodes
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

        all_A, all_B = [], []

        for _ in range(num_episodes):
            _, _, info = self.run_episode(render=False)
            all_A.extend(info["comm_A"])
            all_B.extend(info["comm_B"])

        all_A = np.array(all_A, dtype=np.float32)
        all_B = np.array(all_B, dtype=np.float32)

        analysis = {
            "A_mean": float(all_A.mean()),
            "A_std": float(all_A.std()),
            "A_max": float(all_A.max()),
            "A_min": float(all_A.min()),

            "B_mean": float(all_B.mean()),
            "B_std": float(all_B.std()),
            "B_max": float(all_B.max()),
            "B_min": float(all_B.min()),

            "correlation": float(np.corrcoef(all_A, all_B)[0, 1]) if len(all_A) > 1 else 0.0
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

        r, s, info = self.run_episode(render=False)

        grid = self.env.grid.copy()

        A_path = np.array(info["positions_A"])
        B_path = np.array(info["positions_B"])

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap="gray_r")

        plt.plot(A_path[:, 1], A_path[:, 0], 'r.-', label='Agent A')
        plt.plot(B_path[:, 1], B_path[:, 0], 'b.-', label='Agent B')

        plt.title(f"Trajectory (Success={s})")
        plt.legend()
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
        heat_A = np.zeros((rows, cols))
        heat_B = np.zeros((rows, cols))
        count = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                if self.env.grid[r, c] == 1:  # obstacle
                    continue

                self.env.reset()
                self.env.agent_positions = [(r, c), (r, c)]

                sA = self.env._get_observation(0)
                sB = self.env._get_observation(1)

                aA, comA = self._greedy(self.model_A, sA)
                aB, comB = self._greedy(self.model_B, sB)

                heat_A[r, c] += comA
                heat_B[r, c] += comB
                count[r, c] += 1

        heat_A /= np.maximum(count, 1)
        heat_B /= np.maximum(count, 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Communication Heatmap (Agent A)")
        plt.imshow(heat_A, cmap="viridis")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Communication Heatmap (Agent B)")
        plt.imshow(heat_B, cmap="viridis")
        plt.colorbar()

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

        results = []

        for _ in range(num_configs):
            self.env._initialize_grid()
            r, s, info = self.run_episode(render=False)
            results.append((r, s))

        rewards = [x[0] for x in results]
        successes = [x[1] for x in results]

        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(np.mean(successes)),
            "configs_tested": num_configs
        }

    def _greedy(self, model: nn.Module, obs: np.ndarray) -> Tuple[int, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values, comm = model(obs_t)
        
        return q_values.argmax(1).item(), comm.item()


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

    model_A = AgentDQN(input_dim=11, hidden_dim=64, num_actions=5)
    model_B = AgentDQN(input_dim=11, hidden_dim=64, num_actions=5)

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

    report = {
        "summary": {
            "overall_mean_reward": results["performance"]["mean_reward"],
            "overall_success_rate": results["performance"]["success_rate"],
            "communication_corr": results["communication"]["correlation"]
        },
        "details": results
    }

    with open(save_path, "w") as f:
        json.dump(report, f, indent=4)



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
    os.makedirs("results", exist_ok=True)

    model_A, model_B = load_trained_models(checkpoint_dir)

    env = MultiAgentEnv()

    evaluator = MultiAgentEvaluator(env, model_A, model_B)

    performance = evaluator.evaluate_performance(100)
    comm = evaluator.analyze_communication(20)
    generalization = evaluator.test_generalization(10)

    evaluator.visualize_trajectory("results/trajectory.png")
    evaluator.plot_communication_heatmap("results/comm_heatmap.png")

    results = {
        "performance": performance,
        "communication": comm,
        "generalization": generalization
    }

    create_evaluation_report(results, "results/evaluation_report.json")

    print("Evaluation complete.")
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()