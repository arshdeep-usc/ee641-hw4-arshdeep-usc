"""
Training script for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
from typing import Tuple, Optional
from multi_agent_env import MultiAgentEnv
from models import AgentDQN
from replay_buffer import ReplayBuffer

import random

def apply_observation_mask(obs: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply masking to observation based on ablation mode.

    Args:
        obs: 11-dimensional observation vector
        mode: One of 'independent', 'comm', 'full'

    Returns:
        Masked observation
    """
    # TODO: Implement masking logic
    # 'independent': Set elements 9 and 10 to zero
    # 'comm': Set element 10 to zero
    # 'full': No masking

    obs = obs.copy()

    if mode == "independent":
        obs[9] = 0.0
        obs[10] = 0.0
    elif mode == "comm":
        obs[10] = 0.0

    return obs


class MultiAgentTrainer:
    """
    Trainer for multi-agent DQN system.

    Handles training loop, exploration, and network updates.
    """

    def __init__(self, env: MultiAgentEnv, args):
        """
        Initialize trainer.

        Args:
            env: Multi-agent environment
            args: Training arguments
        """
        self.env = env
        self.args = args

        # Use CPU for small networks
        self.device = torch.device("cpu")

        # TODO: Initialize networks for both agents (remember to .to(self.device))
        # TODO: Initialize target networks (if using)
        # TODO: Initialize optimizers
        # TODO: Initialize replay buffer
        # TODO: Initialize epsilon for exploration

        input_dim = 11
        num_actions = 5

        self.net_A = AgentDQN(input_dim, args.hidden_dim, num_actions).to(self.device)
        self.net_B = AgentDQN(input_dim, args.hidden_dim, num_actions).to(self.device)
        self.target_A = AgentDQN(input_dim, args.hidden_dim, num_actions).to(self.device)
        self.target_B = AgentDQN(input_dim, args.hidden_dim, num_actions).to(self.device)

        self.target_A.load_state_dict(self.net_A.state_dict())
        self.target_B.load_state_dict(self.net_B.state_dict())

        self.optim_A = optim.Adam(self.net_A.parameters(), lr=args.lr)
        self.optim_B = optim.Adam(self.net_B.parameters(), lr=args.lr)

        self.replay = ReplayBuffer(capacity=10000, seed=args.seed)

        self.epsilon = args.epsilon_start

        self.models_dir = "results/agent_models"
        self.logs_dir = "results/training_logs"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.train_log = {"episode": [], "reward": [], "epsilon": []}

        print("Trainer initialized.")

    def select_action(self, state: np.ndarray, network: nn.Module,
                      epsilon: float) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Agent observation (11-dimensional, may need masking)
            network: Agent's DQN
            epsilon: Exploration probability

        Returns:
            action: Selected action
            comm_signal: Communication signal
        """
        # TODO: Apply observation masking based on self.args.mode
        #       masked_state = apply_observation_mask(state, self.args.mode)
        # TODO: With probability epsilon, select random action
        # TODO: Otherwise, select action with highest Q-value
        # TODO: Always get communication signal from network
        # TODO: Return (action, comm_signal)

        masked_state = apply_observation_mask(state, self.args.mode)
        state_t = torch.tensor(masked_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values, comm_signal = network(state_t)
            comm_signal = comm_signal.item()

        if np.random.rand() < epsilon:
            action = np.random.randint(5)
        else:
            action = q_values.argmax(dim=1).item()

        return action, comm_signal

    def update_networks(self, batch_size: int) -> float:
        """
        Sample batch and update both agent networks.

        Args:
            batch_size: Size of training batch

        Returns:
            loss: Combined loss value
        """
        # TODO: Sample batch from replay buffer
        # TODO: Convert to tensors and move to device
        # TODO: Compute Q-values for current states
        # TODO: Compute target Q-values using target networks
        # TODO: Calculate TD loss for both agents
        # TODO: Backpropagate and update networks
        # TODO: Return combined loss

        if len(self.replay) < batch_size:
            return 0.0

        (sA, sB,
         aA, aB,
         cA, cB,
         r,
         nA, nB,
         d) = self.replay.sample(batch_size)

        self.optim_A.zero_grad()
        self.optim_B.zero_grad()

        sA_t = torch.tensor([apply_observation_mask(x, self.args.mode) for x in sA], dtype=torch.float32)
        sB_t = torch.tensor([apply_observation_mask(x, self.args.mode) for x in sB], dtype=torch.float32)
        nA_t = torch.tensor([apply_observation_mask(x, self.args.mode) for x in nA], dtype=torch.float32)
        nB_t = torch.tensor([apply_observation_mask(x, self.args.mode) for x in nB], dtype=torch.float32)
        r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(-1)
        d_t = torch.tensor(d, dtype=torch.float32).unsqueeze(-1)
        aA_t = torch.tensor(aA).long().unsqueeze(-1)
        aB_t = torch.tensor(aB).long().unsqueeze(-1)

        qA, _ = self.net_A(sA_t)
        qB, _ = self.net_B(sB_t)
        qA_selected = qA.gather(1, aA_t)
        qB_selected = qB.gather(1, aB_t)

        #with torch.no_grad():
        next_qA_online, _ = self.net_A(nA_t)
        next_qB_online, _ = self.net_B(nB_t)

        next_actions_A = next_qA_online.argmax(dim=1, keepdim=True)
        next_actions_B = next_qB_online.argmax(dim=1, keepdim=True)

        next_qA_target, _ = self.target_A(nA_t)
        next_qB_target, _ = self.target_B(nB_t)

        qa_target = next_qA_target.gather(1, next_actions_A)
        qb_target = next_qB_target.gather(1, next_actions_B)

        target = r_t + self.args.gamma * (1 - d_t) * 0.5 * (qa_target + qb_target)

        loss_A = F.mse_loss(qA_selected, target)
        loss_B = F.mse_loss(qB_selected, target)
        loss = loss_A + loss_B

        
        loss.backward()
        
        self.optim_A.step()
        self.optim_B.step()

        return loss.item()

    def train_episode(self) -> Tuple[float, bool]:
        """
        Run one training episode.

        Returns:
            episode_reward: Total reward for episode
            success: Whether agents reached target
        """
        # TODO: Reset environment
        # TODO: Initialize episode variables
        # TODO: Run episode until termination:
        #       - Select actions for both agents
        #       - Execute actions in environment
        #       - Store transition in replay buffer
        #       - Update networks if enough samples
        # TODO: Return episode reward and success flag

        sA, sB = self.env.reset()
        episode_reward = 0.0
        success = False

        for _ in range(self.args.max_steps):
            aA, commA = self.select_action(sA, self.net_A, self.epsilon)
            aB, commB = self.select_action(sB, self.net_B, self.epsilon)

            (nA, nB), r, done = self.env.step(aA, aB, commA, commB)

            self.replay.push(
                sA, sB,
                aA, aB,
                commA, commB,
                r,
                nA, nB,
                done
            )

            loss = self.update_networks(self.args.batch_size)

            sA, sB = nA, nB
            episode_reward += r

            if r == 10.0:
                sucess = True
                break

            if done:
                success = (r >= 10.0)
                break

        return episode_reward, success

    def train(self) -> None:
        """
        Main training loop.
        """
        # TODO: Create results directories
        # TODO: Initialize logging
        # TODO: Main training loop:
        #       - Run episodes
        #       - Update epsilon
        #       - Update target networks periodically
        #       - Log progress
        #       - Save checkpoints
        # TODO: Save final models including TorchScript format:
        #       scripted_model = torch.jit.script(self.network_A)
        #       scripted_model.save("dqn_net.pt")


        for episode in range(1, self.args.num_episodes + 1):
            episode_reward, success = self.train_episode()

            if episode % 100 == 0:
                print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Success: {success}")

            self.train_log["episode"].append(episode)
            self.train_log["reward"].append(episode_reward)
            self.train_log["epsilon"].append(self.epsilon)

            log_path = os.path.join(self.logs_dir, "training_log.json")
            with open(log_path, "w") as f:
                json.dump(self.train_log, f, indent=4)

            self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)


            if episode % self.args.target_update == 0:
                self.target_A.load_state_dict(self.net_A.state_dict())
                self.target_B.load_state_dict(self.net_B.state_dict())
                print(f"[Episode {episode}] Target networks updated.")

            if episode % self.args.save_freq == 0:
                torch.save(self.net_A.state_dict(), os.path.join(self.models_dir, "agent_A.pt"))
                torch.save(self.net_B.state_dict(), os.path.join(self.models_dir, "agent_B.pt"))
                print(f"[Episode {episode}] Models saved to {self.models_dir}")


        torch.save(self.net_A.state_dict(), os.path.join(self.models_dir, "agent_A.pt"))
        torch.save(self.net_B.state_dict(), os.path.join(self.models_dir, "agent_B.pt"))
        print(f"Training complete. Final models saved to {self.models_dir}")

        scripted_A = torch.jit.script(self.net_A)
        scripted_B = torch.jit.script(self.net_B)
        scripted_A.save(os.path.join(self.models_dir, "agent_A_scripted.pt"))
        scripted_B.save(os.path.join(self.models_dir, "agent_B_scripted.pt"))


    
    def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            mean_reward: Average reward
            success_rate: Fraction of successful episodes
        """
        # TODO: Set networks to evaluation mode
        # TODO: Run episodes without exploration
        # TODO: Track rewards and successes
        # TODO: Return statistics

        self.net_A.eval()
        self.net_B.eval()

        rewards = []
        successes = []

        for _ in range(num_episodes):
            sA, sB = self.env.reset()
            total_r = 0.0
            success = False

            for _ in range(self.args.max_steps):
                aA, commA = self.select_action(sA, self.net_A, epsilon=0.0)
                aB, commB = self.select_action(sB, self.net_B, epsilon=0.0)

                (nA, nB), r, done = self.env.step(aA, aB, commA, commB)

                sA, sB = nA, nB
                total_r += r

                if done:
                    success = (r >= 10.0)
                    break

            rewards.append(total_r)
            successes.append(success)

        return np.mean(rewards), np.mean(successes)


def main():
    """
    Parse arguments and run training.
    """
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN')

    # Environment parameters
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10],
                       help='Grid dimensions')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                       help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Epsilon decay rate')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--target_update', type=int, default=100,
                       help='Target network update frequency')

    # Ablation study mode
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Model save frequency')

    args = parser.parse_args()

    # TODO: Set random seeds
    # TODO: Create environment
    # TODO: Create trainer
    # TODO: Run training
    # TODO: Final evaluation

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment
    env = MultiAgentEnv(
        grid_size=tuple(args.grid_size),
        max_steps=args.max_steps,
        seed=args.seed
    )

    # Trainer
    trainer = MultiAgentTrainer(env, args)

    # Train
    trainer.train()

    import matplotlib.pyplot as plt
    
    episodes = np.array(trainer.train_log["episode"])
    rewards = np.array(trainer.train_log["reward"])
    
    successes = (rewards >= 2.0).astype(float)
    
    def smooth(x, window=50):
        if window <= 1:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0))
        res = (cumsum[window:] - cumsum[:-window]) / float(window)
        pad = np.full(window - 1, res[0])
        return np.concatenate([pad, res])
    
    rewards_smooth = smooth(rewards, window=50)
    success_smooth = smooth(successes, window=50)
    
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.plot(episodes, rewards_smooth, color="tab:blue", label="Reward (smoothed)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Success (Reward>=2.0) rate", color="tab:green")
    ax2.plot(episodes, success_smooth, color="tab:green", label="Success (smoothed)")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(-0.05, 1.05)
    
    plt.title("Training Reward and Success Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(trainer.logs_dir, "training_reward_success_"+str(args.mode)+".png"))
    plt.close()

    # Evaluate
    mean_r, succ = trainer.evaluate(10)
    print(f"Evaluation: mean_reward={mean_r:.2f}, success_rate={succ:.2f}")

    print(args)


if __name__ == '__main__':
    main()