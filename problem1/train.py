"""
Training script for Value Iteration and Q-Iteration.
"""

import numpy as np
import argparse
import json
import os
from environment import GridWorldEnv
from value_iteration import ValueIteration
from q_iteration import QIteration


def main():
    """
    Run both algorithms and save results.
    """
    parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

    # TODO: Initialize environment with seed
    # TODO: Run Value Iteration
    #       - Create ValueIteration solver
    #       - Solve for optimal values
    #       - Extract policy
    #       - Save results
    # TODO: Run Q-Iteration
    #       - Create QIteration solver
    #       - Solve for optimal Q-values
    #       - Extract policy and values
    #       - Save results
    # TODO: Compare algorithms
    #       - Print convergence statistics
    #       - Check if policies match
    #       - Save comparison results

    env = GridWorldEnv(seed=args.seed)

    print("Running Value Iteration...")
    vi_solver = ValueIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    vi_values, vi_iters = vi_solver.solve(max_iterations=args.max_iter)
    vi_policy = vi_solver.extract_policy(vi_values)

    np.save('results/value_iteration_values.npy', vi_values)
    np.save('results/value_iteration_policy.npy', vi_policy)

    with open('results/value_iteration_stats.json', 'w') as f:
        json.dump({
            'iterations': vi_iters,
            'gamma': args.gamma,
            'epsilon': args.epsilon
        }, f, indent=4)

    print(f"Value Iteration converged in {vi_iters} iterations.")

    print("\nRunning Q-Iteration...")
    qi_solver = QIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    qi_qvalues, qi_iters = qi_solver.solve(max_iterations=args.max_iter)
    qi_policy = qi_solver.extract_policy(qi_qvalues)
    qi_values = qi_solver.extract_values(qi_qvalues)

    # Save Q-Iteration results
    np.save('results/q_iteration_qvalues.npy', qi_qvalues)
    np.save('results/q_iteration_values.npy', qi_values)
    np.save('results/q_iteration_policy.npy', qi_policy)

    with open('results/q_iteration_stats.json', 'w') as f:
        json.dump({
            'iterations': qi_iters,
            'gamma': args.gamma,
            'epsilon': args.epsilon
        }, f, indent=4)

    print(f"Q-Iteration converged in {qi_iters} iterations.")

    print("\nComparing algorithms...")

    policies_match = np.array_equal(vi_policy, qi_policy)
    values_close = np.allclose(vi_values, qi_values, atol=1e-3)

    comparison = {
        "value_iteration_iterations": vi_iters,
        "q_iteration_iterations": qi_iters,
        "policies_match": bool(policies_match),
        "values_similar": bool(values_close),
    }

    with open('results/comparison.json', 'w') as f:
        json.dump(comparison, f, indent=4)

    print("Policies match?" , policies_match)
    print("Values similar?", values_close)
    print("\nResults saved to 'results/' directory.")


if __name__ == '__main__':
    main()