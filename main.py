import numpy as np
import logging
import os

from mdp import MDP
from policy import Policy
from policy_evaluation import MonteCarloPolicyEvaluation
from policy_iteration import PolicyIteration

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), level=logging.INFO)

def main():
    mdp = MDP()

    # Task 1: Monte Carlo Policy Evaluation
    mc_evaluator = MonteCarloPolicyEvaluation(mdp)

    logging.info("=== Monte Carlo Policy Evaluation ===")
    policy_always_invest1 = Policy([1, 1, 1, 0, 0])
    Q_invest1, V_invest1 = mc_evaluator.monte_carlo_policy_evaluation(policy_always_invest1.actions, num_episodes=10000)
    logging.info("Policy: Always Invest")
    logging.info("Value Function V:")
    for state in mdp.S:
        logging.info(f"V[{state}] = {V_invest1[state]:.4f}")
    logging.info("")

    policy_always_invest2 = Policy([0, 0, 0, 0, 0])
    Q_invest2, V_invest2 = mc_evaluator.monte_carlo_policy_evaluation(policy_always_invest2.actions, num_episodes=10000)
    logging.info("Policy: Never Invest")
    logging.info("Value Function V:")
    for state in mdp.S:
        logging.info(f"V[{state}] = {V_invest2[state]:.4f}")
    logging.info("")

    # Task 2: Optimal Policy using Policy Iteration
    logging.info("=== Policy Iteration for Optimal Policy ===")
    policy_iter = PolicyIteration(mdp)
    V_opt, policy_opt, Q_opt = policy_iter.value_iteration()
    logging.info("Optimal Value Function V:")
    for state in mdp.S:
        logging.info(f"V[{state}] = {V_opt[state]:.4f}")
    logging.info("Optimal Policy:")
    for state, action in enumerate(policy_opt):
        action_str = "Invest" if action == 1 else "Do not invest"
        logging.info(f"Policy[{state}] = {action_str}")
    logging.info

if __name__ == '__main__':
    main()