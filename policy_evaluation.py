from collections import defaultdict
import numpy as np

from modules.mdp import MDP

class MonteCarloPolicyEvaluation:
    def __init__(self, mdp):
        """
        Initialize the Monte Carlo Policy Evaluation.

        Parameters:
        - mdp (MDP): The Markov Decision Process instance.
        - discount_factor (float): The discount factor for future rewards.
        """
        self.mdp = mdp
        self.discount_factor = self.mdp.discount_factor
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.V = defaultdict(float)
        self.Q = None


    def monte_carlo_policy_evaluation(self, policy, num_episodes):
      """
      Estimate the value function V^pi(s) using Monte Carlo simulations.

      Parameters:
      - policy (list): The policy to evaluate
      - num_episodes (int): Number of episodes to simulate per state

      Returns:
      - Q (np.ndarray): Estimated action-value function with shape (num_non_terminal_states, num_actions)
      - V (dict): Estimated value function
      """
      self.returns_sum.clear()
      self.returns_count.clear()
      self.V.clear()

      non_terminal_states = [0, 1, 2]
      actions = [0, 1]

      for s in non_terminal_states:
          for a in actions:
            for episode_num in range(num_episodes):
              history = self.mdp.simulate(s, a, policy)

              # compute the discounted cumulative rewards
              G = 0
              for t in range(len(history) - 1, -1, -1):
                s, a, r = history[t]
                G = self.discount_factor * G + r
                self.returns_sum[(s, a)] += G
                self.returns_count[(s, a)] += 1.0
      self.Q = np.zeros((len(non_terminal_states), len(actions)))
      for s in non_terminal_states:
        for a in actions:
          if (s, a) in self.returns_count:
            self.Q[s, a] = self.returns_sum[(s, a)] / self.returns_count[(s, a)]

      for s in self.mdp.S:
        if s in [3, 4]: # terminal states have fixed values
          self.V[s] = 10 if s == 3 else 0
        elif s in non_terminal_states:
          a = policy[s]
          self.V[s] = self.Q[s, a]
        else:
          self.V[s] = 0.0  # If a state was never visited

      return self.Q, self.V