import numpy as np
from modules.mdp import MDP

class PolicyIteration:
    def __init__(self, mdp, tol=1e-10, max_iter=100000):
        """
        Initialize the PolicyIteration class.

        Parameters:
        - mdp (MDP): The MDP instance.
        - tol (float): Tolerance for convergence.
        - max_iter (int): Maximum number of iterations.
        """
        self.mdp = mdp
        self.discount_factor = self.mdp.discount_factor
        self.max_iter = max_iter
        self.tol = tol

    def value_iteration(self):
        """
        Perform value iteration to compute the optimal policy and value function.

        Returns:
        - V (np.ndarray): Optimal value function.
        - policy (np.ndarray): Optimal policy.
        - Q_matrix (np.ndarray): Optimal action-value function.
        """
        num_states = len(self.mdp.S)
        num_actions = len(self.mdp.A)
        V = np.zeros(num_states)  # Initialize value function
        policy = np.zeros(num_states, dtype=int) # initialize policy

        for i in range(self.max_iter):
          V_new = np.zeros_like(V)
          # for each state, compute the value
          for s in range(num_states):
            values = []
            for a in range(num_actions):
              Q_sa = self.mdp.R[a][s] + self.discount_factor * np.dot(self.mdp.P[a][s], V)
              values.append(Q_sa)
            V_new[s] = max(values)
            policy[s] = np.argmax(values)

          # check convergence
          if np.max(np.abs(V - V_new)) < self.tol:
            V = V_new
            break
          V = V_new
        else:
            print("Value iteration did not converge within the maximum number of iterations.")

        # compute Q matrix
        Q_matrix = np.zeros((num_states, num_actions))
        for s in range(num_states):
          for a in range(num_actions):
            Q_matrix[s, a] = self.mdp.R[a][s] + self.discount_factor * np.dot(self.mdp.P[a][s], V)

        return V, policy, Q_matrix
