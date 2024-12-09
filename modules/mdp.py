import numpy as np
from collections import defaultdict

class MDP():
  def __init__(self):
    self.A = [0, 1]
    self.S = [0, 1, 2, 3, 4]
    # Transition Probability Matrix for Action 0 (Do not invest)
    P0 = np.array([[0.5, .15, .15, 0, .20],
                   [0, .5, .0, .25, .25],
                   [0, 0, .15, .05, .8],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1]])
     # Reward Vector for Action 0
    R0 = np.array([0, 0, 0, 10, 0]) # Rewards for transitioning into each state
    # Transition Probability Matrix for Action 1 (Invest)
    P1 = np.array([[0.5, .25, .15, 0, .10],
                   [0, .5, .0, .35, .15],
                   [0, 0, .20, .05, .75],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1]])
    # Reward Vector for Action 1 (Invest)
    R1 = np.array([-0.1, -0.1, -0.1, 10, 0])

    self.P = [P0, P1]
    self.R = [R0, R1]
    self.discount_factor = 0.996

  def step(self, s, a):
    """
    Perform one step in the MDP.

    Parameters:
    - s (int): Current state
    - a (int): Action taken

    Returns:
    - s_prime (int): Next state after taking action a
    - R (float): Reward received after the transition
    - done (bool): Flag indicating if the next state is terminal
    """
    # Choose next state based on transition probabilities for action a
    s_prime = np.random.choice(len(self.S), p=self.P[a][s])
    # Get the immediate reward for transitioning into s_prime via action a
    R = self.R[a][s]
    # Determine if the episode has ended
    if s_prime == 4:  # State 4 is a terminal fail state
      done = True
    else:
      done = False
    return s_prime, R, done

  def simulate(self, s, a, π):
    """
    Simulate an entire episode following policy π.

    Parameters:
    - s (int): Starting state
    - a (int): Starting action
    - π (list): Policy mapping states to actions

    Returns:
    - history (list): List of tuples containing (state, action, reward)
    """
    done = False
    t = 0
    history = []
    while not done:
      if t > 0:
        a = π[s]  # Choose action based on policy after the first step
      s_prime, R, done = self.step(s, a)
      history.append((s, a, R))
      s = s_prime
      t += 1

    return history