# Reinforcement-Learning-policy-evaluation
Project Background: You are the CEO of a biotech company which is considering the development of a new vaccine. Starting at phase 0 (state 0), the drug develpment can stay in the same state or advance to "phase 1 with promising results" (state 1) or advance to "phase 1 with disappointing results" (state 2), or fail completely (state 4). At phase 1, the drug can stay in the same state, fail or become a success (state 3), in which case you will sell its patent to a big pharma company for 10$ million.

These state transitions happen from month to month, and at each state, you have the option to make an additional investment of 100,000$, which increases the chances of success.

After careful study, your analysts develop the program below to simulate different scenarios using statistical data from similar projects.

task 1. Use the Monte Carlo& Dynamic Programming method to find the value functions induced by the following policies:
a) Always invest
b) Never invest

task 2. Use the Monte Carlo& Dynamic Programming method to do policy iteration and find the optimal policy and corresponding value functions

Use a discount factor of 0.9960 to discount future rewards.
