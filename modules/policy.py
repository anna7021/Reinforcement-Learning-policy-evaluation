class Policy:
    def __init__(self, actions):
        self.actions = actions

    def action(self, state):
        return self.actions[state]

# Example usage:
policy_always_invest = Policy([1, 1, 1, 0, 0])
policy_never_invest = [0, 0, 0, 0, 0]