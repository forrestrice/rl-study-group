from numpy.random import normal


class MovingArm:
    def __init__(self):
        self.action_value = normal(0, 1)

    def play(self):
        reward = self.action_value + normal(0, 1)
        self.action_value += normal(0, 0.01)
        return reward

    def __repr__(self):
        return str(self.action_value)


class MovingTestbed:
    def __init__(self, num_arms=10):
        self.arms = [MovingArm() for i in range(num_arms)]
        self.optimal_plays = 0
        self.total_plays = 0

    def play(self, arm):
        optimal_action_value = max([a.action_value for a in self.arms])
        chosen_arm = self.arms[arm]
        if chosen_arm.action_value == optimal_action_value:
            self.optimal_plays += 1
        self.total_plays += 1
        return chosen_arm.play()

    def optimal_percentage(self):
        return 100 * self.optimal_plays / self.total_plays
