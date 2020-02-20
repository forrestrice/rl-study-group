from numpy.random import normal
from numpy import amax


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
        self.arms = normal(0, 1, num_arms)
        self.optimal_plays = 0
        self.total_plays = 0
        self.noise_gen = self._noise()

    def play(self, arm):
        optimal_action_value = amax(self.arms)
        reward = self.arms[arm]
        if reward == optimal_action_value:
            self.optimal_plays += 1
        self.total_plays += 1
        self.arms[arm] += next(self.noise_gen)
        return reward

    def optimal_percentage(self):
        return 100 * self.optimal_plays / self.total_plays

    @staticmethod
    def _noise():
        while True:
            noise_batch = normal(0, 0.01, 200_000)
            for noise in noise_batch:
                yield noise
