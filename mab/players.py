from numpy.random import random, random_integers
from abc import ABC, abstractmethod


class Player(ABC):

    def __init__(self):
        self.total_reward = 0
        self.plays = 0

    @abstractmethod
    def choose_arm(self):
        pass

    @abstractmethod
    def update_estimate(self, chosen_arm, reward):
        pass

    @abstractmethod
    def get_parameter(self):
        pass

    def average_reward(self):
        return self.total_reward / self.plays

    def play(self, testbed):
        chosen_arm = self.choose_arm()
        reward = testbed.play(chosen_arm)
        self.update_estimate(chosen_arm, reward)
        self.total_reward += reward
        self.plays += 1


class EpsilonGreedyPlayer(Player):
    def __init__(self, num_arms=10, epsilon=0.1, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.arm_estimates = {i: 0 for i in range(num_arms)}

    def choose_arm(self):
        greedy_arm = max(self.arm_estimates, key=self.arm_estimates.get)
        if random() < self.epsilon:
            # random_integers draws from inclusive range. Subtract 2 to get the range of n-1 arms
            random_index = random_integers(0, len(self.arm_estimates) - 2)
            return random_index if random_index < greedy_arm else random_index + 1
        else:
            return greedy_arm

    def update_estimate(self, chosen_arm, reward):
        self.arm_estimates[chosen_arm] += self.alpha * (reward - self.arm_estimates[chosen_arm])

    def get_parameter(self):
        return self.epsilon
