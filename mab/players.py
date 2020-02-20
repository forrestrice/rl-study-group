from numpy.random import random, random_integers
from numpy import zeros, argmax
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
        self.arm_estimates = zeros(num_arms)
        self.explore_gen = self.explore_generator()
        self.random_arm_gen = self.random_arm_generator()

    def choose_arm(self):
        greedy_arm = argmax(self.arm_estimates)
        if next(self.explore_gen) < self.epsilon:
            return next(self.random_arm_gen)
        else:
            return greedy_arm

    def update_estimate(self, chosen_arm, reward):
        self.arm_estimates[chosen_arm] += self.alpha * (reward - self.arm_estimates[chosen_arm])

    def get_parameter(self):
        return self.epsilon

    @staticmethod
    def explore_generator():
        while True:
            randoms = random(200_000)
            for rand in randoms:
                yield rand

    def random_arm_generator(self):
        while True:
            random_arms = random_integers(0, len(self.arm_estimates) - 1, 10_000)
            for random_arm in random_arms:
                yield random_arm
