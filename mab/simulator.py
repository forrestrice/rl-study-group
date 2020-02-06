from collections import defaultdict

from mab.bandits import MovingTestbed
from mab.players import EpsilonGreedyPlayer

checkpoint_steps = 100_000
search_iter = 10


def parameter_search(player_generator):
    parameter_reward = defaultdict(float)
    parameter_optimal_ratio = defaultdict(float)
    for player in player_generator:
        testbed = MovingTestbed()
        for step in range(checkpoint_steps):
            player.play(testbed)
        # track metrics for next 100,000 iterations
        checkpoint_reward = player.total_reward
        checkpoint_optimal_plays = testbed.optimal_plays
        for step in range(checkpoint_steps):
            player.play(testbed)
        tracked_reward = player.total_reward - checkpoint_reward
        tracked_optimal = testbed.optimal_plays - checkpoint_optimal_plays
        parameter_reward[player.get_parameter()] += tracked_reward / checkpoint_steps
        parameter_optimal_ratio[player.get_parameter()] += tracked_optimal / checkpoint_steps
        print(player.get_parameter(), tracked_reward / checkpoint_steps, tracked_optimal / checkpoint_steps)
    for p in parameter_reward:
        print(p, parameter_reward[p] / search_iter, parameter_optimal_ratio[p] / search_iter)


def epsilon_player_generator():
    for i in range(0, 10):
        epsilon = i / 100
        for j in range(search_iter):
            yield EpsilonGreedyPlayer(epsilon=epsilon)


parameter_search(epsilon_player_generator())
