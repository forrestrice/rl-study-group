from mab.bandits import MovingTestbed
from mab.players import EpsilonGreedyPlayer

checkpoint_steps = 100_000


def parameter_search(player_generator):
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
        print(player.get_parameter(), tracked_reward / checkpoint_steps, tracked_optimal / checkpoint_steps)


def epsilon_player_generator():
    for i in range(0, 100):
        epsilon = i / 100
        yield EpsilonGreedyPlayer(epsilon=epsilon)


parameter_search(epsilon_player_generator())
