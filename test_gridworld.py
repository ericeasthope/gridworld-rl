#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Tests for functions used by the Q-learning implementation in `gridworld.py`

Assembled by Eric Easthope
'''

import numpy as np
from gridworld import choose_action, run_episode
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# Configure an environment without any obstacles
map_without_obstacle = [
    'SF',
    'FG'
]

# Configure an environment with a single obstacle
map_with_obstacle = [
    'FFG',
    'FHF',
    'SFF'
]

def test_choose_action():
    '''
    Check if choose_action always outputs a number within 0-3 (this is the
    "action space" for an environment with four possible movement directions).

    Two environments are tested. The first environment `map_without_obstacle`
    starts the agent at the 0th environment index, and contains no obstacles.

    The second environment starts the agent at a nonzero environment index, and
    contains a single obstacle. By fixing `epsilon = 0.5`, any one of the four
    possible directions is equally likely to be tested.
    '''

    # Set the probability that the agent acts randomly
    epsilon = 0.5

    # Initialize two "Frozen Lake" environments using the maps specified above
    first_environment = FrozenLakeEnv(desc=map_without_obstacle,
                                      is_slippery=False)
    second_environment = FrozenLakeEnv(desc=map_with_obstacle,
                                       is_slippery=False)

    # Reset the first environment to place the agent at the starting state S
    state = first_environment.reset()

    # Get the size of the first environment's "action space"
    actions = first_environment.action_space.n

    # Initialize an action-value function Q(state, action)
    Q = np.zeros(shape=[
        first_environment.observation_space.n,
        actions
    ])

    # Choose a number corresponding to the direction that the agent moves
    action = choose_action(first_environment, state, Q, epsilon)

    assert action in range(actions);

    # Reset the second environment to place the agent at the starting state S
    state = second_environment.reset()

    # Get the size of the second environment's "action space"
    actions = second_environment.action_space.n

    # Initialize an action-value function Q(state, action)
    Q = np.zeros(shape=[
        second_environment.observation_space.n,
        actions
    ])

    # Choose a number corresponding to the direction that the agent moves
    action = choose_action(second_environment, state, Q, epsilon)

    assert action in range(actions);

def test_run_episode():
    '''
    Check if running an episode where the agent (almost surely) succeeds always
    results in a reward +1, and a step count less than or equal to `max_steps`.

    The first environment `map_without_obstacle` is tested. If the agent
    successfully reaches the goal state G within `max_steps`, the agent must
    receive a nonzero reward.
    '''

    # Limit how many steps the agent may take during a single episode
    max_steps = 50

    # Set the probability that the agent acts randomly
    epsilon = 1.

    # Set a step size (fixed within [0,1])
    step_size = 0.8

    # Initialize two "Frozen Lake" environments using the maps specified above
    first_environment = FrozenLakeEnv(desc=map_without_obstacle,
                                      is_slippery=False)

    # Reset the environment to place the agent at the starting state S
    state = first_environment.reset()

    # Get the size of the environment's "action space"
    actions = first_environment.action_space.n

    # Initialize an action-value function Q(state, action)
    Q = np.zeros(shape=[
        first_environment.observation_space.n,
        actions
    ])

    # Run an episode with the environment map without obstacles
    reward, steps = run_episode(first_environment, Q, step_size=step_size)

    assert reward > 0 and steps <= max_steps;
