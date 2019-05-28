#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sets the Gridworld environments and uses a Q-learning (reinforcement learning)
algorithm derived from OpenAI Gym's Frozen Lake environment to teach an agent
to avoid a moved obstacle and to reach a goal state.

"Winter is in full swing, and the community lake has frozen over. Recalling
that the community offers warm drinks to patrons who reach the lake's gazebo,
you've donned your skates for a day out on the ice.

A large hill blocks your view of the gazebo, but you remember a way around the
hill from last winter. However, the heavy rains last fall have caused
the lake to freeze a little differently. To make matters worse,
the usual way to the gazebo is no longer there. You don't want to ruin your
skates by climbing over the hill.

Perhaps you can find another way to skate to the gazebo?"

    S : your starting state, where you put on your skates
    F : a frozen surface, easy to skate on, but not as rewarding as a warm drink
    H : the hill, which will ruin your skates (you cannot step here)
    G : the gazebo (your goal state), where you receive a reward

Assembled by Eric Easthope
'''

import numpy as np
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# Configure an environment with a way past the obstacle on the right-hand side
left_env_map = [
    'FFFFFFFFG',
    'FFFFFFFFF',
    'FFFFFFFFF',
    'HHHHHHHHF',
    'FFFFFFFFF',
    'FFFSFFFFF'
]

# Configure an environment with a way past the obstacle on the left-hand side
right_env_map = [
    'FFFFFFFFG',
    'FFFFFFFFF',
    'FFFFFFFFF',
    'FHHHHHHHH',
    'FFFFFFFFF',
    'FFFSFFFFF'
]

# Assign numerical values to possible actions (the agent's movements)
directions = dict([
    (0, 'LEFT'),
    (1, 'DOWN'),
    (2, 'RIGHT'),
    (3, 'UP')
])

# Generate references for movements prohibited by the obstacle
left_prohibited_moves = dict(
    zip(range(18,26), ['DOWN' for i in range(8)]) +
    zip(range(36,44), ['UP'   for i in range(8)]) +
    [(35, 'LEFT')]
)

right_prohibited_moves = dict(
    zip(range(19,27), ['DOWN' for i in range(8)]) +
    zip(range(37,45), ['UP' for i in range(8)])   +
    [(27, 'RIGHT')]
)

gamma = 0.95           # Set a discount factor (fixed within [0,1])
epsilon = 0.9          # Set the probability that the agent acts randomly
step_size = 0.85       # Set a step size (fixed within [0,1])

# Set the number of times that we train the agent to reach the goal state
total_episodes = 9000

# Set at which episode the obstacle moves
episode_when_obstacle_moves = 1000

# Limit how many steps the agent may take during a single episode
max_steps = 50

def choose_action(environment, state, Q, epsilon):
    '''
    Choose an action based on:

        * The `environment`
        * The agent's current `state`
        * An action-value function Q(state, action)
        * The probability `epsilon` that the agent acts randomly

    The agent defaults to moving upwards unless an `action` is set by the
    if-else statement below.

    Given some `state` s, the agent chooses a random action with probability
    `epsilon`. Otherwise, the agent chooses whichever action maximizes the
    current estimate by `Q`.

    Returns a number within 0-3 to specify one of four directions
    '''

    action = 3
    if np.random.uniform(0, 1) < epsilon:
        action = environment.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action;

def run_episode(environment, Q, step_size, prohibited_moves=None):
    '''
    Run an episode to move the agent around Gridworld

    The agent begins at the starting state S, and chooses at most `max_steps`
    to try to reach the goal state G. The agent receives a reward +1 for
    reaching the goal state.

    As the agent moves, the action-value function `Q` is iteratively updated to
    favour optimal steps.

    Returns reward, and number of steps to reach goal state G
    '''

    # Reset the environment to place the agent at the starting state S
    state = environment.reset()

    # Take at most `max_steps` to reach the goal state G
    for t in range(max_steps):

        # Make the agent choose which direction to step
        action = choose_action(environment, state, Q, epsilon)

        # Detect if the chosen action hits an obstacle
        if (prohibited_moves is None or
            not (state, directions[action]) in prohibited_moves.viewitems()):

            # Take a step in the environment based on the chosen action
            next_state, reward, done, info = environment.step(action)

            # Update the action-value function `Q`
            Q[state, action] = (Q[state, action] +
                step_size                        *
                (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
                )

            # Update state to the state that is being moved to
            state = next_state

            # Give reward +1 if the goal state G is reached within `max_steps`
            if done:
                return 1, t+1;

    # Give reward 0 if the goal state G is not reached within `max_steps`
    return 0, max_steps;

if __name__ == '__main__':
    '''
    Initialize and run the Q-learning algorithm within Gridworld
    '''

    # Initialize two "Frozen Lake" environments using the maps specified above
    left_environment = FrozenLakeEnv(desc=left_env_map, is_slippery=False)
    right_environment = FrozenLakeEnv(desc=right_env_map, is_slippery=False)

    # Initialize an action-value function Q(state, action)
    Q = np.zeros(shape=[
        left_environment.observation_space.n,
        left_environment.action_space.n
    ])

    # Initialize accounts of reward and steps for every episode
    cumulative_reward = np.zeros(total_episodes)
    number_of_steps = np.zeros(total_episodes)

    # Run episodes with the first (left) environment map
    for episode in range(episode_when_obstacle_moves):
        reward, steps = run_episode(left_environment,
                                    Q, step_size=step_size,
                                    prohibited_moves=left_prohibited_moves)

        # Store the cumulative reward at this episode
        cumulative_reward[episode] = np.max(cumulative_reward) + reward

        # Store the number of steps taken during this episode
        number_of_steps[episode] = steps

    # Run episodes with the second (right) environment map (the obstacle moves)
    for episode in range(episode_when_obstacle_moves, total_episodes):
        reward, steps = run_episode(right_environment,
                                    Q, step_size=step_size,
                                    prohibited_moves=right_prohibited_moves)

        # Store the cumulative reward at this episode
        cumulative_reward[episode] = np.max(cumulative_reward) + reward

        # Store the number of steps taken during this episode
        number_of_steps[episode] = steps

    '''
    Below I plot the outcome of teaching the agent.

    The plot relates the agent's cumulative reward to the number of
    steps that the agent takes during each episode. At first, the agent is seen
    to take many steps with little to no reward.

    However, once the agent reaches the goal state G a number of times, the
    ratio of cumulative reward to number of episodes trends linearly. The agent
    is also taking few steps to reach the goal state. Accordingly, the ratio of
    cumulative reward to number of steps tends to increase.

    At the 1000th episode, whereupon the agent must adapt to a moved obstacle,
    this trend in the reward-step ratio is temporarily disturbed.
    Nevertheless, the agent continues to receive reward, and eventually learns
    a new optimal policy for reaching for the goal state in fewer steps.
    '''

    plt.figure(figsize=(15,10))
    plt.axis([
        0, total_episodes,
        0, np.max(cumulative_reward/number_of_steps) + 1
    ])
    plt.xlabel("Episodes", fontsize=15)
    plt.ylabel("Cumulative Reward / Number of Steps", fontsize=15)
    plt.plot(
        [i for i in range(total_episodes)],
        cumulative_reward/number_of_steps,
    )
    plt.show()
