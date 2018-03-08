import sys
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from scipy.stats import norm, uniform, bernoulli
from cartpole_helpers import render_agent, get_policy_mean_reward, get_q_values


class SimpleLinearAgent(object):
    """A simple linear agent.
    Implements the random guessing and hill climbing optimization methods
    """

    def __init__(self, weights=None):
        if weights:
            self.weights = weights
        else:
            self.weights = norm(loc=0, scale=0.1).rvs(size=4)

    def choose_action(self, observation):
        score = np.dot(self.weights, observation)
        return int(score >= 0)

    def _optimize_policy(self,
                         env,
                         update_func,
                         update_func_opts,
                         n_samples_per_rollout=10,
                         n_iter=1000,
                         early_stop=True):
        """Generic optimization function
        Takes in an update rule. At each iteration applies the update rule
        to get a new set of weights and evaluates the performance of the new
        weights on the environment. If the new weights get a lower average
        score than the old weights, they are discarder; otherwise, they are
        retained.
        """
        curr_reward = get_policy_mean_reward(
            self, env, n_samples_per_rollout)

        for i in range(n_iter):
            curr_weights = self.weights
            self.weights = update_func(update_func_opts)
            new_reward = get_policy_mean_reward(
                self, env, n_samples_per_rollout)

            if new_reward == 200 and early_stop:
                return i+1
            if new_reward > curr_reward:
                curr_reward = new_reward
            else:
                self.weights = curr_weights

        return n_iter

    def rgo(self,
            env,
            param_distributions,
            n_samples_per_rollout=10,
            n_iter=1000,
            early_stop=True):
        """Random guessing optimization"""
        opts = {
            'param_distributions': param_distributions
        }

        def update_func(opts):
            return [rv.rvs() for rv in opts['param_distributions']]

        return self._optimize_policy(
            env, update_func, opts, n_samples_per_rollout, n_iter, early_stop)

    def hco(self, env, n_samples_per_rollout=10, n_iter=1000, early_stop=True):
        """Hill climbing optimization"""
        opts = {
            'weights': self.weights
        }

        def update_func(opts):
            return opts['weights'] + norm(loc=0, scale=0.1).rvs(size=4)

        return self._optimize_policy(
            env, update_func, opts, n_samples_per_rollout, n_iter, early_stop)


class PGLinearAgent(object):
    """A linear agent implementing policy gradient optimization."""

    def __init__(self, learning_rate=0.001):
        self._dense_layer = tf.layers.Dense(2)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

    def get_params(self):
        return self._dense_layer.weights[0].numpy()

    def get_action_logits(self, observations):
        return self._dense_layer(observations)

    def choose_action(self, observation):
        observation_tensor = tf.reshape(observation, [1, 4])
        logits = self.get_action_logits(observation_tensor)
        action_probabilities = tf.nn.softmax(logits)
        return bernoulli.rvs(action_probabilities[0][1])

    def loss(self, observations, actions, q_values):
        observations = tf.convert_to_tensor(observations)
        logits = self.get_action_logits(observations)
        negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=logits)

        weighted_negative_likelihoods = negative_likelihoods * q_values
        return tf.reduce_mean(weighted_negative_likelihoods)

    def policy_rollout(self, env, n_samples, time_horizon=200):
        observations = []
        actions = []
        rewards = np.zeros((n_samples, time_horizon))
        n_timesteps = []

        for i in range(n_samples):
            observation = env.reset()
            for t in range(time_horizon):
                action = self.choose_action(observation)
                observations.append(observation)
                actions.append(action)

                observation, reward, done, info = env.step(action)
                rewards[i, t] = reward
                if done:
                    n_timesteps.append(t+1)
                    break

        q_values = get_q_values(rewards, n_timesteps)
        mean_reward = np.mean(np.sum(rewards, axis=1))

        return (
            np.array(observations),
            np.array(actions),
            np.array(q_values),
            mean_reward)

    def optimize_policy(self,
                        env,
                        n_samples_per_rollout,
                        time_horizon=200,
                        n_iter=1000,
                        early_stop=True,
                        verbose=True):
        grads = tfe.implicit_gradients(self.loss)
        for i in range(n_iter):
            observations, actions, q_values, mean_reward = self.policy_rollout(
                env, n_samples_per_rollout, time_horizon)

            if verbose and (i+1) % 10 == 0:
                print('Iteration {0}. Average reward: {1}'
                      .format(i+1, mean_reward))

            if mean_reward == 200 and early_stop:
                return i+1

            self.optimizer.apply_gradients(
                grads(observations, actions, q_values))

        return n_iter


if __name__ == '__main__':
    tfe.enable_eager_execution()
    solver = sys.argv[1]

    env = gym.make('CartPole-v0')
    t = time.time()

    if solver == 'pgo':
        agent = PGLinearAgent(learning_rate=0.1)
        n_iter = agent.optimize_policy(
            env,
            n_samples_per_rollout=10,
            n_iter=80,
            early_stop=False)

        params = agent.get_params()

    if solver == 'rgo':
        agent = SimpleLinearAgent()
        param_distributions = [
            uniform(loc=-15, scale=30),
            uniform(loc=-1.5, scale=3),
            uniform(loc=-10, scale=20),
            uniform(loc=-1, scale=2)]

        n_iter = agent.rgo(env, param_distributions)
        params = agent.weights

    if solver == 'hco':
        agent = SimpleLinearAgent()
        n_iter = agent.hco(env)
        params = agent.weights

    print('\nElapsed time: {}'.format(time.time()-t))
    print('Optimal parameters found:\n {}'.format(params))
    print('Number of iterations: {}'.format(n_iter))
    print('Average reward: {}'
          .format(get_policy_mean_reward(agent, env, n_iter=100)))

    for _ in range(4):
        render_agent(env, agent)

    env.close()
