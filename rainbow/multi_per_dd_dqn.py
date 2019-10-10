import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque

import collections
import random
import gym


class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

class n_step_memory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.state = collections.deque(maxlen=int(maxlen))
        self.action = collections.deque(maxlen=int(maxlen))
        self.reward = collections.deque(maxlen=int(maxlen))
        self.next_state = collections.deque(maxlen=int(maxlen))
        self.done = collections.deque(maxlen=int(maxlen))

    def append(self, state, next_state, reward, done, action):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)

    def sample(self):
        if len(self.state) == int(self.maxlen):
            done = [self.done[i] for i in range(self.maxlen-1)]
            if True in done:
                return None
            return {'state': np.stack(self.state),
                    'next_state': np.stack(self.next_state),
                    'reward': np.stack(self.reward),
                    'done': np.stack(self.done),
                    'action': np.stack(self.action)}
        else:
            return None

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer2 = tf.keras.layers.Dense(64, activation='relu')
        self.state = tf.keras.layers.Dense(2)
        self.action = tf.keras.layers.Dense(2)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        state = self.state(layer2)
        action = self.action(layer2)
        mean = tf.keras.backend.mean(action, keepdims=True)
        advantage = (action - mean)
        value = state + advantage        
        return value

class Agent:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.99

        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = optimizers.Adam(lr=self.lr, )

        self.batch_size = 64
        self.state_size = 4
        self.action_size = 2
        self.n_step = 5

        self.n_step_memory = n_step_memory(maxlen=self.n_step)
        self.memory = Memory(capacity=2000)

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state, epsilon):
        q_value = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action, q_value

    def append_sample(self, state, action, reward, next_state, done):
        self.n_step_memory.append(state, next_state, reward, done, action)
        n_step_sample = self.n_step_memory.sample()
        if not n_step_sample is None:
            state = n_step_sample['state'][0]
            next_state = n_step_sample['next_state'][-1]
            reward = n_step_sample['reward']
            done = n_step_sample['done'][-1]
            action = n_step_sample['action'][0]

            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)

            main_current_q = np.array(self.dqn_model(state_tensor))[0]
            main_next_q = np.array(self.dqn_model(next_state_tensor))[0]
            target_next_q = np.array(self.dqn_target(next_state_tensor))[0]
            next_action = np.argmax(main_next_q)
            target_next_q = target_next_q[next_action]
            
            main_current_q = main_current_q[action]

            reward = np.sum([np.power(self.gamma, i) * r for i, r in enumerate(reward)])

            td_error = np.abs(reward + np.power(self.gamma, self.n_step) * target_next_q * (1-done) - main_current_q)

            self.memory.add(td_error, (state, action, reward, next_state, done))    

    def update(self):
        minibatch, idxs, IS_weight = self.memory.sample(self.batch_size)
        minibatch = np.array(minibatch)
        state = [i[0] for i in minibatch]
        action = [i[1] for i in minibatch]
        reward = [i[2] for i in minibatch]
        next_state = [i[3] for i in minibatch]
        done = [i[4] for i in minibatch]

        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            reward = tf.convert_to_tensor(reward, dtype=tf.float32)
            action = tf.convert_to_tensor(action, dtype=tf.int32)
            done = tf.convert_to_tensor(np.stack(done), dtype=tf.float32)

            target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(next_state), dtype=tf.float32))
            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(next_state), dtype=tf.float32))
            main_q = tf.stop_gradient(main_q)
            next_action = tf.argmax(main_q, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * target_q, axis=1)

            target_value = (1-done) * np.power(self.gamma, self.n_step) * target_value + reward

            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(state), dtype=tf.float32))
            main_value = tf.reduce_sum(tf.one_hot(action, self.action_size) * main_q, axis=1)

            error = tf.square(main_value - target_value) * 0.5
            error = error * tf.convert_to_tensor(IS_weight, dtype=tf.float32)
            error = tf.reduce_mean(error)

        dqn_grads = tape.gradient(error, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

        state_value = np.array(self.dqn_model(tf.convert_to_tensor(np.vstack(state), dtype=tf.float32)))
        state_value = np.array([sv[a] for a, sv in zip(np.array(action), state_value)])

        td_error = np.abs(target_value - state_value)

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_error[i])  

    def run(self):

        env = gym.make('CartPole-v1')
        episode = 0
        step = 0

        while True:
            state = env.reset()
            done = False
            episode += 1
            epsilon = 1 / (episode * 0.1 + 1)
            score = 0
            while not done:
                step += 1
                action, q_value = self.get_action(state, epsilon)
                next_state, reward, done, info = env.step(action)

                self.append_sample(state, action, reward, next_state, done)
                
                score += reward

                state = next_state

                if step > 1000:
                    self.update()
                    if step % 20 == 0:
                        self.update_target()

            print(episode, score)
        


if __name__ == '__main__':
    agent = Agent()
    agent.run()