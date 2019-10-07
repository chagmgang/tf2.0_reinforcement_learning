import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque

import random
import gym

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(2)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value

class Agent:
    def __init__(self):
        # hyper parameters
        self.lr =0.001
        self.gamma = 0.99

        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = optimizers.Adam(lr=self.lr, )

        self.batch_size = 64
        self.state_size = 4
        self.action_size = 2

        self.memory = deque(maxlen=2000)

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state, epsilon):
        q_value = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value) 
        return action, q_value

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = [i[0] for i in mini_batch]
        actions = [i[1] for i in mini_batch]
        rewards = [i[2] for i in mini_batch]
        next_states = [i[3] for i in mini_batch]
        dones = [i[4] for i in mini_batch]

        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))
            next_action = tf.argmax(target_q, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * target_q, axis=1)

            target_value = (1-dones) * self.gamma * target_value + rewards

            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * main_q, axis=1)

            error = tf.square(main_value - target_value) * 0.5
            error = tf.reduce_mean(error)
            
        dqn_grads = tape.gradient(error, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))


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

                if len(self.memory) > self.batch_size:
                    self.update()
                    if step % 20 == 0:
                        self.update_target()
            print(episode, score)

if __name__ == '__main__':
    agent = Agent()
    agent.run()
