import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model

import random
import gym

class A2C(Model):
    def __init__(self):
        super(A2C, self).__init__()
        self.layer1 = tf.keras.layers.Dense(128, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.layer_a1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_c1 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(2, activation='softmax')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        
        layer_a1 = self.layer_a1(layer2)
        logits = self.logits(layer_a1)

        layer_c1 = self.layer_c1(layer2)
        value = self.value(layer_c1)

        return logits, value

class Agent:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.99

        self.a2c = A2C()
        self.opt = optimizers.Adam(lr=self.lr, )
        
        self.rollout = 32
        self.batch_size = 32
        self.state_size = 4
        self.action_size = 2

    def get_action(self, state):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.a2c(state)
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def update(self, state, next_state, reward, done, action):
        sample_range = np.arange(self.rollout)
        np.random.shuffle(sample_range)
        sample_idx = sample_range[:self.batch_size]

        state = [state[i] for i in sample_idx]
        next_state = [next_state[i] for i in sample_idx]
        reward = [reward[i] for i in sample_idx]
        done = [done[i] for i in sample_idx]
        action = [action[i] for i in sample_idx]

        a2c_variable = self.a2c.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(a2c_variable)
            _, current_value = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
            _, next_value = self.a2c(tf.convert_to_tensor(next_state, dtype=tf.float32))
            current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)
            target = tf.stop_gradient(self.gamma * (1-tf.convert_to_tensor(done, dtype=tf.float32)) * next_value + tf.convert_to_tensor(reward, dtype=tf.float32))
            value_loss = tf.reduce_mean(tf.square(target - current_value) * 0.5)

            policy, _  = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
            entropy = tf.reduce_mean(- policy * tf.math.log(policy+1e-8)) * 0.1
            action = tf.convert_to_tensor(action, dtype=tf.int32)
            onehot_action = tf.one_hot(action, self.action_size)
            action_policy = tf.reduce_sum(onehot_action * policy, axis=1)
            adv = tf.stop_gradient(target - current_value)
            pi_loss = -tf.reduce_mean(tf.math.log(action_policy+1e-8) * adv) - entropy

            total_loss = pi_loss + value_loss

        grads = tape.gradient(total_loss, a2c_variable)
        self.opt.apply_gradients(zip(grads, a2c_variable))

    def run(self):

        env = gym.make('CartPole-v0')
        state = env.reset()
        episode = 0
        score = 0

        while True:
            
            state_list, next_state_list = [], []
            reward_list, done_list, action_list = [], [], []

            for _ in range(self.rollout):
                
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                score += reward

                if done:
                    if score == 200:
                        reward = 1
                    else:
                        reward = -1
                else:
                    reward = 0

                state_list.append(state)
                next_state_list.append(next_state)
                reward_list.append(reward)
                done_list.append(done)
                action_list.append(action)

                state = next_state

                if done:
                    print(episode, score)
                    state = env.reset()
                    episode += 1
                    score = 0
            self.update(
                state=state_list, next_state=next_state_list,
                reward=reward_list, done=done_list, action=action_list)


if __name__ == '__main__':
    agent = Agent()
    agent.run()