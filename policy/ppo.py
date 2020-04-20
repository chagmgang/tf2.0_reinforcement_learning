import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model

import random
import copy
import gym

def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target

class PPO(Model):
    def __init__(self):
        super(PPO, self).__init__()
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
        self.lamda = 0.95

        self.ppo = PPO()
        self.opt = optimizers.Adam(lr=self.lr, )
        
        self.rollout = 128
        self.batch_size = 128
        self.state_size = 4
        self.action_size = 2
        self.epoch = 3
        self.ppo_eps = 0.2
        self.normalize = True

    def get_action(self, state):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.ppo(state)
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def update(self, state, next_state, reward, done, action):

        old_policy, current_value = self.ppo(tf.convert_to_tensor(state, dtype=tf.float32))
        _, next_value = self.ppo(tf.convert_to_tensor(next_state, dtype=tf.float32))
        current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)
        current_value, next_value = np.array(current_value), np.array(next_value)
        old_policy = np.array(old_policy)
        
        adv, target = get_gaes(
            rewards=np.array(reward),
            dones=np.array(done),
            values=current_value,
            next_values=next_value,
            gamma=self.gamma,
            lamda=self.lamda,
            normalize=self.normalize)

        for _ in range(self.epoch):
            sample_range = np.arange(self.rollout)
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]
            
            batch_state = [state[i] for i in sample_idx]
            batch_done = [done[i] for i in sample_idx]
            batch_action = [action[i] for i in sample_idx]
            batch_target = [target[i] for i in sample_idx]
            batch_adv = [adv[i] for i in sample_idx]
            batch_old_policy = [old_policy[i] for i in sample_idx]

            ppo_variable = self.ppo.trainable_variables

            with tf.GradientTape() as tape:
                tape.watch(ppo_variable)
                train_policy, train_current_value = self.ppo(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                train_current_value = tf.squeeze(train_current_value)
                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                train_action = tf.convert_to_tensor(batch_action, dtype=tf.int32)
                train_old_policy = tf.convert_to_tensor(batch_old_policy, dtype=tf.float32)

                entropy = tf.reduce_mean(-train_policy * tf.math.log(train_policy + 1e-8)) * 0.1
                onehot_action = tf.one_hot(train_action, self.action_size)
                selected_prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
                selected_old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=1)
                logpi = tf.math.log(selected_prob + 1e-8)
                logoldpi = tf.math.log(selected_old_prob + 1e-8)

                ratio = tf.exp(logpi - logoldpi)
                clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1-self.ppo_eps, clip_value_max=1+self.ppo_eps)
                minimum = tf.minimum(tf.multiply(train_adv, clipped_ratio), tf.multiply(train_adv, ratio))
                pi_loss = -tf.reduce_mean(minimum) + entropy

                value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))

                total_loss = pi_loss + value_loss

            grads = tape.gradient(total_loss, ppo_variable)
            self.opt.apply_gradients(zip(grads, ppo_variable))

    def run(self):

        env = gym.make('CartPole-v1')
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
                    if score == 500:
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