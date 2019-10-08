import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model

import random
import gym

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.layer_a1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_a2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_a3 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, state):
        layer_a1 = self.layer_a1(state)
        layer_a2 = self.layer_a2(layer_a1)
        layer_a3 = self.layer_a3(layer_a2)
        logits = self.logits(layer_a3)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.layer_c1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_c2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_c3 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        layer_c1 = self.layer_c1(state)
        layer_c2 = self.layer_c2(layer_c1)
        layer_c3 = self.layer_c3(layer_c2)
        value = self.value(layer_c3)
        return value

class Agent:
    def __init__(self):
        self.lr = 0.01
        self.gamma = 0.99

        self.policy = ActorModel()
        self.value = CriticModel()
        self.policy_opt = optimizers.Adam(lr=self.lr, )
        self.value_opt = optimizers.Adam(lr=self.lr, )
        
        self.rollout = 128
        self.batch_size = 128
        self.state_size = 4
        self.action_size = 2

    def get_action(self, state):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy = self.policy(state)
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

        critic_variable = self.value.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)
            current_value = self.value(tf.convert_to_tensor(state, dtype=tf.float32))
            next_value = self.value(tf.convert_to_tensor(next_state, dtype=tf.float32))
            current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)
            target = tf.stop_gradient(self.gamma * (1-tf.convert_to_tensor(done, dtype=tf.float32)) * next_value + tf.convert_to_tensor(reward, dtype=tf.float32))
            value_loss = tf.reduce_mean(tf.square(target - current_value) * 0.5)

        value_grads = tape_critic.gradient(value_loss, critic_variable)
        self.value_opt.apply_gradients(zip(value_grads, critic_variable))

        actor_variable = self.policy.trainable_variables
        with tf.GradientTape() as tape_actor:
            tape_actor.watch(actor_variable)
            policy = self.policy(tf.convert_to_tensor(state, dtype=tf.float32))
            action = tf.convert_to_tensor(action, dtype=tf.int32)
            onehot_action = tf.one_hot(action, self.action_size)
            action_policy = tf.reduce_sum(onehot_action * policy, axis=1)
            adv = tf.stop_gradient(target - current_value)
            pi_loss = -tf.reduce_mean(tf.math.log(action_policy) * adv) - tf.reduce_mean(- policy * tf.math.log(policy)) * 0.01

        actor_grads = tape_actor.gradient(pi_loss, actor_variable)
        self.policy_opt.apply_gradients(zip(actor_grads, actor_variable))


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